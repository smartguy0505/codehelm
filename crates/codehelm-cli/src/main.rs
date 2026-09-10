use std::{
    env, fs,
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, Subcommand, ValueEnum};
use codehelm_core::{
    Agent, AnthropicProvider, ApprovalHandler, Config, ModelProvider, OllamaProvider,
    OpenAiProvider, PolicyApproval, Provider, ReadOnlyApproval, RetryPolicy, SessionRecorder,
    SessionStore, WorkspaceTools, config::ConfigOverrides, discover_workspace, list_checkpoints,
    load_config, load_project_instructions, restore_checkpoint,
};
use codehelm_protocol::AgentEvent;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "codehelm",
    version,
    about = "Provider-neutral, safety-first agentic coding CLI"
)]
struct Cli {
    #[arg(long, global = true, value_enum)]
    provider: Option<ProviderArg>,
    #[arg(long, global = true)]
    model: Option<String>,
    #[arg(long, global = true)]
    max_turns: Option<usize>,
    #[arg(long, global = true)]
    json: bool,
    #[arg(short = 'y', long, global = true)]
    yes: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Clone, ValueEnum)]
enum ProviderArg {
    Openai,
    Anthropic,
    Ollama,
    OpenaiCompatible,
}

impl From<ProviderArg> for Provider {
    fn from(value: ProviderArg) -> Self {
        match value {
            ProviderArg::Openai => Self::Openai,
            ProviderArg::Anthropic => Self::Anthropic,
            ProviderArg::Ollama => Self::Ollama,
            ProviderArg::OpenaiCompatible => Self::OpenaiCompatible,
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    Chat,
    Build {
        task: String,
    },
    Plan {
        task: String,
    },
    Review {
        focus: Option<String>,
    },
    Exec {
        task: String,
    },
    Resume {
        session: Option<String>,
        task: Option<String>,
    },
    Checkpoints,
    Rollback {
        checkpoint: String,
    },
    Init,
    Config,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("codehelm: {error}");
            if matches!(
                error.downcast_ref::<codehelm_core::AgentError>(),
                Some(codehelm_core::AgentError::Cancelled)
            ) {
                ExitCode::from(130)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = fs::canonicalize(std::env::current_dir()?)?;
    if matches!(cli.command, Some(Command::Init)) {
        initialize(&cwd)?;
        println!(
            "Initialized {}",
            cwd.join(".codehelm/config.json").display()
        );
        return Ok(());
    }
    let root = discover_workspace(&cwd)?;

    let config = load_config(
        &root,
        ConfigOverrides {
            provider: cli.provider.map(Into::into),
            model: cli.model,
            max_turns: cli.max_turns,
        },
    )?;

    match cli.command.unwrap_or(Command::Chat) {
        Command::Config => println!("{}", serde_json::to_string_pretty(&config)?),
        Command::Checkpoints => {
            let checkpoints = list_checkpoints(&root)?;
            if checkpoints.is_empty() {
                println!("No recoverable checkpoints.");
            } else {
                println!("{}", checkpoints.join("\n"));
            }
        }
        Command::Rollback { checkpoint } => {
            let id = if checkpoint == "latest" {
                list_checkpoints(&root)?
                    .into_iter()
                    .next()
                    .ok_or("no recoverable checkpoints")?
            } else {
                checkpoint
            };
            let count = restore_checkpoint(&root, &id, &config.permissions)?;
            println!("Restored {count} file(s) from checkpoint {id}.");
        }
        Command::Plan { task } => {
            run_agent(&root, &cwd, &config, &task, "plan", cli.json, cli.yes).await?
        }
        Command::Exec { task } => {
            run_agent(&root, &cwd, &config, &task, "exec", cli.json, cli.yes).await?
        }
        Command::Build { task } => {
            run_agent(&root, &cwd, &config, &task, "build", cli.json, cli.yes).await?
        }
        Command::Resume { session, task } => {
            let store = SessionStore::load(&root, session.as_deref().unwrap_or("latest"))?;
            let mode = store.session().mode.clone();
            let task =
                task.unwrap_or_else(|| "Continue the previous task from the saved context.".into());
            run_agent_with_session(
                &root,
                &cwd,
                &config,
                &task,
                &mode,
                RunOptions {
                    json: cli.json,
                    assume_yes: cli.yes,
                },
                Some(store),
            )
            .await?;
        }
        Command::Review { focus } => {
            let task = focus.map_or_else(
                || "Review the current repository changes.".into(),
                |focus| format!("Review the current repository changes, focusing on {focus}."),
            );
            run_agent(&root, &cwd, &config, &task, "review", cli.json, cli.yes).await?;
        }
        command => print_migration_status(command, &config, cli.json),
    }
    Ok(())
}

async fn run_agent(
    root: &Path,
    cwd: &Path,
    config: &Config,
    task: &str,
    mode: &str,
    json: bool,
    assume_yes: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    run_agent_with_session(
        root,
        cwd,
        config,
        task,
        mode,
        RunOptions { json, assume_yes },
        None,
    )
    .await
}

#[derive(Clone, Copy)]
struct RunOptions {
    json: bool,
    assume_yes: bool,
}

async fn run_agent_with_session(
    root: &Path,
    cwd: &Path,
    config: &Config,
    task: &str,
    mode: &str,
    options: RunOptions,
    saved: Option<SessionStore>,
) -> Result<(), Box<dyn std::error::Error>> {
    let RunOptions { json, assume_yes } = options;
    let retry = RetryPolicy {
        max_retries: config.provider_max_retries,
        base_delay_ms: config.provider_retry_base_ms,
    };
    let provider: Box<dyn ModelProvider> = match config.provider {
        Provider::Openai | Provider::OpenaiCompatible => {
            let key = api_key("OPENAI_API_KEY")?;
            let provider = if let Some(base_url) = &config.base_url {
                OpenAiProvider::with_base_url(key, &config.model, base_url)
            } else {
                OpenAiProvider::new(key, &config.model)
            };
            Box::new(provider.with_retry_policy(retry))
        }
        Provider::Anthropic => {
            let key = api_key("ANTHROPIC_API_KEY")?;
            let provider = if let Some(base_url) = &config.base_url {
                AnthropicProvider::with_base_url(key, &config.model, base_url)
            } else {
                AnthropicProvider::new(key, &config.model)
            };
            Box::new(provider.with_retry_policy(retry))
        }
        Provider::Ollama => {
            let base_url = config
                .base_url
                .clone()
                .or_else(|| env::var("OLLAMA_HOST").ok());
            let provider = if let Some(base_url) = base_url {
                OllamaProvider::with_base_url(&config.model, base_url)
            } else {
                OllamaProvider::new(&config.model)
            };
            Box::new(provider.with_retry_policy(retry))
        }
    };
    let mut tools = WorkspaceTools::new(
        root,
        config.permissions.clone(),
        config.max_tool_output_chars,
    )?;
    let writable = mode == "build";
    if writable {
        tools = tools
            .enable_edits()
            .enable_commands(config.command_timeout_ms);
    }
    let approvals: Box<dyn ApprovalHandler> = if writable {
        let policy = config.permissions.clone();
        Box::new(PolicyApproval::new(
            policy,
            assume_yes,
            move |description: &str| prompt_approval(description, !json),
        ))
    } else {
        Box::new(ReadOnlyApproval)
    };
    let instructions = load_project_instructions(root, cwd, config.max_instruction_chars)?;
    let working_directory = cwd.strip_prefix(root).unwrap_or(cwd).display();
    let system = format!(
        "You are CodeHelm, a careful coding agent. Mode: {mode}. Workspace paths are relative to the project root. Invocation directory: {working_directory}. Inspect before editing, make focused changes, and verify your work. The rollback_edits tool can restore every file changed during this run. Never invent tool results.\n\nProject instructions:\n{instructions}"
    );
    let resumed = saved.is_some();
    let store = match saved {
        Some(store) => store,
        None => SessionStore::create(root, mode, config.provider.to_string(), &config.model)?,
    };
    let recorder = SessionRecorder::new(store);
    let session_event = AgentEvent::SessionStarted {
        id: recorder.id(),
        resumed,
    };
    recorder.append_event(&session_event)?;
    render_event(session_event, json);
    let initial_items = recorder.items();
    let cancel_recorder = recorder.clone();
    let event_recorder = recorder.clone();
    let mut events = move |event: AgentEvent| {
        if let Err(error) = event_recorder.append_event(&event) {
            eprintln!("codehelm: {error}");
        }
        render_event(event, json);
    };
    let agent = Agent::new(
        provider,
        tools,
        approvals,
        &mut events,
        system,
        config.max_turns,
    )
    .with_token_budget(config.max_total_tokens);
    let agent = if resumed {
        agent.with_items(initial_items)
    } else {
        agent
    };
    let mut agent = agent.with_store(recorder);
    tokio::select! {
        result = agent.run(task) => { result?; }
        signal = tokio::signal::ctrl_c() => {
            signal?;
            let event = AgentEvent::Cancelled { reason: "interrupt".into() };
            cancel_recorder.append_event(&event)?;
            render_event(event, json);
            return Err(codehelm_core::AgentError::Cancelled.into());
        }
    }
    Ok(())
}

fn prompt_approval(description: &str, interactive: bool) -> bool {
    if !interactive || !io::stdin().is_terminal() {
        return false;
    }
    eprint!("Approve {description}? [y/N] ");
    let _ = io::stderr().flush();
    let mut answer = String::new();
    io::stdin().read_line(&mut answer).is_ok()
        && matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

fn api_key(provider_variable: &str) -> Result<String, Box<dyn std::error::Error>> {
    env::var("CODEHELM_API_KEY")
        .or_else(|_| env::var(provider_variable))
        .map_err(|_| format!("set {provider_variable} or CODEHELM_API_KEY").into())
}

fn render_event(event: AgentEvent, json: bool) {
    if json {
        println!(
            "{}",
            serde_json::to_string(&event).expect("event serializes")
        );
        return;
    }
    match event {
        AgentEvent::ModelDelta { delta } => {
            print!("{delta}");
            let _ = io::stdout().flush();
        }
        AgentEvent::SessionStarted { id, resumed } => {
            eprintln!("session {id}{}", if resumed { " (resumed)" } else { "" });
        }
        AgentEvent::Cancelled { reason } => eprintln!("cancelled: {reason}"),
        AgentEvent::ModelComplete { .. } => println!(),
        AgentEvent::ProviderRetry {
            attempt,
            delay_ms,
            reason,
        } => eprintln!("provider retry {attempt} in {delay_ms}ms: {reason}"),
        AgentEvent::Usage { usage } => eprintln!(
            "usage: {} input + {} output = {} tokens",
            usage.input_tokens,
            usage.output_tokens,
            usage.total()
        ),
        AgentEvent::ToolStart { tool, reason, .. } => {
            if let Some(reason) = reason {
                eprintln!("→ {tool}: {reason}");
            } else {
                eprintln!("→ {tool}");
            }
        }
        AgentEvent::ToolDenied { tool, reason } => eprintln!("denied {tool}: {reason}"),
        _ => {}
    }
}

fn initialize(cwd: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let directory = cwd.join(".codehelm");
    fs::create_dir_all(&directory)?;
    write_new(
        directory.join("config.json"),
        &format!("{}\n", serde_json::to_string_pretty(&Config::default())?),
    )?;
    write_new(
        cwd.join("AGENTS.md"),
        "# Agent instructions\n\nDescribe project conventions, test commands, and architectural constraints here.\n",
    )?;
    Ok(())
}

fn write_new(path: PathBuf, content: &str) -> std::io::Result<()> {
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut file) => std::io::Write::write_all(&mut file, content.as_bytes()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error),
    }
}

fn print_migration_status(command: Command, config: &Config, json: bool) {
    let name = match command {
        Command::Chat => "chat",
        Command::Build { .. } => "build",
        Command::Plan { .. } => "plan",
        Command::Review { .. } => "review",
        Command::Exec { .. } => "exec",
        Command::Init
        | Command::Config
        | Command::Checkpoints
        | Command::Rollback { .. }
        | Command::Resume { .. } => {
            unreachable!()
        }
    };
    if json {
        println!(
            "{}",
            serde_json::json!({
                "type": "migration_status",
                "command": name,
                "provider": config.provider,
                "model": config.model,
                "implemented": false,
                "message": "Rust command surface is ready; the agent runtime is the next migration slice"
            })
        );
    } else {
        println!(
            "Rust command '{name}' is configured for {}/{}.",
            config.provider, config.model
        );
        println!(
            "The Rust agent runtime is the next migration slice; use the JavaScript CLI for live agent runs meanwhile."
        );
    }
}
