use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, Subcommand, ValueEnum};
use codehelm_core::{
    Agent, AnthropicProvider, ApprovalHandler, BuildApproval, Config, ModelProvider,
    OpenAiProvider, Provider, ReadOnlyApproval, SessionRecorder, SessionStore, WorkspaceTools,
    config::ConfigOverrides, list_checkpoints, load_config, restore_checkpoint,
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
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = std::env::current_dir()?;
    if matches!(cli.command, Some(Command::Init)) {
        initialize(&cwd)?;
        println!(
            "Initialized {}",
            cwd.join(".codehelm/config.json").display()
        );
        return Ok(());
    }

    let config = load_config(
        &cwd,
        ConfigOverrides {
            provider: cli.provider.map(Into::into),
            model: cli.model,
            max_turns: cli.max_turns,
        },
    )?;

    match cli.command.unwrap_or(Command::Chat) {
        Command::Config => println!("{}", serde_json::to_string_pretty(&config)?),
        Command::Checkpoints => {
            let checkpoints = list_checkpoints(&cwd)?;
            if checkpoints.is_empty() {
                println!("No recoverable checkpoints.");
            } else {
                println!("{}", checkpoints.join("\n"));
            }
        }
        Command::Rollback { checkpoint } => {
            let id = if checkpoint == "latest" {
                list_checkpoints(&cwd)?
                    .into_iter()
                    .next()
                    .ok_or("no recoverable checkpoints")?
            } else {
                checkpoint
            };
            let count = restore_checkpoint(&cwd, &id, &config.permissions)?;
            println!("Restored {count} file(s) from checkpoint {id}.");
        }
        Command::Plan { task } => run_agent(&cwd, &config, &task, "plan", cli.json).await?,
        Command::Exec { task } => run_agent(&cwd, &config, &task, "exec", cli.json).await?,
        Command::Build { task } => run_agent(&cwd, &config, &task, "build", cli.json).await?,
        Command::Resume { session, task } => {
            let store = SessionStore::load(&cwd, session.as_deref().unwrap_or("latest"))?;
            let mode = store.session().mode.clone();
            let task =
                task.unwrap_or_else(|| "Continue the previous task from the saved context.".into());
            run_agent_with_session(&cwd, &config, &task, &mode, cli.json, Some(store)).await?;
        }
        Command::Review { focus } => {
            let task = focus.map_or_else(
                || "Review the current repository changes.".into(),
                |focus| format!("Review the current repository changes, focusing on {focus}."),
            );
            run_agent(&cwd, &config, &task, "review", cli.json).await?;
        }
        command => print_migration_status(command, &config, cli.json),
    }
    Ok(())
}

async fn run_agent(
    cwd: &Path,
    config: &Config,
    task: &str,
    mode: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    run_agent_with_session(cwd, config, task, mode, json, None).await
}

async fn run_agent_with_session(
    cwd: &Path,
    config: &Config,
    task: &str,
    mode: &str,
    json: bool,
    saved: Option<SessionStore>,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider: Box<dyn ModelProvider> = match config.provider {
        Provider::Openai | Provider::OpenaiCompatible => {
            let key = api_key("OPENAI_API_KEY")?;
            let provider = if let Some(base_url) = &config.base_url {
                OpenAiProvider::with_base_url(key, &config.model, base_url)
            } else {
                OpenAiProvider::new(key, &config.model)
            };
            Box::new(provider)
        }
        Provider::Anthropic => {
            let key = api_key("ANTHROPIC_API_KEY")?;
            let provider = if let Some(base_url) = &config.base_url {
                AnthropicProvider::with_base_url(key, &config.model, base_url)
            } else {
                AnthropicProvider::new(key, &config.model)
            };
            Box::new(provider)
        }
        Provider::Ollama => return Err("the native Ollama adapter is not implemented yet".into()),
    };
    let mut tools = WorkspaceTools::new(
        cwd,
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
        Box::new(BuildApproval)
    } else {
        Box::new(ReadOnlyApproval)
    };
    let instructions = project_instructions(cwd)?;
    let system = format!(
        "You are CodeHelm, a careful coding agent. Mode: {mode}. Inspect before editing, make focused changes, and verify your work. The rollback_edits tool can restore every file changed during this run. Never invent tool results.\n\nProject instructions:\n{instructions}"
    );
    let resumed = saved.is_some();
    let store = match saved {
        Some(store) => store,
        None => SessionStore::create(cwd, mode, config.provider.to_string(), &config.model)?,
    };
    let recorder = SessionRecorder::new(store);
    let session_event = AgentEvent::SessionStarted {
        id: recorder.id(),
        resumed,
    };
    recorder.append_event(&session_event)?;
    render_event(session_event, json);
    let initial_items = recorder.items();
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
    );
    let agent = if resumed {
        agent.with_items(initial_items)
    } else {
        agent
    };
    let mut agent = agent.with_store(recorder);
    agent.run(task).await?;
    Ok(())
}

fn api_key(provider_variable: &str) -> Result<String, Box<dyn std::error::Error>> {
    env::var("CODEHELM_API_KEY")
        .or_else(|_| env::var(provider_variable))
        .map_err(|_| format!("set {provider_variable} or CODEHELM_API_KEY").into())
}

fn project_instructions(cwd: &Path) -> io::Result<String> {
    let mut sections = Vec::new();
    for name in ["AGENTS.md", "CLAUDE.md"] {
        match fs::read_to_string(cwd.join(name)) {
            Ok(content) => sections.push(format!("## {name}\n{content}")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(if sections.is_empty() {
        "(none)".into()
    } else {
        sections.join("\n\n")
    })
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
        AgentEvent::ModelComplete { .. } => println!(),
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
