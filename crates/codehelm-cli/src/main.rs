use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, Subcommand, ValueEnum};
use codehelm_core::{Config, Provider, config::ConfigOverrides, load_config};
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
    Build { task: String },
    Plan { task: String },
    Review { focus: Option<String> },
    Exec { task: String },
    Resume { session: Option<String> },
    Init,
    Config,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("codehelm: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
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
        command => print_migration_status(command, &config, cli.json),
    }
    Ok(())
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
        Command::Resume { .. } => "resume",
        Command::Init | Command::Config => unreachable!(),
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
