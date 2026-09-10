# CodeHelm Agent CLI

CodeHelm is a provider-neutral, safety-first coding agent that runs in your terminal. It combines useful patterns from modern agent CLIs: plan/build/review modes, explicit permissions, Git-aware tools, project instructions, resumable sessions, local-model support, and machine-readable output.

This repository is migrating from its dependency-free JavaScript MVP to a production-oriented Rust implementation. The JavaScript agent remains usable while Rust components reach feature parity.

## Rust migration

The Rust workspace currently provides:

- A typed CLI surface for chat, plan, build, review, exec, resume, init, and config
- Layered global, project, and command-line configuration
- Provider and operating-mode types
- Sensitive-path policies and command risk classification
- Canonical workspace containment with symlink escape protection
- A stable, serializable agent action and event protocol
- A provider-neutral async agent loop with incremental model events
- Injected provider, tool, approval, and event boundaries with deterministic tests
- Native OpenAI Responses and Anthropic Messages streaming adapters, wired to the Rust CLI
- Structured tool-call identity preserved across provider turns
- Multiple tool calls per model turn with deterministic authorization and execution
- Workspace-contained Rust tools for listing, reading, and regex search

Build and inspect the Rust CLI:

```bash
cargo build --release
./target/release/codehelm --help
./target/release/codehelm config
OPENAI_API_KEY="..." ./target/release/codehelm plan "explain this architecture"
```

The Rust CLI supports live `plan`, `review`, `exec`, and transactional `build` runs through OpenAI, Anthropic, and local Ollama models. Build mode snapshots original file contents, writes atomically, and can roll back every edit from the current run.

Rust build mode can also run configured allowlisted commands directly, without invoking a shell. Commands are bounded by configured timeout and output limits; OS-level filesystem and network isolation remains on the roadmap.

Commands outside `allowCommands` require an interactive confirmation. Headless runs deny these ask-level actions unless `--yes` is supplied; denylisted commands and protected paths remain blocked even with `--yes`. Paths matching optional `permissions.askWrite` patterns use the same approval flow.

Every Rust build stores original file contents under `.codehelm/checkpoints` before the first write. Interrupted runs can be recovered with `codehelm checkpoints` and `codehelm rollback latest`.

Rust sessions are atomically persisted under `.codehelm/sessions` after every conversation transition, with an append-only NDJSON event journal. Continue the latest run with `codehelm resume latest "continue with the failing test"`.

Pressing Ctrl-C cancels an active provider stream or command, journals the interruption, preserves the last durable conversation state and edit checkpoint, and exits with status 130.

Transient connection failures, timeouts, rate limits, and server errors are retried before streaming begins with bounded exponential backoff. Retry attempts are journaled and exposed in NDJSON output; partial streams are never replayed automatically.

When launched from a nested directory, the Rust CLI discovers the nearest CodeHelm or Git project root. It loads `AGENTS.md` and `CLAUDE.md` files from the root down to the invocation directory, with deeper instructions applied later and a configurable `maxInstructionChars` context limit.

## Features

- OpenAI, Anthropic, Ollama, and OpenAI-compatible providers
- Interactive terminal sessions and non-interactive execution
- Read-only `plan` and `review` modes
- Workspace-contained file access, including symlink escape protection
- Sensitive-file and dangerous-command deny rules
- Read, search, focused replacement, file creation, shell, and Git tools
- `AGENTS.md` and `CLAUDE.md` project instructions
- Persistent sessions under `.codehelm/sessions`
- NDJSON events for scripts and CI
- Zero runtime dependencies; Node.js 20+

## Quick start

```bash
cd codehelm
npm link
cd /path/to/your/project
codehelm init
export OPENAI_API_KEY="your-key"
codehelm plan "add passwordless authentication"
codehelm build "implement the approved authentication plan"
codehelm review
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY="your-key"
codehelm build --provider anthropic --model claude-sonnet-4-6 "fix the failing tests"
```

For a local Ollama model:

```bash
cargo run --release -p codehelm-cli -- build --provider ollama --model qwen3-coder "explain and improve this project"
```

For OpenAI-compatible gateways, set `provider` to `openai-compatible`, provide `baseUrl` in `.codehelm/config.json`, and set `CODEHELM_API_KEY`.

## Commands

```text
codehelm                         interactive build session
codehelm plan "task"             read-only investigation
codehelm build "task"            implement, test, and review
codehelm review                  inspect current Git changes
codehelm checkpoints             list recoverable edit checkpoints
codehelm rollback latest         restore an interrupted build
codehelm resume latest "task"   continue a saved Rust conversation
codehelm exec "task" --json      headless execution with NDJSON events
codehelm resume latest           continue the latest session
codehelm init                    create project configuration
```

Shell commands outside the configured allowlist require confirmation. In non-interactive environments they are denied unless `--yes` is supplied. Denylisted commands are always blocked.

## Configuration

Project configuration lives in `.codehelm/config.json`. Global defaults can be placed in `~/.config/codehelm/config.json`. Project values override global values, and CLI flags override both.

```json
{
  "provider": "openai",
  "model": "gpt-5-mini",
  "maxTurns": 20,
  "maxInstructionChars": 100000,
  "providerMaxRetries": 3,
  "providerRetryBaseMs": 500,
  "permissions": {
    "allowCommands": ["git status", "git diff", "npm test"],
    "denyCommands": ["rm", "sudo", "git reset --hard"],
    "askWrite": ["generated/**"],
    "denyRead": [".env", ".env.*", "**/*.pem", "**/*.key"],
    "denyWrite": [".git/**", ".env", "**/*.key"]
  }
}
```

## Architecture

```text
crates/codehelm-cli/       Rust command-line frontend
crates/codehelm-core/      Rust configuration and security policy
crates/codehelm-protocol/  Rust agent actions and event messages
bin/codehelm.js          executable entry point
src/cli.js           commands, interactive UX, approvals
src/agent.js         provider-neutral tool-use loop
src/providers.js     model provider adapters
src/tools.js         repository, editing, shell, and Git tools
src/permissions.js   path containment and policy decisions
src/session.js       resumable local session storage
src/config.js        layered project/global configuration
```

The agent uses a provider-neutral JSON action protocol, keeping the execution engine independent of any vendor-specific tool-calling format.

## Development

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
npm test
npm run check
node ./bin/codehelm.js --help
```

## Roadmap

- OS-level command sandboxing and explicit interactive approvals
- Patch-based edits with visual diff approval
- Git worktree checkpoints and rollback
- MCP client and plugin/skill system
- ACP server for editor integration
- Parallel subagents with isolated worktrees
- Tree-sitter/LSP context ranking
- Token and cost budgets
- OS-level sandbox adapters
