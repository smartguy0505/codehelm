import readline from "node:readline/promises";
import process from "node:process";
import { Agent } from "./agent.js";
import { loadConfig, initializeProject } from "./config.js";
import { createProvider } from "./providers.js";
import { SessionStore } from "./session.js";
import { createTools } from "./tools.js";

const HELP = `CodeHelm — provider-neutral agentic coding CLI

Usage:
  codehelm [chat]                         Start an interactive build session
  codehelm build "task"                  Implement and verify a task
  codehelm plan "task"                   Read-only investigation and plan
  codehelm review ["focus"]              Review the current Git diff
  codehelm exec "task"                   Non-interactive build session
  codehelm resume [session-id|latest]     Continue a saved session
  codehelm init                           Create .codehelm/config.json and AGENTS.md

Options:
  --provider <openai|anthropic|ollama|openai-compatible>
  --model <model-name>
  --max-turns <number>
  --json                              Emit NDJSON events
  --yes                               Approve non-denied commands (use carefully)
  --help

Environment:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, CODEHELM_API_KEY, OPENAI_BASE_URL, OLLAMA_HOST`;

function parseArgs(argv) {
  const flags = {};
  const positional = [];
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (item === "--json") flags.json = true;
    else if (item === "--yes" || item === "-y") flags.yes = true;
    else if (item === "--help" || item === "-h") flags.help = true;
    else if (item === "--provider") flags.provider = argv[++index];
    else if (item === "--model") flags.model = argv[++index];
    else if (item === "--max-turns") flags.maxTurns = argv[++index];
    else positional.push(item);
  }
  return { flags, positional };
}

function humanEmit(event) {
  if (event.type === "tool_start") console.error(`→ ${event.tool} ${JSON.stringify(event.args)}`);
  if (event.type === "tool_result") console.error(`✓ ${event.tool}`);
  if (event.type === "tool_denied") console.error(`! ${event.tool}: ${event.result}`);
}

export async function main(argv) {
  const { flags, positional } = parseArgs(argv);
  if (flags.help) return console.log(HELP);
  const cwd = process.cwd();
  const command = ["chat", "build", "plan", "review", "exec", "resume", "init"].includes(positional[0]) ? positional.shift() : "chat";
  if (command === "init") {
    const result = await initializeProject(cwd);
    console.log(`Initialized ${result.configFile}\nInstructions: ${result.instructionsFile}`);
    return;
  }

  const config = await loadConfig(cwd, flags);
  const provider = createProvider(config);
  const tools = createTools({ cwd, config });
  const store = new SessionStore(cwd);
  const requestedMode = command === "plan" ? "plan" : command === "review" ? "review" : "build";
  const session = command === "resume"
    ? await store.load(positional.shift() || "latest")
    : store.create(requestedMode, config.provider, config.model);
  const rl = readline.createInterface({ input: process.stdin, output: process.stderr });
  const emit = flags.json ? (event) => console.log(JSON.stringify({ sessionId: session.id, ...event })) : humanEmit;
  const approve = async (tool, args, reason) => {
    if (flags.yes) return true;
    if (!process.stdin.isTTY) return false;
    const answer = await rl.question(`Allow ${tool} ${JSON.stringify(args)}? (${reason}) [y/N] `);
    return /^y(es)?$/i.test(answer.trim());
  };
  const agent = new Agent({ cwd, config, provider, tools, session, store, approve, emit });

  try {
    if (command === "review" && positional.length === 0) positional.push("Review the current Git status and diff. Report actionable findings with file and line references.");
    const initialPrompt = positional.join(" ").trim();
    if (initialPrompt) {
      const result = await agent.run(initialPrompt);
      if (!flags.json) console.log(result);
    }
    if (["chat", "build", "resume"].includes(command) && process.stdin.isTTY) {
      console.error(`CodeHelm session ${session.id} · ${session.mode} · ${config.provider}/${config.model}`);
      while (true) {
        const prompt = (await rl.question("codehelm> ")).trim();
        if (!prompt) continue;
        if (["/exit", "/quit"].includes(prompt)) break;
        const result = await agent.run(prompt);
        if (!flags.json) console.log(result);
      }
    } else if (!initialPrompt) {
      throw new Error(`A task is required for ${command} mode`);
    }
  } finally {
    rl.close();
  }
}

export { parseArgs };
