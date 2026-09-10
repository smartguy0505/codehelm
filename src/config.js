import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

export const DEFAULT_CONFIG = Object.freeze({
  provider: "openai",
  model: "gpt-5-mini",
  maxTurns: 20,
  commandTimeoutMs: 120_000,
  maxToolOutputChars: 30_000,
  permissions: {
    allowCommands: ["git status", "git diff", "git log", "npm test", "npm run", "pnpm test", "pytest", "cargo test", "go test"],
    denyCommands: ["rm", "sudo", "shutdown", "reboot", "mkfs", "dd", "git reset --hard", "git clean", "git push --force"],
    denyRead: [".env", ".env.*", "**/.env", "**/.env.*", "*.pem", "**/*.pem", "*.key", "**/*.key", "**/id_rsa", "**/id_ed25519"],
    denyWrite: [".git/**", ".env", ".env.*", "**/.env", "**/.env.*", "*.pem", "**/*.pem", "*.key", "**/*.key"]
  }
});

function merge(base, override) {
  if (!override || typeof override !== "object" || Array.isArray(override)) return override ?? base;
  const result = { ...base };
  for (const [key, value] of Object.entries(override)) {
    result[key] = value && typeof value === "object" && !Array.isArray(value)
      ? merge(base?.[key] ?? {}, value)
      : value;
  }
  return result;
}

async function readJson(file) {
  try {
    return JSON.parse(await fs.readFile(file, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return {};
    throw new Error(`Cannot read config ${file}: ${error.message}`);
  }
}

export async function loadConfig(cwd, flags = {}) {
  const globalFile = path.join(os.homedir(), ".config", "codehelm", "config.json");
  const projectFile = path.join(cwd, ".codehelm", "config.json");
  const globalConfig = await readJson(globalFile);
  const projectConfig = await readJson(projectFile);
  const flagConfig = {};
  if (flags.provider) flagConfig.provider = flags.provider;
  if (flags.model) flagConfig.model = flags.model;
  if (flags.maxTurns) flagConfig.maxTurns = Number(flags.maxTurns);
  return merge(merge(merge(DEFAULT_CONFIG, globalConfig), projectConfig), flagConfig);
}

export async function initializeProject(cwd) {
  const dir = path.join(cwd, ".codehelm");
  await fs.mkdir(dir, { recursive: true });
  const configFile = path.join(dir, "config.json");
  const instructionsFile = path.join(cwd, "AGENTS.md");
  try {
    await fs.writeFile(configFile, `${JSON.stringify(DEFAULT_CONFIG, null, 2)}\n`, { flag: "wx" });
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
  }
  try {
    await fs.writeFile(instructionsFile, "# Agent instructions\n\nDescribe project conventions, test commands, and architectural constraints here.\n", { flag: "wx" });
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
  }
  return { configFile, instructionsFile };
}
