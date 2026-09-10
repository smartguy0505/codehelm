import { exec as execCallback } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { resolveInside } from "./permissions.js";

const exec = promisify(execCallback);

function truncate(value, limit) {
  if (value.length <= limit) return value;
  return `${value.slice(0, limit)}\n… truncated ${value.length - limit} characters`;
}

function ensureRealPathInside(root, realPath, requested) {
  const relative = path.relative(root, realPath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) throw new Error(`Symlink escapes workspace: ${requested}`);
}

async function resolveExistingInside(root, requested) {
  const resolved = resolveInside(root, requested);
  const realPath = await fs.realpath(resolved.absolute);
  ensureRealPathInside(root, realPath, requested);
  return { ...resolved, absolute: realPath };
}

async function resolveWriteInside(root, requested) {
  const resolved = resolveInside(root, requested);
  try {
    const realTarget = await fs.realpath(resolved.absolute);
    ensureRealPathInside(root, realTarget, requested);
    return { ...resolved, absolute: realTarget };
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  let ancestor = path.dirname(resolved.absolute);
  while (true) {
    try {
      const realAncestor = await fs.realpath(ancestor);
      ensureRealPathInside(root, realAncestor, requested);
      return resolved;
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
      const parent = path.dirname(ancestor);
      if (parent === ancestor) throw error;
      ancestor = parent;
    }
  }
}

async function listRecursive(root, current, depth, output) {
  if (depth < 0) return;
  const entries = await fs.readdir(current, { withFileTypes: true });
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if ([".git", "node_modules", "dist", "build", ".codehelm"].includes(entry.name)) continue;
    const full = path.join(current, entry.name);
    output.push(path.relative(root, full));
    if (entry.isDirectory()) await listRecursive(root, full, depth - 1, output);
  }
}

export function createTools({ cwd, config }) {
  return {
    async list_files(args = {}) {
      const { absolute } = await resolveExistingInside(cwd, args.path ?? ".");
      const output = [];
      await listRecursive(cwd, absolute, Math.min(Number(args.depth ?? 3), 8), output);
      return output.join("\n") || "(empty)";
    },

    async read_file(args) {
      const { absolute } = await resolveExistingInside(cwd, args.path);
      const data = await fs.readFile(absolute, "utf8");
      const lines = data.split("\n");
      const start = Math.max(1, Number(args.startLine ?? 1));
      const end = Math.min(lines.length, Number(args.endLine ?? start + 399));
      return lines.slice(start - 1, end).map((line, index) => `${start + index}: ${line}`).join("\n");
    },

    async search(args) {
      await resolveExistingInside(cwd, args.path ?? ".");
      const searchPath = args.path ?? ".";
      const command = `rg --line-number --color never --glob '!node_modules' --glob '!.git' -- ${shellQuote(args.pattern)} ${shellQuote(searchPath)}`;
      try {
        const { stdout } = await exec(command, { cwd, timeout: 30_000, maxBuffer: 2_000_000 });
        return truncate(stdout.trim(), config.maxToolOutputChars) || "(no matches)";
      } catch (error) {
        if (error.code === 1) return "(no matches)";
        throw error;
      }
    },

    async write_file(args) {
      const { absolute, relative } = await resolveWriteInside(cwd, args.path);
      await fs.mkdir(path.dirname(absolute), { recursive: true });
      await fs.writeFile(absolute, String(args.content), "utf8");
      return `Wrote ${relative} (${Buffer.byteLength(String(args.content))} bytes)`;
    },

    async replace_in_file(args) {
      const { absolute, relative } = await resolveExistingInside(cwd, args.path);
      const data = await fs.readFile(absolute, "utf8");
      const oldText = String(args.oldText);
      const occurrences = data.split(oldText).length - 1;
      if (occurrences !== 1) throw new Error(`Expected oldText exactly once in ${relative}; found ${occurrences}`);
      const updated = data.replace(oldText, String(args.newText));
      await fs.writeFile(absolute, updated, "utf8");
      return `Updated ${relative}`;
    },

    async run_command(args) {
      const { stdout, stderr } = await exec(args.command, {
        cwd,
        timeout: config.commandTimeoutMs,
        maxBuffer: 4_000_000,
        env: { ...process.env, CODEHELM_AGENT: "1" }
      });
      return truncate([stdout, stderr].filter(Boolean).join("\n").trim(), config.maxToolOutputChars) || "Command completed successfully.";
    },

    async git_status() {
      const { stdout } = await exec("git status --short", { cwd, timeout: 20_000 });
      return stdout.trim() || "Working tree clean.";
    },

    async git_diff(args = {}) {
      const command = args.staged ? "git diff --cached --no-ext-diff" : "git diff --no-ext-diff";
      const { stdout } = await exec(command, { cwd, timeout: 20_000, maxBuffer: 4_000_000 });
      return truncate(stdout, config.maxToolOutputChars) || "No diff.";
    }
  };
}

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

export const TOOL_DESCRIPTIONS = {
  list_files: { path: "relative directory (default .)", depth: "optional depth, default 3" },
  read_file: { path: "relative file", startLine: "optional", endLine: "optional" },
  search: { pattern: "regular expression", path: "relative path (default .)" },
  write_file: { path: "relative file", content: "complete content" },
  replace_in_file: { path: "relative file", oldText: "exact unique text", newText: "replacement" },
  run_command: { command: "shell command" },
  git_status: {},
  git_diff: { staged: "optional boolean" }
};
