import path from "node:path";

function globToRegExp(glob) {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replaceAll("**", "\0")
    .replaceAll("*", "[^/]*")
    .replaceAll("\0", ".*")
    .replaceAll("?", ".");
  return new RegExp(`^${escaped}$`);
}

export function matchesAny(value, patterns = []) {
  const normalized = value.split(path.sep).join("/").replace(/^\.\//, "");
  return patterns.some((pattern) => globToRegExp(pattern).test(normalized));
}

export function resolveInside(root, requested) {
  if (typeof requested !== "string" || requested.includes("\0")) throw new Error("Invalid path");
  const absolute = path.resolve(root, requested);
  const relative = path.relative(root, absolute);
  if (relative.startsWith("..") || path.isAbsolute(relative)) throw new Error(`Path escapes workspace: ${requested}`);
  return { absolute, relative: relative || "." };
}

export function commandRisk(command, policy) {
  const normalized = command.trim().replace(/\s+/g, " ");
  if (!normalized) return { decision: "deny", reason: "empty command" };
  if (/[;&|`]|\$\(|\n/.test(normalized)) return { decision: "deny", reason: "shell composition is disabled" };
  if (policy.denyCommands.some((item) => normalized === item || normalized.startsWith(`${item} `))) {
    return { decision: "deny", reason: "command denylist" };
  }
  if (policy.allowCommands.some((item) => normalized === item || normalized.startsWith(`${item} `))) {
    return { decision: "allow", reason: "command allowlist" };
  }
  return { decision: "ask", reason: "command is not allowlisted" };
}

export function authorizeTool(mode, name, args, policy) {
  const writeTools = new Set(["write_file", "replace_in_file"]);
  if ((mode === "plan" || mode === "review") && writeTools.has(name)) {
    return { decision: "deny", reason: `${mode} mode is read-only` };
  }
  if (name === "run_command") {
    if (mode === "plan" || mode === "review") {
      const risk = commandRisk(args.command ?? "", policy);
      return risk.decision === "allow" ? risk : { decision: "deny", reason: `${mode} mode only permits allowlisted commands` };
    }
    return commandRisk(args.command ?? "", policy);
  }
  if (["read_file", "write_file", "replace_in_file"].includes(name)) {
    const target = String(args.path ?? "").replaceAll("\\", "/");
    const patterns = name === "read_file" ? policy.denyRead : policy.denyWrite;
    if (matchesAny(target, patterns)) return { decision: "deny", reason: "path denylist" };
  }
  return { decision: "allow", reason: "safe built-in tool" };
}
