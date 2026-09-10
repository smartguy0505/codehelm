import fs from "node:fs/promises";
import path from "node:path";
import { authorizeTool } from "./permissions.js";
import { TOOL_DESCRIPTIONS } from "./tools.js";

export function parseAgentAction(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
  const candidates = [fenced, text, text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1)].filter(Boolean);
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate.trim());
      if (parsed && ["tool", "final"].includes(parsed.type)) return parsed;
    } catch {}
  }
  return { type: "final", message: text.trim() };
}

async function projectInstructions(cwd) {
  const names = ["AGENTS.md", "CLAUDE.md"];
  const parts = [];
  for (const name of names) {
    try {
      parts.push(`## ${name}\n${await fs.readFile(path.join(cwd, name), "utf8")}`);
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  return parts.join("\n\n");
}

function systemPrompt(mode, instructions) {
  const modeText = {
    plan: "Investigate and produce a concrete implementation plan. Do not modify files.",
    build: "Complete the requested task. Inspect before editing, make focused changes, and verify with tests.",
    review: "Review the repository changes. Do not modify files. Prioritize correctness, security, and missing tests."
  }[mode];
  return `You are CodeHelm, a careful autonomous coding agent working in a local repository.

Mode: ${mode}. ${modeText}

You operate one action at a time. Every response MUST be one JSON object, without markdown:
{"type":"tool","tool":"read_file","args":{"path":"src/app.js"},"reason":"why this is needed"}
or
{"type":"final","message":"concise outcome, files changed, and verification"}

Available tools and argument shapes:
${JSON.stringify(TOOL_DESCRIPTIONS, null, 2)}

Rules:
- Never invent tool results.
- Use relative workspace paths only.
- Read relevant files before editing them.
- Prefer replace_in_file for focused edits.
- Check git_status/git_diff before finishing build or review work.
- Run suitable tests when changes are made.
- If blocked, explain the exact blocker in the final response.

Project instructions:
${instructions || "(none)"}`;
}

export class Agent {
  constructor({ cwd, config, provider, tools, session, store, approve, emit }) {
    Object.assign(this, { cwd, config, provider, tools, session, store, approve, emit });
  }

  async run(prompt) {
    const instructions = await projectInstructions(this.cwd);
    const messages = [
      { role: "system", content: systemPrompt(this.session.mode, instructions) },
      ...this.session.messages,
      { role: "user", content: prompt }
    ];
    this.session.messages.push({ role: "user", content: prompt });

    for (let turn = 1; turn <= this.config.maxTurns; turn += 1) {
      this.emit({ type: "turn", turn });
      const raw = await this.provider.complete(messages);
      const action = parseAgentAction(raw);
      messages.push({ role: "assistant", content: JSON.stringify(action) });
      this.session.messages.push({ role: "assistant", content: JSON.stringify(action) });

      if (action.type === "final") {
        const event = { type: "final", message: action.message, turn };
        this.session.events.push(event);
        await this.store.save(this.session);
        this.emit(event);
        return action.message;
      }

      const tool = this.tools[action.tool];
      if (!tool) {
        messages.push({ role: "user", content: `Tool error: Unknown tool ${action.tool}` });
        continue;
      }

      const authorization = authorizeTool(this.session.mode, action.tool, action.args ?? {}, this.config.permissions);
      let allowed = authorization.decision === "allow";
      if (authorization.decision === "ask") {
        allowed = await this.approve(action.tool, action.args ?? {}, authorization.reason);
      }
      if (!allowed || authorization.decision === "deny") {
        const result = `Permission denied: ${authorization.reason}`;
        this.emit({ type: "tool_denied", tool: action.tool, args: action.args, result });
        messages.push({ role: "user", content: `Tool result for ${action.tool}: ${result}` });
        continue;
      }

      this.emit({ type: "tool_start", tool: action.tool, args: action.args, reason: action.reason });
      let result;
      try {
        result = await tool(action.args ?? {});
      } catch (error) {
        result = `Tool error: ${error.message}`;
      }
      const event = { type: "tool_result", tool: action.tool, result };
      this.session.events.push(event);
      this.emit(event);
      messages.push({ role: "user", content: `Tool result for ${action.tool}:\n${result}` });
      await this.store.save(this.session);
    }
    throw new Error(`Agent exceeded ${this.config.maxTurns} turns`);
  }
}
