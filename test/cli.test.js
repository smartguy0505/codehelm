import test from "node:test";
import assert from "node:assert/strict";
import { parseArgs } from "../src/cli.js";

test("parses provider, model, max turns, and mode", () => {
  const parsed = parseArgs(["plan", "--provider", "ollama", "--model", "qwen", "--max-turns", "7", "inspect repo"]);
  assert.deepEqual(parsed.flags, { provider: "ollama", model: "qwen", maxTurns: "7" });
  assert.deepEqual(parsed.positional, ["plan", "inspect repo"]);
});
