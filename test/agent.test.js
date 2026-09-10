import test from "node:test";
import assert from "node:assert/strict";
import { parseAgentAction } from "../src/agent.js";

test("parses direct JSON tool actions", () => {
  assert.deepEqual(parseAgentAction('{"type":"tool","tool":"git_status","args":{}}'), {
    type: "tool",
    tool: "git_status",
    args: {}
  });
});

test("parses fenced JSON", () => {
  const action = parseAgentAction('```json\n{"type":"final","message":"done"}\n```');
  assert.equal(action.message, "done");
});

test("treats ordinary provider text as a final answer", () => {
  assert.deepEqual(parseAgentAction("Could not continue"), { type: "final", message: "Could not continue" });
});
