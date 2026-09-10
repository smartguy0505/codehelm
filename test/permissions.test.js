import test from "node:test";
import assert from "node:assert/strict";
import { authorizeTool, commandRisk, matchesAny, resolveInside } from "../src/permissions.js";
import { DEFAULT_CONFIG } from "../src/config.js";

test("resolveInside contains paths within the workspace", () => {
  assert.equal(resolveInside("/tmp/project", "src/app.js").absolute, "/tmp/project/src/app.js");
  assert.throws(() => resolveInside("/tmp/project", "../secret"), /escapes workspace/);
});

test("sensitive glob patterns match nested keys", () => {
  assert.equal(matchesAny(".env", [".env"]), true);
  assert.equal(matchesAny("secrets/server.pem", ["**/*.pem"]), true);
  assert.equal(matchesAny("server.pem", ["*.pem"]), true);
  assert.equal(matchesAny("src/app.js", ["**/*.pem"]), false);
});

test("command policy allows, asks, and denies", () => {
  const policy = DEFAULT_CONFIG.permissions;
  assert.equal(commandRisk("npm test", policy).decision, "allow");
  assert.equal(commandRisk("node script.js", policy).decision, "ask");
  assert.equal(commandRisk("rm -rf output", policy).decision, "deny");
  assert.equal(commandRisk("echo safe && rm -rf output", policy).decision, "deny");
});

test("plan mode denies writes", () => {
  const result = authorizeTool("plan", "write_file", { path: "src/app.js" }, DEFAULT_CONFIG.permissions);
  assert.equal(result.decision, "deny");
});
