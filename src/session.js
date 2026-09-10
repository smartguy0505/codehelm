import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

export class SessionStore {
  constructor(cwd) {
    this.directory = path.join(cwd, ".codehelm", "sessions");
  }

  async save(session) {
    await fs.mkdir(this.directory, { recursive: true });
    session.updatedAt = new Date().toISOString();
    await fs.writeFile(path.join(this.directory, `${session.id}.json`), `${JSON.stringify(session, null, 2)}\n`, "utf8");
  }

  async load(id) {
    const file = id === "latest" ? await this.latestFile() : path.join(this.directory, `${id}.json`);
    if (!file) throw new Error("No saved sessions found");
    return JSON.parse(await fs.readFile(file, "utf8"));
  }

  async latestFile() {
    try {
      const names = (await fs.readdir(this.directory)).filter((name) => name.endsWith(".json"));
      const entries = await Promise.all(names.map(async (name) => ({
        name,
        stat: await fs.stat(path.join(this.directory, name))
      })));
      entries.sort((a, b) => b.stat.mtimeMs - a.stat.mtimeMs);
      return entries[0] ? path.join(this.directory, entries[0].name) : null;
    } catch (error) {
      if (error.code === "ENOENT") return null;
      throw error;
    }
  }

  create(mode, provider, model) {
    return {
      id: crypto.randomUUID().slice(0, 12),
      mode,
      provider,
      model,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: [],
      events: []
    };
  }
}
