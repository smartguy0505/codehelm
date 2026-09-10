function envKey(provider) {
  const names = {
    openai: "OPENAI_API_KEY",
    anthropic: "ANTHROPIC_API_KEY",
    ollama: null
  };
  const name = names[provider];
  return process.env.CODEHELM_API_KEY || (name ? process.env[name] : undefined);
}

async function postJson(url, body, headers = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(300_000)
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`Provider request failed (${response.status}): ${text.slice(0, 800)}`);
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Provider returned invalid JSON: ${text.slice(0, 400)}`);
  }
}

function openAIAdapter(config) {
  const apiKey = envKey("openai");
  if (!apiKey) throw new Error("Set OPENAI_API_KEY or CODEHELM_API_KEY");
  const baseUrl = (config.baseUrl || process.env.OPENAI_BASE_URL || "https://api.openai.com/v1").replace(/\/$/, "");
  return {
    async complete(messages) {
      const data = await postJson(`${baseUrl}/chat/completions`, {
        model: config.model,
        messages,
        temperature: 0
      }, { authorization: `Bearer ${apiKey}` });
      return data.choices?.[0]?.message?.content ?? "";
    }
  };
}

function anthropicAdapter(config) {
  const apiKey = envKey("anthropic");
  if (!apiKey) throw new Error("Set ANTHROPIC_API_KEY or CODEHELM_API_KEY");
  const baseUrl = (config.baseUrl || "https://api.anthropic.com").replace(/\/$/, "");
  return {
    async complete(messages) {
      const system = messages.filter((item) => item.role === "system").map((item) => item.content).join("\n\n");
      const conversation = messages.filter((item) => item.role !== "system").map((item) => ({
        role: item.role === "assistant" ? "assistant" : "user",
        content: item.content
      }));
      const data = await postJson(`${baseUrl}/v1/messages`, {
        model: config.model,
        system,
        messages: conversation,
        max_tokens: 8192,
        temperature: 0
      }, {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01"
      });
      return data.content?.filter((item) => item.type === "text").map((item) => item.text).join("\n") ?? "";
    }
  };
}

function ollamaAdapter(config) {
  const baseUrl = (config.baseUrl || process.env.OLLAMA_HOST || "http://127.0.0.1:11434").replace(/\/$/, "");
  return {
    async complete(messages) {
      const data = await postJson(`${baseUrl}/api/chat`, {
        model: config.model,
        messages,
        stream: false,
        options: { temperature: 0 }
      });
      return data.message?.content ?? "";
    }
  };
}

export function createProvider(config) {
  if (config.provider === "openai" || config.provider === "openai-compatible") return openAIAdapter(config);
  if (config.provider === "anthropic") return anthropicAdapter(config);
  if (config.provider === "ollama") return ollamaAdapter(config);
  throw new Error(`Unknown provider: ${config.provider}`);
}
