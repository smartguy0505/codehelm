use std::collections::BTreeMap;

use async_trait::async_trait;
use codehelm_protocol::{
    AgentAction, AgentEvent, ConversationItem, ModelRequest, Role, TokenUsage, ToolCallRequest,
};
use futures_util::StreamExt;
use reqwest::{Client, RequestBuilder, Response, StatusCode, header::RETRY_AFTER};
use serde_json::{Value, json};

use crate::agent::{AgentError, EventSink, ModelProvider};

const MAX_PROVIDER_FRAME_BYTES: usize = 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 800;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub base_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 500,
        }
    }
}

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    retry: RetryPolicy,
    request_timeout: std::time::Duration,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_base_url(api_key, model, "https://api.openai.com/v1")
    }

    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            retry: RetryPolicy::default(),
            request_timeout: std::time::Duration::from_secs(300),
        }
    }

    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    pub fn with_request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

#[async_trait(?Send)]
impl ModelProvider for OpenAiProvider {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError> {
        let request = self
            .client
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .timeout(self.request_timeout)
            .json(&openai_request(&self.model, request));
        let response = send_with_retry(request, self.retry, events).await?;
        let mut state = OpenAiState::default();
        stream_sse(response, |event| state.consume(event, events)).await?;
        state.finish()
    }
}

#[derive(Default)]
struct OpenAiState {
    text: String,
    tools: Vec<ToolCallRequest>,
}

impl OpenAiState {
    fn consume(&mut self, event: Value, events: &mut dyn EventSink) -> Result<(), AgentError> {
        match event["type"].as_str() {
            Some("response.output_text.delta") => {
                if let Some(delta) = event["delta"].as_str() {
                    self.text.push_str(delta);
                    events.emit(AgentEvent::ModelDelta {
                        delta: delta.into(),
                    });
                }
            }
            Some("response.output_item.done")
                if event["item"]["type"].as_str() == Some("function_call") =>
            {
                let item = &event["item"];
                let id = required_str(item, "call_id")?;
                let name = required_str(item, "name")?;
                let args = parse_args(item["arguments"].as_str().unwrap_or("{}"))?;
                self.tools.push(ToolCallRequest {
                    id,
                    tool: name,
                    args,
                    reason: None,
                });
            }
            Some("response.completed") => emit_usage(&event["response"]["usage"], events),
            Some("error" | "response.failed") => {
                return Err(AgentError::Provider(event.to_string()));
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<AgentAction, AgentError> {
        if !self.tools.is_empty() {
            Ok(tool_action(self.tools))
        } else if self.text.is_empty() {
            Err(AgentError::Provider(
                "OpenAI returned no text or tool call".into(),
            ))
        } else {
            Ok(AgentAction::Final { message: self.text })
        }
    }
}

fn openai_request(model: &str, request: &ModelRequest) -> Value {
    let instructions = request
        .items
        .iter()
        .filter_map(|item| match item {
            ConversationItem::Message {
                role: Role::System,
                content,
            } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let input = request
        .items
        .iter()
        .filter_map(|item| match item {
            ConversationItem::Message {
                role: Role::System, ..
            } => None,
            ConversationItem::Message { role, content } => Some(json!({
                "role": role_name(*role), "content": content
            })),
            ConversationItem::ToolCall { id, name, args } => Some(json!({
                "type": "function_call", "call_id": id, "name": name,
                "arguments": args.to_string()
            })),
            ConversationItem::ToolResult { id, output } => Some(json!({
                "type": "function_call_output", "call_id": id, "output": output
            })),
        })
        .collect::<Vec<_>>();
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function", "name": tool.name, "description": tool.description,
            "parameters": tool.input_schema, "strict": false
            })
        })
        .collect::<Vec<_>>();
    json!({
        "model": model, "instructions": instructions, "input": input,
        "tools": tools, "tool_choice": "auto", "stream": true, "store": false
    })
}

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: usize,
    retry: RetryPolicy,
    request_timeout: std::time::Duration,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_base_url(api_key, model, "https://api.anthropic.com")
    }

    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            max_tokens: 8_192,
            retry: RetryPolicy::default(),
            request_timeout: std::time::Duration::from_secs(300),
        }
    }

    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    pub fn with_request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

#[async_trait(?Send)]
impl ModelProvider for AnthropicProvider {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError> {
        let request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .timeout(self.request_timeout)
            .json(&anthropic_request(&self.model, self.max_tokens, request));
        let response = send_with_retry(request, self.retry, events).await?;
        let mut state = AnthropicState::default();
        stream_sse(response, |event| state.consume(event, events)).await?;
        state.finish()
    }
}

#[derive(Default)]
struct AnthropicState {
    text: String,
    tools: BTreeMap<u64, AnthropicTool>,
    input_tokens: u64,
    cached_input_tokens: u64,
}

#[derive(Default)]
struct AnthropicTool {
    id: String,
    name: String,
    args: String,
}

impl AnthropicState {
    fn consume(&mut self, event: Value, events: &mut dyn EventSink) -> Result<(), AgentError> {
        match event["type"].as_str() {
            Some("message_start") => {
                let usage = &event["message"]["usage"];
                self.input_tokens = token_count(usage, "input_tokens")
                    .saturating_add(token_count(usage, "cache_creation_input_tokens"))
                    .saturating_add(token_count(usage, "cache_read_input_tokens"));
                self.cached_input_tokens = token_count(usage, "cache_read_input_tokens");
            }
            Some("message_delta") => {
                events.emit(AgentEvent::Usage {
                    usage: TokenUsage {
                        input_tokens: self.input_tokens,
                        output_tokens: token_count(&event["usage"], "output_tokens"),
                        cached_input_tokens: self.cached_input_tokens,
                        reasoning_tokens: 0,
                    },
                });
            }
            Some("content_block_start") if event["content_block"]["type"] == "tool_use" => {
                let index = event["index"].as_u64().unwrap_or(self.tools.len() as u64);
                self.tools.insert(
                    index,
                    AnthropicTool {
                        id: required_str(&event["content_block"], "id")?,
                        name: required_str(&event["content_block"], "name")?,
                        args: String::new(),
                    },
                );
            }
            Some("content_block_delta") => match event["delta"]["type"].as_str() {
                Some("text_delta") => {
                    if let Some(delta) = event["delta"]["text"].as_str() {
                        self.text.push_str(delta);
                        events.emit(AgentEvent::ModelDelta {
                            delta: delta.into(),
                        });
                    }
                }
                Some("input_json_delta") => {
                    if let Some(delta) = event["delta"]["partial_json"].as_str() {
                        let index = event["index"].as_u64().unwrap_or(0);
                        let tool = self.tools.get_mut(&index).ok_or_else(|| {
                            AgentError::Provider(format!(
                                "Anthropic sent arguments for unknown tool block {index}"
                            ))
                        })?;
                        tool.args.push_str(delta);
                    }
                }
                _ => {}
            },
            Some("error") => return Err(AgentError::Provider(event.to_string())),
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<AgentAction, AgentError> {
        if !self.tools.is_empty() {
            let calls = self
                .tools
                .into_values()
                .map(|tool| {
                    let args = parse_args(if tool.args.is_empty() {
                        "{}"
                    } else {
                        &tool.args
                    })?;
                    Ok(ToolCallRequest {
                        id: tool.id,
                        tool: tool.name,
                        args,
                        reason: None,
                    })
                })
                .collect::<Result<Vec<_>, AgentError>>()?;
            Ok(tool_action(calls))
        } else if !self.text.is_empty() {
            Ok(AgentAction::Final { message: self.text })
        } else {
            Err(AgentError::Provider(
                "Anthropic returned no text or tool call".into(),
            ))
        }
    }
}

fn anthropic_request(model: &str, max_tokens: usize, request: &ModelRequest) -> Value {
    let system = request
        .items
        .iter()
        .filter_map(|item| match item {
            ConversationItem::Message {
                role: Role::System,
                content,
            } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut messages: Vec<Value> = Vec::new();
    for item in &request.items {
        let (role, block) = match item {
            ConversationItem::Message {
                role: Role::System, ..
            } => continue,
            ConversationItem::Message { role, content } => {
                (role_name(*role), json!({"type": "text", "text": content}))
            }
            ConversationItem::ToolCall { id, name, args } => (
                "assistant",
                json!({"type": "tool_use", "id": id, "name": name, "input": args}),
            ),
            ConversationItem::ToolResult { id, output } => (
                "user",
                json!({"type": "tool_result", "tool_use_id": id, "content": output}),
            ),
        };
        if messages.last().and_then(|message| message["role"].as_str()) == Some(role) {
            messages
                .last_mut()
                .and_then(|message| message["content"].as_array_mut())
                .expect("content array")
                .push(block);
        } else {
            messages.push(json!({"role": role, "content": [block]}));
        }
    }
    let tools = request.tools.iter().map(|tool| json!({
        "name": tool.name, "description": tool.description, "input_schema": tool.input_schema
    })).collect::<Vec<_>>();
    json!({
        "model": model, "system": system, "messages": messages, "tools": tools,
        "max_tokens": max_tokens, "stream": true
    })
}

pub struct OllamaProvider {
    client: Client,
    base_url: String,
    model: String,
    retry: RetryPolicy,
    request_timeout: std::time::Duration,
}

impl OllamaProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base_url(model, "http://127.0.0.1:11434")
    }

    pub fn with_base_url(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let base_url = if base_url.contains("://") {
            base_url
        } else {
            format!("http://{base_url}")
        };
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.into(),
            retry: RetryPolicy::default(),
            request_timeout: std::time::Duration::from_secs(300),
        }
    }

    pub fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    pub fn with_request_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

#[async_trait(?Send)]
impl ModelProvider for OllamaProvider {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError> {
        let request = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .timeout(self.request_timeout)
            .json(&ollama_request(&self.model, request));
        let response = send_with_retry(request, self.retry, events).await?;
        let mut state = OllamaState::default();
        stream_ndjson(response, |chunk| state.consume(chunk, events)).await?;
        state.finish()
    }
}

#[derive(Default)]
struct OllamaState {
    text: String,
    tools: Vec<ToolCallRequest>,
    calls_seen: usize,
}

impl OllamaState {
    fn consume(&mut self, chunk: Value, events: &mut dyn EventSink) -> Result<(), AgentError> {
        if let Some(error) = chunk["error"].as_str() {
            return Err(AgentError::Provider(error.into()));
        }
        if chunk["done"].as_bool() == Some(true) {
            events.emit(AgentEvent::Usage {
                usage: TokenUsage {
                    input_tokens: token_count(&chunk, "prompt_eval_count"),
                    output_tokens: token_count(&chunk, "eval_count"),
                    ..TokenUsage::default()
                },
            });
        }
        if let Some(delta) = chunk["message"]["content"]
            .as_str()
            .filter(|text| !text.is_empty())
        {
            self.text.push_str(delta);
            events.emit(AgentEvent::ModelDelta {
                delta: delta.into(),
            });
        }
        if let Some(calls) = chunk["message"]["tool_calls"].as_array() {
            for call in calls {
                let function = &call["function"];
                let name = required_str(function, "name")?;
                let args = function["arguments"].clone();
                let index = function["index"]
                    .as_u64()
                    .map_or(self.calls_seen, |value| value as usize);
                self.calls_seen += 1;
                self.tools.push(ToolCallRequest {
                    id: format!("ollama-{index}"),
                    tool: name,
                    args,
                    reason: None,
                });
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<AgentAction, AgentError> {
        if !self.tools.is_empty() {
            Ok(tool_action(self.tools))
        } else if self.text.is_empty() {
            Err(AgentError::Provider(
                "Ollama returned no text or tool call".into(),
            ))
        } else {
            Ok(AgentAction::Final { message: self.text })
        }
    }
}

fn ollama_request(model: &str, request: &ModelRequest) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    for item in &request.items {
        match item {
            ConversationItem::Message { role, content } => messages.push(json!({
                "role": role_name(*role), "content": content
            })),
            ConversationItem::ToolCall { name, args, .. } => {
                let call = json!({
                    "type": "function", "function": {"name": name, "arguments": args}
                });
                if let Some(calls) = messages
                    .last_mut()
                    .filter(|message| message["role"] == "assistant")
                    .and_then(|message| message["tool_calls"].as_array_mut())
                {
                    calls.push(call);
                } else {
                    messages.push(json!({
                        "role": "assistant", "content": "", "tool_calls": [call]
                    }));
                }
            }
            ConversationItem::ToolResult { id, output } => {
                let name = request
                    .items
                    .iter()
                    .rev()
                    .find_map(|item| match item {
                        ConversationItem::ToolCall {
                            id: call_id, name, ..
                        } if call_id == id => Some(name.as_str()),
                        _ => None,
                    })
                    .unwrap_or("unknown");
                messages.push(json!({"role": "tool", "tool_name": name, "content": output}));
            }
        }
    }
    let tools = request.tools.iter().map(|tool| json!({
        "type": "function", "function": {
            "name": tool.name, "description": tool.description, "parameters": tool.input_schema
        }
    })).collect::<Vec<_>>();
    json!({"model": model, "messages": messages, "tools": tools, "stream": true})
}

fn tool_action(mut calls: Vec<ToolCallRequest>) -> AgentAction {
    if calls.len() == 1 {
        let call = calls.pop().expect("one call");
        AgentAction::Tool {
            id: call.id,
            tool: call.tool,
            args: call.args,
            reason: call.reason,
        }
    } else {
        AgentAction::Tools { calls }
    }
}

fn token_count(value: &Value, field: &str) -> u64 {
    value[field].as_u64().unwrap_or(0)
}

fn emit_usage(value: &Value, events: &mut dyn EventSink) {
    events.emit(AgentEvent::Usage {
        usage: TokenUsage {
            input_tokens: token_count(value, "input_tokens"),
            output_tokens: token_count(value, "output_tokens"),
            cached_input_tokens: token_count(&value["input_tokens_details"], "cached_tokens"),
            reasoning_tokens: token_count(&value["output_tokens_details"], "reasoning_tokens"),
        },
    });
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

async fn send_with_retry(
    request: RequestBuilder,
    policy: RetryPolicy,
    events: &mut dyn EventSink,
) -> Result<Response, AgentError> {
    for retry in 0..=policy.max_retries {
        let attempt = request.try_clone().ok_or_else(|| {
            AgentError::Provider("provider request body cannot be retried".into())
        })?;
        match attempt.send().await {
            Ok(response) if retryable_status(response.status()) && retry < policy.max_retries => {
                let reason = format!("HTTP {}", response.status());
                let delay_ms = retry_after_ms(&response)
                    .unwrap_or_else(|| backoff_ms(policy.base_delay_ms, retry));
                events.emit(AgentEvent::ProviderRetry {
                    attempt: retry + 1,
                    delay_ms,
                    reason,
                });
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            Ok(response) => return Ok(response),
            Err(error) if retryable_error(&error) && retry < policy.max_retries => {
                let delay_ms = backoff_ms(policy.base_delay_ms, retry);
                events.emit(AgentEvent::ProviderRetry {
                    attempt: retry + 1,
                    delay_ms,
                    reason: error.to_string(),
                });
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
            Err(error) => return Err(provider_error(error)),
        }
    }
    unreachable!("retry loop always returns on its final attempt")
}

fn retryable_status(status: StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429) || status.is_server_error()
}

fn retryable_error(error: &reqwest::Error) -> bool {
    error.is_connect() || error.is_timeout()
}

fn retry_after_ms(response: &Response) -> Option<u64> {
    response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()
        .map(|seconds| seconds.saturating_mul(1_000).min(60_000))
}

fn backoff_ms(base: u64, retry: usize) -> u64 {
    base.saturating_mul(1_u64.checked_shl(retry.min(16) as u32).unwrap_or(u64::MAX))
        .min(30_000)
}

async fn stream_sse(
    response: Response,
    mut consume: impl FnMut(Value) -> Result<(), AgentError>,
) -> Result<(), AgentError> {
    let status = response.status();
    if !status.is_success() {
        return Err(http_status_error(response).await);
    }
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        buffer.extend_from_slice(&chunk.map_err(provider_error)?);
        while let Some((end, delimiter_len)) = event_boundary(&buffer) {
            let frame = buffer.drain(..end).collect::<Vec<_>>();
            buffer.drain(..delimiter_len);
            let frame = String::from_utf8(frame)
                .map_err(|error| AgentError::Provider(format!("invalid SSE UTF-8: {error}")))?;
            for line in frame.lines() {
                if let Some(data) = line.strip_prefix("data:").map(str::trim) {
                    if data != "[DONE]" {
                        consume(serde_json::from_str(data).map_err(|error| {
                            AgentError::Provider(format!("invalid SSE JSON: {error}"))
                        })?)?;
                    }
                }
            }
        }
        ensure_frame_limit(&buffer)?;
    }
    Ok(())
}

async fn stream_ndjson(
    response: Response,
    mut consume: impl FnMut(Value) -> Result<(), AgentError>,
) -> Result<(), AgentError> {
    let status = response.status();
    if !status.is_success() {
        return Err(http_status_error(response).await);
    }
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    while let Some(chunk) = stream.next().await {
        buffer.extend_from_slice(&chunk.map_err(provider_error)?);
        while let Some(end) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer.drain(..end).collect::<Vec<_>>();
            buffer.drain(..1);
            consume_json_line(&line, &mut consume)?;
        }
        ensure_frame_limit(&buffer)?;
    }
    consume_json_line(&buffer, &mut consume)
}

fn ensure_frame_limit(buffer: &[u8]) -> Result<(), AgentError> {
    if buffer.len() > MAX_PROVIDER_FRAME_BYTES {
        Err(AgentError::Provider(format!(
            "provider stream frame exceeded {MAX_PROVIDER_FRAME_BYTES} bytes"
        )))
    } else {
        Ok(())
    }
}

async fn http_status_error(response: Response) -> AgentError {
    let status = response.status();
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while body.len() < MAX_ERROR_BODY_BYTES {
        let Some(chunk) = stream.next().await else {
            break;
        };
        match chunk {
            Ok(chunk) => {
                let keep = (MAX_ERROR_BODY_BYTES - body.len()).min(chunk.len());
                body.extend_from_slice(&chunk[..keep]);
            }
            Err(error) => return provider_error(error),
        }
    }
    AgentError::Provider(format!("HTTP {status}: {}", String::from_utf8_lossy(&body)))
}

fn consume_json_line(
    line: &[u8],
    consume: &mut impl FnMut(Value) -> Result<(), AgentError>,
) -> Result<(), AgentError> {
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(());
    }
    let value = serde_json::from_slice(line)
        .map_err(|error| AgentError::Provider(format!("invalid NDJSON: {error}")))?;
    consume(value)
}

fn event_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2))
        .or_else(|| {
            bytes
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| (index, 4))
        })
}

fn required_str(value: &Value, field: &str) -> Result<String, AgentError> {
    value[field]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| AgentError::Provider(format!("provider event omitted {field}")))
}

fn parse_args(input: &str) -> Result<Value, AgentError> {
    serde_json::from_str(input)
        .map_err(|error| AgentError::Provider(format!("invalid tool arguments: {error}")))
}

fn provider_error(error: impl std::fmt::Display) -> AgentError {
    AgentError::Provider(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_events_normalize_tool_calls() {
        let mut state = OpenAiState::default();
        state
            .consume(
                json!({"type":"response.output_item.done","item":{
                    "type":"function_call","call_id":"call_1","name":"read_file",
                    "arguments":"{\"path\":\"README.md\"}"
                }}),
                &mut |_| {},
            )
            .unwrap();
        assert!(
            matches!(state.finish().unwrap(), AgentAction::Tool { id, tool, .. } if id == "call_1" && tool == "read_file")
        );
    }

    #[test]
    fn provider_frames_have_a_hard_memory_boundary() {
        assert!(ensure_frame_limit(&vec![0; MAX_PROVIDER_FRAME_BYTES]).is_ok());
        let error = ensure_frame_limit(&vec![0; MAX_PROVIDER_FRAME_BYTES + 1]).unwrap_err();
        assert!(error.to_string().contains("frame exceeded"));
    }

    #[test]
    fn provider_request_deadlines_are_configurable() {
        let timeout = std::time::Duration::from_millis(1234);
        assert_eq!(
            OpenAiProvider::new("key", "model")
                .with_request_timeout(timeout)
                .request_timeout,
            timeout
        );
        assert_eq!(
            AnthropicProvider::new("key", "model")
                .with_request_timeout(timeout)
                .request_timeout,
            timeout
        );
        assert_eq!(
            OllamaProvider::new("model")
                .with_request_timeout(timeout)
                .request_timeout,
            timeout
        );
    }

    #[test]
    fn provider_usage_is_normalized() {
        let mut openai = OpenAiState::default();
        let mut openai_events = Vec::new();
        openai
            .consume(
                json!({"type":"response.completed","response":{"usage":{
                    "input_tokens":20,"output_tokens":7,
                    "input_tokens_details":{"cached_tokens":5},
                    "output_tokens_details":{"reasoning_tokens":3}
                }}}),
                &mut |event| openai_events.push(event),
            )
            .unwrap();
        assert!(matches!(
            openai_events.as_slice(),
            [AgentEvent::Usage { usage }]
                if *usage == TokenUsage {
                    input_tokens: 20,
                    output_tokens: 7,
                    cached_input_tokens: 5,
                    reasoning_tokens: 3,
                }
        ));

        let mut anthropic = AnthropicState::default();
        let mut anthropic_events = Vec::new();
        anthropic
            .consume(
                json!({"type":"message_start","message":{"usage":{
                    "input_tokens":10,"cache_creation_input_tokens":4,
                    "cache_read_input_tokens":6
                }}}),
                &mut |event| anthropic_events.push(event),
            )
            .unwrap();
        anthropic
            .consume(
                json!({"type":"message_delta","usage":{"output_tokens":8}}),
                &mut |event| anthropic_events.push(event),
            )
            .unwrap();
        assert!(matches!(
            anthropic_events.as_slice(),
            [AgentEvent::Usage { usage }]
                if *usage == TokenUsage {
                    input_tokens: 20,
                    output_tokens: 8,
                    cached_input_tokens: 6,
                    reasoning_tokens: 0,
                }
        ));

        let mut ollama = OllamaState::default();
        let mut ollama_events = Vec::new();
        ollama
            .consume(
                json!({"message":{"content":"ok"},"done":true,
                    "prompt_eval_count":12,"eval_count":3}),
                &mut |event| ollama_events.push(event),
            )
            .unwrap();
        assert!(ollama_events.iter().any(|event| matches!(
            event,
            AgentEvent::Usage { usage } if usage.input_tokens == 12 && usage.output_tokens == 3
        )));
    }

    #[test]
    fn anthropic_partial_json_normalizes_tool_calls() {
        let mut state = AnthropicState::default();
        let mut sink = |_| {};
        state
            .consume(
                json!({"type":"content_block_start","content_block":{
                    "type":"tool_use","id":"tool_1","name":"read_file"
                }}),
                &mut sink,
            )
            .unwrap();
        state
            .consume(
                json!({"type":"content_block_delta","delta":{
                    "type":"input_json_delta","partial_json":"{\"path\":\"README.md\"}"
                }}),
                &mut sink,
            )
            .unwrap();
        assert!(
            matches!(state.finish().unwrap(), AgentAction::Tool { id, tool, .. } if id == "tool_1" && tool == "read_file")
        );
    }

    #[test]
    fn request_keeps_tool_call_identity() {
        let request = ModelRequest {
            items: vec![
                ConversationItem::ToolCall {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    args: json!({"path":"a"}),
                },
                ConversationItem::ToolResult {
                    id: "call_1".into(),
                    output: "text".into(),
                },
            ],
            tools: vec![],
        };
        let body = openai_request("model", &request);
        assert_eq!(body["input"][1]["call_id"], "call_1");
    }

    #[test]
    fn ollama_chunks_normalize_text_and_tool_calls() {
        let mut text_state = OllamaState::default();
        let mut deltas = Vec::new();
        text_state
            .consume(
                json!({"message":{"content":"hello"},"done":false}),
                &mut |event| deltas.push(event),
            )
            .unwrap();
        assert_eq!(
            text_state.finish().unwrap(),
            AgentAction::Final {
                message: "hello".into()
            }
        );
        assert!(matches!(&deltas[0], AgentEvent::ModelDelta { delta } if delta == "hello"));

        let mut tool_state = OllamaState::default();
        tool_state
            .consume(
                json!({"message":{"tool_calls":[{"function":{
                    "index":0,"name":"read_file","arguments":{"path":"README.md"}
                }}]},"done":true}),
                &mut |_| {},
            )
            .unwrap();
        assert!(matches!(
            tool_state.finish().unwrap(),
            AgentAction::Tool { id, tool, args, .. }
                if id == "ollama-0" && tool == "read_file" && args["path"] == "README.md"
        ));
    }

    #[test]
    fn ollama_request_reconstructs_tool_result_name() {
        let request = ModelRequest {
            items: vec![
                ConversationItem::ToolCall {
                    id: "internal-1".into(),
                    name: "search".into(),
                    args: json!({"pattern":"needle"}),
                },
                ConversationItem::ToolResult {
                    id: "internal-1".into(),
                    output: "match".into(),
                },
            ],
            tools: vec![],
        };
        let body = ollama_request("qwen3", &request);
        assert_eq!(body["messages"][1]["tool_name"], "search");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn ollama_groups_batch_calls_into_one_assistant_message() {
        let request = ModelRequest {
            items: vec![
                ConversationItem::ToolCall {
                    id: "a".into(),
                    name: "read_file".into(),
                    args: json!({}),
                },
                ConversationItem::ToolCall {
                    id: "b".into(),
                    name: "search".into(),
                    args: json!({}),
                },
            ],
            tools: vec![],
        };
        let body = ollama_request("qwen3", &request);
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(
            body["messages"][0]["tool_calls"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn ollama_host_without_scheme_is_normalized() {
        let provider = OllamaProvider::with_base_url("model", "localhost:11434");
        assert_eq!(provider.base_url, "http://localhost:11434");
    }

    #[test]
    fn openai_preserves_multiple_tool_calls() {
        let mut state = OpenAiState::default();
        for (id, name) in [("call_1", "read_file"), ("call_2", "search")] {
            state
                .consume(
                    json!({"type":"response.output_item.done","item":{
                        "type":"function_call","call_id":id,"name":name,"arguments":"{}"
                    }}),
                    &mut |_| {},
                )
                .unwrap();
        }
        assert!(
            matches!(state.finish().unwrap(), AgentAction::Tools { calls } if calls.len() == 2)
        );
    }

    #[test]
    fn anthropic_keeps_arguments_separate_for_multiple_tools() {
        let mut state = AnthropicState::default();
        let mut sink = |_| {};
        for (index, id, name) in [(0, "one", "read_file"), (1, "two", "search")] {
            state
                .consume(
                    json!({"type":"content_block_start","index":index,"content_block":{
                        "type":"tool_use","id":id,"name":name
                    }}),
                    &mut sink,
                )
                .unwrap();
            state
                .consume(
                    json!({"type":"content_block_delta","index":index,"delta":{
                        "type":"input_json_delta","partial_json":"{}"
                    }}),
                    &mut sink,
                )
                .unwrap();
        }
        assert!(
            matches!(state.finish().unwrap(), AgentAction::Tools { calls } if calls.len() == 2 && calls[1].tool == "search")
        );
    }

    #[test]
    fn retry_policy_is_bounded_and_status_specific() {
        assert!(retryable_status(StatusCode::REQUEST_TIMEOUT));
        assert!(retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(retryable_status(StatusCode::BAD_GATEWAY));
        assert!(!retryable_status(StatusCode::BAD_REQUEST));
        assert!(!retryable_status(StatusCode::UNAUTHORIZED));
        assert_eq!(backoff_ms(500, 0), 500);
        assert_eq!(backoff_ms(500, 3), 4_000);
        assert_eq!(backoff_ms(500, 20), 30_000);
    }
}
