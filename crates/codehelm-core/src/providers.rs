use std::collections::BTreeMap;

use async_trait::async_trait;
use codehelm_protocol::{
    AgentAction, AgentEvent, ConversationItem, ModelRequest, Role, ToolCallRequest,
};
use futures_util::StreamExt;
use reqwest::{Client, Response};
use serde_json::{Value, json};

use crate::agent::{AgentError, EventSink, ModelProvider};

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
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
        }
    }
}

#[async_trait(?Send)]
impl ModelProvider for OpenAiProvider {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError> {
        let response = self
            .client
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&openai_request(&self.model, request))
            .send()
            .await
            .map_err(provider_error)?;
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
        }
    }
}

#[async_trait(?Send)]
impl ModelProvider for AnthropicProvider {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError> {
        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&anthropic_request(&self.model, self.max_tokens, request))
            .send()
            .await
            .map_err(provider_error)?;
        let mut state = AnthropicState::default();
        stream_sse(response, |event| state.consume(event, events)).await?;
        state.finish()
    }
}

#[derive(Default)]
struct AnthropicState {
    text: String,
    tools: BTreeMap<u64, AnthropicTool>,
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
        }
    }
}

#[async_trait(?Send)]
impl ModelProvider for OllamaProvider {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError> {
        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&ollama_request(&self.model, request))
            .send()
            .await
            .map_err(provider_error)?;
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

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

async fn stream_sse(
    response: Response,
    mut consume: impl FnMut(Value) -> Result<(), AgentError>,
) -> Result<(), AgentError> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.map_err(provider_error)?;
        return Err(AgentError::Provider(format!(
            "HTTP {status}: {}",
            body.chars().take(800).collect::<String>()
        )));
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
    }
    Ok(())
}

async fn stream_ndjson(
    response: Response,
    mut consume: impl FnMut(Value) -> Result<(), AgentError>,
) -> Result<(), AgentError> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.map_err(provider_error)?;
        return Err(AgentError::Provider(format!(
            "HTTP {status}: {}",
            body.chars().take(800).collect::<String>()
        )));
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
    }
    consume_json_line(&buffer, &mut consume)
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
}
