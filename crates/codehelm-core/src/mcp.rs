use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    process::Stdio,
    time::Duration,
};

use codehelm_protocol::ToolSpec;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
};

use crate::{agent::AgentError, config::McpServerConfig};

const MCP_PROTOCOL_VERSION: &str = "2025-11-25";
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2024-11-05"];
const MAX_MESSAGE_BYTES: usize = 1024 * 1024;

pub struct McpManager {
    clients: Vec<McpClient>,
    routes: BTreeMap<String, (usize, String)>,
    specs: Vec<ToolSpec>,
}

impl McpManager {
    pub async fn connect(
        configs: &BTreeMap<String, McpServerConfig>,
        cwd: &Path,
    ) -> Result<Self, AgentError> {
        let mut clients = Vec::new();
        let mut routes = BTreeMap::new();
        let mut specs = Vec::new();
        for (server_name, config) in configs {
            let client = McpClient::connect(server_name, config, cwd).await?;
            let client_index = clients.len();
            for (remote_name, spec) in &client.tools {
                let exposed = format!("mcp__{server_name}__{remote_name}");
                if routes
                    .insert(exposed.clone(), (client_index, remote_name.clone()))
                    .is_some()
                {
                    return Err(mcp_error(server_name, "duplicate exposed tool name"));
                }
                specs.push(ToolSpec {
                    name: exposed,
                    description: format!(
                        "MCP server `{server_name}`: {}",
                        spec.description
                            .as_deref()
                            .unwrap_or("No description provided")
                    ),
                    input_schema: spec.input_schema.clone(),
                });
            }
            clients.push(client);
        }
        Ok(Self {
            clients,
            routes,
            specs,
        })
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    pub fn contains(&self, name: &str) -> bool {
        self.routes.contains_key(name)
    }

    pub async fn call(&mut self, name: &str, arguments: &Value) -> Result<String, AgentError> {
        let (client_index, remote_name) = self
            .routes
            .get(name)
            .cloned()
            .ok_or_else(|| mcp_error(name, "unknown MCP tool"))?;
        self.clients[client_index]
            .call(&remote_name, arguments)
            .await
    }
}

struct McpTool {
    description: Option<String>,
    input_schema: Value,
}

struct McpClient {
    name: String,
    child: Child,
    rpc: RpcConnection<BufReader<ChildStdout>, ChildStdin>,
    timeout: Duration,
    tools: BTreeMap<String, McpTool>,
}

impl McpClient {
    async fn connect(name: &str, config: &McpServerConfig, cwd: &Path) -> Result<Self, AgentError> {
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .current_dir(cwd)
            .env_clear()
            .envs(base_environment())
            .envs(&config.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let mut child = command
            .spawn()
            .map_err(|error| mcp_error(name, format!("failed to start: {error}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| mcp_error(name, "stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| mcp_error(name, "stdout unavailable"))?;
        let timeout = Duration::from_millis(config.timeout_ms);
        let mut client = Self {
            name: name.to_owned(),
            child,
            rpc: RpcConnection::new(BufReader::new(stdout), stdin),
            timeout,
            tools: BTreeMap::new(),
        };
        client.initialize().await?;
        client.tools = client.list_tools().await?;
        Ok(client)
    }

    async fn initialize(&mut self) -> Result<(), AgentError> {
        let result = self
            .rpc
            .request(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name":"codehelm","version":env!("CARGO_PKG_VERSION")}
                }),
                self.timeout,
            )
            .await
            .map_err(|error| mcp_error(&self.name, error))?;
        let version = result["protocolVersion"]
            .as_str()
            .ok_or_else(|| mcp_error(&self.name, "initialize omitted protocolVersion"))?;
        if !SUPPORTED_PROTOCOL_VERSIONS.contains(&version) {
            return Err(mcp_error(
                &self.name,
                format!("unsupported negotiated protocol version {version}"),
            ));
        }
        if !result["capabilities"]["tools"].is_object() {
            return Err(mcp_error(&self.name, "server does not advertise tools"));
        }
        self.rpc
            .notify("notifications/initialized", None)
            .await
            .map_err(|error| mcp_error(&self.name, error))
    }

    async fn list_tools(&mut self) -> Result<BTreeMap<String, McpTool>, AgentError> {
        let mut tools = BTreeMap::new();
        let mut cursor: Option<String> = None;
        let mut seen_cursors = BTreeSet::new();
        loop {
            let params = cursor
                .as_ref()
                .map_or_else(|| json!({}), |cursor| json!({"cursor":cursor}));
            let result = self
                .rpc
                .request("tools/list", params, self.timeout)
                .await
                .map_err(|error| mcp_error(&self.name, error))?;
            let listed = result["tools"]
                .as_array()
                .ok_or_else(|| mcp_error(&self.name, "tools/list omitted tools array"))?;
            for tool in listed {
                let name = tool["name"]
                    .as_str()
                    .filter(|name| valid_tool_name(name))
                    .ok_or_else(|| mcp_error(&self.name, "server returned invalid tool name"))?;
                let schema = tool
                    .get("inputSchema")
                    .filter(|schema| schema.is_object())
                    .cloned()
                    .ok_or_else(|| {
                        mcp_error(&self.name, format!("tool {name} has invalid inputSchema"))
                    })?;
                if tools
                    .insert(
                        name.to_owned(),
                        McpTool {
                            description: tool["description"].as_str().map(str::to_owned),
                            input_schema: schema,
                        },
                    )
                    .is_some()
                {
                    return Err(mcp_error(&self.name, format!("duplicate tool {name}")));
                }
            }
            cursor = result["nextCursor"].as_str().map(str::to_owned);
            let Some(next) = &cursor else { break };
            if !seen_cursors.insert(next.clone()) {
                return Err(mcp_error(&self.name, "tools/list repeated a cursor"));
            }
        }
        Ok(tools)
    }

    async fn call(&mut self, name: &str, arguments: &Value) -> Result<String, AgentError> {
        let result = self
            .rpc
            .request(
                "tools/call",
                json!({"name":name,"arguments":arguments}),
                self.timeout,
            )
            .await
            .map_err(|error| mcp_error(&self.name, error))?;
        let content = result["content"]
            .as_array()
            .ok_or_else(|| mcp_error(&self.name, "tools/call omitted content array"))?;
        let output = content
            .iter()
            .map(|item| {
                item["text"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| item.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n");
        if result["isError"].as_bool() == Some(true) {
            Err(mcp_error(&self.name, output))
        } else {
            Ok(if output.is_empty() {
                "(empty MCP result)".into()
            } else {
                output
            })
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

struct RpcConnection<R, W> {
    reader: R,
    writer: W,
    next_id: u64,
}

impl<R, W> RpcConnection<R, W>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    fn new(reader: R, writer: W) -> Self {
        Self {
            reader,
            writer,
            next_id: 1,
        }
    }

    async fn request(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))
            .await?;
        match tokio::time::timeout(timeout, self.wait_for_response(id)).await {
            Ok(result) => result,
            Err(_) => {
                let _ = self
                    .notify(
                        "notifications/cancelled",
                        Some(json!({"requestId":id,"reason":"CodeHelm request timeout"})),
                    )
                    .await;
                Err(format!(
                    "request {method} timed out after {}ms",
                    timeout.as_millis()
                ))
            }
        }
    }

    async fn notify(&mut self, method: &str, params: Option<Value>) -> Result<(), String> {
        let mut message = json!({"jsonrpc":"2.0","method":method});
        if let Some(params) = params {
            message["params"] = params;
        }
        self.send(&message).await
    }

    async fn wait_for_response(&mut self, id: u64) -> Result<Value, String> {
        loop {
            let message = self.read_message().await?;
            if message["id"].as_u64() == Some(id) {
                if let Some(error) = message.get("error") {
                    return Err(format!("JSON-RPC error: {error}"));
                }
                return message
                    .get("result")
                    .cloned()
                    .ok_or_else(|| "JSON-RPC response omitted result".into());
            }
            if message.get("id").is_some() && message["method"].is_string() {
                let response_id = message["id"].clone();
                self.send(&json!({
                    "jsonrpc":"2.0","id":response_id,
                    "error":{"code":-32601,"message":"Client method not supported"}
                }))
                .await?;
            }
        }
    }

    async fn send(&mut self, message: &Value) -> Result<(), String> {
        let mut encoded = serde_json::to_vec(message).map_err(|error| error.to_string())?;
        encoded.push(b'\n');
        self.writer
            .write_all(&encoded)
            .await
            .map_err(|error| error.to_string())?;
        self.writer.flush().await.map_err(|error| error.to_string())
    }

    async fn read_message(&mut self) -> Result<Value, String> {
        let mut line = Vec::new();
        loop {
            let available = self
                .reader
                .fill_buf()
                .await
                .map_err(|error| error.to_string())?;
            if available.is_empty() {
                return Err("MCP server closed stdout".into());
            }
            let take = available
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(available.len(), |position| position + 1);
            if line.len().saturating_add(take) > MAX_MESSAGE_BYTES {
                return Err(format!("MCP message exceeded {MAX_MESSAGE_BYTES} bytes"));
            }
            line.extend_from_slice(&available[..take]);
            self.reader.consume(take);
            if line.last() == Some(&b'\n') {
                break;
            }
        }
        serde_json::from_slice(&line).map_err(|error| format!("invalid MCP JSON: {error}"))
    }
}

fn valid_tool_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        })
}

fn base_environment() -> BTreeMap<String, std::ffi::OsString> {
    ["PATH", "HOME", "USERPROFILE", "TMPDIR", "TEMP", "TMP"]
        .into_iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (name.to_owned(), value)))
        .collect()
}

fn mcp_error(server: &str, message: impl std::fmt::Display) -> AgentError {
    AgentError::Tool {
        tool: format!("mcp:{server}"),
        message: message.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn negotiates_and_calls_a_stdio_tool() {
        let script = r#"
read -r request
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"1"}}}'
read -r initialized
read -r list
printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo","description":"Echo text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}]}}'
read -r call
printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"hello from MCP"}],"isError":false}}'
"#;
        let config = McpServerConfig {
            command: "sh".into(),
            args: vec!["-c".into(), script.into()],
            timeout_ms: 2_000,
            ..McpServerConfig::default()
        };
        let mut configs = BTreeMap::new();
        configs.insert("fixture".into(), config);
        let mut manager = McpManager::connect(&configs, Path::new(".")).await.unwrap();

        assert_eq!(manager.specs().len(), 1);
        assert_eq!(manager.specs()[0].name, "mcp__fixture__echo");
        assert!(manager.contains("mcp__fixture__echo"));
        assert_eq!(
            manager
                .call("mcp__fixture__echo", &json!({"text":"hello"}))
                .await
                .unwrap(),
            "hello from MCP"
        );
    }

    #[test]
    fn validates_spec_tool_names() {
        assert!(valid_tool_name("admin.tools-list_2"));
        assert!(!valid_tool_name("contains spaces"));
        assert!(!valid_tool_name(""));
        assert!(!valid_tool_name(&"x".repeat(129)));
    }
}
