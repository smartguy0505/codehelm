//! Stable messages exchanged between the agent core, frontends, and plugins.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentAction {
    Tool {
        tool: String,
        #[serde(default)]
        args: Value,
        #[serde(default)]
        reason: Option<String>,
    },
    Final {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    Turn {
        turn: usize,
    },
    ToolStart {
        tool: String,
        args: Value,
        reason: Option<String>,
    },
    ToolResult {
        tool: String,
        result: String,
    },
    ToolDenied {
        tool: String,
        reason: String,
    },
    Final {
        message: String,
        turn: usize,
    },
}

pub fn parse_action(input: &str) -> Result<AgentAction, serde_json::Error> {
    if let Ok(action) = serde_json::from_str(input.trim()) {
        return Ok(action);
    }

    if let Some(start) = input.find("```") {
        let after_open = &input[start + 3..];
        let body = after_open
            .strip_prefix("json")
            .unwrap_or(after_open)
            .trim_start_matches([' ', '\t', '\r', '\n']);
        if let Some(end) = body.find("```") {
            return serde_json::from_str(body[..end].trim());
        }
    }

    serde_json::from_str(input.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_direct_action() {
        let action = parse_action(r#"{"type":"final","message":"done"}"#).unwrap();
        assert_eq!(
            action,
            AgentAction::Final {
                message: "done".into()
            }
        );
    }

    #[test]
    fn parses_fenced_action() {
        let action =
            parse_action("```json\n{\"type\":\"final\",\"message\":\"done\"}\n```").unwrap();
        assert_eq!(
            action,
            AgentAction::Final {
                message: "done".into()
            }
        );
    }
}
