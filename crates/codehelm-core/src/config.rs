use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Plan,
    Build,
    Review,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Provider {
    Openai,
    Anthropic,
    Ollama,
    OpenaiCompatible,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Ollama => "ollama",
            Self::OpenaiCompatible => "openai-compatible",
        };
        formatter.write_str(value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Config {
    pub provider: Provider,
    pub model: String,
    pub base_url: Option<String>,
    pub max_turns: usize,
    pub command_timeout_ms: u64,
    pub max_tool_output_chars: usize,
    pub max_instruction_chars: usize,
    pub max_skill_chars: usize,
    pub max_context_chars: usize,
    pub provider_max_retries: usize,
    pub provider_retry_base_ms: u64,
    pub provider_timeout_ms: u64,
    pub max_total_tokens: Option<u64>,
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
    pub permissions: crate::permissions::PermissionPolicy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: Provider::Openai,
            model: "gpt-5-mini".into(),
            base_url: None,
            max_turns: 20,
            command_timeout_ms: 120_000,
            max_tool_output_chars: 30_000,
            max_instruction_chars: 100_000,
            max_skill_chars: 200_000,
            max_context_chars: 400_000,
            provider_max_retries: 3,
            provider_retry_base_ms: 500,
            provider_timeout_ms: 300_000,
            max_total_tokens: None,
            mcp_servers: BTreeMap::new(),
            permissions: Default::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct McpServerConfig {
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub timeout_ms: u64,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            args: Vec::new(),
            env: BTreeMap::new(),
            timeout_ms: 60_000,
        }
    }
}

#[derive(Debug, Default)]
pub struct ConfigOverrides {
    pub provider: Option<Provider>,
    pub model: Option<String>,
    pub max_turns: Option<usize>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read configuration {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid configuration {path}: {source}")]
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("merged configuration is invalid: {0}")]
    Invalid(serde_json::Error),
    #[error("invalid configuration: {0}")]
    Validation(String),
}

pub fn load_config(cwd: &Path, overrides: ConfigOverrides) -> Result<Config, ConfigError> {
    let mut value = serde_json::to_value(Config::default()).expect("default config serializes");

    if let Some(home) = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) {
        merge_file(
            &mut value,
            &PathBuf::from(home).join(".config/codehelm/config.json"),
        )?;
    }
    merge_file(&mut value, &cwd.join(".codehelm/config.json"))?;

    if let Some(provider) = overrides.provider {
        value["provider"] = serde_json::to_value(provider).expect("provider serializes");
    }
    if let Some(model) = overrides.model {
        value["model"] = Value::String(model);
    }
    if let Some(max_turns) = overrides.max_turns {
        value["maxTurns"] = Value::from(max_turns);
    }

    let config: Config = serde_json::from_value(value).map_err(ConfigError::Invalid)?;
    config.validate()?;
    Ok(config)
}

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model.trim().is_empty() {
            return Err(ConfigError::Validation("model must not be empty".into()));
        }
        validate_range("maxTurns", self.max_turns as u64, 1, 1_000)?;
        validate_range("commandTimeoutMs", self.command_timeout_ms, 1, 86_400_000)?;
        validate_range("providerTimeoutMs", self.provider_timeout_ms, 1, 86_400_000)?;
        validate_range(
            "maxToolOutputChars",
            self.max_tool_output_chars as u64,
            1,
            10_000_000,
        )?;
        validate_range(
            "maxInstructionChars",
            self.max_instruction_chars as u64,
            1,
            10_000_000,
        )?;
        validate_range("maxSkillChars", self.max_skill_chars as u64, 1, 10_000_000)?;
        validate_range(
            "maxContextChars",
            self.max_context_chars as u64,
            1,
            10_000_000,
        )?;
        validate_range(
            "providerMaxRetries",
            self.provider_max_retries as u64,
            0,
            20,
        )?;
        if self.provider_max_retries > 0 {
            validate_range(
                "providerRetryBaseMs",
                self.provider_retry_base_ms,
                1,
                60_000,
            )?;
        }
        if self.max_total_tokens == Some(0) {
            return Err(ConfigError::Validation(
                "maxTotalTokens must be greater than zero when set".into(),
            ));
        }
        if let Some(base_url) = &self.base_url {
            let parsed = reqwest::Url::parse(base_url).map_err(|error| {
                ConfigError::Validation(format!("baseUrl is not a valid URL: {error}"))
            })?;
            if !matches!(parsed.scheme(), "http" | "https") {
                return Err(ConfigError::Validation(
                    "baseUrl must use http or https".into(),
                ));
            }
        }
        self.permissions
            .validate()
            .map_err(|error| ConfigError::Validation(error.to_string()))?;
        for (name, server) in &self.mcp_servers {
            if !valid_mcp_name(name) {
                return Err(ConfigError::Validation(format!(
                    "MCP server name `{name}` must contain only letters, digits, underscores, or hyphens"
                )));
            }
            if server.command.trim().is_empty() {
                return Err(ConfigError::Validation(format!(
                    "MCP server `{name}` command must not be empty"
                )));
            }
            validate_range(
                &format!("mcpServers.{name}.timeoutMs"),
                server.timeout_ms,
                1,
                86_400_000,
            )?;
            if server.env.keys().any(|key| {
                key.is_empty()
                    || !key
                        .chars()
                        .all(|character| character == '_' || character.is_ascii_alphanumeric())
            }) {
                return Err(ConfigError::Validation(format!(
                    "MCP server `{name}` contains an invalid environment variable name"
                )));
            }
        }
        Ok(())
    }
}

fn valid_mcp_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|character| {
            character == '_' || character == '-' || character.is_ascii_alphanumeric()
        })
}

fn validate_range(name: &str, value: u64, min: u64, max: u64) -> Result<(), ConfigError> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(ConfigError::Validation(format!(
            "{name} must be between {min} and {max}"
        )))
    }
}

fn merge_file(target: &mut Value, path: &Path) -> Result<(), ConfigError> {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(ConfigError::Read {
                path: path.into(),
                source,
            });
        }
    };
    let value = serde_json::from_str(&source).map_err(|source| ConfigError::Parse {
        path: path.into(),
        source,
    })?;
    merge_value(target, value);
    Ok(())
}

fn merge_value(target: &mut Value, source: Value) {
    match (target, source) {
        (Value::Object(target), Value::Object(source)) => {
            for (key, value) in source {
                merge_value(target.entry(key).or_insert(Value::Null), value);
            }
        }
        (target, source) => *target = source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_values_merge_without_losing_defaults() {
        let mut target = serde_json::json!({"a": 1, "nested": {"left": true, "right": true}});
        merge_value(&mut target, serde_json::json!({"nested": {"right": false}}));
        assert_eq!(
            target,
            serde_json::json!({"a": 1, "nested": {"left": true, "right": false}})
        );
    }

    #[test]
    fn default_configuration_is_valid() {
        let value = serde_json::to_value(Config::default()).unwrap();
        let restored: Config = serde_json::from_value(value).unwrap();
        assert_eq!(restored, Config::default());
        restored.validate().unwrap();
    }

    #[test]
    fn rejects_unsafe_or_nonsensical_limits() {
        let mut config = Config {
            max_turns: 0,
            ..Config::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("maxTurns")
        );
        config = Config {
            provider_max_retries: 21,
            ..Config::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("providerMaxRetries")
        );
        config = Config {
            max_total_tokens: Some(0),
            ..Config::default()
        };
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("maxTotalTokens")
        );
    }

    #[test]
    fn validates_provider_base_url() {
        let mut config = Config {
            base_url: Some("file:///tmp/provider".into()),
            ..Config::default()
        };
        assert!(config.validate().is_err());
        config.base_url = Some("http://127.0.0.1:11434".into());
        config.validate().unwrap();
    }

    #[test]
    fn validates_mcp_process_configuration() {
        let mut config = Config::default();
        config.mcp_servers.insert(
            "bad name".into(),
            McpServerConfig {
                command: "server".into(),
                ..McpServerConfig::default()
            },
        );
        assert!(config.validate().is_err());
        config.mcp_servers.clear();
        config
            .mcp_servers
            .insert("valid-server".into(), McpServerConfig::default());
        assert!(config.validate().is_err());
    }
}
