use std::{
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
    pub max_context_chars: usize,
    pub provider_max_retries: usize,
    pub provider_retry_base_ms: u64,
    pub max_total_tokens: Option<u64>,
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
            max_context_chars: 400_000,
            provider_max_retries: 3,
            provider_retry_base_ms: 500,
            max_total_tokens: None,
            permissions: Default::default(),
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

    serde_json::from_value(value).map_err(ConfigError::Invalid)
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
    }
}
