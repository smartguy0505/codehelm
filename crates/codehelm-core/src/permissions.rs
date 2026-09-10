use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use globset::{Glob, GlobSet, GlobSetBuilder};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PermissionPolicy {
    pub allow_commands: Vec<String>,
    pub deny_commands: Vec<String>,
    pub deny_read: Vec<String>,
    pub ask_write: Vec<String>,
    pub deny_write: Vec<String>,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            allow_commands: [
                "git status",
                "git diff",
                "git log",
                "npm test",
                "npm run",
                "pnpm test",
                "pytest",
                "cargo test",
                "go test",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            deny_commands: [
                "rm",
                "sudo",
                "shutdown",
                "reboot",
                "mkfs",
                "dd",
                "git reset --hard",
                "git clean",
                "git push --force",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            deny_read: [
                ".env",
                ".env.*",
                "**/.env",
                "**/.env.*",
                "*.pem",
                "**/*.pem",
                "*.key",
                "**/*.key",
                "**/id_rsa",
                "**/id_ed25519",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            ask_write: Vec::new(),
            deny_write: [
                ".git/**",
                ".env",
                ".env.*",
                "**/.env",
                "**/.env.*",
                "*.pem",
                "**/*.pem",
                "*.key",
                "**/*.key",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Ask,
    Deny,
}

#[derive(Debug, Error)]
pub enum PermissionError {
    #[error("path escapes workspace: {0}")]
    PathEscape(String),
    #[error("failed to resolve path {path}: {source}")]
    Resolve {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid permission glob {pattern}: {source}")]
    InvalidGlob {
        pattern: String,
        source: globset::Error,
    },
}

impl PermissionPolicy {
    pub fn command_decision(&self, command: &str) -> Decision {
        let normalized = command.split_whitespace().collect::<Vec<_>>().join(" ");
        if normalized.is_empty()
            || normalized.contains([';', '&', '|', '`', '\n'])
            || normalized.contains("$(")
        {
            return Decision::Deny;
        }
        if prefix_match(&normalized, &self.deny_commands) {
            Decision::Deny
        } else if prefix_match(&normalized, &self.allow_commands) {
            Decision::Allow
        } else {
            Decision::Ask
        }
    }

    pub fn read_decision(&self, path: &Path) -> Result<Decision, PermissionError> {
        Ok(if build_globs(&self.deny_read)?.is_match(normalize(path)) {
            Decision::Deny
        } else {
            Decision::Allow
        })
    }

    pub fn write_decision(&self, path: &Path) -> Result<Decision, PermissionError> {
        let path = normalize(path);
        Ok(if build_globs(&self.deny_write)?.is_match(&path) {
            Decision::Deny
        } else if build_globs(&self.ask_write)?.is_match(&path) {
            Decision::Ask
        } else {
            Decision::Allow
        })
    }
}

pub fn resolve_inside(root: &Path, requested: &Path) -> Result<PathBuf, PermissionError> {
    if requested.is_absolute()
        || requested
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(PermissionError::PathEscape(requested.display().to_string()));
    }
    let root = fs::canonicalize(root).map_err(|source| PermissionError::Resolve {
        path: root.into(),
        source,
    })?;
    let candidate = root.join(requested);

    if candidate.exists() {
        let canonical =
            fs::canonicalize(&candidate).map_err(|source| PermissionError::Resolve {
                path: candidate.clone(),
                source,
            })?;
        if !canonical.starts_with(&root) {
            return Err(PermissionError::PathEscape(requested.display().to_string()));
        }
        return Ok(canonical);
    }

    let mut ancestor = candidate.parent();
    while let Some(path) = ancestor {
        if path.exists() {
            let canonical = fs::canonicalize(path).map_err(|source| PermissionError::Resolve {
                path: path.into(),
                source,
            })?;
            if !canonical.starts_with(&root) {
                return Err(PermissionError::PathEscape(requested.display().to_string()));
            }
            break;
        }
        ancestor = path.parent();
    }
    Ok(candidate)
}

fn prefix_match(command: &str, entries: &[String]) -> bool {
    entries.iter().any(|entry| {
        command == entry
            || command
                .strip_prefix(entry)
                .is_some_and(|rest| rest.starts_with(' '))
    })
}

fn build_globs(patterns: &[String]) -> Result<GlobSet, PermissionError> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern).map_err(|source| PermissionError::InvalidGlob {
            pattern: pattern.clone(),
            source,
        })?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|source| PermissionError::InvalidGlob {
            pattern: "<set>".into(),
            source,
        })
}

fn normalize(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_are_classified_safely() {
        let policy = PermissionPolicy::default();
        assert_eq!(policy.command_decision("cargo test"), Decision::Allow);
        assert_eq!(policy.command_decision("node script.js"), Decision::Ask);
        assert_eq!(policy.command_decision("rm -rf build"), Decision::Deny);
        assert_eq!(
            policy.command_decision("echo ok && rm -rf build"),
            Decision::Deny
        );
    }

    #[test]
    fn sensitive_paths_are_denied() {
        let policy = PermissionPolicy::default();
        assert_eq!(
            policy.read_decision(Path::new("nested/.env")).unwrap(),
            Decision::Deny
        );
        assert_eq!(
            policy.write_decision(Path::new("server.pem")).unwrap(),
            Decision::Deny
        );
        assert_eq!(
            policy.read_decision(Path::new("src/main.rs")).unwrap(),
            Decision::Allow
        );
    }

    #[test]
    fn write_ask_patterns_do_not_override_denials() {
        let policy = PermissionPolicy {
            ask_write: vec!["generated/**".into(), ".env".into()],
            ..PermissionPolicy::default()
        };
        assert_eq!(
            policy
                .write_decision(Path::new("generated/code.rs"))
                .unwrap(),
            Decision::Ask
        );
        assert_eq!(
            policy.write_decision(Path::new(".env")).unwrap(),
            Decision::Deny
        );
    }

    #[test]
    fn parent_paths_are_rejected() {
        assert!(matches!(
            resolve_inside(Path::new("."), Path::new("../secret")),
            Err(PermissionError::PathEscape(_))
        ));
    }
}
