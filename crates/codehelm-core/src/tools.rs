use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use async_trait::async_trait;
use codehelm_protocol::ToolSpec;
use serde_json::{Value, json};

use crate::{
    agent::{AgentError, ApprovalHandler, ToolExecutor},
    permissions::{Decision, PermissionPolicy, resolve_inside},
};

pub struct WorkspaceTools {
    root: PathBuf,
    policy: PermissionPolicy,
    max_output_chars: usize,
}

impl WorkspaceTools {
    pub fn new(
        root: impl Into<PathBuf>,
        policy: PermissionPolicy,
        max_output_chars: usize,
    ) -> Result<Self, AgentError> {
        let root = fs::canonicalize(root.into()).map_err(|error| tool_error("workspace", error))?;
        Ok(Self {
            root,
            policy,
            max_output_chars,
        })
    }

    fn relative(&self, requested: &str) -> Result<(PathBuf, PathBuf), AgentError> {
        let relative = PathBuf::from(requested);
        if self
            .policy
            .read_decision(&relative)
            .map_err(|error| tool_error("permission", error))?
            == Decision::Deny
        {
            return Err(tool_error(
                "permission",
                format!("read denied: {requested}"),
            ));
        }
        let absolute =
            resolve_inside(&self.root, &relative).map_err(|error| tool_error("path", error))?;
        Ok((relative, absolute))
    }

    fn list_files(&self, args: &Value) -> Result<String, AgentError> {
        let requested = string_arg(args, "path").unwrap_or(".");
        let depth = integer_arg(args, "depth").unwrap_or(3).min(8) as usize;
        let (_, directory) = self.relative(requested)?;
        if !directory.is_dir() {
            return Err(tool_error(
                "list_files",
                format!("not a directory: {requested}"),
            ));
        }
        let mut output = Vec::new();
        self.walk(&directory, depth, &mut output)?;
        output.sort();
        Ok(self.truncate(if output.is_empty() {
            "(empty)".into()
        } else {
            output.join("\n")
        }))
    }

    fn walk(
        &self,
        directory: &Path,
        depth: usize,
        output: &mut Vec<String>,
    ) -> Result<(), AgentError> {
        if depth == 0 {
            return Ok(());
        }
        let entries = fs::read_dir(directory).map_err(|error| tool_error("list_files", error))?;
        for entry in entries {
            let entry = entry.map_err(|error| tool_error("list_files", error))?;
            let path = entry.path();
            let relative = path
                .strip_prefix(&self.root)
                .expect("walk remains inside root");
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some(".git" | ".codehelm" | "node_modules" | "target" | "dist" | "build")
            ) {
                continue;
            }
            if self
                .policy
                .read_decision(relative)
                .map_err(|error| tool_error("permission", error))?
                == Decision::Deny
            {
                continue;
            }
            let file_type = entry
                .file_type()
                .map_err(|error| tool_error("list_files", error))?;
            if file_type.is_symlink() && resolve_inside(&self.root, relative).is_err() {
                continue;
            }
            output.push(normalize(relative));
            if file_type.is_dir() {
                self.walk(&path, depth - 1, output)?;
            }
        }
        Ok(())
    }

    fn read_file(&self, args: &Value) -> Result<String, AgentError> {
        let requested = required_arg(args, "path")?;
        let (_, file) = self.relative(requested)?;
        let data = fs::read_to_string(&file).map_err(|error| tool_error("read_file", error))?;
        let lines = data.lines().collect::<Vec<_>>();
        let start = integer_arg(args, "startLine").unwrap_or(1).max(1) as usize;
        let default_end = start.saturating_add(399);
        let end = integer_arg(args, "endLine")
            .map_or(default_end, |value| value as usize)
            .min(lines.len());
        if start > end.saturating_add(1) {
            return Ok("(no lines)".into());
        }
        let result = lines
            .iter()
            .enumerate()
            .skip(start - 1)
            .take(end.saturating_sub(start - 1))
            .map(|(index, line)| format!("{}: {line}", index + 1))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(self.truncate(if result.is_empty() {
            "(empty)".into()
        } else {
            result
        }))
    }

    fn search(&self, args: &Value) -> Result<String, AgentError> {
        let pattern = required_arg(args, "pattern")?;
        let requested = string_arg(args, "path").unwrap_or(".");
        let (_, search_path) = self.relative(requested)?;
        let output = Command::new("rg")
            .args(["--json", "--line-number", "--color", "never", "--"])
            .arg(pattern)
            .arg(&search_path)
            .current_dir(&self.root)
            .output()
            .map_err(|error| tool_error("search", error))?;
        if !output.status.success() && output.status.code() != Some(1) {
            return Err(tool_error(
                "search",
                String::from_utf8_lossy(&output.stderr),
            ));
        }
        let mut matches = Vec::new();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let Ok(event) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if event["type"] != "match" {
                continue;
            }
            let Some(path) = event["data"]["path"]["text"].as_str() else {
                continue;
            };
            let absolute = Path::new(path);
            let relative = absolute.strip_prefix(&self.root).unwrap_or(absolute);
            if self
                .policy
                .read_decision(relative)
                .map_err(|error| tool_error("permission", error))?
                == Decision::Deny
            {
                continue;
            }
            let line_number = event["data"]["line_number"].as_u64().unwrap_or(0);
            let text = event["data"]["lines"]["text"]
                .as_str()
                .unwrap_or("")
                .trim_end();
            matches.push(format!("{}:{line_number}:{text}", normalize(relative)));
        }
        Ok(self.truncate(if matches.is_empty() {
            "(no matches)".into()
        } else {
            matches.join("\n")
        }))
    }

    fn truncate(&self, value: String) -> String {
        if value.chars().count() <= self.max_output_chars {
            return value;
        }
        let kept = value
            .chars()
            .take(self.max_output_chars)
            .collect::<String>();
        format!("{kept}\n… output truncated")
    }
}

#[async_trait(?Send)]
impl ToolExecutor for WorkspaceTools {
    fn specs(&self) -> Vec<ToolSpec> {
        vec![
            spec(
                "list_files",
                "List workspace files recursively",
                json!({
                    "type":"object", "properties":{"path":{"type":"string"},"depth":{"type":"integer","minimum":1,"maximum":8}}, "additionalProperties":false
                }),
            ),
            spec(
                "read_file",
                "Read a line range from a UTF-8 workspace file",
                json!({
                    "type":"object", "properties":{"path":{"type":"string"},"startLine":{"type":"integer"},"endLine":{"type":"integer"}}, "required":["path"], "additionalProperties":false
                }),
            ),
            spec(
                "search",
                "Search workspace text with a regular expression",
                json!({
                    "type":"object", "properties":{"pattern":{"type":"string"},"path":{"type":"string"}}, "required":["pattern"], "additionalProperties":false
                }),
            ),
        ]
    }

    async fn execute(&mut self, tool: &str, args: &Value) -> Result<String, AgentError> {
        match tool {
            "list_files" => self.list_files(args),
            "read_file" => self.read_file(args),
            "search" => self.search(args),
            _ => Err(tool_error(tool, "unknown or unavailable tool")),
        }
    }
}

pub struct ReadOnlyApproval;

#[async_trait(?Send)]
impl ApprovalHandler for ReadOnlyApproval {
    async fn approve(&mut self, tool: &str, _args: &Value, _reason: Option<&str>) -> bool {
        matches!(tool, "list_files" | "read_file" | "search")
    }
}

fn spec(name: &str, description: &str, input_schema: Value) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: description.into(),
        input_schema,
    }
}

fn required_arg<'a>(args: &'a Value, name: &str) -> Result<&'a str, AgentError> {
    string_arg(args, name).ok_or_else(|| tool_error(name, "missing string argument"))
}

fn string_arg<'a>(args: &'a Value, name: &str) -> Option<&'a str> {
    args[name].as_str()
}
fn integer_arg(args: &Value, name: &str) -> Option<u64> {
    args[name].as_u64()
}
fn normalize(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
fn tool_error(tool: &str, error: impl std::fmt::Display) -> AgentError {
    AgentError::Tool {
        tool: tool.into(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn workspace() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "codehelm-tools-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(path.join("src")).unwrap();
        fs::write(path.join("src/lib.rs"), "first\nneedle\nthird\n").unwrap();
        fs::write(path.join(".env"), "SECRET=value\n").unwrap();
        path
    }

    #[tokio::test]
    async fn reads_lines_and_hides_sensitive_files() {
        let root = workspace();
        let mut tools = WorkspaceTools::new(&root, PermissionPolicy::default(), 1_000).unwrap();
        let output = tools
            .execute(
                "read_file",
                &json!({"path":"src/lib.rs","startLine":2,"endLine":2}),
            )
            .await
            .unwrap();
        assert_eq!(output, "2: needle");
        let list = tools
            .execute("list_files", &json!({"depth":3}))
            .await
            .unwrap();
        assert!(list.contains("src/lib.rs"));
        assert!(!list.contains(".env"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn rejects_parent_paths() {
        let root = workspace();
        let mut tools = WorkspaceTools::new(&root, PermissionPolicy::default(), 1_000).unwrap();
        assert!(
            tools
                .execute("read_file", &json!({"path":"../secret"}))
                .await
                .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
