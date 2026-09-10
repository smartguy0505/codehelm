use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use async_trait::async_trait;
use codehelm_protocol::ToolSpec;
use serde_json::{Value, json};

use crate::{
    agent::{AgentError, ApprovalHandler, ToolExecutor},
    checkpoint::Checkpoint,
    permissions::{Decision, PermissionPolicy, resolve_inside},
};

pub struct WorkspaceTools {
    root: PathBuf,
    policy: PermissionPolicy,
    max_output_chars: usize,
    writable: bool,
    checkpoint: Option<Checkpoint>,
    command_timeout_ms: Option<u64>,
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
            writable: false,
            checkpoint: None,
            command_timeout_ms: None,
        })
    }

    pub fn enable_edits(mut self) -> Self {
        self.writable = true;
        self.checkpoint = Some(Checkpoint::new(&self.root));
        self
    }

    pub fn enable_commands(mut self, timeout_ms: u64) -> Self {
        self.command_timeout_ms = Some(timeout_ms);
        self
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

    fn git_status(&self) -> Result<String, AgentError> {
        let raw = self.run_git_bytes(["status", "--porcelain=v1", "-z", "--branch"])?;
        let records = raw
            .split(|byte| *byte == 0)
            .filter(|record| !record.is_empty())
            .collect::<Vec<_>>();
        let mut lines = Vec::new();
        let mut index = 0;
        while index < records.len() {
            let record = records[index];
            if record.starts_with(b"## ") {
                lines.push(String::from_utf8_lossy(record).into_owned());
                index += 1;
                continue;
            }
            if record.len() < 4 {
                index += 1;
                continue;
            }
            let status = &record[..2];
            let path = &record[3..];
            if self.git_path_visible(path)? {
                lines.push(format!(
                    "{} {}",
                    String::from_utf8_lossy(status),
                    String::from_utf8_lossy(path)
                ));
            }
            index += 1;
            if status.contains(&b'R') || status.contains(&b'C') {
                index += 1;
            }
        }
        Ok(self.truncate(if lines.is_empty() {
            "(no changes)".into()
        } else {
            lines.join("\n")
        }))
    }

    fn git_diff(&self, args: &Value) -> Result<String, AgentError> {
        let staged = args["staged"].as_bool().unwrap_or(false);
        let requested = string_arg(args, "path");
        let mut names = vec!["diff", "--name-only", "-z"];
        if staged {
            names.push("--cached");
        }
        names.push("--");
        if let Some(path) = requested {
            self.relative(path)?;
            names.push(path);
        }
        let changed = self.run_git_bytes(names)?;
        let mut output = String::new();
        for path in changed
            .split(|byte| *byte == 0)
            .filter(|path| !path.is_empty())
        {
            let path = String::from_utf8_lossy(path);
            if !self.git_path_visible(path.as_bytes())? {
                continue;
            }
            let mut diff = vec!["diff", "--no-ext-diff", "--unified=3"];
            if staged {
                diff.push("--cached");
            }
            diff.extend(["--", path.as_ref()]);
            output.push_str(&String::from_utf8_lossy(&self.run_git_bytes(diff)?));
        }
        Ok(self.truncate(if output.is_empty() {
            "(no changes)".into()
        } else {
            output.trim_end().to_owned()
        }))
    }

    fn git_path_visible(&self, path: &[u8]) -> Result<bool, AgentError> {
        let relative = PathBuf::from(String::from_utf8_lossy(path).as_ref());
        Ok(self
            .policy
            .read_decision(&relative)
            .map_err(|error| tool_error("permission", error))?
            != Decision::Deny
            && resolve_inside(&self.root, &relative).is_ok())
    }

    fn run_git_bytes<I, S>(&self, args: I) -> Result<Vec<u8>, AgentError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.root)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .output()
            .map_err(|error| tool_error("git", error))?;
        if !output.status.success() {
            return Err(tool_error(
                "git",
                String::from_utf8_lossy(&output.stderr).trim(),
            ));
        }
        Ok(output.stdout)
    }

    fn write_path(&mut self, requested: &str, content: &[u8]) -> Result<String, AgentError> {
        if !self.writable {
            return Err(tool_error("write_file", "editing is disabled"));
        }
        let relative = PathBuf::from(requested);
        if self
            .policy
            .write_decision(&relative)
            .map_err(|error| tool_error("permission", error))?
            == Decision::Deny
        {
            return Err(tool_error(
                "permission",
                format!("write denied: {requested}"),
            ));
        }
        let target =
            resolve_inside(&self.root, &relative).map_err(|error| tool_error("path", error))?;
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| tool_error("checkpoint", "editing checkpoint is unavailable"))?;
        checkpoint.capture(&relative, &target)?;
        let checkpoint_id = checkpoint.id().to_owned();
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|error| tool_error("write_file", error))?;
        }
        atomic_write(&target, content)?;
        Ok(format!(
            "Wrote {} ({} bytes; checkpoint {checkpoint_id})",
            normalize(&relative),
            content.len()
        ))
    }

    fn write_file(&mut self, args: &Value) -> Result<String, AgentError> {
        let path = required_arg(args, "path")?.to_owned();
        let content = required_arg(args, "content")?.as_bytes().to_vec();
        self.write_path(&path, &content)
    }

    fn edit_preview(&self, tool: &str, args: &Value) -> Result<Option<String>, AgentError> {
        let (path, proposed) = match tool {
            "write_file" => (
                required_arg(args, "path")?.to_owned(),
                required_arg(args, "content")?.to_owned(),
            ),
            "replace_in_file" => {
                let path = required_arg(args, "path")?.to_owned();
                let old = required_arg(args, "oldText")?;
                let new = required_arg(args, "newText")?;
                let (_, target) = self.relative(&path)?;
                let content = fs::read_to_string(target)
                    .map_err(|error| tool_error("replace_in_file", error))?;
                if content.matches(old).count() != 1 {
                    return Err(tool_error(
                        "replace_in_file",
                        "oldText must occur exactly once",
                    ));
                }
                (path, content.replacen(old, new, 1))
            }
            _ => return Ok(None),
        };
        let relative = PathBuf::from(&path);
        if self
            .policy
            .write_decision(&relative)
            .map_err(|error| tool_error("permission", error))?
            == Decision::Deny
        {
            return Err(tool_error("permission", format!("write denied: {path}")));
        }
        let target =
            resolve_inside(&self.root, &relative).map_err(|error| tool_error("path", error))?;
        let current = if target.exists() {
            if self
                .policy
                .read_decision(&relative)
                .map_err(|error| tool_error("permission", error))?
                == Decision::Deny
            {
                return Err(tool_error("permission", format!("read denied: {path}")));
            }
            fs::read_to_string(&target).map_err(|error| tool_error(tool, error))?
        } else {
            String::new()
        };
        if current == proposed {
            return Ok(Some(format!("--- a/{path}\n+++ b/{path}\n(no changes)")));
        }
        Ok(Some(
            self.truncate(unified_diff(&path, &current, &proposed)),
        ))
    }

    fn replace_in_file(&mut self, args: &Value) -> Result<String, AgentError> {
        let path = required_arg(args, "path")?.to_owned();
        let old = required_arg(args, "oldText")?;
        let new = required_arg(args, "newText")?;
        let (_, target) = self.relative(&path)?;
        let content =
            fs::read_to_string(target).map_err(|error| tool_error("replace_in_file", error))?;
        if content.matches(old).count() != 1 {
            return Err(tool_error(
                "replace_in_file",
                "oldText must occur exactly once",
            ));
        }
        self.write_path(&path, content.replacen(old, new, 1).as_bytes())
    }

    fn rollback(&mut self) -> Result<String, AgentError> {
        let checkpoint = self
            .checkpoint
            .take()
            .ok_or_else(|| tool_error("rollback", "editing checkpoint is unavailable"))?;
        let count = checkpoint.restore(&self.policy)?;
        self.checkpoint = Some(Checkpoint::new(&self.root));
        Ok(format!("Rolled back {count} file(s)"))
    }

    async fn run_command(&self, args: &Value) -> Result<String, AgentError> {
        let command = required_arg(args, "command")?;
        if self.policy.command_decision(command) == Decision::Deny {
            return Err(tool_error("run_command", "command is denied by policy"));
        }
        let parts = shell_words::split(command)
            .map_err(|error| tool_error("run_command", format!("invalid command: {error}")))?;
        let (program, arguments) = parts
            .split_first()
            .ok_or_else(|| tool_error("run_command", "command is empty"))?;
        let timeout_ms = self
            .command_timeout_ms
            .ok_or_else(|| tool_error("run_command", "command execution is disabled"))?;
        let child = tokio::process::Command::new(program)
            .args(arguments)
            .current_dir(&self.root)
            .env("CODEHELM_AGENT", "1")
            .kill_on_drop(true)
            .output();
        let output = tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), child)
            .await
            .map_err(|_| tool_error("run_command", format!("timed out after {timeout_ms}ms")))?
            .map_err(|error| tool_error("run_command", error))?;
        let combined = format!(
            "exit code: {}\n{}{}",
            output
                .status
                .code()
                .map_or_else(|| "signal".into(), |code| code.to_string()),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(self.truncate(combined.trim_end().to_owned()))
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
        let mut specs = vec![
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
            spec(
                "git_status",
                "Show the current Git branch and concise working-tree status",
                json!({"type":"object","properties":{},"additionalProperties":false}),
            ),
            spec(
                "git_diff",
                "Show an unstaged or staged Git diff, optionally limited to one path",
                json!({
                    "type":"object",
                    "properties":{"staged":{"type":"boolean"},"path":{"type":"string"}},
                    "additionalProperties":false
                }),
            ),
        ];
        if self.writable {
            specs.extend([
                spec("write_file", "Atomically create or replace a workspace file", json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"],"additionalProperties":false})),
                spec("replace_in_file", "Atomically replace one exact text occurrence", json!({"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["path","oldText","newText"],"additionalProperties":false})),
                spec("rollback_edits", "Rollback all files changed during this run", json!({"type":"object","properties":{},"additionalProperties":false})),
            ]);
        }
        if self.command_timeout_ms.is_some() {
            specs.push(spec("run_command", "Run one policy-approved command without a shell", json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"],"additionalProperties":false})));
        }
        specs
    }

    fn preview(&self, tool: &str, args: &Value) -> Result<Option<String>, AgentError> {
        self.edit_preview(tool, args)
    }

    async fn execute(&mut self, tool: &str, args: &Value) -> Result<String, AgentError> {
        match tool {
            "list_files" => self.list_files(args),
            "read_file" => self.read_file(args),
            "search" => self.search(args),
            "git_status" => self.git_status(),
            "git_diff" => self.git_diff(args),
            "write_file" => self.write_file(args),
            "replace_in_file" => self.replace_in_file(args),
            "rollback_edits" if self.writable => self.rollback(),
            "run_command" => self.run_command(args).await,
            _ => Err(tool_error(tool, "unknown or unavailable tool")),
        }
    }
}

pub struct ReadOnlyApproval;

#[async_trait(?Send)]
impl ApprovalHandler for ReadOnlyApproval {
    async fn approve(&mut self, tool: &str, _args: &Value, _reason: Option<&str>) -> bool {
        matches!(
            tool,
            "list_files" | "read_file" | "search" | "git_status" | "git_diff"
        )
    }
}

pub struct PolicyApproval<F> {
    policy: PermissionPolicy,
    assume_yes: bool,
    prompt: F,
}

impl<F> PolicyApproval<F> {
    pub fn new(policy: PermissionPolicy, assume_yes: bool, prompt: F) -> Self {
        Self {
            policy,
            assume_yes,
            prompt,
        }
    }

    fn decision(&self, tool: &str, args: &Value) -> Decision {
        match tool {
            "list_files" | "read_file" | "search" | "git_status" | "git_diff"
            | "rollback_edits" => Decision::Allow,
            "write_file" | "replace_in_file" => {
                args["path"].as_str().map_or(Decision::Deny, |path| {
                    self.policy
                        .write_decision(Path::new(path))
                        .unwrap_or(Decision::Deny)
                })
            }
            "run_command" => args["command"].as_str().map_or(Decision::Deny, |command| {
                self.policy.command_decision(command)
            }),
            _ => Decision::Deny,
        }
    }

    fn description(tool: &str, args: &Value) -> String {
        match tool {
            "run_command" => format!("run command `{}`", args["command"].as_str().unwrap_or("?")),
            "write_file" | "replace_in_file" => {
                format!("{tool} `{}`", args["path"].as_str().unwrap_or("?"))
            }
            _ => tool.to_owned(),
        }
    }
}

#[async_trait(?Send)]
impl<F: FnMut(&str) -> bool> ApprovalHandler for PolicyApproval<F> {
    async fn approve(&mut self, tool: &str, args: &Value, _reason: Option<&str>) -> bool {
        match self.decision(tool, args) {
            Decision::Allow => true,
            Decision::Ask => self.assume_yes || (self.prompt)(&Self::description(tool, args)),
            Decision::Deny => false,
        }
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

fn atomic_write(target: &Path, content: &[u8]) -> Result<(), AgentError> {
    let temporary = target.with_extension(format!("codehelm-{}.tmp", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| tool_error("atomic_write", error))?;
    if let Err(error) = file.write_all(content).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(tool_error("atomic_write", error));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, target) {
        let _ = fs::remove_file(&temporary);
        return Err(tool_error("atomic_write", error));
    }
    Ok(())
}

fn unified_diff(path: &str, old: &str, new: &str) -> String {
    let old_lines = old.lines().collect::<Vec<_>>();
    let new_lines = new.lines().collect::<Vec<_>>();
    let prefix = old_lines
        .iter()
        .zip(&new_lines)
        .take_while(|(left, right)| left == right)
        .count();
    let max_suffix = old_lines.len().min(new_lines.len()).saturating_sub(prefix);
    let suffix = old_lines
        .iter()
        .rev()
        .zip(new_lines.iter().rev())
        .take(max_suffix)
        .take_while(|(left, right)| left == right)
        .count();
    let context_start = prefix.saturating_sub(3);
    let old_end = old_lines
        .len()
        .saturating_sub(suffix)
        .saturating_add(3)
        .min(old_lines.len());
    let new_end = new_lines
        .len()
        .saturating_sub(suffix)
        .saturating_add(3)
        .min(new_lines.len());
    let mut diff = format!(
        "--- a/{path}\n+++ b/{path}\n@@ -{},{} +{},{} @@\n",
        context_start + 1,
        old_end.saturating_sub(context_start),
        context_start + 1,
        new_end.saturating_sub(context_start)
    );
    for line in &old_lines[context_start..prefix] {
        diff.push_str(&format!(" {line}\n"));
    }
    for line in &old_lines[prefix..old_lines.len().saturating_sub(suffix)] {
        diff.push_str(&format!("-{line}\n"));
    }
    for line in &new_lines[prefix..new_lines.len().saturating_sub(suffix)] {
        diff.push_str(&format!("+{line}\n"));
    }
    for line in &new_lines[new_lines.len().saturating_sub(suffix)..new_end] {
        diff.push_str(&format!(" {line}\n"));
    }
    diff.trim_end().to_owned()
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

    fn git(root: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(root)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("HOME", root)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed");
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

    #[tokio::test]
    async fn edits_can_be_rolled_back() {
        let root = workspace();
        let mut tools = WorkspaceTools::new(&root, PermissionPolicy::default(), 1_000)
            .unwrap()
            .enable_edits();
        tools
            .execute(
                "replace_in_file",
                &json!({"path":"src/lib.rs","oldText":"needle","newText":"changed"}),
            )
            .await
            .unwrap();
        tools
            .execute("write_file", &json!({"path":"new.txt","content":"new"}))
            .await
            .unwrap();
        tools.execute("rollback_edits", &json!({})).await.unwrap();
        assert!(
            fs::read_to_string(root.join("src/lib.rs"))
                .unwrap()
                .contains("needle")
        );
        assert!(!root.join("new.txt").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn edit_preview_is_unified_and_side_effect_free() {
        let root = workspace();
        let tools = WorkspaceTools::new(&root, PermissionPolicy::default(), 1_000)
            .unwrap()
            .enable_edits();
        let preview = tools
            .preview(
                "replace_in_file",
                &json!({"path":"src/lib.rs","oldText":"needle","newText":"changed"}),
            )
            .unwrap()
            .unwrap();
        assert!(preview.starts_with("--- a/src/lib.rs\n+++ b/src/lib.rs\n@@"));
        assert!(preview.contains("-needle\n+changed"));
        assert_eq!(
            fs::read_to_string(root.join("src/lib.rs")).unwrap(),
            "first\nneedle\nthird\n"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn edit_preview_does_not_expose_sensitive_files() {
        let root = workspace();
        let tools = WorkspaceTools::new(&root, PermissionPolicy::default(), 1_000)
            .unwrap()
            .enable_edits();
        let error = tools
            .preview(
                "write_file",
                &json!({"path":".env","content":"replacement"}),
            )
            .unwrap_err();
        assert!(error.to_string().contains("write denied"));
        assert!(!error.to_string().contains("SECRET=value"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn commands_are_shell_free_and_policy_gated() {
        let root = workspace();
        let mut policy = PermissionPolicy::default();
        policy.allow_commands.push("rustc --version".into());
        let mut tools = WorkspaceTools::new(&root, policy, 1_000)
            .unwrap()
            .enable_commands(5_000);
        let result = tools
            .execute("run_command", &json!({"command":"rustc --version"}))
            .await
            .unwrap();
        assert!(result.starts_with("exit code: 0\nrustc "));
        assert!(
            tools
                .execute(
                    "run_command",
                    &json!({"command":"rustc --version && echo unsafe"})
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("denied by policy")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn readonly_git_tools_filter_sensitive_paths() {
        let root = workspace();
        git(&root, &["init", "--quiet"]);
        git(&root, &["config", "user.email", "test@codehelm.local"]);
        git(&root, &["config", "user.name", "CodeHelm Test"]);
        git(&root, &["add", "src/lib.rs", ".env"]);
        git(&root, &["commit", "--quiet", "-m", "initial"]);
        fs::write(root.join("src/lib.rs"), "visible change\n").unwrap();
        fs::write(root.join(".env"), "SECRET=changed\n").unwrap();

        let mut tools = WorkspaceTools::new(&root, PermissionPolicy::default(), 10_000).unwrap();
        let status = tools.execute("git_status", &json!({})).await.unwrap();
        assert!(status.contains("src/lib.rs"));
        assert!(!status.contains(".env"));
        let diff = tools.execute("git_diff", &json!({})).await.unwrap();
        assert!(diff.contains("visible change"));
        assert!(!diff.contains("SECRET"));
        assert!(
            tools
                .execute("git_diff", &json!({"path":"../outside"}))
                .await
                .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn policy_approval_prompts_for_unknown_commands_and_never_bypasses_denials() {
        let mut prompts = Vec::new();
        let mut approval =
            PolicyApproval::new(PermissionPolicy::default(), false, |message: &str| {
                prompts.push(message.to_owned());
                true
            });
        assert!(
            approval
                .approve("run_command", &json!({"command":"node script.js"}), None)
                .await
        );
        assert!(
            !approval
                .approve("run_command", &json!({"command":"rm -rf build"}), None)
                .await
        );
        drop(approval);
        assert_eq!(prompts.len(), 1);

        let mut yes = PolicyApproval::new(PermissionPolicy::default(), true, |_: &str| false);
        assert!(
            yes.approve("run_command", &json!({"command":"node script.js"}), None)
                .await
        );
        assert!(
            !yes.approve("run_command", &json!({"command":"sudo reboot"}), None)
                .await
        );
    }
}
