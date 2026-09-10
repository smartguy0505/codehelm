use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{
    agent::AgentError,
    permissions::{Decision, PermissionPolicy, resolve_inside},
};

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    id: String,
    entries: Vec<Entry>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Entry {
    path: String,
    existed: bool,
}

pub struct Checkpoint {
    root: PathBuf,
    directory: PathBuf,
    manifest: Manifest,
}

impl Checkpoint {
    pub fn new(root: &Path) -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let id = format!("{millis}-{}", std::process::id());
        Self {
            root: root.to_owned(),
            directory: root.join(".codehelm/checkpoints").join(&id),
            manifest: Manifest {
                id,
                entries: Vec::new(),
            },
        }
    }

    pub fn id(&self) -> &str {
        &self.manifest.id
    }

    pub fn capture(&mut self, relative: &Path, target: &Path) -> Result<(), AgentError> {
        let path = normalized(relative)?;
        if self.manifest.entries.iter().any(|entry| entry.path == path) {
            return Ok(());
        }
        let original = match fs::read(target) {
            Ok(data) => Some(data),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(checkpoint_error(error)),
        };
        if let Some(data) = &original {
            let backup = self.directory.join("files").join(relative);
            if let Some(parent) = backup.parent() {
                fs::create_dir_all(parent).map_err(checkpoint_error)?;
            }
            atomic_write(&backup, data)?;
        }
        self.manifest.entries.push(Entry {
            path,
            existed: original.is_some(),
        });
        fs::create_dir_all(&self.directory).map_err(checkpoint_error)?;
        let data = serde_json::to_vec_pretty(&self.manifest).map_err(checkpoint_error)?;
        atomic_write(&self.directory.join("manifest.json"), &data)
    }

    pub fn restore(self, policy: &PermissionPolicy) -> Result<usize, AgentError> {
        restore_checkpoint(&self.root, &self.manifest.id, policy)
    }
}

pub fn list_checkpoints(root: &Path) -> Result<Vec<String>, AgentError> {
    let directory = root.join(".codehelm/checkpoints");
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(checkpoint_error(error)),
    };
    let mut ids = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().join("manifest.json").is_file())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect::<Vec<_>>();
    ids.sort();
    ids.reverse();
    Ok(ids)
}

pub fn restore_checkpoint(
    root: &Path,
    id: &str,
    policy: &PermissionPolicy,
) -> Result<usize, AgentError> {
    if id.is_empty()
        || !id.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '-' || character == '_'
        })
    {
        return Err(checkpoint_error("invalid checkpoint id"));
    }
    let root = fs::canonicalize(root).map_err(checkpoint_error)?;
    let directory = root.join(".codehelm/checkpoints").join(id);
    let data = fs::read(directory.join("manifest.json")).map_err(checkpoint_error)?;
    let manifest: Manifest = serde_json::from_slice(&data).map_err(checkpoint_error)?;
    if manifest.id != id {
        return Err(checkpoint_error("checkpoint id does not match manifest"));
    }
    for entry in &manifest.entries {
        let relative = PathBuf::from(&entry.path);
        if policy.write_decision(&relative).map_err(checkpoint_error)? == Decision::Deny {
            return Err(checkpoint_error(format!("restore denied: {}", entry.path)));
        }
        let target = resolve_inside(&root, &relative).map_err(checkpoint_error)?;
        if entry.existed {
            let backup = directory.join("files").join(&relative);
            let original = fs::read(backup).map_err(checkpoint_error)?;
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(checkpoint_error)?;
            }
            atomic_write(&target, &original)?;
        } else {
            match fs::remove_file(target) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(checkpoint_error(error)),
            }
        }
    }
    let count = manifest.entries.len();
    fs::remove_dir_all(directory).map_err(checkpoint_error)?;
    Ok(count)
}

fn normalized(path: &Path) -> Result<String, AgentError> {
    path.to_str()
        .map(|value| value.replace('\\', "/"))
        .ok_or_else(|| checkpoint_error("checkpoint paths must be UTF-8"))
}

fn atomic_write(target: &Path, content: &[u8]) -> Result<(), AgentError> {
    let temporary = target.with_extension(format!("codehelm-{}.tmp", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(checkpoint_error)?;
    if let Err(error) = file.write_all(content).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(checkpoint_error(error));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, target) {
        let _ = fs::remove_file(&temporary);
        return Err(checkpoint_error(error));
    }
    Ok(())
}

fn checkpoint_error(error: impl std::fmt::Display) -> AgentError {
    AgentError::Tool {
        tool: "checkpoint".into(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "codehelm-checkpoint-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn durable_checkpoint_restores_after_owner_is_dropped() {
        let root = workspace();
        let target = root.join("file.txt");
        fs::write(&target, "before").unwrap();
        let id = {
            let mut checkpoint = Checkpoint::new(&root);
            checkpoint.capture(Path::new("file.txt"), &target).unwrap();
            let id = checkpoint.id().to_owned();
            fs::write(&target, "after").unwrap();
            id
        };
        assert_eq!(list_checkpoints(&root).unwrap(), vec![id.clone()]);
        assert_eq!(
            restore_checkpoint(&root, &id, &PermissionPolicy::default()).unwrap(),
            1
        );
        assert_eq!(fs::read_to_string(target).unwrap(), "before");
        assert!(list_checkpoints(&root).unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_ids_cannot_traverse() {
        let root = workspace();
        assert!(restore_checkpoint(&root, "../escape", &PermissionPolicy::default()).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
