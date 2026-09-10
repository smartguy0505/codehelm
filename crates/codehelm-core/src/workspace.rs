use std::{
    fs,
    path::{Path, PathBuf},
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("failed to resolve workspace path {path}: {source}")]
    Resolve {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("working directory {cwd} is outside workspace {root}")]
    Outside { root: PathBuf, cwd: PathBuf },
    #[error("instruction file escapes workspace: {0}")]
    InstructionEscape(PathBuf),
    #[error("failed to read instruction file {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("project instructions exceed the configured {limit} character limit")]
    TooLarge { limit: usize },
}

pub fn discover_workspace(start: &Path) -> Result<PathBuf, WorkspaceError> {
    let start = fs::canonicalize(start).map_err(|source| WorkspaceError::Resolve {
        path: start.to_owned(),
        source,
    })?;
    for directory in start.ancestors() {
        if directory.join(".codehelm/config.json").is_file() || directory.join(".git").exists() {
            return Ok(directory.to_owned());
        }
    }
    Ok(start)
}

pub fn load_project_instructions(
    root: &Path,
    cwd: &Path,
    max_chars: usize,
) -> Result<String, WorkspaceError> {
    let root = fs::canonicalize(root).map_err(|source| WorkspaceError::Resolve {
        path: root.to_owned(),
        source,
    })?;
    let cwd = fs::canonicalize(cwd).map_err(|source| WorkspaceError::Resolve {
        path: cwd.to_owned(),
        source,
    })?;
    let relative = cwd
        .strip_prefix(&root)
        .map_err(|_| WorkspaceError::Outside {
            root: root.clone(),
            cwd: cwd.clone(),
        })?;
    let mut directories = vec![root.clone()];
    let mut current = root.clone();
    for component in relative.components() {
        current.push(component);
        directories.push(current.clone());
    }

    let mut sections = Vec::new();
    let mut total = 0;
    for directory in directories {
        for name in ["AGENTS.md", "CLAUDE.md"] {
            let path = directory.join(name);
            let content = match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
                Err(source) => return Err(WorkspaceError::Read { path, source }),
            };
            let canonical = fs::canonicalize(&path).map_err(|source| WorkspaceError::Read {
                path: path.clone(),
                source,
            })?;
            if !canonical.starts_with(&root) {
                return Err(WorkspaceError::InstructionEscape(path));
            }
            total += content.chars().count();
            if total > max_chars {
                return Err(WorkspaceError::TooLarge { limit: max_chars });
            }
            let label = canonical
                .strip_prefix(&root)
                .unwrap_or(&canonical)
                .display();
            sections.push(format!("## {label}\n{content}"));
        }
    }
    Ok(if sections.is_empty() {
        "(none)".into()
    } else {
        sections.join("\n\n")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn workspace() -> PathBuf {
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "codehelm-workspace-{id}-{}",
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(path.join("packages/app/src")).unwrap();
        fs::create_dir(path.join(".git")).unwrap();
        path
    }

    #[test]
    fn discovers_repository_from_nested_directory() {
        let root = workspace();
        assert_eq!(
            discover_workspace(&root.join("packages/app/src")).unwrap(),
            root
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn instructions_are_loaded_root_to_leaf() {
        let root = workspace();
        fs::write(root.join("AGENTS.md"), "root").unwrap();
        fs::write(root.join("packages/AGENTS.md"), "package").unwrap();
        fs::write(root.join("packages/app/CLAUDE.md"), "app").unwrap();
        let instructions =
            load_project_instructions(&root, &root.join("packages/app/src"), 100).unwrap();
        assert!(instructions.find("root").unwrap() < instructions.find("package").unwrap());
        assert!(instructions.find("package").unwrap() < instructions.find("app").unwrap());
        assert!(matches!(
            load_project_instructions(&root, &root, 2),
            Err(WorkspaceError::TooLarge { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }
}
