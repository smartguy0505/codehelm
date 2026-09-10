use std::{collections::BTreeMap, fs, path::Path};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SkillError {
    #[error("failed to inspect skill path {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("skill file escapes its package directory: {0}")]
    Escape(std::path::PathBuf),
    #[error("invalid skill {path}: {message}")]
    Invalid {
        path: std::path::PathBuf,
        message: String,
    },
    #[error("duplicate skill name `{0}`")]
    Duplicate(String),
    #[error("skill instructions exceed the configured {limit} character limit")]
    TooLarge { limit: usize },
}

#[derive(Debug, Clone)]
struct Skill {
    description: String,
    content: String,
}

#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    skills: BTreeMap<String, Skill>,
}

impl SkillRegistry {
    pub fn discover(root: &Path, max_chars: usize) -> Result<Self, SkillError> {
        let root = fs::canonicalize(root).map_err(|source| SkillError::Io {
            path: root.to_owned(),
            source,
        })?;
        let mut registry = Self::default();
        let mut total_chars: usize = 0;
        for relative in [".agents/skills", ".codehelm/skills"] {
            let directory = root.join(relative);
            if !directory.exists() {
                continue;
            }
            let canonical_directory =
                fs::canonicalize(&directory).map_err(|source| SkillError::Io {
                    path: directory.clone(),
                    source,
                })?;
            if !canonical_directory.starts_with(&root) {
                return Err(SkillError::Escape(directory));
            }
            let entries = fs::read_dir(&canonical_directory).map_err(|source| SkillError::Io {
                path: canonical_directory.clone(),
                source,
            })?;
            for entry in entries {
                let entry = entry.map_err(|source| SkillError::Io {
                    path: canonical_directory.clone(),
                    source,
                })?;
                if !entry
                    .file_type()
                    .map_err(|source| SkillError::Io {
                        path: entry.path(),
                        source,
                    })?
                    .is_dir()
                {
                    continue;
                }
                let package = fs::canonicalize(entry.path()).map_err(|source| SkillError::Io {
                    path: entry.path(),
                    source,
                })?;
                if !package.starts_with(&canonical_directory) {
                    return Err(SkillError::Escape(entry.path()));
                }
                let path = package.join("SKILL.md");
                if !path.is_file() {
                    continue;
                }
                let canonical = fs::canonicalize(&path).map_err(|source| SkillError::Io {
                    path: path.clone(),
                    source,
                })?;
                if !canonical.starts_with(&package) {
                    return Err(SkillError::Escape(path));
                }
                let content = fs::read_to_string(&canonical).map_err(|source| SkillError::Io {
                    path: canonical.clone(),
                    source,
                })?;
                total_chars = total_chars.saturating_add(content.chars().count());
                if total_chars > max_chars {
                    return Err(SkillError::TooLarge { limit: max_chars });
                }
                let (name, description) = parse_metadata(&canonical, &content)?;
                if registry
                    .skills
                    .insert(
                        name.clone(),
                        Skill {
                            description,
                            content,
                        },
                    )
                    .is_some()
                {
                    return Err(SkillError::Duplicate(name));
                }
            }
        }
        Ok(registry)
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn list(&self) -> String {
        self.skills
            .iter()
            .map(|(name, skill)| format!("{name}: {}", skill.description))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn read(&self, name: &str) -> Option<&str> {
        self.skills.get(name).map(|skill| skill.content.as_str())
    }
}

fn parse_metadata(path: &Path, content: &str) -> Result<(String, String), SkillError> {
    let normalized = content.replace("\r\n", "\n");
    let metadata = normalized
        .strip_prefix("---\n")
        .and_then(|rest| rest.split_once("\n---\n").map(|(metadata, _)| metadata))
        .ok_or_else(|| SkillError::Invalid {
            path: path.to_owned(),
            message: "expected YAML frontmatter with name and description".into(),
        })?;
    let mut name = None;
    let mut description = None;
    for line in metadata.lines() {
        if let Some((key, value)) = line.split_once(':') {
            let value = value
                .trim()
                .trim_matches(|character| matches!(character, '\'' | '"'));
            match key.trim() {
                "name" => name = Some(value.to_owned()),
                "description" => description = Some(value.to_owned()),
                _ => {}
            }
        }
    }
    let name = name
        .filter(|name| valid_name(name))
        .ok_or_else(|| SkillError::Invalid {
            path: path.to_owned(),
            message: "name must use 1-64 letters, digits, underscores, or hyphens".into(),
        })?;
    let description = description
        .filter(|value| !value.is_empty() && value.chars().count() <= 500)
        .ok_or_else(|| SkillError::Invalid {
            path: path.to_owned(),
            message: "description must contain 1-500 characters".into(),
        })?;
    Ok((name, description))
}

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c == '_' || c == '-' || c.is_ascii_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn workspace() -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "codehelm-skills-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join(".agents/skills/testing")).unwrap();
        root
    }

    #[test]
    fn discovers_metadata_and_loads_instructions_on_demand() {
        let root = workspace();
        fs::write(root.join(".agents/skills/testing/SKILL.md"), "---\nname: testing\ndescription: Run focused tests\n---\n\n# Instructions\nUse cargo test.\n").unwrap();
        let registry = SkillRegistry::discover(&root, 10_000).unwrap();
        assert_eq!(registry.list(), "testing: Run focused tests");
        assert!(registry.read("testing").unwrap().contains("Use cargo test"));
        assert!(registry.read("missing").is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_duplicates_and_oversized_packages() {
        let root = workspace();
        let content = "---\nname: same\ndescription: one\n---\nbody";
        fs::write(root.join(".agents/skills/testing/SKILL.md"), content).unwrap();
        fs::create_dir_all(root.join(".codehelm/skills/duplicate")).unwrap();
        fs::write(root.join(".codehelm/skills/duplicate/SKILL.md"), content).unwrap();
        assert!(matches!(
            SkillRegistry::discover(&root, 10_000),
            Err(SkillError::Duplicate(_))
        ));
        assert!(matches!(
            SkillRegistry::discover(&root, 1),
            Err(SkillError::TooLarge { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_skill_file_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = workspace();
        let outside = root.with_extension("outside-skill");
        fs::write(
            &outside,
            "---\nname: escaped\ndescription: escaped\n---\nsecret",
        )
        .unwrap();
        symlink(&outside, root.join(".agents/skills/testing/SKILL.md")).unwrap();
        assert!(matches!(
            SkillRegistry::discover(&root, 10_000),
            Err(SkillError::Escape(_))
        ));
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(outside).unwrap();
    }
}
