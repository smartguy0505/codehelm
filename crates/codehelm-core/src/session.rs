use std::{
    cell::RefCell,
    fs,
    io::Write,
    path::{Path, PathBuf},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use codehelm_protocol::{AgentEvent, ConversationItem};
use serde::{Deserialize, Serialize};

use crate::agent::{AgentError, ConversationStore};

const SCHEMA_VERSION: u32 = 1;
static SESSION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub schema_version: u32,
    pub id: String,
    pub mode: String,
    pub provider: String,
    pub model: String,
    pub created_at_ms: u128,
    pub updated_at_ms: u128,
    pub items: Vec<ConversationItem>,
}

#[derive(Debug)]
pub struct SessionStore {
    directory: PathBuf,
    session: Session,
}

impl SessionStore {
    pub fn create(
        root: &Path,
        mode: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, AgentError> {
        let now = now_ms();
        let sequence = SESSION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let session = Session {
            schema_version: SCHEMA_VERSION,
            id: format!("{now}-{}-{sequence}", std::process::id()),
            mode: mode.into(),
            provider: provider.into(),
            model: model.into(),
            created_at_ms: now,
            updated_at_ms: now,
            items: Vec::new(),
        };
        let mut store = Self {
            directory: root.join(".codehelm/sessions"),
            session,
        };
        store.persist()?;
        Ok(store)
    }

    pub fn load(root: &Path, id: &str) -> Result<Self, AgentError> {
        let directory = root.join(".codehelm/sessions");
        let id = if id == "latest" {
            latest_id(&directory)?
        } else {
            validate_id(id)?.to_owned()
        };
        let data = fs::read(directory.join(format!("{id}.json"))).map_err(session_error)?;
        let session: Session = serde_json::from_slice(&data)
            .map_err(|error| session_error(format!("session {id} is corrupt: {error}")))?;
        if session.schema_version != SCHEMA_VERSION {
            return Err(session_error(format!(
                "unsupported session schema {}",
                session.schema_version
            )));
        }
        if session.id != id {
            return Err(session_error("session id does not match filename"));
        }
        Ok(Self { directory, session })
    }

    pub fn id(&self) -> &str {
        &self.session.id
    }
    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn append_event(&self, event: &AgentEvent) -> Result<(), AgentError> {
        fs::create_dir_all(&self.directory).map_err(session_error)?;
        let path = self.directory.join(format!("{}.jsonl", self.session.id));
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(session_error)?;
        serde_json::to_writer(&mut file, event).map_err(session_error)?;
        file.write_all(b"\n")
            .and_then(|_| file.sync_data())
            .map_err(session_error)
    }

    fn save_items(&mut self, items: &[ConversationItem]) -> Result<(), AgentError> {
        self.session.items = items.to_vec();
        self.session.updated_at_ms = now_ms();
        self.persist()
    }

    fn persist(&mut self) -> Result<(), AgentError> {
        fs::create_dir_all(&self.directory).map_err(session_error)?;
        let target = self.directory.join(format!("{}.json", self.session.id));
        let data = serde_json::to_vec_pretty(&self.session).map_err(session_error)?;
        atomic_write(&target, &data)
    }
}

#[derive(Clone)]
pub struct SessionRecorder(Rc<RefCell<SessionStore>>);

impl SessionRecorder {
    pub fn new(store: SessionStore) -> Self {
        Self(Rc::new(RefCell::new(store)))
    }
    pub fn id(&self) -> String {
        self.0.borrow().id().to_owned()
    }
    pub fn items(&self) -> Vec<ConversationItem> {
        self.0.borrow().session().items.clone()
    }
    pub fn append_event(&self, event: &AgentEvent) -> Result<(), AgentError> {
        self.0.borrow().append_event(event)
    }
}

impl ConversationStore for SessionRecorder {
    fn save(&mut self, items: &[ConversationItem]) -> Result<(), AgentError> {
        self.0.borrow_mut().save_items(items)
    }
}

fn latest_id(directory: &Path) -> Result<String, AgentError> {
    let entries = fs::read_dir(directory).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            session_error("no saved sessions")
        } else {
            session_error(error)
        }
    })?;
    let mut latest: Option<(u128, String)> = None;
    for entry in entries.filter_map(Result::ok) {
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(data) = fs::read(entry.path()) else {
            continue;
        };
        let Ok(session) = serde_json::from_slice::<Session>(&data) else {
            continue;
        };
        if session.schema_version == SCHEMA_VERSION
            && latest
                .as_ref()
                .is_none_or(|item| session.updated_at_ms > item.0)
        {
            latest = Some((session.updated_at_ms, session.id));
        }
    }
    latest
        .map(|item| item.1)
        .ok_or_else(|| session_error("no valid saved sessions"))
}

fn validate_id(id: &str) -> Result<&str, AgentError> {
    if !id.is_empty()
        && id.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '-' || character == '_'
        })
    {
        Ok(id)
    } else {
        Err(session_error("invalid session id"))
    }
}

fn atomic_write(target: &Path, content: &[u8]) -> Result<(), AgentError> {
    let temporary = target.with_extension(format!("codehelm-{}.tmp", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(session_error)?;
    if let Err(error) = file.write_all(content).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(session_error(error));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary, target) {
        let _ = fs::remove_file(&temporary);
        return Err(session_error(error));
    }
    Ok(())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
fn session_error(error: impl std::fmt::Display) -> AgentError {
    AgentError::Session(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use codehelm_protocol::Role;

    fn workspace() -> PathBuf {
        let sequence = SESSION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("codehelm-session-test-{}-{sequence}", now_ms()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn atomically_saves_and_resumes_latest_session() {
        let root = workspace();
        let store = SessionStore::create(&root, "build", "openai", "model").unwrap();
        let id = store.id().to_owned();
        let mut recorder = SessionRecorder::new(store);
        recorder
            .save(&[ConversationItem::Message {
                role: Role::User,
                content: "task".into(),
            }])
            .unwrap();
        recorder
            .append_event(&AgentEvent::Turn { turn: 1 })
            .unwrap();
        let loaded = SessionStore::load(&root, "latest").unwrap();
        assert_eq!(loaded.id(), id);
        assert_eq!(loaded.session().items.len(), 1);
        assert!(
            root.join(format!(".codehelm/sessions/{id}.jsonl"))
                .is_file()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_traversal_and_corrupt_sessions() {
        let root = workspace();
        assert!(SessionStore::load(&root, "../escape").is_err());
        fs::create_dir_all(root.join(".codehelm/sessions")).unwrap();
        fs::write(root.join(".codehelm/sessions/broken.json"), b"{").unwrap();
        assert!(
            SessionStore::load(&root, "broken")
                .unwrap_err()
                .to_string()
                .contains("corrupt")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
