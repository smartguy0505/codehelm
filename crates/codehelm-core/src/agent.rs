use async_trait::async_trait;
use codehelm_protocol::{
    AgentAction, AgentEvent, ConversationItem, ModelRequest, Role, TokenUsage, ToolCallRequest,
    ToolSpec,
};
use serde_json::Value;
use thiserror::Error;

use crate::context::bounded_context;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("provider failed: {0}")]
    Provider(String),
    #[error("tool {tool} failed: {message}")]
    Tool { tool: String, message: String },
    #[error("agent exceeded {0} turns")]
    MaxTurns(usize),
    #[error("session persistence failed: {0}")]
    Session(String),
    #[error("agent run cancelled")]
    Cancelled,
    #[error("agent exceeded token budget of {limit} tokens (used {used})")]
    TokenBudget { limit: u64, used: u64 },
}

/// Provider boundary. Implementations translate vendor-native streaming and tool
/// calls into the stable CodeHelm protocol before returning an action.
#[async_trait(?Send)]
pub trait ModelProvider {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError>;
}

#[async_trait(?Send)]
impl<T: ModelProvider + ?Sized> ModelProvider for Box<T> {
    async fn respond(
        &mut self,
        request: &ModelRequest,
        events: &mut dyn EventSink,
    ) -> Result<AgentAction, AgentError> {
        (**self).respond(request, events).await
    }
}

#[async_trait(?Send)]
pub trait ToolExecutor {
    fn specs(&self) -> Vec<ToolSpec>;

    async fn execute(&mut self, tool: &str, args: &Value) -> Result<String, AgentError>;
}

#[async_trait(?Send)]
pub trait ApprovalHandler {
    async fn approve(&mut self, tool: &str, args: &Value, reason: Option<&str>) -> bool;
}

#[async_trait(?Send)]
impl<T: ApprovalHandler + ?Sized> ApprovalHandler for Box<T> {
    async fn approve(&mut self, tool: &str, args: &Value, reason: Option<&str>) -> bool {
        (**self).approve(tool, args, reason).await
    }
}

pub trait EventSink {
    fn emit(&mut self, event: AgentEvent);
}

pub trait ConversationStore {
    fn save(&mut self, items: &[ConversationItem]) -> Result<(), AgentError>;
}

pub struct NoopStore;

impl ConversationStore for NoopStore {
    fn save(&mut self, _items: &[ConversationItem]) -> Result<(), AgentError> {
        Ok(())
    }
}

impl<F> EventSink for F
where
    F: FnMut(AgentEvent),
{
    fn emit(&mut self, event: AgentEvent) {
        self(event);
    }
}

pub struct Agent<P, T, A, E, S = NoopStore> {
    provider: P,
    tools: T,
    approvals: A,
    events: E,
    items: Vec<ConversationItem>,
    max_turns: usize,
    store: S,
    max_total_tokens: Option<u64>,
    total_tokens: u64,
    max_context_chars: usize,
}

impl<P, T, A, E> Agent<P, T, A, E, NoopStore>
where
    P: ModelProvider,
    T: ToolExecutor,
    A: ApprovalHandler,
    E: EventSink,
{
    pub fn new(
        provider: P,
        tools: T,
        approvals: A,
        events: E,
        system_prompt: impl Into<String>,
        max_turns: usize,
    ) -> Self {
        Self {
            provider,
            tools,
            approvals,
            events,
            items: vec![ConversationItem::Message {
                role: Role::System,
                content: system_prompt.into(),
            }],
            max_turns,
            store: NoopStore,
            max_total_tokens: None,
            total_tokens: 0,
            max_context_chars: usize::MAX,
        }
    }

    pub fn with_items(mut self, items: Vec<ConversationItem>) -> Self {
        self.items = items;
        self
    }

    pub fn with_store<S: ConversationStore>(self, store: S) -> Agent<P, T, A, E, S> {
        Agent {
            provider: self.provider,
            tools: self.tools,
            approvals: self.approvals,
            events: self.events,
            items: self.items,
            max_turns: self.max_turns,
            store,
            max_total_tokens: self.max_total_tokens,
            total_tokens: self.total_tokens,
            max_context_chars: self.max_context_chars,
        }
    }

    pub fn with_token_budget(mut self, max_total_tokens: Option<u64>) -> Self {
        self.max_total_tokens = max_total_tokens;
        self
    }

    pub fn with_context_limit(mut self, max_context_chars: usize) -> Self {
        self.max_context_chars = max_context_chars;
        self
    }
}

impl<P, T, A, E, S> Agent<P, T, A, E, S>
where
    P: ModelProvider,
    T: ToolExecutor,
    A: ApprovalHandler,
    E: EventSink,
    S: ConversationStore,
{
    pub async fn run(&mut self, prompt: impl Into<String>) -> Result<String, AgentError> {
        self.items.push(ConversationItem::Message {
            role: Role::User,
            content: prompt.into(),
        });
        self.store.save(&self.items)?;

        for turn in 1..=self.max_turns {
            self.events.emit(AgentEvent::Turn { turn });
            self.events.emit(AgentEvent::ModelStart { turn });
            let context = bounded_context(&self.items, self.max_context_chars);
            if context.removed_items > 0 {
                self.events.emit(AgentEvent::ContextCompacted {
                    removed_items: context.removed_items,
                    retained_items: context.items.len(),
                    estimated_chars: context.estimated_chars,
                });
            }
            let request = ModelRequest {
                items: context.items,
                tools: self.tools.specs(),
            };
            let mut usage = TokenUsage::default();
            let mut events = UsageTrackingSink {
                inner: &mut self.events,
                usage: &mut usage,
            };
            let action = self.provider.respond(&request, &mut events).await?;
            self.total_tokens = self.total_tokens.saturating_add(usage.total());
            self.events.emit(AgentEvent::ModelComplete { turn });
            if let Some(limit) = self
                .max_total_tokens
                .filter(|limit| self.total_tokens > *limit)
            {
                return Err(AgentError::TokenBudget {
                    limit,
                    used: self.total_tokens,
                });
            }

            match action {
                AgentAction::Final { message } => {
                    self.items.push(ConversationItem::Message {
                        role: Role::Assistant,
                        content: message.clone(),
                    });
                    self.store.save(&self.items)?;
                    self.events.emit(AgentEvent::Final {
                        message: message.clone(),
                        turn,
                    });
                    return Ok(message);
                }
                action @ (AgentAction::Tool { .. } | AgentAction::Tools { .. }) => {
                    let mut calls = match action {
                        AgentAction::Tool {
                            id,
                            tool,
                            args,
                            reason,
                        } => vec![ToolCallRequest {
                            id,
                            tool,
                            args,
                            reason,
                        }],
                        AgentAction::Tools { calls } => calls,
                        AgentAction::Final { .. } => unreachable!(),
                    };
                    if calls.is_empty() {
                        return Err(AgentError::Provider(
                            "provider returned an empty tool batch".into(),
                        ));
                    }
                    for (index, call) in calls.iter_mut().enumerate() {
                        if call.id.is_empty() {
                            call.id = format!("codehelm-{turn}-{index}");
                        }
                        self.items.push(ConversationItem::ToolCall {
                            id: call.id.clone(),
                            name: call.tool.clone(),
                            args: call.args.clone(),
                        });
                    }
                    self.store.save(&self.items)?;

                    for call in calls {
                        let ToolCallRequest {
                            id,
                            tool,
                            args,
                            reason,
                        } = call;
                        self.events.emit(AgentEvent::ToolStart {
                            tool: tool.clone(),
                            args: args.clone(),
                            reason: reason.clone(),
                        });
                        if !self
                            .approvals
                            .approve(&tool, &args, reason.as_deref())
                            .await
                        {
                            let message = "permission denied".to_owned();
                            self.events.emit(AgentEvent::ToolDenied {
                                tool: tool.clone(),
                                reason: message.clone(),
                            });
                            self.push_tool_result(&id, &message);
                            self.store.save(&self.items)?;
                            continue;
                        }

                        let result = match self.tools.execute(&tool, &args).await {
                            Ok(result) => result,
                            Err(error) => format!("Tool error: {error}"),
                        };
                        self.events.emit(AgentEvent::ToolResult {
                            tool,
                            result: result.clone(),
                        });
                        self.push_tool_result(&id, &result);
                        self.store.save(&self.items)?;
                    }
                }
            }
        }
        Err(AgentError::MaxTurns(self.max_turns))
    }

    fn push_tool_result(&mut self, id: &str, result: &str) {
        self.items.push(ConversationItem::ToolResult {
            id: id.to_owned(),
            output: result.to_owned(),
        });
    }
}

struct UsageTrackingSink<'a, E> {
    inner: &'a mut E,
    usage: &'a mut TokenUsage,
}

impl<E: EventSink> EventSink for UsageTrackingSink<'_, E> {
    fn emit(&mut self, event: AgentEvent) {
        if let AgentEvent::Usage { usage } = &event {
            *self.usage = *usage;
        }
        self.inner.emit(event);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    struct MockProvider(VecDeque<AgentAction>);

    #[async_trait(?Send)]
    impl ModelProvider for MockProvider {
        async fn respond(
            &mut self,
            _request: &ModelRequest,
            events: &mut dyn EventSink,
        ) -> Result<AgentAction, AgentError> {
            events.emit(AgentEvent::ModelDelta { delta: "ok".into() });
            Ok(self.0.pop_front().expect("mock response"))
        }
    }

    struct MockTools;

    #[async_trait(?Send)]
    impl ToolExecutor for MockTools {
        fn specs(&self) -> Vec<ToolSpec> {
            vec![ToolSpec {
                name: "read_file".into(),
                description: "Read a file".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }]
        }

        async fn execute(&mut self, tool: &str, _args: &Value) -> Result<String, AgentError> {
            assert_eq!(tool, "read_file");
            Ok("contents".into())
        }
    }

    struct Allow;

    #[async_trait(?Send)]
    impl ApprovalHandler for Allow {
        async fn approve(&mut self, _tool: &str, _args: &Value, _reason: Option<&str>) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn executes_tool_and_finishes() {
        let provider = MockProvider(VecDeque::from([
            AgentAction::Tool {
                id: "call-1".into(),
                tool: "read_file".into(),
                args: serde_json::json!({"path": "README.md"}),
                reason: Some("inspect".into()),
            },
            AgentAction::Final {
                message: "done".into(),
            },
        ]));
        let mut events = Vec::new();
        let mut agent = Agent::new(
            provider,
            MockTools,
            Allow,
            |event| events.push(event),
            "system",
            3,
        );

        assert_eq!(agent.run("task").await.unwrap(), "done");
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolResult { result, .. } if result == "contents")
        ));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ModelDelta { delta } if delta == "ok"))
        );
    }

    #[tokio::test]
    async fn stops_at_turn_limit() {
        let provider = MockProvider(VecDeque::from([AgentAction::Tool {
            id: "call-1".into(),
            tool: "read_file".into(),
            args: Value::Null,
            reason: None,
        }]));
        let mut agent = Agent::new(provider, MockTools, Allow, |_| {}, "system", 1);

        assert!(matches!(
            agent.run("task").await,
            Err(AgentError::MaxTurns(1))
        ));
    }

    #[tokio::test]
    async fn preserves_and_executes_every_tool_in_a_batch() {
        let provider = MockProvider(VecDeque::from([
            AgentAction::Tools {
                calls: vec![
                    ToolCallRequest {
                        id: "a".into(),
                        tool: "read_file".into(),
                        args: Value::Null,
                        reason: None,
                    },
                    ToolCallRequest {
                        id: "b".into(),
                        tool: "read_file".into(),
                        args: Value::Null,
                        reason: None,
                    },
                ],
            },
            AgentAction::Final {
                message: "done".into(),
            },
        ]));
        let mut events = Vec::new();
        let mut agent = Agent::new(
            provider,
            MockTools,
            Allow,
            |event| events.push(event),
            "system",
            2,
        );
        assert_eq!(agent.run("task").await.unwrap(), "done");
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolResult { .. }))
                .count(),
            2
        );
    }

    struct UsageProvider;

    #[async_trait(?Send)]
    impl ModelProvider for UsageProvider {
        async fn respond(
            &mut self,
            _request: &ModelRequest,
            events: &mut dyn EventSink,
        ) -> Result<AgentAction, AgentError> {
            events.emit(AgentEvent::Usage {
                usage: TokenUsage {
                    input_tokens: 8,
                    output_tokens: 5,
                    ..TokenUsage::default()
                },
            });
            Ok(AgentAction::Tool {
                id: "blocked".into(),
                tool: "read_file".into(),
                args: Value::Null,
                reason: None,
            })
        }
    }

    #[tokio::test]
    async fn token_budget_stops_before_tool_execution() {
        let mut events = Vec::new();
        let mut agent = Agent::new(
            UsageProvider,
            MockTools,
            Allow,
            |event| events.push(event),
            "system",
            2,
        )
        .with_token_budget(Some(10));

        assert!(matches!(
            agent.run("task").await,
            Err(AgentError::TokenBudget {
                limit: 10,
                used: 13
            })
        ));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolStart { .. }))
        );
    }
}
