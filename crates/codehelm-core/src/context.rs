use codehelm_protocol::{ConversationItem, Role};

#[derive(Debug, Clone, PartialEq)]
pub struct ContextWindow {
    pub items: Vec<ConversationItem>,
    pub removed_items: usize,
    pub estimated_chars: usize,
}

/// Build a bounded request context without splitting a tool call from its result.
/// Full history remains owned by the session store; this only shapes provider input.
pub fn bounded_context(items: &[ConversationItem], max_chars: usize) -> ContextWindow {
    let systems = items
        .iter()
        .filter(|item| {
            matches!(
                item,
                ConversationItem::Message {
                    role: Role::System,
                    ..
                }
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut units: Vec<Vec<ConversationItem>> = Vec::new();
    for item in items.iter().filter(|item| {
        !matches!(
            item,
            ConversationItem::Message {
                role: Role::System,
                ..
            }
        )
    }) {
        if matches!(item, ConversationItem::Message { .. }) || units.is_empty() {
            units.push(Vec::new());
        }
        units.last_mut().expect("context unit").push(item.clone());
    }

    let system_chars = systems.iter().map(item_chars).sum::<usize>();
    let mut selected = Vec::new();
    let mut chars = system_chars;
    for unit in units.into_iter().rev() {
        let unit_chars = unit.iter().map(item_chars).sum::<usize>();
        if !selected.is_empty() && chars.saturating_add(unit_chars) > max_chars {
            break;
        }
        chars = chars.saturating_add(unit_chars);
        selected.push(unit);
    }
    selected.reverse();

    let mut bounded = systems;
    bounded.extend(selected.into_iter().flatten());
    ContextWindow {
        removed_items: items.len().saturating_sub(bounded.len()),
        estimated_chars: chars,
        items: bounded,
    }
}

fn item_chars(item: &ConversationItem) -> usize {
    match item {
        ConversationItem::Message { content, .. } => content.len(),
        ConversationItem::ToolCall { id, name, args } => id
            .len()
            .saturating_add(name.len())
            .saturating_add(args.to_string().len()),
        ConversationItem::ToolResult { id, output } => id.len().saturating_add(output.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn retains_system_and_recent_complete_tool_unit() {
        let items = vec![
            ConversationItem::Message {
                role: Role::System,
                content: "system".into(),
            },
            ConversationItem::Message {
                role: Role::User,
                content: "old request".repeat(20),
            },
            ConversationItem::Message {
                role: Role::Assistant,
                content: "old answer".repeat(20),
            },
            ConversationItem::Message {
                role: Role::User,
                content: "new".into(),
            },
            ConversationItem::ToolCall {
                id: "1".into(),
                name: "read".into(),
                args: json!({}),
            },
            ConversationItem::ToolResult {
                id: "1".into(),
                output: "result".into(),
            },
        ];

        let window = bounded_context(&items, 40);
        assert_eq!(window.removed_items, 2);
        assert_eq!(window.items.len(), 4);
        assert!(matches!(
            window.items[1],
            ConversationItem::Message {
                role: Role::User,
                ..
            }
        ));
        assert!(matches!(window.items[2], ConversationItem::ToolCall { .. }));
        assert!(matches!(
            window.items[3],
            ConversationItem::ToolResult { .. }
        ));
    }

    #[test]
    fn always_retains_latest_unit_when_it_exceeds_limit() {
        let items = vec![ConversationItem::Message {
            role: Role::User,
            content: "oversized".repeat(20),
        }];
        assert_eq!(bounded_context(&items, 1).items, items);
    }
}
