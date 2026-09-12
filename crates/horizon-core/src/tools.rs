//! Tool schema shared between the policy and the environment runner.

use serde_json::{json, Value};

pub const TOOL_NAMES: &[&str] = &[
    "read_file",
    "list_files",
    "search",
    "run_command",
    "write_file",
    "finish",
];

pub const COMMAND_ALLOWLIST: &[&str] = &[
    "ls", "cat", "rg", "grep", "head", "tail", "wc", "find", "python", "python3", "pytest",
];

pub fn is_known_tool(name: &str) -> bool {
    TOOL_NAMES.contains(&name)
}

pub fn is_allowed_command(argv0: &str) -> bool {
    COMMAND_ALLOWLIST.contains(&argv0)
}

/// Internal source of truth: name, description, JSON-Schema parameters.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        }),
        json!({
            "name": "list_files",
            "description": "List files under a workspace directory.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}}, "required": []}
        }),
        json!({
            "name": "search",
            "description": "Search file contents for a regex (ripgrep-style).",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}}, "required": ["pattern"]}
        }),
        json!({
            "name": "run_command",
            "description": "Run a shell command from the workspace allowlist. Allowed: ls, cat, rg, grep, head, tail, wc, find, python, pytest.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
        }),
        json!({
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file inside the workspace. Path must stay within the workspace; no symlinks, no deletes.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
        }),
        json!({
            "name": "finish",
            "description": "Signal that the task is complete. Provide a final summary.",
            "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}
        }),
    ]
}

/// Tools wrapped in OpenAI's function-tool envelope.
pub fn openai_tool_definitions() -> Vec<Value> {
    tool_definitions()
        .into_iter()
        .map(|t| {
            json!({
                "type": "function",
                "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            })
        })
        .collect()
}

/// Parsed policy action: `{"tool": name, "input": {...}}` (or `name`/`args`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub tool: String,
    pub input: serde_json::Map<String, Value>,
}

pub fn parse_action(action: &str) -> Option<ToolCall> {
    let payload: Value = serde_json::from_str(action).ok()?;
    let obj = payload.as_object()?;
    let tool = obj
        .get("tool")
        .or_else(|| obj.get("name"))?
        .as_str()?
        .to_string();
    let input = match obj.get("input").or_else(|| obj.get("args")) {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => return None,
    };
    Some(ToolCall { tool, input })
}

pub fn tool_name_of(action: &str) -> Option<String> {
    parse_action(action).map(|c| c.tool)
}

pub fn is_finish(action: &str) -> bool {
    tool_name_of(action).as_deref() == Some("finish")
}
