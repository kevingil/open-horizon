//! Minimal OpenAI-compatible chat client.
//!
//! Works against OpenAI, SGLang, vLLM, Ollama, and Anthropic's compat
//! endpoint. Token accounting folds the three usage shapes seen in the
//! wild: Anthropic-via-compat top-level cache fields, OpenAI
//! `prompt_tokens_details.cached_tokens`, and OpenAI
//! `completion_tokens_details.reasoning_tokens`.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use horizon_core::pricing::TokenUsage;

#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub base_url: String,
    pub api_key: String,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(Value::String(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: &str, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(Value::String(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallOut {
    pub id: String,
    pub name: String,
    pub input: Map<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatOutcome {
    pub text: String,
    pub tool_call: Option<ToolCallOut>,
    pub usage: TokenUsage,
    pub reasoning_tokens: u64,
    pub raw_message: Value,
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("provider returned {status}: {body}")]
    Status { status: u16, body: String },
    #[error("malformed response: {0}")]
    Malformed(String),
}

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    config: OpenAiConfig,
}

impl OpenAiClient {
    pub fn new(config: OpenAiConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("reqwest client builds");
        Self { http, config }
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// One chat completion with tools. Retries with exponential backoff
    /// (0.5s doubling, capped at 8s) up to `max_retries` times.
    pub async fn chat_with_retry(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[Value],
        max_tokens: u32,
        extra_body: &Map<String, Value>,
        max_retries: u32,
    ) -> Result<ChatOutcome, OpenAiError> {
        let mut attempt = 0u32;
        loop {
            match self
                .chat(model, messages, tools, max_tokens, extra_body)
                .await
            {
                Ok(out) => return Ok(out),
                Err(e) if attempt >= max_retries => return Err(e),
                Err(e) => {
                    let delay = (0.5f64 * 2f64.powi(attempt as i32)).min(8.0);
                    tracing::warn!(attempt, delay, error = %e, "llm.retry");
                    tokio::time::sleep(Duration::from_secs_f64(delay)).await;
                    attempt += 1;
                }
            }
        }
    }

    pub async fn chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[Value],
        max_tokens: u32,
        extra_body: &Map<String, Value>,
    ) -> Result<ChatOutcome, OpenAiError> {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
            body["tool_choice"] = Value::String("auto".into());
        }
        if let Value::Object(map) = &mut body {
            for (k, v) in extra_body {
                map.insert(k.clone(), v.clone());
            }
        }
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(OpenAiError::Status {
                status: status.as_u16(),
                body: text.chars().take(500).collect(),
            });
        }
        let value: Value =
            serde_json::from_str(&text).map_err(|e| OpenAiError::Malformed(e.to_string()))?;
        Ok(parse_response(&value))
    }
}

pub fn parse_response(value: &Value) -> ChatOutcome {
    let message = value
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or(Value::Null);
    let text = extract_text(&message);
    let tool_call = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .and_then(|calls| calls.first())
        .and_then(|call| {
            let func = call.get("function")?;
            let name = func.get("name")?.as_str()?.to_string();
            let raw_args = func.get("arguments").and_then(Value::as_str).unwrap_or("");
            let input = if raw_args.is_empty() {
                Map::new()
            } else {
                match serde_json::from_str::<Value>(raw_args) {
                    Ok(Value::Object(m)) => m,
                    Ok(other) => Map::from_iter([("_raw".to_string(), other)]),
                    Err(_) => {
                        Map::from_iter([("_raw".to_string(), Value::String(raw_args.to_string()))])
                    }
                }
            };
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(ToolCallOut { id, name, input })
        });
    let (usage, reasoning_tokens) = parse_usage(value.get("usage"));
    ChatOutcome {
        text,
        tool_call,
        usage,
        reasoning_tokens,
        raw_message: message,
    }
}

fn extract_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string(),
        _ => String::new(),
    }
}

fn u64_at(v: Option<&Value>, key: &str) -> u64 {
    v.and_then(|u| u.get(key))
        .and_then(|x| x.as_u64().or_else(|| x.as_f64().map(|f| f.max(0.0) as u64)))
        .unwrap_or(0)
}

fn parse_usage(usage: Option<&Value>) -> (TokenUsage, u64) {
    let prompt = u64_at(usage, "prompt_tokens");
    let completion = u64_at(usage, "completion_tokens");
    let cache_write = u64_at(usage, "cache_creation_input_tokens");
    let cache_read_anthropic = u64_at(usage, "cache_read_input_tokens");
    let cache_read_openai = u64_at(
        usage.and_then(|u| u.get("prompt_tokens_details")),
        "cached_tokens",
    );
    let reasoning = u64_at(
        usage.and_then(|u| u.get("completion_tokens_details")),
        "reasoning_tokens",
    );
    let cache_read = cache_read_anthropic + cache_read_openai;
    (
        TokenUsage {
            input_tokens: prompt
                .saturating_sub(cache_read)
                .saturating_sub(cache_write),
            output_tokens: completion,
            cache_write_tokens: cache_write,
            cache_read_tokens: cache_read,
        },
        reasoning,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_call_and_usage() {
        let v = json!({
            "choices": [{"message": {"role": "assistant", "content": null,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 3}}
        });
        let out = parse_response(&v);
        let call = out.tool_call.unwrap();
        assert_eq!(call.name, "read_file");
        assert_eq!(call.input["path"], "README.md");
        assert_eq!(out.usage.input_tokens, 60);
        assert_eq!(out.usage.cache_read_tokens, 40);
        assert_eq!(out.usage.output_tokens, 10);
        assert_eq!(out.reasoning_tokens, 3);
    }

    #[test]
    fn parses_text_parts_and_anthropic_cache() {
        let v = json!({
            "choices": [{"message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 10}
        });
        let out = parse_response(&v);
        assert_eq!(out.text, "done");
        assert!(out.tool_call.is_none());
        assert_eq!(out.usage.input_tokens, 20);
        assert_eq!(out.usage.cache_write_tokens, 20);
    }
}
