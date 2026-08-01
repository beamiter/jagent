//! Native provider tool-calling for the agent action protocol.
//!
//! The text protocol ([`crate::session::parse_action`]) asks the model to
//! reply with exactly one JSON object and parses that object strictly. This
//! module is an alternative *wire encoding* of the very same three actions —
//! `run`, `say`, `done` — carried by each provider's native tool-calling, so
//! the provider enforces the schema instead of the prompt asking politely.
//!
//! Nothing about the agent's guarantees changes. A `run` tool call is still
//! only a proposal; it is converted into the same
//! [`crate::session::ParsedAction`] values the text protocol produces and fed
//! into the same state machine, so approval still returns an
//! [`crate::session::ApprovedCommand`] that the caller must deliberately act
//! on (invariant #1). Malformed arguments and unknown tool names fail closed
//! exactly like malformed JSON (invariant #2), and every ingested string is
//! byte-bounded (invariant #3).
//!
//! # Mixing text and tool calls
//!
//! Providers routinely emit a prose preamble in the same reply as a tool call.
//! The rule applied here is fixed and total:
//!
//! - **Exactly one** tool call per reply. Zero tool calls fail closed with
//!   [`ParseError::NoToolCall`] *even when text is present* — prose is never
//!   promoted to an action, the same way prose alone fails the text protocol.
//!   Two or more fail closed with [`ParseError::MultipleToolCalls`]: the state
//!   machine advances one action per turn, and picking one silently would be a
//!   guess about which command the user is being asked to review.
//! - Accompanying text is **never silently dropped**. It is folded into the
//!   action's `thought`, ahead of any `thought` argument the call itself
//!   carried, and therefore surfaces as a visible
//!   [`crate::session::Turn::AssistantThought`] in the transcript. Over-long
//!   preamble is elided to the thought budget rather than rejected, because
//!   model prose is not a protocol field.
//!
//! # Provider support
//!
//! Anthropic and OpenAI-compatible endpoints get provider-correct schemas.
//! Ollama's `/api/chat` has no standard tool-calling in the protocol this
//! crate targets, so native mode returns
//! [`ProviderError::InvalidConfiguration`] there rather than emitting a
//! request that would silently degrade to prose.

use crate::provider::{Provider, ProviderError, Usage, MAX_MODEL_TEXT_BYTES};
use crate::session::{parse_tool_action, ParseError, ParsedAction, MAX_THOUGHT_BYTES};
use crate::text::elide_middle;
use serde_json::{json, Value};
use std::io::{self, Write};

/// Tool name for the command-proposal action.
pub const TOOL_RUN: &str = "run";
/// Tool name for the "say something without proposing a command" action.
pub const TOOL_SAY: &str = "say";
/// Tool name for the task-completion action.
pub const TOOL_DONE: &str = "done";
/// Every tool name this crate advertises, in schema order.
pub const AGENT_TOOL_NAMES: [&str; 3] = [TOOL_RUN, TOOL_SAY, TOOL_DONE];

/// Byte cap for one tool call's raw JSON arguments.
pub const MAX_TOOL_ARGUMENTS_BYTES: usize = 64 * 1024;
/// Byte cap for a tool name. Longer names are unknown by construction.
pub const MAX_TOOL_NAME_BYTES: usize = 64;
/// Byte cap for the provider-assigned call id, which is never interpreted.
pub const MAX_TOOL_ID_BYTES: usize = 256;
/// Cap on tool calls retained from one provider response before failing
/// closed. The protocol accepts exactly one; this only bounds parsing memory.
pub const MAX_STREAM_TOOL_CALLS: usize = 8;

const NATIVE_TOOLS_UNSUPPORTED: &str =
    "Ollama's /api/chat endpoint has no standard tool-calling in the protocol this crate \
     targets; use AgentProtocol::Text for Ollama";

/// Which wire encoding the agent loop uses to carry actions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AgentProtocol {
    /// Today's JSON-in-text protocol: the model replies with exactly one JSON
    /// object, parsed by [`crate::session::parse_action`]. Requests built in
    /// this mode are byte-identical to [`crate::provider::build_chat_request`].
    #[default]
    Text,
    /// The provider's native tool-calling carries the action. Unsupported on
    /// Ollama, which fails with [`ProviderError::InvalidConfiguration`].
    NativeTools,
}

/// One tool call extracted from a model reply, in provider-neutral form.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolCall {
    /// Provider-assigned call id, bounded and otherwise uninterpreted. Empty
    /// when the provider did not supply one.
    pub id: String,
    /// The tool name the model invoked. Untrusted: anything outside
    /// [`AGENT_TOOL_NAMES`] fails closed at [`ToolResponse::to_action`].
    pub name: String,
    /// Raw JSON object text carrying the arguments, exactly as delivered
    /// (Anthropic re-serializes its `input` object; OpenAI passes its
    /// `arguments` string through; streaming concatenates the fragments).
    /// An empty string means "no arguments" and is treated as `{}`.
    pub arguments: String,
}

/// A model reply parsed for native tool calls, alongside the same
/// token-limit and usage facts [`crate::provider::ChatResponse`] carries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolResponse {
    /// Assistant text accompanying the tool call; may be empty. Multiple text
    /// blocks are joined with `"\n"`, matching
    /// [`crate::provider::parse_chat_response_full`].
    pub text: String,
    pub calls: Vec<ToolCall>,
    pub reached_token_limit: bool,
    pub usage: Option<Usage>,
}

impl ToolResponse {
    /// Assemble a reply from parts — the shape a streaming caller has after
    /// folding [`crate::stream::StreamEvent`]s.
    pub fn new(text: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            text: text.into(),
            calls,
            reached_token_limit: false,
            usage: None,
        }
    }

    /// Resolve this reply to exactly one action, applying the mixed
    /// text-and-tool rule documented at the module level. Fails closed on
    /// token-limited output, zero or multiple tool calls, unknown names, and
    /// malformed arguments.
    pub fn to_action(&self) -> Result<ParsedAction, ParseError> {
        if self.reached_token_limit {
            return Err(ParseError::TruncatedResponse);
        }
        let call = match self.calls.as_slice() {
            [] => return Err(ParseError::NoToolCall),
            [call] => call,
            calls => return Err(ParseError::MultipleToolCalls(calls.len())),
        };
        let action = parse_tool_action(&call.name, &call.arguments)?;
        Ok(with_preamble(action, &self.text))
    }
}

/// Fold assistant prose that accompanied a tool call into the action's
/// thought, ahead of any `thought` argument, so it stays visible instead of
/// being dropped.
fn with_preamble(action: ParsedAction, text: &str) -> ParsedAction {
    let preamble = text.trim();
    if preamble.is_empty() {
        return action;
    }
    let preamble = elide_middle(preamble, MAX_THOUGHT_BYTES);
    let mut action = action;
    let thought = match &mut action {
        ParsedAction::Run { thought, .. }
        | ParsedAction::Say { thought, .. }
        | ParsedAction::Done { thought, .. } => thought,
    };
    *thought = Some(match thought.take() {
        Some(existing) => elide_middle(&format!("{preamble}\n\n{existing}"), MAX_THOUGHT_BYTES),
        None => preamble,
    });
    action
}

/// The provider-shaped `tools` array advertising `run`, `say`, and `done`.
///
/// Anthropic emits `{name, description, input_schema}`; OpenAI-compatible
/// emits `{type: "function", function: {name, description, parameters}}`.
pub fn agent_tools(provider: Provider) -> Result<Value, ProviderError> {
    if provider == Provider::Ollama {
        return Err(ProviderError::InvalidConfiguration(
            NATIVE_TOOLS_UNSUPPORTED.into(),
        ));
    }
    Ok(Value::Array(
        tool_specs()
            .iter()
            .map(|spec| match provider {
                Provider::Anthropic => json!({
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.schema(),
                }),
                _ => json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.schema(),
                    },
                }),
            })
            .collect(),
    ))
}

/// The provider-shaped `tool_choice` that requires the model to answer with
/// one of the agent tools rather than prose.
pub fn agent_tool_choice(provider: Provider) -> Result<Value, ProviderError> {
    match provider {
        // `any` = "use one of these tools"; disabling parallel use makes the
        // "exactly one call" rule the provider's job as well as ours.
        Provider::Anthropic => Ok(json!({"type": "any", "disable_parallel_tool_use": true})),
        Provider::OpenAiCompatible => Ok(Value::String("required".into())),
        Provider::Ollama => Err(ProviderError::InvalidConfiguration(
            NATIVE_TOOLS_UNSUPPORTED.into(),
        )),
    }
}

/// Top-level request-body fields native mode adds, in the order they are
/// merged. Everything else about the request is unchanged from text mode.
pub(crate) fn agent_body_fields(
    provider: Provider,
) -> Result<Vec<(&'static str, Value)>, ProviderError> {
    let mut fields = vec![
        ("tools", agent_tools(provider)?),
        ("tool_choice", agent_tool_choice(provider)?),
    ];
    if provider == Provider::OpenAiCompatible {
        // Servers that do not implement this ignore it; those that do stop
        // emitting the multi-call replies the protocol has to reject.
        fields.push(("parallel_tool_calls", json!(false)));
    }
    Ok(fields)
}

struct ToolSpec {
    name: &'static str,
    description: &'static str,
    field: &'static str,
    field_description: &'static str,
}

impl ToolSpec {
    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                self.field: {"type": "string", "description": self.field_description},
            },
            "required": [self.field],
            "additionalProperties": false,
        })
    }
}

/// The schemas deliberately do **not** advertise a `thought` field: visible
/// prose alongside the call already carries reasoning (see the module docs),
/// and an advertised thought field invites hidden reasoning the text protocol
/// also discourages. Ingestion still tolerates one, exactly as
/// [`crate::session::parse_action`] does.
fn tool_specs() -> [ToolSpec; 3] {
    [
        ToolSpec {
            name: TOOL_RUN,
            description: "Propose exactly one shell command for the user to review. This is a \
                          proposal only: the application never executes it without explicit \
                          per-command user approval. Propose one focused command, then wait for \
                          its exit status and output.",
            field: "command",
            field_description: "One visible command line: no newlines, carriage returns, or \
                                other control characters.",
        },
        ToolSpec {
            name: TOOL_SAY,
            description: "Send a note or a clarifying question to the user without proposing a \
                          command. Use this when the request is ambiguous.",
            field: "message",
            field_description: "The question or note to show the user.",
        },
        ToolSpec {
            name: TOOL_DONE,
            description: "Finish the task with a short summary. Use only when the task is \
                          actually complete.",
            field: "message",
            field_description: "A short summary of what was accomplished.",
        },
    ]
}

/// Extract text and native tool calls from one non-streaming reply.
///
/// This is the tool-mode counterpart of
/// [`crate::provider::parse_chat_response_full`]. Unlike that function an
/// empty text body is not an error — a reply that is nothing but a tool call
/// is the normal case. Deciding what the calls *mean* is
/// [`ToolResponse::to_action`]'s job, so every fail-closed rule lives in one
/// place.
pub fn parse_tool_response(
    provider: Provider,
    response: &Value,
) -> Result<ToolResponse, ProviderError> {
    let (text, calls) = match provider {
        Provider::Anthropic => anthropic_parts(response)?,
        Provider::OpenAiCompatible => openai_parts(response)?,
        Provider::Ollama => {
            return Err(ProviderError::InvalidConfiguration(
                NATIVE_TOOLS_UNSUPPORTED.into(),
            ))
        }
    };
    if text.len() > MAX_MODEL_TEXT_BYTES {
        return Err(ProviderError::ResponseTooLarge {
            limit: MAX_MODEL_TEXT_BYTES,
        });
    }
    Ok(ToolResponse {
        text,
        calls,
        reached_token_limit: crate::provider::reached_token_limit(provider, response),
        usage: crate::provider::parse_usage(provider, response),
    })
}

fn anthropic_parts(response: &Value) -> Result<(String, Vec<ToolCall>), ProviderError> {
    let mut calls = Vec::new();
    let parts = response
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let text = crate::provider::join_model_text(
        parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str)),
    )?;
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {}
            Some("tool_use") => {
                ensure_tool_call_capacity(&calls)?;
                let arguments = match part.get("input") {
                    None | Some(Value::Null) => String::new(),
                    Some(input @ Value::Object(_)) => serialize_tool_arguments(input)?,
                    Some(_) => {
                        return Err(malformed("tool_use input is not a JSON object"));
                    }
                };
                calls.push(ToolCall {
                    id: bounded_id(part.get("id")),
                    name: bounded_name(part.get("name"))?,
                    arguments,
                });
            }
            // thinking and future block types carry no action.
            _ => {}
        }
    }
    Ok((text, calls))
}

fn openai_parts(response: &Value) -> Result<(String, Vec<ToolCall>), ProviderError> {
    let message = response.pointer("/choices/0/message");
    let text = match message.and_then(|message| message.get("content")) {
        Some(content) => crate::provider::content_text(content)?.unwrap_or_default(),
        None => String::new(),
    };
    let mut calls = Vec::new();
    match message.and_then(|message| message.get("tool_calls")) {
        None | Some(Value::Null) => {}
        Some(Value::Array(entries)) => {
            for entry in entries {
                ensure_tool_call_capacity(&calls)?;
                let function = entry
                    .get("function")
                    .ok_or_else(|| malformed("tool_calls entry has no function object"))?;
                // The spec sends arguments as a JSON *string*; several
                // OpenAI-compatible servers send the object directly. Both are
                // accepted here and re-parsed strictly either way.
                let arguments = match function.get("arguments") {
                    None | Some(Value::Null) => String::new(),
                    Some(Value::String(arguments)) => bounded_tool_arguments(arguments)?,
                    Some(object @ Value::Object(_)) => serialize_tool_arguments(object)?,
                    Some(_) => {
                        return Err(malformed(
                            "tool call arguments are neither a JSON string nor an object",
                        ))
                    }
                };
                calls.push(ToolCall {
                    id: bounded_id(entry.get("id")),
                    name: bounded_name(function.get("name"))?,
                    arguments,
                });
            }
        }
        Some(_) => return Err(malformed("tool_calls is not an array")),
    }
    Ok((text, calls))
}

fn malformed(detail: &str) -> ProviderError {
    ProviderError::MalformedResponse(detail.to_string())
}

fn ensure_tool_call_capacity(calls: &[ToolCall]) -> Result<(), ProviderError> {
    if calls.len() >= MAX_STREAM_TOOL_CALLS {
        return Err(malformed(&format!(
            "response contains more than {MAX_STREAM_TOOL_CALLS} tool calls"
        )));
    }
    Ok(())
}

fn bounded_tool_arguments(arguments: &str) -> Result<String, ProviderError> {
    if arguments.len() > MAX_TOOL_ARGUMENTS_BYTES {
        return Err(ProviderError::ResponseTooLarge {
            limit: MAX_TOOL_ARGUMENTS_BYTES,
        });
    }
    Ok(arguments.to_string())
}

/// Serialize provider-native argument objects through a bounded writer. Using
/// `Value::to_string()` and checking afterwards would transiently allocate the
/// entire attacker-controlled object, including JSON escaping expansion.
fn serialize_tool_arguments(arguments: &Value) -> Result<String, ProviderError> {
    let mut writer = BoundedJsonWriter::new(MAX_TOOL_ARGUMENTS_BYTES);
    if let Err(error) = serde_json::to_writer(&mut writer, arguments) {
        if writer.exceeded {
            return Err(ProviderError::ResponseTooLarge {
                limit: MAX_TOOL_ARGUMENTS_BYTES,
            });
        }
        return Err(ProviderError::MalformedResponse(format!(
            "could not serialize tool arguments: {error}"
        )));
    }
    String::from_utf8(writer.bytes)
        .map_err(|_| malformed("serialized tool arguments are not valid UTF-8"))
}

struct BoundedJsonWriter {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl BoundedJsonWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            exceeded: false,
        }
    }
}

impl Write for BoundedJsonWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("JSON output exceeds its byte limit"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn bounded_id(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_str)
        .map(|id| elide_middle(id, MAX_TOOL_ID_BYTES))
        .unwrap_or_default()
}

fn bounded_name(value: Option<&Value>) -> Result<String, ProviderError> {
    let name = value
        .and_then(Value::as_str)
        .ok_or_else(|| malformed("tool call has no string name"))?;
    if name.len() > MAX_TOOL_NAME_BYTES {
        return Err(malformed("tool name exceeds its byte limit"));
    }
    Ok(name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{
        build_agent_chat_request, build_agent_chat_request_streaming, build_chat_request,
        build_chat_request_streaming, ChatConfig, Message, Role,
    };

    fn config(provider: Provider) -> ChatConfig {
        ChatConfig {
            provider,
            api_key: Some("test-key".into()),
            model: "test-model".into(),
            base_url: provider.default_base_url().into(),
            max_tokens: 512,
            temperature: None,
        }
    }

    fn history() -> [Message; 1] {
        [Message {
            role: Role::User,
            text: "hello".into(),
        }]
    }

    #[test]
    fn text_mode_requests_are_byte_identical_to_the_0_4_builders() {
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let config = config(provider);
            assert_eq!(
                build_agent_chat_request(&config, Some("sys"), &history(), AgentProtocol::Text)
                    .unwrap(),
                build_chat_request(&config, Some("sys"), &history()).unwrap(),
                "{provider:?}"
            );
            assert_eq!(
                build_agent_chat_request_streaming(
                    &config,
                    Some("sys"),
                    &history(),
                    AgentProtocol::Text
                )
                .unwrap(),
                build_chat_request_streaming(&config, Some("sys"), &history()).unwrap(),
                "{provider:?}"
            );
        }

        // Pinned bytes, so a change inside build_chat_request itself is caught
        // here too and not only relative to a moving target.
        let request = build_agent_chat_request(
            &config(Provider::Anthropic),
            Some("sys"),
            &history(),
            AgentProtocol::Text,
        )
        .unwrap();
        assert_eq!(
            request.body,
            r#"{"max_tokens":512,"messages":[{"content":"hello","role":"user"}],"model":"test-model","system":"sys"}"#
        );
    }

    #[test]
    fn native_mode_only_adds_tool_fields() {
        for provider in [Provider::Anthropic, Provider::OpenAiCompatible] {
            let config = config(provider);
            let text = build_chat_request(&config, Some("sys"), &history()).unwrap();
            let native = build_agent_chat_request(
                &config,
                Some("sys"),
                &history(),
                AgentProtocol::NativeTools,
            )
            .unwrap();
            assert_eq!(native.url, text.url, "{provider:?}");
            assert_eq!(native.headers, text.headers, "{provider:?}");

            let mut native_body: Value = serde_json::from_str(&native.body).unwrap();
            let text_body: Value = serde_json::from_str(&text.body).unwrap();
            let object = native_body.as_object_mut().unwrap();
            assert!(object.remove("tools").is_some(), "{provider:?}");
            assert!(object.remove("tool_choice").is_some(), "{provider:?}");
            object.remove("parallel_tool_calls");
            assert_eq!(native_body, text_body, "{provider:?}");
        }
    }

    #[test]
    fn anthropic_tool_schema_is_provider_correct() {
        let request = build_agent_chat_request(
            &config(Provider::Anthropic),
            Some("sys"),
            &history(),
            AgentProtocol::NativeTools,
        )
        .unwrap();
        let body: Value = serde_json::from_str(&request.body).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        let names: Vec<&str> = tools
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, AGENT_TOOL_NAMES.to_vec());
        assert_eq!(tools[0]["input_schema"]["type"], "object");
        assert_eq!(
            tools[0]["input_schema"]["properties"]["command"]["type"],
            "string"
        );
        assert_eq!(tools[0]["input_schema"]["required"], json!(["command"]));
        assert_eq!(
            tools[0]["input_schema"]["additionalProperties"],
            json!(false)
        );
        // Anthropic never uses the OpenAI envelope.
        assert!(tools[0].get("function").is_none());
        assert_eq!(
            body["tool_choice"],
            json!({"type": "any", "disable_parallel_tool_use": true})
        );
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn openai_tool_schema_is_provider_correct() {
        let request = build_agent_chat_request(
            &config(Provider::OpenAiCompatible),
            Some("sys"),
            &history(),
            AgentProtocol::NativeTools,
        )
        .unwrap();
        let body: Value = serde_json::from_str(&request.body).unwrap();
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[1]["type"], "function");
        assert_eq!(tools[1]["function"]["name"], "say");
        assert_eq!(
            tools[1]["function"]["parameters"]["properties"]["message"]["type"],
            "string"
        );
        assert_eq!(
            tools[1]["function"]["parameters"]["required"],
            json!(["message"])
        );
        // No Anthropic-style top-level input_schema.
        assert!(tools[1].get("input_schema").is_none());
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["parallel_tool_calls"], false);
    }

    #[test]
    fn ollama_native_mode_fails_with_a_clear_configuration_error() {
        for build in [
            build_agent_chat_request(
                &config(Provider::Ollama),
                Some("sys"),
                &history(),
                AgentProtocol::NativeTools,
            ),
            build_agent_chat_request_streaming(
                &config(Provider::Ollama),
                Some("sys"),
                &history(),
                AgentProtocol::NativeTools,
            ),
        ] {
            match build {
                Err(ProviderError::InvalidConfiguration(message)) => {
                    assert!(message.contains("Ollama"), "{message}");
                    assert!(message.contains("AgentProtocol::Text"), "{message}");
                }
                other => panic!("expected InvalidConfiguration, got {other:?}"),
            }
        }
        assert!(matches!(
            agent_tools(Provider::Ollama),
            Err(ProviderError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            parse_tool_response(Provider::Ollama, &json!({"message": {"content": "hi"}})),
            Err(ProviderError::InvalidConfiguration(_))
        ));
        // Text mode on Ollama keeps working.
        assert!(build_agent_chat_request(
            &config(Provider::Ollama),
            Some("sys"),
            &history(),
            AgentProtocol::Text
        )
        .is_ok());
    }

    fn anthropic_reply(blocks: Value) -> Value {
        json!({
            "content": blocks,
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 11, "output_tokens": 22},
        })
    }

    fn openai_reply(message: Value) -> Value {
        json!({
            "choices": [{"message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 22},
        })
    }

    #[test]
    fn anthropic_tool_use_becomes_the_same_parsed_action() {
        let reply = anthropic_reply(json!([
            {"type": "tool_use", "id": "toolu_1", "name": "run",
             "input": {"command": "ls -la"}},
        ]));
        let parsed = parse_tool_response(Provider::Anthropic, &reply).unwrap();
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].id, "toolu_1");
        assert_eq!(parsed.calls[0].name, "run");
        assert_eq!(
            parsed.usage,
            Some(Usage {
                input_tokens: Some(11),
                output_tokens: Some(22),
            })
        );
        assert!(!parsed.reached_token_limit);
        assert_eq!(
            parsed.to_action().unwrap(),
            ParsedAction::Run {
                thought: None,
                command: "ls -la".into(),
            }
        );

        let reply = anthropic_reply(json!([
            {"type": "tool_use", "id": "toolu_2", "name": "done",
             "input": {"message": "all set"}},
        ]));
        assert_eq!(
            parse_tool_response(Provider::Anthropic, &reply)
                .unwrap()
                .to_action()
                .unwrap(),
            ParsedAction::Done {
                thought: None,
                message: "all set".into(),
            }
        );
    }

    #[test]
    fn openai_tool_calls_become_the_same_parsed_action() {
        let reply = openai_reply(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "say", "arguments": "{\"message\":\"which repo?\"}"},
            }],
        }));
        let parsed = parse_tool_response(Provider::OpenAiCompatible, &reply).unwrap();
        assert_eq!(parsed.text, "");
        assert_eq!(parsed.calls[0].id, "call_1");
        assert_eq!(
            parsed.to_action().unwrap(),
            ParsedAction::Say {
                thought: None,
                message: "which repo?".into(),
            }
        );

        // Servers that hand back an arguments object instead of a string.
        let reply = openai_reply(json!({
            "tool_calls": [{
                "id": "call_2",
                "function": {"name": "run", "arguments": {"command": "pwd"}},
            }],
        }));
        assert_eq!(
            parse_tool_response(Provider::OpenAiCompatible, &reply)
                .unwrap()
                .to_action()
                .unwrap(),
            ParsedAction::Run {
                thought: None,
                command: "pwd".into(),
            }
        );
    }

    #[test]
    fn text_alongside_a_tool_call_is_kept_as_the_thought() {
        let reply = anthropic_reply(json!([
            {"type": "text", "text": "Checking the tree first."},
            {"type": "tool_use", "id": "toolu_1", "name": "run",
             "input": {"command": "git status"}},
        ]));
        let parsed = parse_tool_response(Provider::Anthropic, &reply).unwrap();
        assert_eq!(parsed.text, "Checking the tree first.");
        assert_eq!(
            parsed.to_action().unwrap(),
            ParsedAction::Run {
                thought: Some("Checking the tree first.".into()),
                command: "git status".into(),
            }
        );

        // A thought argument does not displace the prose; both survive.
        let reply = openai_reply(json!({
            "content": "preamble",
            "tool_calls": [{
                "function": {
                    "name": "run",
                    "arguments": "{\"command\":\"ls\",\"thought\":\"argument thought\"}",
                },
            }],
        }));
        let ParsedAction::Run { thought, .. } =
            parse_tool_response(Provider::OpenAiCompatible, &reply)
                .unwrap()
                .to_action()
                .unwrap()
        else {
            panic!("expected a run action");
        };
        assert_eq!(thought.as_deref(), Some("preamble\n\nargument thought"));
    }

    #[test]
    fn oversized_preamble_is_elided_rather_than_rejected() {
        let reply = anthropic_reply(json!([
            {"type": "text", "text": "x".repeat(MAX_THOUGHT_BYTES * 3)},
            {"type": "tool_use", "name": "run", "input": {"command": "ls"}},
        ]));
        let ParsedAction::Run { thought, .. } = parse_tool_response(Provider::Anthropic, &reply)
            .unwrap()
            .to_action()
            .unwrap()
        else {
            panic!("expected a run action");
        };
        let thought = thought.expect("preamble kept");
        assert!(thought.len() <= MAX_THOUGHT_BYTES);
        assert!(thought.contains("bytes elided"));
    }

    #[test]
    fn text_without_a_tool_call_fails_closed() {
        let reply = anthropic_reply(json!([
            {"type": "text", "text": "I would run rm -rf / for you."},
        ]));
        let parsed = parse_tool_response(Provider::Anthropic, &reply).unwrap();
        assert_eq!(parsed.to_action(), Err(ParseError::NoToolCall));

        let reply = openai_reply(json!({"content": "just prose"}));
        let parsed = parse_tool_response(Provider::OpenAiCompatible, &reply).unwrap();
        assert_eq!(parsed.to_action(), Err(ParseError::NoToolCall));
    }

    #[test]
    fn multiple_tool_calls_fail_closed() {
        let reply = anthropic_reply(json!([
            {"type": "tool_use", "id": "a", "name": "run", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "b", "name": "run", "input": {"command": "rm -rf /"}},
        ]));
        let parsed = parse_tool_response(Provider::Anthropic, &reply).unwrap();
        assert_eq!(parsed.to_action(), Err(ParseError::MultipleToolCalls(2)));
    }

    #[test]
    fn unknown_tool_names_fail_closed() {
        for name in [
            "shell",
            "RUN",
            "run\u{0}",
            &"r".repeat(MAX_TOOL_NAME_BYTES + 1),
        ] {
            let reply = openai_reply(json!({
                "tool_calls": [{"function": {"name": name, "arguments": "{\"command\":\"ls\"}"}}],
            }));
            let action = match parse_tool_response(Provider::OpenAiCompatible, &reply) {
                // An over-long name is refused at the wire layer.
                Err(ProviderError::MalformedResponse(_)) => continue,
                Ok(parsed) => parsed.to_action(),
                Err(other) => panic!("unexpected provider error for {name:?}: {other}"),
            };
            assert!(
                matches!(&action, Err(ParseError::UnknownAction(reported)) if reported == name),
                "expected UnknownAction for {name:?}, got {action:?}"
            );
        }

        // Surrounding whitespace is trimmed exactly as the text protocol trims
        // its action field, so this stays a known name rather than failing.
        let reply = openai_reply(json!({
            "tool_calls": [{"function": {"name": " run ", "arguments": "{\"command\":\"ls\"}"}}],
        }));
        assert!(parse_tool_response(Provider::OpenAiCompatible, &reply)
            .unwrap()
            .to_action()
            .is_ok());

        let reply = anthropic_reply(json!([
            {"type": "tool_use", "id": "x", "name": 7, "input": {}},
        ]));
        assert!(matches!(
            parse_tool_response(Provider::Anthropic, &reply),
            Err(ProviderError::MalformedResponse(_))
        ));
    }

    #[test]
    fn malformed_arguments_fail_closed() {
        let cases = [
            ("{\"command\":", "invalid JSON"),
            ("[]", "must be an object"),
            ("{}", "missing required field 'command'"),
            (
                "{\"command\":\"ls\",\"extra\":1}",
                "unexpected field 'extra'",
            ),
            ("{\"command\":\"\"}", "must not be empty"),
            ("{\"command\":7}", "must be a string"),
            (
                "{\"command\":\"ls\\nwhoami\"}",
                "must be exactly one visible line",
            ),
            ("{\"command\":\"ls\\twhoami\"}", "control character"),
        ];
        for (arguments, expected) in cases {
            let reply = openai_reply(json!({
                "tool_calls": [{"function": {"name": "run", "arguments": arguments}}],
            }));
            let parsed = parse_tool_response(Provider::OpenAiCompatible, &reply).unwrap();
            let error = parsed.to_action().expect_err(arguments).to_string();
            assert!(error.contains(expected), "{arguments}: {error}");
        }

        // Oversized arguments are refused, not truncated into a command.
        let reply = openai_reply(json!({
            "tool_calls": [{"function": {
                "name": "run",
                "arguments": format!("{{\"command\":\"{}\"}}", "a".repeat(MAX_TOOL_ARGUMENTS_BYTES)),
            }}],
        }));
        assert!(matches!(
            parse_tool_response(Provider::OpenAiCompatible, &reply),
            Err(ProviderError::ResponseTooLarge {
                limit: MAX_TOOL_ARGUMENTS_BYTES
            })
        ));
    }

    #[test]
    fn native_response_parsing_bounds_calls_text_and_argument_serialization() {
        let calls: Vec<Value> = (0..=MAX_STREAM_TOOL_CALLS)
            .map(|index| {
                json!({
                    "id": format!("call_{index}"),
                    "function": {"name": "run", "arguments": "{}"},
                })
            })
            .collect();
        let reply = openai_reply(json!({"tool_calls": calls}));
        assert!(matches!(
            parse_tool_response(Provider::OpenAiCompatible, &reply),
            Err(ProviderError::MalformedResponse(message))
                if message.contains("more than")
        ));

        let half = "x".repeat(MAX_MODEL_TEXT_BYTES / 2);
        let reply = anthropic_reply(json!([
            {"type": "text", "text": half},
            {"type": "text", "text": half},
        ]));
        assert!(matches!(
            parse_tool_response(Provider::Anthropic, &reply),
            Err(ProviderError::ResponseTooLarge {
                limit: MAX_MODEL_TEXT_BYTES
            })
        ));

        // The decoded object is under the limit, but JSON escaping would
        // expand it far past the wire budget. The bounded writer must stop
        // serialization without materializing that expanded string.
        let reply = anthropic_reply(json!([{
            "type": "tool_use",
            "name": "say",
            "input": {"message": "\u{0}".repeat(MAX_TOOL_ARGUMENTS_BYTES / 2)},
        }]));
        assert!(matches!(
            parse_tool_response(Provider::Anthropic, &reply),
            Err(ProviderError::ResponseTooLarge {
                limit: MAX_TOOL_ARGUMENTS_BYTES
            })
        ));
    }

    #[test]
    fn token_limit_and_empty_replies_mirror_the_text_parser() {
        let reply = json!({
            "content": [{"type": "tool_use", "name": "run", "input": {"command": "ls"}}],
            "stop_reason": "max_tokens",
        });
        let parsed = parse_tool_response(Provider::Anthropic, &reply).unwrap();
        assert!(parsed.reached_token_limit);
        assert_eq!(parsed.usage, None);
        assert_eq!(parsed.to_action(), Err(ParseError::TruncatedResponse));

        // An empty reply is not an error here (a bare tool call has no text);
        // it fails closed one layer up instead.
        let parsed = parse_tool_response(Provider::Anthropic, &json!({"content": []})).unwrap();
        assert_eq!(parsed.text, "");
        assert_eq!(parsed.to_action(), Err(ParseError::NoToolCall));
    }
}
