//! Protocol-aware, provider-neutral response handling.
//!
//! [`AgentResponse`] is the common high-level result for non-streaming and
//! streaming agent traffic. It keeps the selected [`AgentProtocol`] attached
//! to the corresponding response shape, so integrations do not need to pair
//! text parsing with [`ChatResponse`] or native-tool parsing with
//! [`ToolResponse`] by hand.
//!
//! [`AgentStream`] folds the existing low-level [`StreamEvent`] values into an
//! [`AgentResponse`] without changing which events are returned to the caller.
//! A folded response is available only after [`StreamEvent::Done`]. Protocol
//! errors, truncated streams, and native tool calls in text mode fail closed.

use crate::provider::{
    decode_response_value, parse_chat_response_full, ChatResponse, Provider, ProviderError, Usage,
};
use crate::session::{parse_action, ParseError, ParsedAction};
use crate::stream::{StreamEvent, StreamParser};
use crate::tools::{parse_tool_response, AgentProtocol, ToolCall, ToolResponse};
use serde_json::Value;

const TEXT_RESPONSE_CONTAINED_TOOL_CALLS: &str =
    "text protocol response contained native tool calls";
const TEXT_STREAM_CONTAINED_TOOL_CALL: &str = "text protocol stream contained a native tool call";
const STREAM_NOT_COMPLETE: &str = "stream did not reach a completion event";

/// One complete agent response in the wire protocol selected for its request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentResponse {
    /// A JSON-in-text protocol response.
    Text(ChatResponse),
    /// A provider-native tool-calling response.
    NativeTools(ToolResponse),
}

impl AgentResponse {
    /// Parse one bounded encoded response according to `protocol`.
    ///
    /// The encoded body is decoded exactly once. The resulting [`Value`] is
    /// then handed to the matching protocol parser, preserving the shared
    /// pre-allocation response-envelope limit.
    pub fn parse_bytes(
        provider: Provider,
        protocol: AgentProtocol,
        body: &[u8],
    ) -> Result<Self, ProviderError> {
        let value = decode_response_value(body)?;
        Self::parse_value(provider, protocol, &value)
    }

    /// Parse an already decoded, trusted or transport-bounded response value
    /// according to `protocol`.
    ///
    /// Text mode explicitly rejects provider-native calls rather than
    /// extracting adjacent prose and silently discarding an action.
    pub fn parse_value(
        provider: Provider,
        protocol: AgentProtocol,
        value: &Value,
    ) -> Result<Self, ProviderError> {
        match protocol {
            AgentProtocol::Text => {
                ensure_text_response_has_no_tool_calls(provider, value)?;
                parse_chat_response_full(provider, value).map(Self::Text)
            }
            AgentProtocol::NativeTools => {
                parse_tool_response(provider, value).map(Self::NativeTools)
            }
        }
    }

    /// Resolve this response to the single action consumed by an
    /// [`crate::session::AgentSession`].
    ///
    /// A text response stopped at the provider token limit is never parsed as
    /// an action, even when its partial text happens to be valid JSON.
    pub fn to_action(&self) -> Result<ParsedAction, ParseError> {
        match self {
            Self::Text(response) => {
                if response.reached_token_limit {
                    return Err(ParseError::TruncatedResponse);
                }
                parse_action(&response.text)
            }
            Self::NativeTools(response) => response.to_action(),
        }
    }

    /// Assistant prose carried by this response.
    pub fn text(&self) -> &str {
        match self {
            Self::Text(response) => &response.text,
            Self::NativeTools(response) => &response.text,
        }
    }

    /// Whether the provider stopped at its configured output-token limit.
    pub fn reached_token_limit(&self) -> bool {
        match self {
            Self::Text(response) => response.reached_token_limit,
            Self::NativeTools(response) => response.reached_token_limit,
        }
    }

    /// Provider-reported token usage, when available.
    pub fn usage(&self) -> Option<Usage> {
        match self {
            Self::Text(response) => response.usage,
            Self::NativeTools(response) => response.usage,
        }
    }
}

/// A protocol-aware accumulator around one low-level [`StreamParser`].
///
/// [`Self::push`] and [`Self::finish`] return the parser's events unchanged so
/// integrations can continue rendering deltas as they arrive. In parallel,
/// the wrapper accumulates the normalized response. Call [`Self::into_response`]
/// after observing [`StreamEvent::Done`] to obtain it.
#[derive(Debug)]
pub struct AgentStream {
    parser: StreamParser,
    protocol: AgentProtocol,
    text: String,
    calls: Vec<ToolCall>,
    reached_token_limit: bool,
    usage: Option<Usage>,
    done: bool,
    failure: Option<String>,
}

impl AgentStream {
    /// Start parsing one streaming response for `provider` and `protocol`.
    pub fn new(provider: Provider, protocol: AgentProtocol) -> Self {
        Self {
            parser: StreamParser::new(provider),
            protocol,
            text: String::new(),
            calls: Vec::new(),
            reached_token_limit: false,
            usage: None,
            done: false,
            failure: None,
        }
    }

    /// Feed the next raw response-body chunk and return the underlying parser
    /// events unchanged.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<StreamEvent> {
        let events = self.parser.push(bytes);
        self.accumulate(&events);
        events
    }

    /// Signal EOF and return the underlying parser events unchanged.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let events = self.parser.finish();
        self.accumulate(&events);
        events
    }

    /// Convert a successfully completed stream into its protocol-specific
    /// high-level response.
    ///
    /// A low-level protocol error, a stream that has not emitted
    /// [`StreamEvent::Done`], or a tool call observed in text mode is returned
    /// as [`ProviderError::MalformedResponse`].
    pub fn into_response(self) -> Result<AgentResponse, ProviderError> {
        if let Some(message) = self.failure {
            return Err(ProviderError::MalformedResponse(message));
        }
        if !self.done {
            return Err(ProviderError::MalformedResponse(
                STREAM_NOT_COMPLETE.to_string(),
            ));
        }
        match self.protocol {
            AgentProtocol::Text => Ok(AgentResponse::Text(ChatResponse {
                text: self.text,
                reached_token_limit: self.reached_token_limit,
                usage: self.usage,
            })),
            AgentProtocol::NativeTools => Ok(AgentResponse::NativeTools(ToolResponse {
                text: self.text,
                calls: self.calls,
                reached_token_limit: self.reached_token_limit,
                usage: self.usage,
            })),
        }
    }

    fn accumulate(&mut self, events: &[StreamEvent]) {
        for event in events {
            match event {
                StreamEvent::TextDelta(delta) => self.text.push_str(delta),
                StreamEvent::ReachedTokenLimit => self.reached_token_limit = true,
                StreamEvent::ToolCall(call) => {
                    self.calls.push(call.clone());
                    if self.protocol == AgentProtocol::Text && self.failure.is_none() {
                        self.failure = Some(TEXT_STREAM_CONTAINED_TOOL_CALL.to_string());
                    }
                }
                StreamEvent::Usage(usage) => self.usage = Some(*usage),
                StreamEvent::Done => self.done = true,
                StreamEvent::Protocol(message) => {
                    if self.failure.is_none() {
                        self.failure = Some(message.clone());
                    }
                }
            }
        }
    }
}

fn ensure_text_response_has_no_tool_calls(
    provider: Provider,
    response: &Value,
) -> Result<(), ProviderError> {
    let has_calls = match provider {
        Provider::Anthropic => response
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts
                    .iter()
                    .any(|part| part.get("type").and_then(Value::as_str) == Some("tool_use"))
            }),
        Provider::OpenAiCompatible => {
            nonempty_tool_calls(response.pointer("/choices/0/message/tool_calls"))?
                || response
                    .pointer("/choices/0/message/function_call")
                    .is_some_and(|call| !call.is_null())
        }
        Provider::Ollama => {
            nonempty_tool_calls(response.pointer("/message/tool_calls"))?
                || nonempty_tool_calls(response.get("tool_calls"))?
        }
    };
    if has_calls {
        return Err(ProviderError::MalformedResponse(
            TEXT_RESPONSE_CONTAINED_TOOL_CALLS.to_string(),
        ));
    }
    Ok(())
}

fn nonempty_tool_calls(value: Option<&Value>) -> Result<bool, ProviderError> {
    match value {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Array(calls)) => Ok(!calls.is_empty()),
        Some(_) => Err(ProviderError::MalformedResponse(
            "native tool_calls field is not an array".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::ParsedAction;
    use serde_json::json;

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
        }
    }

    #[test]
    fn text_bytes_parse_to_an_action_and_expose_metadata() {
        let body = json!({
            "content": [{
                "type": "text",
                "text": "{\"action\":\"say\",\"message\":\"hello\"}",
            }],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 9},
        })
        .to_string();

        let response =
            AgentResponse::parse_bytes(Provider::Anthropic, AgentProtocol::Text, body.as_bytes())
                .unwrap();
        assert_eq!(
            response.to_action().unwrap(),
            ParsedAction::Say {
                thought: None,
                message: "hello".into(),
            }
        );
        assert_eq!(
            response.text(),
            "{\"action\":\"say\",\"message\":\"hello\"}"
        );
        assert!(!response.reached_token_limit());
        assert_eq!(response.usage(), Some(usage(7, 9)));
        assert!(matches!(response, AgentResponse::Text(_)));
    }

    #[test]
    fn text_token_limit_precedes_action_parsing() {
        let value = json!({
            "choices": [{
                "message": {"content": "{\"action\":\"done\",\"message\":\"looks complete\"}"},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        });
        let response =
            AgentResponse::parse_value(Provider::OpenAiCompatible, AgentProtocol::Text, &value)
                .unwrap();

        assert!(response.reached_token_limit());
        assert_eq!(response.usage(), Some(usage(2, 3)));
        assert_eq!(response.to_action(), Err(ParseError::TruncatedResponse));
    }

    #[test]
    fn native_bytes_parse_once_into_a_tool_response() {
        let body = json!({
            "content": [
                {"type": "text", "text": "Inspecting first."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "run",
                    "input": {"command": "pwd"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 11, "output_tokens": 13},
        })
        .to_string();

        let response = AgentResponse::parse_bytes(
            Provider::Anthropic,
            AgentProtocol::NativeTools,
            body.as_bytes(),
        )
        .unwrap();
        assert!(matches!(response, AgentResponse::NativeTools(_)));
        assert_eq!(response.text(), "Inspecting first.");
        assert_eq!(response.usage(), Some(usage(11, 13)));
        assert_eq!(
            response.to_action().unwrap(),
            ParsedAction::Run {
                thought: Some("Inspecting first.".into()),
                command: "pwd".into(),
            }
        );
    }

    #[test]
    fn text_mode_rejects_native_tool_calls_for_every_provider_shape() {
        let fixtures = [
            (
                Provider::Anthropic,
                json!({
                    "content": [
                        {"type": "text", "text": "prose"},
                        {"type": "tool_use", "name": "run", "input": {"command": "pwd"}},
                    ],
                }),
            ),
            (
                Provider::OpenAiCompatible,
                json!({
                    "choices": [{"message": {
                        "content": "prose",
                        "tool_calls": [{"function": {"name": "run", "arguments": "{}"}}],
                    }}],
                }),
            ),
            (
                Provider::Ollama,
                json!({
                    "message": {
                        "content": "prose",
                        "tool_calls": [{"function": {"name": "run", "arguments": {}}}],
                    },
                }),
            ),
        ];

        for (provider, value) in fixtures {
            assert!(matches!(
                AgentResponse::parse_value(provider, AgentProtocol::Text, &value),
                Err(ProviderError::MalformedResponse(message))
                    if message == TEXT_RESPONSE_CONTAINED_TOOL_CALLS
            ));
        }
    }

    #[test]
    fn text_mode_accepts_an_explicitly_empty_tool_call_array() {
        let value = json!({
            "choices": [{
                "message": {
                    "content": "{\"action\":\"done\",\"message\":\"ok\"}",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
        });
        let response =
            AgentResponse::parse_value(Provider::OpenAiCompatible, AgentProtocol::Text, &value)
                .unwrap();
        assert!(matches!(
            response.to_action(),
            Ok(ParsedAction::Done { .. })
        ));
    }

    fn text_stream_body(finish_reason: &str) -> String {
        let frame = |value: Value| format!("data: {value}\n\n");
        [
            frame(json!({
                "choices": [{
                    "delta": {"content": "{\"action\":\"done\","},
                    "finish_reason": null,
                }],
            })),
            frame(json!({
                "choices": [{
                    "delta": {"content": "\"message\":\"complete\"}"},
                    "finish_reason": null,
                }],
            })),
            frame(json!({
                "choices": [{"delta": {}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 19},
            })),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat()
    }

    fn tool_stream_body() -> String {
        concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"run\",\"arguments\":\"{\\\"command\\\":\\\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"pwd\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":23,\"completion_tokens\":29}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string()
    }

    #[test]
    fn stream_returns_low_level_events_unchanged_while_folding_text() {
        let body = text_stream_body("stop");
        let split = body.len() / 2;
        let chunks = [&body.as_bytes()[..split], &body.as_bytes()[split..]];

        let mut direct = StreamParser::new(Provider::OpenAiCompatible);
        let mut expected = Vec::new();
        expected.extend(direct.push(chunks[0]));
        expected.extend(direct.push(chunks[1]));

        let mut stream = AgentStream::new(Provider::OpenAiCompatible, AgentProtocol::Text);
        let mut actual = Vec::new();
        actual.extend(stream.push(chunks[0]));
        actual.extend(stream.push(chunks[1]));
        assert_eq!(actual, expected);
        assert!(actual.contains(&StreamEvent::Done));

        let response = stream.into_response().unwrap();
        assert_eq!(response.usage(), Some(usage(17, 19)));
        assert_eq!(
            response.to_action().unwrap(),
            ParsedAction::Done {
                thought: None,
                message: "complete".into(),
            }
        );
    }

    #[test]
    fn streamed_text_at_the_token_limit_never_becomes_an_action() {
        let mut stream = AgentStream::new(Provider::OpenAiCompatible, AgentProtocol::Text);
        let events = stream.push(text_stream_body("length").as_bytes());
        assert!(events.contains(&StreamEvent::ReachedTokenLimit));
        assert!(events.contains(&StreamEvent::Done));

        let response = stream.into_response().unwrap();
        assert!(response.reached_token_limit());
        assert_eq!(response.to_action(), Err(ParseError::TruncatedResponse));
    }

    #[test]
    fn native_stream_folds_tool_calls_and_metadata() {
        let mut stream = AgentStream::new(Provider::OpenAiCompatible, AgentProtocol::NativeTools);
        let events = stream.push(tool_stream_body().as_bytes());
        assert!(events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_))));
        assert!(events.contains(&StreamEvent::Done));

        let response = stream.into_response().unwrap();
        assert!(matches!(response, AgentResponse::NativeTools(_)));
        assert_eq!(response.usage(), Some(usage(23, 29)));
        assert_eq!(
            response.to_action().unwrap(),
            ParsedAction::Run {
                thought: None,
                command: "pwd".into(),
            }
        );
    }

    #[test]
    fn text_stream_with_a_tool_call_fails_closed() {
        let mut stream = AgentStream::new(Provider::OpenAiCompatible, AgentProtocol::Text);
        let events = stream.push(tool_stream_body().as_bytes());
        // AgentStream remains a transparent event wrapper, but refuses to
        // promote the mismatched event sequence into an AgentResponse.
        assert!(events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_))));
        assert!(events.contains(&StreamEvent::Done));
        assert!(matches!(
            stream.into_response(),
            Err(ProviderError::MalformedResponse(message))
                if message == TEXT_STREAM_CONTAINED_TOOL_CALL
        ));
    }

    #[test]
    fn protocol_error_and_unfinished_streams_do_not_produce_responses() {
        let mut malformed = AgentStream::new(Provider::OpenAiCompatible, AgentProtocol::Text);
        let events = malformed.push(b"data: {not json}\n\n");
        assert!(events
            .iter()
            .any(|event| matches!(event, StreamEvent::Protocol(_))));
        assert!(matches!(
            malformed.into_response(),
            Err(ProviderError::MalformedResponse(_))
        ));

        let mut unfinished = AgentStream::new(Provider::OpenAiCompatible, AgentProtocol::Text);
        let events = unfinished.push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        );
        assert_eq!(events, vec![StreamEvent::TextDelta("partial".into())]);
        assert!(matches!(
            unfinished.into_response(),
            Err(ProviderError::MalformedResponse(message)) if message == STREAM_NOT_COMPLETE
        ));
    }

    #[test]
    fn finish_events_are_folded_before_conversion() {
        let mut stream = AgentStream::new(Provider::OpenAiCompatible, AgentProtocol::Text);
        stream.push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        );
        let events = stream.finish();
        assert!(events
            .iter()
            .any(|event| matches!(event, StreamEvent::Protocol(_))));
        assert!(matches!(
            stream.into_response(),
            Err(ProviderError::MalformedResponse(_))
        ));
    }
}
