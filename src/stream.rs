//! Sans-IO incremental parsing of streaming chat responses.
//!
//! [`StreamParser`] consumes the raw HTTP response body of a streaming chat
//! request (see [`crate::provider::build_chat_request_streaming`]) as it
//! arrives and yields [`StreamEvent`]s. It is a push parser: the integration
//! feeds whatever byte chunks its transport produces — split mid-line,
//! mid-UTF-8 sequence, or mid-frame — and calls [`StreamParser::finish`] at
//! EOF. One parser instance handles exactly one response.
//!
//! Wire formats:
//!
//! - **Anthropic** Messages SSE: `content_block_delta`/`text_delta` frames
//!   carry text, `message_delta` carries `stop_reason` and usage,
//!   `message_stop` terminates.
//! - **OpenAI-compatible** chat.completions SSE: `data:` JSON frames carry
//!   `choices[0].delta.content` and `finish_reason`, plus an optional usage
//!   frame; `data: [DONE]` terminates.
//! - **Ollama** `/api/chat` NDJSON: one JSON object per line with
//!   `message.content` deltas; the `"done": true` frame carries `done_reason`
//!   and token counts and terminates.
//!
//! Semantics match [`crate::provider::parse_chat_response_full`]: the
//! concatenated [`StreamEvent::TextDelta`]s equal the text it would extract
//! from the equivalent non-streaming response (including the `"\n"` join
//! between multiple Anthropic text blocks), [`StreamEvent::ReachedTokenLimit`]
//! mirrors `reached_token_limit`, and [`StreamEvent::Usage`] mirrors its
//! usage extraction.
//!
//! Fail closed (jagent invariant #2): a malformed frame, an invalid-UTF-8
//! frame, a provider-reported stream error, or an exceeded bound emits one
//! [`StreamEvent::Protocol`], after which the parser is inert and ignores all
//! further input. Bounds (invariant #3): a single buffered line/frame and the
//! cumulative delivered text are each capped at [`MAX_MODEL_TEXT_BYTES`].
//!
//! UTF-8 handling: input is buffered as bytes and split only at `\n`, so a
//! multi-byte sequence split across pushes is reassembled before decoding. A
//! *complete* frame that still fails UTF-8 validation is rejected as
//! malformed rather than decoded lossily — every supported wire format is
//! JSON, which is valid UTF-8 by definition, so lossy replacement characters
//! could only ever corrupt model text.

use crate::provider::{Provider, Usage, MAX_MODEL_TEXT_BYTES};
use crate::text::elide_middle;
use serde_json::Value;

/// Detail quoted from provider-reported stream errors is elided to this many
/// bytes before being embedded in a [`StreamEvent::Protocol`] message.
const MAX_ERROR_DETAIL_BYTES: usize = 256;

/// One parsed increment of a streaming chat response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// The next fragment of assistant text, in order. Concatenating every
    /// delta of a completed stream yields exactly the text that
    /// [`crate::provider::parse_chat_response_full`] would extract from the
    /// equivalent non-streaming response.
    TextDelta(String),
    /// The provider stopped at the output-token limit — Anthropic
    /// `stop_reason` `"max_tokens"`, OpenAI `finish_reason` `"length"`,
    /// Ollama `done_reason` `"length"` — the same condition
    /// [`crate::provider::ChatResponse::reached_token_limit`] reports.
    /// Emitted at most once per stream.
    ReachedTokenLimit,
    /// Token usage reported by the provider. Emitted at most once, directly
    /// before [`StreamEvent::Done`], and only when the provider reported at
    /// least one count; a truncated stream never reports usage.
    Usage(Usage),
    /// The provider's completion frame arrived (`message_stop`, `[DONE]`, or
    /// `"done": true`). The parser is inert afterwards; a subsequent
    /// [`StreamParser::finish`] returns no further events.
    Done,
    /// A malformed frame, invalid UTF-8, a provider-reported error, or an
    /// exceeded byte bound. Fail closed: the parser is inert afterwards and
    /// ignores all further input.
    Protocol(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Streaming,
    Finished,
    Failed,
}

/// Push parser for one streaming chat response body. See the module
/// documentation for wire formats, bounds, and failure semantics.
#[derive(Debug)]
pub struct StreamParser {
    provider: Provider,
    phase: Phase,
    /// Bytes of the current, not-yet-newline-terminated line.
    line: Vec<u8>,
    /// SSE only: accumulated `data:` payload of the event being assembled.
    event_data: String,
    event_has_data: bool,
    delivered_text_bytes: usize,
    /// Anthropic only: a text content block was opened; a later one is
    /// joined with `"\n"` to match the non-streaming extraction.
    saw_text_block: bool,
    /// The provider signaled end-of-message (`stop_reason`/`finish_reason`)
    /// even if its closing sentinel frame has not arrived yet.
    saw_message_end: bool,
    reached_token_limit: bool,
    usage: Usage,
}

impl StreamParser {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            phase: Phase::Streaming,
            line: Vec::new(),
            event_data: String::new(),
            event_has_data: false,
            delivered_text_bytes: 0,
            saw_text_block: false,
            saw_message_end: false,
            reached_token_limit: false,
            usage: Usage::default(),
        }
    }

    /// Feed the next chunk of raw response body bytes. Chunk boundaries are
    /// arbitrary. Returns the events completed by this chunk; after
    /// [`StreamEvent::Done`] or [`StreamEvent::Protocol`] the parser is inert
    /// and always returns no events.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        for &byte in bytes {
            if self.phase != Phase::Streaming {
                break;
            }
            if byte == b'\n' {
                let line = std::mem::take(&mut self.line);
                self.handle_line(&line, &mut events);
            } else {
                self.line.push(byte);
                if self.line.len() > MAX_MODEL_TEXT_BYTES {
                    self.fail(
                        &format!("stream frame exceeds the {MAX_MODEL_TEXT_BYTES} byte limit"),
                        &mut events,
                    );
                }
            }
        }
        events
    }

    /// Signal EOF. Flushes any buffered final line or SSE event, then emits
    /// [`StreamEvent::Done`] if the provider had already signaled
    /// end-of-message (some OpenAI-compatible servers omit the `[DONE]`
    /// sentinel), or a truncation [`StreamEvent::Protocol`] if the stream
    /// ended mid-message. Idempotent: later calls return no events.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if self.phase != Phase::Streaming {
            return events;
        }
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.handle_line(&line, &mut events);
        }
        // An SSE stream may end at EOF instead of a final blank line.
        if self.phase == Phase::Streaming && self.event_has_data {
            let data = std::mem::take(&mut self.event_data);
            self.event_has_data = false;
            self.dispatch_sse_frame(&data, &mut events);
        }
        if self.phase == Phase::Streaming {
            if self.saw_message_end {
                self.complete(&mut events);
            } else {
                self.fail(
                    "stream ended before the provider's completion frame; \
                     the response is truncated",
                    &mut events,
                );
            }
        }
        events
    }

    fn handle_line(&mut self, line: &[u8], events: &mut Vec<StreamEvent>) {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let Ok(line) = std::str::from_utf8(line) else {
            self.fail("stream frame is not valid UTF-8", events);
            return;
        };
        match self.provider {
            Provider::Anthropic | Provider::OpenAiCompatible => self.handle_sse_line(line, events),
            Provider::Ollama => self.handle_ndjson_line(line, events),
        }
    }

    fn handle_sse_line(&mut self, line: &str, events: &mut Vec<StreamEvent>) {
        if line.is_empty() {
            if self.event_has_data {
                let data = std::mem::take(&mut self.event_data);
                self.event_has_data = false;
                self.dispatch_sse_frame(&data, events);
            }
            return;
        }
        if line.starts_with(':') {
            return; // SSE comment; used as a keep-alive by several providers.
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        // Only `data:` carries payload. `event:` names duplicate the JSON
        // `type` field for Anthropic and are absent for OpenAI; `id:`,
        // `retry:`, and unknown fields are ignored per the SSE model.
        if field != "data" {
            return;
        }
        if self.event_has_data {
            self.event_data.push('\n');
        }
        self.event_has_data = true;
        self.event_data.push_str(value);
        if self.event_data.len() > MAX_MODEL_TEXT_BYTES {
            self.fail(
                &format!("stream frame exceeds the {MAX_MODEL_TEXT_BYTES} byte limit"),
                events,
            );
        }
    }

    fn dispatch_sse_frame(&mut self, data: &str, events: &mut Vec<StreamEvent>) {
        match self.provider {
            Provider::Anthropic => self.anthropic_frame(data, events),
            // Ollama never assembles SSE events; the arm is unreachable but
            // harmless, and avoids a panic path in an untrusted-data parser.
            Provider::OpenAiCompatible | Provider::Ollama => self.openai_frame(data, events),
        }
    }

    fn anthropic_frame(&mut self, data: &str, events: &mut Vec<StreamEvent>) {
        let Ok(frame) = serde_json::from_str::<Value>(data) else {
            self.fail("malformed JSON in stream frame", events);
            return;
        };
        let Some(kind) = frame.get("type").and_then(Value::as_str) else {
            self.fail("stream frame is missing its type", events);
            return;
        };
        match kind {
            "message_start" => {
                self.merge_usage(
                    frame.pointer("/message/usage"),
                    "input_tokens",
                    "output_tokens",
                );
            }
            "content_block_start" => {
                let block = frame.get("content_block");
                let block_type = block
                    .and_then(|block| block.get("type"))
                    .and_then(Value::as_str);
                if block_type == Some("text") {
                    if self.saw_text_block {
                        // parse_chat_response_full joins multiple text blocks
                        // with "\n"; mirror it so accumulation matches.
                        self.deliver_text("\n", events);
                    }
                    self.saw_text_block = true;
                    if let Some(text) = block
                        .and_then(|block| block.get("text"))
                        .and_then(Value::as_str)
                    {
                        self.deliver_text(text, events);
                    }
                }
            }
            "content_block_delta" => {
                let delta = frame.get("delta");
                let delta_type = delta
                    .and_then(|delta| delta.get("type"))
                    .and_then(Value::as_str);
                if delta_type == Some("text_delta") {
                    match delta
                        .and_then(|delta| delta.get("text"))
                        .and_then(Value::as_str)
                    {
                        Some(text) => self.deliver_text(text, events),
                        None => self.fail("text_delta frame carries no text", events),
                    }
                }
                // thinking/input_json deltas are non-text content; the
                // non-streaming parser ignores those blocks as well.
            }
            "message_delta" => {
                self.merge_usage(frame.get("usage"), "input_tokens", "output_tokens");
                if let Some(stop_reason) =
                    frame.pointer("/delta/stop_reason").and_then(Value::as_str)
                {
                    self.saw_message_end = true;
                    if stop_reason == "max_tokens" {
                        self.emit_token_limit(events);
                    }
                }
            }
            "message_stop" => self.complete(events),
            "error" => {
                let detail = frame
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unspecified");
                self.fail(
                    &format!(
                        "provider error: {}",
                        elide_middle(detail, MAX_ERROR_DETAIL_BYTES)
                    ),
                    events,
                );
            }
            // ping, content_block_stop, and future event types carry no
            // text, stop reason, or usage.
            _ => {}
        }
    }

    fn openai_frame(&mut self, data: &str, events: &mut Vec<StreamEvent>) {
        if data.trim() == "[DONE]" {
            self.complete(events);
            return;
        }
        let Ok(frame) = serde_json::from_str::<Value>(data) else {
            self.fail("malformed JSON in stream frame", events);
            return;
        };
        if !frame.is_object() {
            self.fail("stream frame is not a JSON object", events);
            return;
        }
        if let Some(error) = frame.get("error") {
            let detail = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error.as_str())
                .unwrap_or("unspecified");
            self.fail(
                &format!(
                    "provider error: {}",
                    elide_middle(detail, MAX_ERROR_DETAIL_BYTES)
                ),
                events,
            );
            return;
        }
        match frame.pointer("/choices/0/delta/content") {
            Some(Value::String(text)) => self.deliver_text(text, events),
            Some(Value::Null) | None => {}
            Some(_) => {
                self.fail("delta content is not a string", events);
                return;
            }
        }
        self.merge_usage(frame.get("usage"), "prompt_tokens", "completion_tokens");
        if let Some(finish_reason) = frame
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
        {
            self.saw_message_end = true;
            if finish_reason == "length" {
                self.emit_token_limit(events);
            }
        }
    }

    fn handle_ndjson_line(&mut self, line: &str, events: &mut Vec<StreamEvent>) {
        if line.trim().is_empty() {
            return;
        }
        let Ok(frame) = serde_json::from_str::<Value>(line) else {
            self.fail("malformed JSON in stream frame", events);
            return;
        };
        if !frame.is_object() {
            self.fail("stream frame is not a JSON object", events);
            return;
        }
        if let Some(error) = frame.get("error").and_then(Value::as_str) {
            self.fail(
                &format!(
                    "provider error: {}",
                    elide_middle(error, MAX_ERROR_DETAIL_BYTES)
                ),
                events,
            );
            return;
        }
        match frame.pointer("/message/content") {
            Some(Value::String(text)) => self.deliver_text(text, events),
            Some(Value::Null) | None => {}
            Some(_) => {
                self.fail("message content is not a string", events);
                return;
            }
        }
        if frame.get("done").and_then(Value::as_bool) == Some(true) {
            if frame.get("done_reason").and_then(Value::as_str) == Some("length") {
                self.emit_token_limit(events);
            }
            self.merge_usage(Some(&frame), "prompt_eval_count", "eval_count");
            self.complete(events);
        }
    }

    /// Emit a text delta, enforcing the cumulative text bound. Empty deltas
    /// are dropped; they carry no information.
    fn deliver_text(&mut self, text: &str, events: &mut Vec<StreamEvent>) {
        if self.phase != Phase::Streaming || text.is_empty() {
            return;
        }
        let total = self.delivered_text_bytes.saturating_add(text.len());
        if total > MAX_MODEL_TEXT_BYTES {
            self.fail(
                &format!("streamed text exceeds the {MAX_MODEL_TEXT_BYTES} byte limit"),
                events,
            );
            return;
        }
        self.delivered_text_bytes = total;
        events.push(StreamEvent::TextDelta(text.to_string()));
    }

    fn emit_token_limit(&mut self, events: &mut Vec<StreamEvent>) {
        if self.phase != Phase::Streaming || self.reached_token_limit {
            return;
        }
        self.reached_token_limit = true;
        events.push(StreamEvent::ReachedTokenLimit);
    }

    /// Record any usage counts the value carries; later frames overwrite
    /// earlier ones (Anthropic reports input in `message_start` and the
    /// final output count in `message_delta`).
    fn merge_usage(&mut self, usage: Option<&Value>, input_key: &str, output_key: &str) {
        let Some(usage) = usage else { return };
        if let Some(input) = usage.get(input_key).and_then(Value::as_u64) {
            self.usage.input_tokens = Some(input);
        }
        if let Some(output) = usage.get(output_key).and_then(Value::as_u64) {
            self.usage.output_tokens = Some(output);
        }
    }

    fn complete(&mut self, events: &mut Vec<StreamEvent>) {
        if self.phase != Phase::Streaming {
            return;
        }
        if self.usage.input_tokens.is_some() || self.usage.output_tokens.is_some() {
            events.push(StreamEvent::Usage(self.usage));
        }
        events.push(StreamEvent::Done);
        self.phase = Phase::Finished;
        self.line = Vec::new();
        self.event_data = String::new();
    }

    fn fail(&mut self, message: &str, events: &mut Vec<StreamEvent>) {
        if self.phase != Phase::Streaming {
            return;
        }
        events.push(StreamEvent::Protocol(message.to_string()));
        self.phase = Phase::Failed;
        self.line = Vec::new();
        self.event_data = String::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::parse_chat_response_full;
    use serde_json::json;

    const TEXT: &str = "Hello, 编译 world 🙂";

    fn sse(event: &str, data: &Value) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }

    fn openai_line(data: &Value) -> String {
        format!("data: {data}\n\n")
    }

    fn ndjson_line(data: &Value) -> String {
        format!("{data}\n")
    }

    fn anthropic_body(stop_reason: &str) -> String {
        [
            sse(
                "message_start",
                &json!({"type": "message_start", "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 25, "output_tokens": 1},
                }}),
            ),
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": "Hello, "}}),
            ),
            sse("ping", &json!({"type": "ping"})),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": "编译"}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": " world 🙂"}}),
            ),
            sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            ),
            sse(
                "message_delta",
                &json!({"type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                        "usage": {"output_tokens": 15}}),
            ),
            sse("message_stop", &json!({"type": "message_stop"})),
        ]
        .concat()
    }

    fn openai_body(finish_reason: &str) -> String {
        [
            openai_line(&json!({"id": "c1", "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"content": "Hello, "}, "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"content": "编译 world 🙂"}, "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {}, "finish_reason": finish_reason},
            ]})),
            openai_line(&json!({"choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12}})),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat()
    }

    fn ollama_body(done_reason: &str) -> String {
        [
            ndjson_line(&json!({"model": "m",
                "message": {"role": "assistant", "content": "Hello, "}, "done": false})),
            ndjson_line(&json!({"model": "m",
                "message": {"role": "assistant", "content": "编译 world 🙂"}, "done": false})),
            ndjson_line(&json!({"model": "m",
                "message": {"role": "assistant", "content": ""},
                "done": true, "done_reason": done_reason,
                "prompt_eval_count": 26, "eval_count": 298})),
        ]
        .concat()
    }

    fn happy_body(provider: Provider) -> String {
        match provider {
            Provider::Anthropic => anthropic_body("end_turn"),
            Provider::OpenAiCompatible => openai_body("stop"),
            Provider::Ollama => ollama_body("stop"),
        }
    }

    fn limit_body(provider: Provider) -> String {
        match provider {
            Provider::Anthropic => anthropic_body("max_tokens"),
            Provider::OpenAiCompatible => openai_body("length"),
            Provider::Ollama => ollama_body("length"),
        }
    }

    fn happy_usage(provider: Provider) -> Usage {
        let (input, output) = match provider {
            Provider::Anthropic => (25, 15),
            Provider::OpenAiCompatible => (9, 12),
            Provider::Ollama => (26, 298),
        };
        Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
        }
    }

    /// Push the whole body in one chunk, then finish.
    fn run(provider: Provider, body: &str) -> Vec<StreamEvent> {
        let mut parser = StreamParser::new(provider);
        let mut events = parser.push(body.as_bytes());
        events.extend(parser.finish());
        events
    }

    #[derive(Debug, Default, PartialEq)]
    struct Folded {
        text: String,
        reached_token_limit: bool,
        usage: Option<Usage>,
        done: bool,
        protocol: Option<String>,
    }

    fn fold(events: &[StreamEvent]) -> Folded {
        let mut folded = Folded::default();
        for event in events {
            match event {
                StreamEvent::TextDelta(delta) => folded.text.push_str(delta),
                StreamEvent::ReachedTokenLimit => folded.reached_token_limit = true,
                StreamEvent::Usage(usage) => folded.usage = Some(*usage),
                StreamEvent::Done => folded.done = true,
                StreamEvent::Protocol(message) => folded.protocol = Some(message.clone()),
            }
        }
        folded
    }

    const ALL_PROVIDERS: [Provider; 3] = [
        Provider::Anthropic,
        Provider::OpenAiCompatible,
        Provider::Ollama,
    ];

    #[test]
    fn anthropic_happy_path_yields_exact_events() {
        let events = run(Provider::Anthropic, &anthropic_body("end_turn"));
        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("Hello, ".into()),
                StreamEvent::TextDelta("编译".into()),
                StreamEvent::TextDelta(" world 🙂".into()),
                StreamEvent::Usage(happy_usage(Provider::Anthropic)),
                StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn openai_happy_path_yields_exact_events() {
        let events = run(Provider::OpenAiCompatible, &openai_body("stop"));
        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("Hello, ".into()),
                StreamEvent::TextDelta("编译 world 🙂".into()),
                StreamEvent::Usage(happy_usage(Provider::OpenAiCompatible)),
                StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn ollama_happy_path_yields_exact_events() {
        let events = run(Provider::Ollama, &ollama_body("stop"));
        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("Hello, ".into()),
                StreamEvent::TextDelta("编译 world 🙂".into()),
                StreamEvent::Usage(happy_usage(Provider::Ollama)),
                StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn byte_by_byte_chunking_matches_whole_body() {
        // Feeding one byte at a time splits every line, SSE event, and
        // multi-byte UTF-8 character across pushes.
        for provider in ALL_PROVIDERS {
            let body = happy_body(provider);
            let mut parser = StreamParser::new(provider);
            let mut events = Vec::new();
            for byte in body.as_bytes() {
                events.extend(parser.push(std::slice::from_ref(byte)));
            }
            events.extend(parser.finish());
            assert_eq!(events, run(provider, &body), "{provider:?}");
        }
    }

    #[test]
    fn crlf_line_endings_are_tolerated() {
        for provider in ALL_PROVIDERS {
            let body = happy_body(provider).replace('\n', "\r\n");
            assert_eq!(run(provider, &body), run(provider, &happy_body(provider)));
        }
    }

    #[test]
    fn token_limit_frames_emit_reached_token_limit_once() {
        for provider in ALL_PROVIDERS {
            let events = run(provider, &limit_body(provider));
            let count = events
                .iter()
                .filter(|event| **event == StreamEvent::ReachedTokenLimit)
                .count();
            assert_eq!(count, 1, "{provider:?}");
            let folded = fold(&events);
            assert_eq!(folded.text, TEXT, "{provider:?}");
            assert!(folded.done, "{provider:?}");
            assert_eq!(folded.protocol, None, "{provider:?}");
        }
    }

    #[test]
    fn done_sentinels_make_the_parser_inert() {
        for provider in ALL_PROVIDERS {
            let body = happy_body(provider);
            let mut parser = StreamParser::new(provider);
            let folded = fold(&parser.push(body.as_bytes()));
            assert!(folded.done, "{provider:?}");
            // Trailing bytes after the completion frame are ignored.
            assert_eq!(parser.push(body.as_bytes()), vec![], "{provider:?}");
            assert_eq!(parser.finish(), vec![], "{provider:?}");
            assert_eq!(parser.finish(), vec![], "{provider:?}");
        }
    }

    #[test]
    fn malformed_json_fails_closed_and_goes_inert() {
        let bodies = [
            (
                Provider::Anthropic,
                "event: message_start\ndata: {broken\n\n",
            ),
            (Provider::OpenAiCompatible, "data: {broken\n\n"),
            (Provider::Ollama, "{broken\n"),
        ];
        for (provider, body) in bodies {
            let mut parser = StreamParser::new(provider);
            let events = parser.push(body.as_bytes());
            assert_eq!(
                events,
                vec![StreamEvent::Protocol(
                    "malformed JSON in stream frame".into()
                )],
                "{provider:?}"
            );
            // Inert: even a well-formed follow-up body is ignored.
            assert_eq!(
                parser.push(happy_body(provider).as_bytes()),
                vec![],
                "{provider:?}"
            );
            assert_eq!(parser.finish(), vec![], "{provider:?}");
        }
    }

    #[test]
    fn malformed_frame_shapes_fail_closed() {
        // Anthropic: a data frame with no type field.
        let events = run(Provider::Anthropic, "data: {\"foo\": 1}\n\n");
        assert!(matches!(&events[0], StreamEvent::Protocol(_)));

        // Anthropic: text_delta without its text.
        let body = sse(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta"}}),
        );
        let events = run(Provider::Anthropic, &body);
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "text_delta frame carries no text".into()
            )]
        );

        // OpenAI: delta content that is not a string.
        let body = openai_line(&json!({"choices": [
            {"index": 0, "delta": {"content": 42}, "finish_reason": null},
        ]}));
        let events = run(Provider::OpenAiCompatible, &body);
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "delta content is not a string".into()
            )]
        );

        // Ollama: a frame that is valid JSON but not an object.
        let events = run(Provider::Ollama, "5\n");
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "stream frame is not a JSON object".into()
            )]
        );
    }

    #[test]
    fn provider_error_frames_fail_closed() {
        let body = sse(
            "error",
            &json!({"type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"}}),
        );
        let events = run(Provider::Anthropic, &body);
        assert_eq!(
            events,
            vec![StreamEvent::Protocol("provider error: Overloaded".into())]
        );

        let events = run(
            Provider::Ollama,
            &ndjson_line(&json!({"error": "model not found"})),
        );
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "provider error: model not found".into()
            )]
        );

        let events = run(
            Provider::OpenAiCompatible,
            &openai_line(&json!({"error": {"message": "quota exceeded"}})),
        );
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "provider error: quota exceeded".into()
            )]
        );
    }

    #[test]
    fn oversized_line_fails_closed_and_goes_inert() {
        let mut parser = StreamParser::new(Provider::Ollama);
        let events = parser.push(&vec![b'a'; MAX_MODEL_TEXT_BYTES + 1]);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], StreamEvent::Protocol(message) if message.contains("byte limit"))
        );
        assert_eq!(parser.push(b"\n"), vec![]);
        assert_eq!(parser.finish(), vec![]);
    }

    #[test]
    fn cumulative_text_over_budget_fails_closed() {
        // Each frame is comfortably under the per-line bound; only the
        // accumulated text exceeds MAX_MODEL_TEXT_BYTES.
        let delta = "a".repeat(100 * 1024);
        let frame = ndjson_line(&json!({"message": {"content": delta}, "done": false}));
        let mut parser = StreamParser::new(Provider::Ollama);
        let mut events = Vec::new();
        for _ in 0..3 {
            events.extend(parser.push(frame.as_bytes()));
        }
        let text_deltas = events
            .iter()
            .filter(|event| matches!(event, StreamEvent::TextDelta(_)))
            .count();
        assert_eq!(text_deltas, 2);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Protocol(message)) if message.contains("byte limit")
        ));
        assert_eq!(parser.finish(), vec![]);
    }

    #[test]
    fn invalid_utf8_frame_fails_closed() {
        let mut parser = StreamParser::new(Provider::Ollama);
        let events = parser.push(&[0xff, 0xfe, b'\n']);
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "stream frame is not valid UTF-8".into()
            )]
        );
        assert_eq!(parser.push(b"{}\n"), vec![]);
    }

    #[test]
    fn finish_on_truncated_stream_reports_truncation() {
        for provider in ALL_PROVIDERS {
            let body = happy_body(provider);
            // Cut the stream in the middle of the text, before any
            // stop reason or completion frame.
            let cut = body.len() / 2;
            let mut parser = StreamParser::new(provider);
            parser.push(&body.as_bytes()[..cut]);
            let events = parser.finish();
            let folded = fold(&events);
            assert!(!folded.done, "{provider:?}");
            assert!(folded.protocol.is_some(), "{provider:?}");
        }
    }

    #[test]
    fn finish_completes_a_message_whose_end_was_signaled_without_a_sentinel() {
        // Some OpenAI-compatible servers omit the trailing `data: [DONE]`.
        // A received finish_reason means the message is complete, so EOF is
        // not truncation.
        let body = openai_body("stop");
        let body = body.strip_suffix("data: [DONE]\n\n").unwrap();
        let events = run(Provider::OpenAiCompatible, body);
        let folded = fold(&events);
        assert_eq!(folded.text, TEXT);
        assert!(folded.done);
        assert_eq!(folded.usage, Some(happy_usage(Provider::OpenAiCompatible)));
        assert_eq!(folded.protocol, None);

        // Same for an Anthropic stream cut after message_delta.
        let body = anthropic_body("end_turn");
        let body = body.strip_suffix("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
        let events = run(Provider::Anthropic, body.unwrap());
        let folded = fold(&events);
        assert!(folded.done);
        assert_eq!(folded.usage, Some(happy_usage(Provider::Anthropic)));
    }

    #[test]
    fn finish_flushes_a_final_line_without_a_trailing_newline() {
        let body = ollama_body("stop");
        let body = body.strip_suffix('\n').unwrap();
        let folded = fold(&run(Provider::Ollama, body));
        assert!(folded.done);
        assert_eq!(folded.usage, Some(happy_usage(Provider::Ollama)));

        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\
                    \"finish_reason\":null}]}\n\ndata: [DONE]";
        let folded = fold(&run(Provider::OpenAiCompatible, body));
        assert_eq!(folded.text, "hi");
        assert!(folded.done);
    }

    #[test]
    fn sse_comments_unknown_fields_and_unknown_events_are_ignored() {
        let noise = ": keep-alive\nid: 3\nretry: 100\nnoise\n\
                     event: block_heartbeat\ndata: {\"type\": \"block_heartbeat\"}\n\n";
        let body = format!("{noise}{}", anthropic_body("end_turn"));
        assert_eq!(
            run(Provider::Anthropic, &body),
            run(Provider::Anthropic, &anthropic_body("end_turn"))
        );

        // Blank lines between NDJSON frames are tolerated.
        let body = ollama_body("stop").replace('\n', "\n\n");
        assert_eq!(
            run(Provider::Ollama, &body),
            run(Provider::Ollama, &ollama_body("stop"))
        );
    }

    #[test]
    fn streaming_accumulation_matches_non_streaming_parse() {
        // Anthropic: two text blocks (joined with "\n" when non-streaming)
        // plus an ignored thinking block, ending at the token limit.
        let streaming = [
            sse(
                "message_start",
                &json!({"type": "message_start", "message": {
                    "usage": {"input_tokens": 25, "output_tokens": 1}}}),
            ),
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "thinking", "thinking": ""}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
            ),
            sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            ),
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 1,
                        "content_block": {"type": "text", "text": ""}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 1,
                        "delta": {"type": "text_delta", "text": "first"}}),
            ),
            sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 1}),
            ),
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 2,
                        "content_block": {"type": "text", "text": "sec"}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 2,
                        "delta": {"type": "text_delta", "text": "ond"}}),
            ),
            sse(
                "message_delta",
                &json!({"type": "message_delta",
                        "delta": {"stop_reason": "max_tokens"},
                        "usage": {"output_tokens": 15}}),
            ),
            sse("message_stop", &json!({"type": "message_stop"})),
        ]
        .concat();
        let non_streaming = json!({
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 25, "output_tokens": 15},
        });
        assert_stream_matches_parse(Provider::Anthropic, &streaming, &non_streaming);

        let non_streaming = json!({
            "choices": [{"message": {"content": TEXT}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12},
        });
        assert_stream_matches_parse(
            Provider::OpenAiCompatible,
            &openai_body("length"),
            &non_streaming,
        );

        let non_streaming = json!({
            "message": {"content": TEXT},
            "done": true,
            "done_reason": "length",
            "prompt_eval_count": 26,
            "eval_count": 298,
        });
        assert_stream_matches_parse(Provider::Ollama, &ollama_body("length"), &non_streaming);
    }

    fn assert_stream_matches_parse(provider: Provider, streaming: &str, non_streaming: &Value) {
        let folded = fold(&run(provider, streaming));
        let parsed = parse_chat_response_full(provider, non_streaming).expect("parse");
        assert_eq!(folded.text, parsed.text, "{provider:?}");
        assert_eq!(
            folded.reached_token_limit, parsed.reached_token_limit,
            "{provider:?}"
        );
        assert_eq!(folded.usage, parsed.usage, "{provider:?}");
        assert!(folded.done, "{provider:?}");
        assert_eq!(folded.protocol, None, "{provider:?}");
    }
}
