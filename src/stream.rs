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
//! frame, payload after an end signal, an empty completed text response, a
//! provider-reported stream error, or an exceeded bound emits one
//! [`StreamEvent::Protocol`], after which the parser is inert and ignores all
//! further input. Bounds (invariant #3): the raw response, number of decoded
//! frames, a single buffered line/frame, cumulative delivered text/tool
//! arguments, and the number of calls are capped. Completed tool blocks remain
//! private until the enclosing response succeeds, so a later error cannot
//! leave an actionable call behind.
//!
//! UTF-8 handling: input is buffered as bytes and split only at `\n`, so a
//! multi-byte sequence split across pushes is reassembled before decoding. A
//! *complete* frame that still fails UTF-8 validation is rejected as
//! malformed rather than decoded lossily — every supported wire format is
//! JSON, which is valid UTF-8 by definition, so lossy replacement characters
//! could only ever corrupt model text.

use crate::provider::{Provider, Usage, MAX_MODEL_TEXT_BYTES};
use crate::text::elide_middle;
use crate::tools::{
    ToolCall, MAX_STREAM_TOOL_CALLS, MAX_TOOL_ARGUMENTS_BYTES, MAX_TOOL_ID_BYTES,
    MAX_TOOL_NAME_BYTES,
};
use serde_json::Value;

/// Detail quoted from provider-reported stream errors is elided to this many
/// bytes before being embedded in a [`StreamEvent::Protocol`] message.
const MAX_ERROR_DETAIL_BYTES: usize = 256;

/// Maximum raw body bytes one [`StreamParser`] will inspect for a response.
///
/// Streaming JSON carries framing overhead beyond the retained 256-KiB model
/// text budget, but it must not become an unlimited sequence of keep-alives or
/// semantically irrelevant frames. Integrations should use the same ceiling at
/// their transport boundary where possible; the parser enforces it again so
/// its public sans-IO API is safe on its own.
pub const MAX_STREAM_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// Maximum JSON/SSE payload frames decoded for one response.
///
/// This independently bounds CPU and repeated temporary `serde_json::Value`
/// allocations when an attacker sends many tiny frames inside the raw-byte
/// ceiling. SSE comments and blank lines consume the raw-byte budget but are
/// not decoded frames.
pub const MAX_STREAM_FRAMES: usize = 4096;

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
    /// A native tool call from a response that finished successfully.
    /// Anthropic's `tool_use` block must first reach `content_block_stop`, and
    /// the enclosing response must then complete (`message_stop`, or EOF after
    /// an explicit stop reason); OpenAI-compatible calls are finalized at the
    /// response completion marker. Calls are
    /// emitted in the order they were opened, always immediately before
    /// [`StreamEvent::Usage`] and [`StreamEvent::Done`]. A later malformed or
    /// truncated frame therefore cannot leave an already-published call for a
    /// caller to act on. The value is identical in shape to what
    /// [`crate::tools::parse_tool_response`] extracts from the equivalent
    /// non-streaming reply, so
    /// [`crate::tools::ToolResponse::to_action`] applies unchanged.
    ToolCall(ToolCall),
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
    raw_response_bytes: usize,
    decoded_frames: usize,
    delivered_text_bytes: usize,
    /// Whether any delivered text survives Unicode whitespace trimming. This
    /// lets completion mirror the non-streaming parser's empty-response rule
    /// without retaining another copy of the streamed text.
    saw_non_whitespace_text: bool,
    /// Anthropic only: a text content block was opened; a later one is
    /// joined with `"\n"` to match the non-streaming extraction.
    saw_text_block: bool,
    /// The provider signaled end-of-message (`stop_reason`/`finish_reason`)
    /// even if its closing sentinel frame has not arrived yet.
    saw_message_end: bool,
    reached_token_limit: bool,
    usage: Usage,
    /// Tool calls still accumulating, in the order they were opened. Keyed by
    /// the provider's own index (Anthropic content-block index, OpenAI
    /// `tool_calls[].index`).
    pending_tools: Vec<PendingToolCall>,
    /// Calls whose provider-specific block is closed but whose enclosing
    /// response has not completed yet. Holding these back makes publication
    /// transactional: a later malformed frame or truncated response can still
    /// fail closed without leaking an actionable call to the integration.
    completed_tools: Vec<PendingToolCall>,
    /// Cumulative bytes of tool-call arguments seen, bounded like text.
    delivered_tool_bytes: usize,
}

#[derive(Debug)]
struct PendingToolCall {
    index: u64,
    id: String,
    name: String,
    arguments: String,
}

impl StreamParser {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            phase: Phase::Streaming,
            line: Vec::new(),
            event_data: String::new(),
            event_has_data: false,
            raw_response_bytes: 0,
            decoded_frames: 0,
            delivered_text_bytes: 0,
            saw_non_whitespace_text: false,
            saw_text_block: false,
            saw_message_end: false,
            reached_token_limit: false,
            usage: Usage::default(),
            pending_tools: Vec::new(),
            completed_tools: Vec::new(),
            delivered_tool_bytes: 0,
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
            if self.raw_response_bytes == MAX_STREAM_RESPONSE_BYTES {
                self.fail(
                    &format!("stream response exceeds the {MAX_STREAM_RESPONSE_BYTES} byte limit"),
                    &mut events,
                );
                break;
            }
            self.raw_response_bytes += 1;
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
        if !self.begin_decoded_frame(events) {
            return;
        }
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
        if self.saw_message_end && !matches!(kind, "message_stop" | "ping") {
            self.fail("Anthropic streamed data after its stop reason", events);
            return;
        }
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
                if block_type == Some("tool_use") {
                    // `input` in this frame is always `{}`; the arguments
                    // arrive as input_json_delta fragments.
                    let name = block
                        .and_then(|block| block.get("name"))
                        .and_then(Value::as_str);
                    let Some(name) = name else {
                        self.fail("tool_use block carries no name", events);
                        return;
                    };
                    let id = block
                        .and_then(|block| block.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let Some(index) = block_index(&frame) else {
                        self.fail("tool_use block carries no valid index", events);
                        return;
                    };
                    self.open_tool_call(index, id, name, events);
                    return;
                }
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
                } else if delta_type == Some("input_json_delta") {
                    match delta
                        .and_then(|delta| delta.get("partial_json"))
                        .and_then(Value::as_str)
                    {
                        Some(fragment) => match block_index(&frame) {
                            Some(index) => self.extend_tool_call(index, fragment, events),
                            None => {
                                self.fail("tool-call argument delta carries no valid index", events)
                            }
                        },
                        None => self.fail("input_json_delta frame carries no partial_json", events),
                    }
                }
                // thinking deltas are non-text content; the non-streaming
                // parser ignores those blocks as well.
            }
            "content_block_stop" => match block_index(&frame) {
                Some(index) => self.close_tool_call(index),
                None => self.fail("content_block_stop carries no valid index", events),
            },
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
            // ping and future event types carry no text, stop reason, usage,
            // or tool call.
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
        if self.saw_message_end
            && frame
                .pointer("/choices/0/delta")
                .and_then(Value::as_object)
                .is_some_and(|delta| !delta.is_empty())
        {
            self.fail(
                "OpenAI-compatible streamed data after its finish reason",
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
        match frame.pointer("/choices/0/delta/tool_calls") {
            Some(Value::Array(entries)) => {
                // Cloned so the accumulator can borrow `self` mutably; a delta
                // frame carries at most a handful of small fragments.
                for entry in entries.clone() {
                    self.accumulate_openai_tool_call(&entry, events);
                    if self.phase != Phase::Streaming {
                        return;
                    }
                }
            }
            Some(Value::Null) | None => {}
            Some(_) => {
                self.fail("delta tool_calls is not an array", events);
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
        if !self.begin_decoded_frame(events) {
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

    fn begin_decoded_frame(&mut self, events: &mut Vec<StreamEvent>) -> bool {
        if self.decoded_frames == MAX_STREAM_FRAMES {
            self.fail(
                &format!("stream contains more than {MAX_STREAM_FRAMES} decoded frames"),
                events,
            );
            return false;
        }
        self.decoded_frames += 1;
        true
    }

    /// Merge one OpenAI-compatible `tool_calls` delta entry. Fragments are
    /// keyed by the entry's `index`; `id` and `name` normally arrive whole in
    /// the opening fragment but are appended defensively, and `arguments`
    /// fragments concatenate into the raw JSON object text.
    fn accumulate_openai_tool_call(&mut self, entry: &Value, events: &mut Vec<StreamEvent>) {
        let Some(index) = entry.get("index").and_then(Value::as_u64) else {
            self.fail("tool call carries no valid index", events);
            return;
        };
        let id = match entry.get("id") {
            Some(Value::String(id)) => id.as_str(),
            Some(Value::Null) | None => "",
            Some(_) => {
                self.fail("tool call id is not a string", events);
                return;
            }
        };
        let function = entry.get("function");
        let name = match function.and_then(|function| function.get("name")) {
            Some(Value::String(name)) => name.as_str(),
            Some(Value::Null) | None => "",
            Some(_) => {
                self.fail("tool call name is not a string", events);
                return;
            }
        };
        let arguments = match function.and_then(|function| function.get("arguments")) {
            Some(Value::String(arguments)) => arguments.as_str(),
            Some(Value::Null) | None => "",
            Some(_) => {
                self.fail("tool call arguments are not a string", events);
                return;
            }
        };
        match self.pending_index(index) {
            None => self.open_tool_call(index, id, name, events),
            Some(position) => {
                self.pending_tools[position].id.push_str(id);
                self.pending_tools[position].name.push_str(name);
                self.check_tool_identifiers(position, events);
            }
        }
        if self.phase != Phase::Streaming {
            return;
        }
        self.extend_tool_call(index, arguments, events);
    }

    fn pending_index(&self, index: u64) -> Option<usize> {
        self.pending_tools
            .iter()
            .position(|pending| pending.index == index)
    }

    /// Start accumulating a tool call, enforcing the whole-response call
    /// bound. Anthropic calls close independently, so counting only currently
    /// open blocks would let a response stream an unbounded sequence of calls.
    fn open_tool_call(&mut self, index: u64, id: &str, name: &str, events: &mut Vec<StreamEvent>) {
        if self.phase != Phase::Streaming {
            return;
        }
        if self
            .pending_tools
            .len()
            .saturating_add(self.completed_tools.len())
            >= MAX_STREAM_TOOL_CALLS
        {
            self.fail(
                &format!("stream contains more than {MAX_STREAM_TOOL_CALLS} tool calls"),
                events,
            );
            return;
        }
        if self.pending_index(index).is_some()
            || self
                .completed_tools
                .iter()
                .any(|completed| completed.index == index)
        {
            self.fail("stream reused a tool-call index", events);
            return;
        }
        self.pending_tools.push(PendingToolCall {
            index,
            id: id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
        });
        let position = self.pending_tools.len() - 1;
        self.check_tool_identifiers(position, events);
    }

    fn check_tool_identifiers(&mut self, position: usize, events: &mut Vec<StreamEvent>) -> bool {
        let pending = &self.pending_tools[position];
        if pending.name.len() > MAX_TOOL_NAME_BYTES {
            self.fail("streamed tool name exceeds its byte limit", events);
            return false;
        }
        if pending.id.len() > MAX_TOOL_ID_BYTES {
            self.fail("streamed tool call id exceeds its byte limit", events);
            return false;
        }
        true
    }

    /// Append an arguments fragment, enforcing the per-call and cumulative
    /// bounds. A fragment for a call that was never opened fails closed.
    fn extend_tool_call(&mut self, index: u64, fragment: &str, events: &mut Vec<StreamEvent>) {
        if self.phase != Phase::Streaming || fragment.is_empty() {
            return;
        }
        let Some(position) = self.pending_index(index) else {
            self.fail(
                "tool-call arguments arrived for an unopened tool call",
                events,
            );
            return;
        };
        let total = self.delivered_tool_bytes.saturating_add(fragment.len());
        if total > MAX_MODEL_TEXT_BYTES {
            self.fail(
                &format!("streamed tool arguments exceed the {MAX_MODEL_TEXT_BYTES} byte limit"),
                events,
            );
            return;
        }
        self.delivered_tool_bytes = total;
        let pending = &mut self.pending_tools[position];
        pending.arguments.push_str(fragment);
        if pending.arguments.len() > MAX_TOOL_ARGUMENTS_BYTES {
            self.fail(
                &format!(
                    "one tool call's arguments exceed the {MAX_TOOL_ARGUMENTS_BYTES} byte limit"
                ),
                events,
            );
        }
    }

    /// Finish the tool call accumulating at `index`, if any. Anthropic closes
    /// each `tool_use` block explicitly; other content blocks close with no
    /// pending call and are a no-op here. Finished calls remain private until
    /// the enclosing response completes successfully.
    fn close_tool_call(&mut self, index: u64) {
        if self.phase != Phase::Streaming {
            return;
        }
        let Some(position) = self.pending_index(index) else {
            return;
        };
        let pending = self.pending_tools.remove(position);
        self.completed_tools.push(pending);
    }

    /// Finish every still-open tool call, in the order they were opened. Used
    /// only for OpenAI-compatible completion frames, whose protocol has no
    /// per-call terminator. Anthropic calls must close explicitly.
    fn finish_open_tool_calls(&mut self) {
        for pending in std::mem::take(&mut self.pending_tools) {
            self.completed_tools.push(pending);
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
        self.saw_non_whitespace_text |= text.chars().any(|character| !character.is_whitespace());
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
        if self.provider == Provider::Anthropic && !self.pending_tools.is_empty() {
            self.fail(
                "Anthropic message ended before a tool_use block was closed",
                events,
            );
            return;
        }
        if self.provider == Provider::OpenAiCompatible {
            self.finish_open_tool_calls();
        }
        if self.reached_token_limit && !self.completed_tools.is_empty() {
            self.fail(
                "tool response reached the provider output limit and may be truncated",
                events,
            );
            return;
        }
        if self.completed_tools.is_empty() && !self.saw_non_whitespace_text {
            self.fail("the model returned an empty response", events);
            return;
        }
        for pending in std::mem::take(&mut self.completed_tools) {
            events.push(StreamEvent::ToolCall(finished_tool_call(pending)));
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
        // A half-accumulated call is never emitted: an incomplete argument
        // fragment, or a fully accumulated call from a response that later
        // failed, must not become a reviewable command.
        self.pending_tools = Vec::new();
        self.completed_tools = Vec::new();
    }
}

/// Anthropic's block index and OpenAI's `tool_calls[].index` both live here.
fn block_index(frame: &Value) -> Option<u64> {
    frame.get("index").and_then(Value::as_u64)
}

fn finished_tool_call(pending: PendingToolCall) -> ToolCall {
    ToolCall {
        id: pending.id,
        name: pending.name,
        // Normalize "no fragments arrived" to the empty object, so the value
        // matches what the non-streaming parser reports for `input: {}`.
        arguments: if pending.arguments.trim().is_empty() {
            "{}".to_string()
        } else {
            pending.arguments
        },
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
        calls: Vec<ToolCall>,
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
                StreamEvent::ToolCall(call) => folded.calls.push(call.clone()),
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
    fn empty_completed_streams_fail_like_non_streaming_responses() {
        let bodies = [
            (
                Provider::Anthropic,
                [
                    sse(
                        "message_delta",
                        &json!({"type": "message_delta",
                                "delta": {"stop_reason": "end_turn"}}),
                    ),
                    sse("message_stop", &json!({"type": "message_stop"})),
                ]
                .concat(),
            ),
            (
                Provider::OpenAiCompatible,
                [
                    openai_line(&json!({"choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"},
                    ]})),
                    "data: [DONE]\n\n".to_string(),
                ]
                .concat(),
            ),
            (
                Provider::Ollama,
                ndjson_line(&json!({"message": {"content": ""}, "done": true})),
            ),
        ];

        for (provider, body) in bodies {
            assert_eq!(
                run(provider, &body),
                vec![StreamEvent::Protocol(
                    "the model returned an empty response".into()
                )],
                "{provider:?}"
            );
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
    fn raw_stream_envelope_accepts_its_limit_and_rejects_the_next_byte() {
        let final_frame = ndjson_line(&json!({
            "message": {"content": "ok"},
            "done": true,
        }));
        let padding_len = MAX_STREAM_RESPONSE_BYTES - final_frame.len();

        let mut exact = vec![b'\n'; padding_len];
        exact.extend_from_slice(final_frame.as_bytes());
        let mut parser = StreamParser::new(Provider::Ollama);
        assert_eq!(
            parser.push(&exact),
            vec![StreamEvent::TextDelta("ok".into()), StreamEvent::Done]
        );

        let mut oversized = vec![b'\n'; padding_len + 1];
        oversized.extend_from_slice(final_frame.as_bytes());
        let mut parser = StreamParser::new(Provider::Ollama);
        let events = parser.push(&oversized);
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::Protocol(message)] if message.contains("stream response")
        ));
        assert_eq!(parser.finish(), vec![]);
    }

    #[test]
    fn tiny_irrelevant_frames_have_an_independent_count_budget() {
        let ignored = openai_line(&json!({}));
        let mut parser = StreamParser::new(Provider::OpenAiCompatible);
        assert!(parser
            .push(ignored.repeat(MAX_STREAM_FRAMES).as_bytes())
            .is_empty());

        let events = parser.push(ignored.as_bytes());
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::Protocol(message)] if message.contains("decoded frames")
        ));
        assert_eq!(parser.push(ignored.as_bytes()), vec![]);
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

    /// An Anthropic tool_use block whose JSON arguments arrive in fragments,
    /// after a prose preamble in a separate text block.
    fn anthropic_tool_body() -> String {
        [
            sse(
                "message_start",
                &json!({"type": "message_start", "message": {
                    "usage": {"input_tokens": 25, "output_tokens": 1}}}),
            ),
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": "Checking 编译 first."}}),
            ),
            sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            ),
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 1,
                        "content_block": {"type": "tool_use", "id": "toolu_1",
                                          "name": "run", "input": {}}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": "{\"comm"}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": "and\": \"ls "}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": "-la\"}"}}),
            ),
            sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 1}),
            ),
            sse(
                "message_delta",
                &json!({"type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 15}}),
            ),
            sse("message_stop", &json!({"type": "message_stop"})),
        ]
        .concat()
    }

    /// The OpenAI-compatible equivalent: index-keyed `tool_calls` deltas whose
    /// `arguments` string arrives in fragments.
    fn openai_tool_body() -> String {
        [
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "Checking 编译 first."},
                 "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "type": "function",
                     "function": {"name": "run", "arguments": ""}},
                ]}, "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "{\"comm"}},
                ]}, "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "and\": \"ls "}},
                ]}, "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "function": {"arguments": "-la\"}"}},
                ]}, "finish_reason": null},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
            ], "usage": {"prompt_tokens": 9, "completion_tokens": 12}})),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat()
    }

    fn expected_run_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "run".into(),
            arguments: "{\"command\": \"ls -la\"}".into(),
        }
    }

    #[test]
    fn split_tool_call_deltas_accumulate_into_one_call() {
        for (provider, body, id) in [
            (Provider::Anthropic, anthropic_tool_body(), "toolu_1"),
            (Provider::OpenAiCompatible, openai_tool_body(), "call_1"),
        ] {
            let events = run(provider, &body);
            let folded = fold(&events);
            assert_eq!(folded.text, "Checking 编译 first.", "{provider:?}");
            assert_eq!(folded.calls, vec![expected_run_call(id)], "{provider:?}");
            assert!(folded.done, "{provider:?}");
            assert_eq!(folded.protocol, None, "{provider:?}");

            // The completed call is ordered before Usage and Done.
            let call_at = events
                .iter()
                .position(|event| matches!(event, StreamEvent::ToolCall(_)))
                .expect("tool call event");
            let done_at = events
                .iter()
                .position(|event| *event == StreamEvent::Done)
                .expect("done event");
            assert!(call_at < done_at, "{provider:?}");

            // Byte-at-a-time delivery splits every fragment and still yields
            // exactly the same events.
            let mut parser = StreamParser::new(provider);
            let mut split = Vec::new();
            for byte in body.as_bytes() {
                split.extend(parser.push(std::slice::from_ref(byte)));
            }
            split.extend(parser.finish());
            assert_eq!(split, events, "{provider:?}");
        }
    }

    #[test]
    fn streamed_tool_call_matches_the_non_streaming_reply() {
        use crate::tools::{parse_tool_response, ToolResponse};

        let non_streaming = json!({
            "content": [
                {"type": "text", "text": "Checking 编译 first."},
                {"type": "tool_use", "id": "toolu_1", "name": "run",
                 "input": {"command": "ls -la"}},
            ],
            "stop_reason": "tool_use",
        });
        let parsed = parse_tool_response(Provider::Anthropic, &non_streaming).unwrap();
        let folded = fold(&run(Provider::Anthropic, &anthropic_tool_body()));
        let streamed = ToolResponse::new(folded.text, folded.calls);
        assert_eq!(streamed.text, parsed.text);
        assert_eq!(streamed.calls.len(), parsed.calls.len());
        assert_eq!(streamed.calls[0].id, parsed.calls[0].id);
        assert_eq!(streamed.calls[0].name, parsed.calls[0].name);
        // The same action, and therefore the same session behavior.
        assert_eq!(streamed.to_action(), parsed.to_action());
        assert!(streamed.to_action().is_ok());
    }

    #[test]
    fn truncated_tool_call_is_never_emitted() {
        for (provider, body) in [
            (Provider::Anthropic, anthropic_tool_body()),
            (Provider::OpenAiCompatible, openai_tool_body()),
        ] {
            // Cut inside the argument fragments, before the call is closed.
            let cut = body.find("-la\\\"}").expect("fragment marker");
            let mut parser = StreamParser::new(provider);
            let mut events = parser.push(&body.as_bytes()[..cut]);
            events.extend(parser.finish());
            let folded = fold(&events);
            assert_eq!(folded.calls, Vec::new(), "{provider:?}");
            assert!(folded.protocol.is_some(), "{provider:?}");
            assert!(!folded.done, "{provider:?}");
        }
    }

    #[test]
    fn closed_anthropic_call_is_withheld_if_the_response_later_fails() {
        let body = [
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "tool_use", "id": "toolu_1",
                                          "name": "run", "input": {}}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": "{\"command\":\"rm -rf data\"}"}}),
            ),
            sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            ),
            "data: not-json\n\n".to_string(),
        ]
        .concat();

        let events = run(Provider::Anthropic, &body);
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "malformed JSON in stream frame".into()
            )]
        );
    }

    #[test]
    fn payload_after_a_provider_end_signal_fails_closed() {
        let anthropic = [
            sse(
                "message_delta",
                &json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
            ),
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "tool_use", "id": "late",
                                          "name": "run", "input": {}}}),
            ),
            sse("message_stop", &json!({"type": "message_stop"})),
        ]
        .concat();
        assert_eq!(
            run(Provider::Anthropic, &anthropic),
            vec![StreamEvent::Protocol(
                "Anthropic streamed data after its stop reason".into()
            )]
        );

        let openai = [
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "late",
                     "function": {"name": "run",
                                  "arguments": "{\"command\":\"rm -rf data\"}"}},
                ]}, "finish_reason": null},
            ]})),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat();
        let events = run(Provider::OpenAiCompatible, &openai);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Protocol(message)) if message.contains("after its finish reason")
        ));
        assert!(!events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_) | StreamEvent::Done)));

        // OpenAI may legitimately send one usage-only frame after the choice
        // has finished. It carries no response payload and remains accepted.
        let usage_tail = [
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"},
            ]})),
            openai_line(&json!({"choices": [],
                                "usage": {"prompt_tokens": 3, "completion_tokens": 1}})),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat();
        let folded = fold(&run(Provider::OpenAiCompatible, &usage_tail));
        assert!(folded.done);
        assert_eq!(
            folded.usage,
            Some(Usage {
                input_tokens: Some(3),
                output_tokens: Some(1),
            })
        );
    }

    #[test]
    fn sequential_anthropic_calls_share_the_whole_response_limit() {
        let mut body = String::new();
        for index in 0..=MAX_STREAM_TOOL_CALLS {
            body.push_str(&sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use",
                                          "id": format!("toolu_{index}"),
                                          "name": "run", "input": {}}}),
            ));
            body.push_str(&sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": index,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": "{\"command\":\"true\"}"}}),
            ));
            body.push_str(&sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": index}),
            ));
        }

        let events = run(Provider::Anthropic, &body);
        assert!(matches!(
            events.as_slice(),
            [StreamEvent::Protocol(message)] if message.contains("more than")
        ));
    }

    #[test]
    fn anthropic_message_stop_does_not_flush_an_unclosed_tool_block() {
        let body = [
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 4,
                        "content_block": {"type": "tool_use", "id": "toolu_1",
                                          "name": "run", "input": {}}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 4,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": "{\"command\":\"rm -rf data\"}"}}),
            ),
            // A valid Anthropic stream must close index 4 before ending the
            // message. Treating message_stop like OpenAI's global call
            // terminator would promote this malformed incremental state.
            sse(
                "message_delta",
                &json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}}),
            ),
            sse("message_stop", &json!({"type": "message_stop"})),
        ]
        .concat();

        let events = run(Provider::Anthropic, &body);
        assert_eq!(
            events,
            vec![StreamEvent::Protocol(
                "Anthropic message ended before a tool_use block was closed".into()
            )]
        );
        assert!(!events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_) | StreamEvent::Done)));
    }

    #[test]
    fn token_limited_stream_never_publishes_a_tool_call() {
        let body = [
            sse(
                "content_block_start",
                &json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "tool_use", "id": "toolu_1",
                                          "name": "run", "input": {}}}),
            ),
            sse(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": "{\"command\":\"rm -rf /\"}"}}),
            ),
            sse(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": 0}),
            ),
            sse(
                "message_delta",
                &json!({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}),
            ),
            sse("message_stop", &json!({"type": "message_stop"})),
        ]
        .concat();

        let events = run(Provider::Anthropic, &body);
        assert_eq!(events.first(), Some(&StreamEvent::ReachedTokenLimit));
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Protocol(message)) if message.contains("may be truncated")
        ));
        assert!(!events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_) | StreamEvent::Done)));
    }

    #[test]
    fn malformed_tool_frames_fail_closed() {
        // Anthropic: a tool_use block with no name.
        let body = sse(
            "content_block_start",
            &json!({"type": "content_block_start", "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_1"}}),
        );
        assert_eq!(
            run(Provider::Anthropic, &body),
            vec![StreamEvent::Protocol(
                "tool_use block carries no name".into()
            )]
        );

        // Provider indexes bind later argument fragments to the call they
        // belong to. Defaulting a missing/negative index to zero could merge
        // distinct malformed calls into one reviewable command.
        for index in [Value::Null, json!(-1)] {
            let mut frame = json!({"type": "content_block_start",
                                   "content_block": {"type": "tool_use",
                                                     "id": "toolu_1", "name": "run"}});
            if !index.is_null() {
                frame["index"] = index;
            }
            assert!(matches!(
                run(Provider::Anthropic, &sse("content_block_start", &frame)).as_slice(),
                [StreamEvent::Protocol(message)] if message.contains("valid index")
            ));
        }

        // Anthropic: an input_json_delta with no payload.
        let body = sse(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": 0,
                    "delta": {"type": "input_json_delta"}}),
        );
        assert_eq!(
            run(Provider::Anthropic, &body),
            vec![StreamEvent::Protocol(
                "input_json_delta frame carries no partial_json".into()
            )]
        );

        // Anthropic: arguments for a block that was never opened as a tool.
        let body = sse(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": 3,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        );
        assert_eq!(
            run(Provider::Anthropic, &body),
            vec![StreamEvent::Protocol(
                "tool-call arguments arrived for an unopened tool call".into()
            )]
        );

        // OpenAI: non-string arguments in a delta.
        let body = openai_line(&json!({"choices": [
            {"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"name": "run", "arguments": 42}},
            ]}},
        ]}));
        assert_eq!(
            run(Provider::OpenAiCompatible, &body),
            vec![StreamEvent::Protocol(
                "tool call arguments are not a string".into()
            )]
        );

        let body = openai_line(&json!({"choices": [
            {"index": 0, "delta": {"tool_calls": [
                {"id": "call_1", "function": {"name": "run", "arguments": "{}"}},
            ]}},
        ]}));
        assert!(matches!(
            run(Provider::OpenAiCompatible, &body).as_slice(),
            [StreamEvent::Protocol(message)] if message.contains("valid index")
        ));

        // OpenAI: tool_calls that is not an array.
        let body = openai_line(&json!({"choices": [
            {"index": 0, "delta": {"tool_calls": "run"}},
        ]}));
        assert_eq!(
            run(Provider::OpenAiCompatible, &body),
            vec![StreamEvent::Protocol(
                "delta tool_calls is not an array".into()
            )]
        );
    }

    #[test]
    fn anthropic_tool_indexes_cannot_be_reused_after_a_block_closes() {
        let call = |id: &str| {
            [
                sse(
                    "content_block_start",
                    &json!({"type": "content_block_start", "index": 2,
                            "content_block": {"type": "tool_use", "id": id,
                                              "name": "run", "input": {}}}),
                ),
                sse(
                    "content_block_delta",
                    &json!({"type": "content_block_delta", "index": 2,
                            "delta": {"type": "input_json_delta",
                                      "partial_json": "{\"command\":\"true\"}"}}),
                ),
                sse(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": 2}),
                ),
            ]
            .concat()
        };
        let body = [call("toolu_1"), call("toolu_2")].concat();
        assert_eq!(
            run(Provider::Anthropic, &body),
            vec![StreamEvent::Protocol(
                "stream reused a tool-call index".into()
            )]
        );
    }

    #[test]
    fn oversized_tool_arguments_fail_closed() {
        let fragment = "a".repeat(32 * 1024);
        let mut parser = StreamParser::new(Provider::OpenAiCompatible);
        let mut events = parser.push(
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "run", "arguments": ""}},
                ]}},
            ]}))
            .as_bytes(),
        );
        for _ in 0..3 {
            events.extend(
                parser.push(
                    openai_line(&json!({"choices": [
                        {"index": 0, "delta": {"tool_calls": [
                            {"index": 0, "function": {"arguments": fragment}},
                        ]}},
                    ]}))
                    .as_bytes(),
                ),
            );
        }
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Protocol(message)) if message.contains("byte limit")
        ));
        assert!(!events
            .iter()
            .any(|event| matches!(event, StreamEvent::ToolCall(_))));
        assert_eq!(parser.finish(), vec![]);
    }

    #[test]
    fn parallel_tool_calls_are_surfaced_separately_for_the_caller_to_reject() {
        // The parser reports what arrived; the "exactly one call" rule lives
        // in ToolResponse::to_action so streaming and non-streaming agree.
        let body = [
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "a", "function": {"name": "run", "arguments": "{}"}},
                    {"index": 1, "id": "b", "function": {"name": "say", "arguments": "{}"}},
                ]}},
            ]})),
            openai_line(&json!({"choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
            ]})),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat();
        let folded = fold(&run(Provider::OpenAiCompatible, &body));
        assert_eq!(folded.calls.len(), 2);
        assert_eq!(folded.calls[0].id, "a");
        assert_eq!(folded.calls[1].id, "b");
        assert_eq!(
            crate::tools::ToolResponse::new("", folded.calls).to_action(),
            Err(crate::session::ParseError::MultipleToolCalls(2))
        );
    }

    #[test]
    fn text_only_streams_are_unchanged_by_tool_support() {
        for provider in ALL_PROVIDERS {
            let folded = fold(&run(provider, &happy_body(provider)));
            assert_eq!(folded.calls, Vec::new(), "{provider:?}");
            assert_eq!(folded.text, TEXT, "{provider:?}");
        }
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
