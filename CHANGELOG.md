# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Exported `is_unsafe_invisible_char` so integrations can apply the crate's own
  invisible/bidirectional rule instead of re-deriving it. It is the only such
  check on a model-proposed command, and while it was crate-private every
  consumer kept a private copy that could only diverge from it; a review card,
  or a shell branch that inserts a proposal into the input buffer, can now call
  the same predicate the proposal was validated with. The set may be widened,
  never narrowed, and a test sweeps every Unicode scalar to hold it to that in
  both directions.
- Added the narrow `validate_no_duplicate_members(&[u8])` structural preflight
  used by jagent's own wire decoders, so integrations can apply the identical
  recursive uniqueness rule to other already byte-bounded JSON trust
  boundaries before typed deserialization. Development tests force
  serde_json's map-backed `arbitrary_precision` number and `raw_value` paths so
  downstream feature unification cannot silently change that contract.
- Added `HttpRequest::transport_metrics` and `validate_transport` with a
  content-free `HttpRequestMetrics` report. Generated requests now carry
  explicit URL, header-count, aggregate-header-byte, JSON-object, and body
  postconditions before leaving the sans-I/O boundary. Directly constructed or
  mutated requests receive the same HTTPS/loopback-HTTP authority checks, and
  ambiguous duplicate JSON fields are rejected recursively.

- Added versioned `AgentCapabilities` discovery for protocol/delivery
  negotiation across split terminal and shell integrations. Its bounded ASCII
  token supports capability subsets, strict parsing, and preference-ordered
  negotiation; `prepare_agent_request` consults the same provider capability
  table before constructing a request.
- Added exact protocol/delivery pairs in capability token version 2 while
  retaining strict version-1 decoding and compatibility-first v1 emission.
  Peer-version-aware and explicit v2 APIs prevent a new endpoint from sending
  an unsupported-version token to an unprobed 0.7 peer. Added
  `CommandExecutionOutcome` plus `observe_execution` so integrations cannot
  turn setup failures into synthetic exit codes.

- Added a runnable no-I/O NativeTools streaming example and a production
  integration guide covering transport, response decoding, review, failure,
  cancellation, and persistence boundaries.
- Added contributor and private security-reporting guidance.
- Added a dedicated Rust 1.86 CI lane, packaged-crate verification, executable
  example checks, pull-request dependency review, and weekly Cargo and GitHub
  Actions update checks.
- Added `AgentResponse::protocol`, `AgentStream::protocol`, and
  `AgentStream::is_complete` for integration diagnostics, plus an error source
  from `SessionError::Protocol` to its underlying `ParseError`.

### Changed

- Provider extension values are measured through the JSON serializer before
  cloning or request assembly, with independent per-value and cumulative
  encoded budgets. Extension failures and duplicate/reserved-name diagnostics
  no longer reflect caller-controlled names or values.
- Provider configuration rejects port zero and caps implausible output-token
  requests, while generated headers are checked as unique canonical lowercase
  HTTP tokens with printable-ASCII values and exactly one JSON content type.

- API keys are now validated as exact, non-empty printable-ASCII HTTP header
  values. Configuration with surrounding/internal whitespace, Unicode, or
  other transport-dependent bytes is rejected instead of being silently
  trimmed or delegated to an HTTP client's inconsistent header parser.
- Declared Rust 1.86 as the crate MSRV, added docs.rs and repository metadata,
  forbade unsafe Rust, and restricted published package contents to the public
  source, examples, guides, policies, changelog, and licenses.
- Pinned every third-party CI action to a full commit SHA and made CI
  credentials read-only and non-persistent.
- Provider generation-limit reporting now includes Anthropic context-window
  exhaustion as incomplete output, including the high-level action guard and
  streaming event path.
- Selected-block and environment contexts are now budgeted after JSON encoding;
  local output elision is reported and untrusted closing-tag prefixes are
  escaped so values cannot terminate their prompt envelope.
- High-confidence redaction now covers more provider/package tokens, truncated
  private-key blocks, named secret settings and headers, and signed/OAuth URL
  parameters while remaining idempotent.
- Dangerous-command warnings now recognize shell boundaries, substitutions,
  wrappers, top-level targets, destructive Git/infrastructure/service/storage
  operations, and review-smuggling control or invisible characters.

### Fixed

- Rejected serde_json's private RawValue sentinel during structural preflight.
  With the dependency's `raw_value` feature unified into a downstream graph,
  native `Value` decoding otherwise reparsed the sentinel's string value as a
  new unchecked JSON document, allowing nested duplicate members to bypass the
  first pass and collapse with last-value-wins semantics.

- `HttpRequest::validate_transport` now rejects duplicate JSON object members
  recursively, including nested messages, tools, and provider options. The
  earlier top-level-only preflight left directly constructed or mutated bodies
  with parser-dependent first-value/last-value semantics below the root.
- Complete provider responses, SSE/NDJSON frames, JSON-in-text actions, and
  string-valued native-tool arguments now reject duplicate JSON object members
  at every depth. The shared duplicate-aware preflight prevents
  first-value/last-value parser differences from changing a completion state
  or proposed command without allocating a second response tree.
- The protocol-aware request path now rejects a second history omission at
  runtime in release builds. A future drift between preparation and provider
  JSON-wire budgets can no longer silently make `AgentRequestReport`
  understate lost context.
- Corrected migration guidance to distinguish the high-level request path's
  default secret redaction from the low-level builders' explicit preparation
  policy.
- Streaming SSE/NDJSON framing now accepts CRLF delimiters split across
  transport chunks, bare-CR record endings, and an initial SSE UTF-8 BOM.
- Ollama error payloads are surfaced as structured provider failures instead
  of falling through to unrelated response-shape errors.
- Base URL validation now requires an ASCII DNS hostname or canonical IP
  literal and rejects ambiguous numeric, encoded, empty-label, and otherwise
  transport-dependent host spellings.

### Compatibility

- High-level `AgentResponse` decoding now requires one complete,
  protocol-consistent provider envelope before action parsing. The low-level
  native-tool parser retains harmless sparse-fixture compatibility while
  rejecting explicit failures, ambiguous choices, and completion metadata
  inconsistent with a present call; the legacy low-level chat parser remains
  a tolerant text extractor.

### Security

- The invisible/bidirectional character set that gates every model-proposed
  command, provider model name, endpoint URL, and generated header value now
  matches the whole supplementary tag plane (`U+E0000..=U+E0FFF`) and the
  reserved specials (`U+FFF0..=U+FFF8`) rather than only the tag characters and
  variation selectors Unicode has assigned. Unassigned code points such as
  `U+E0000` and `U+E0080` have general category `Cn`, so `char::is_control` is
  false and they are not whitespace; a reply like
  `{"action":"run","command":"ls -la /etc\u{E0000}"}` previously passed
  `validate_command` and reached the approval card carrying text the reviewer
  could not see. Assigned neighbours (`U+FFF9..=U+FFFB` interlinear annotation
  anchors, `U+13430` onward) remain valid command text.
- High-level response handling now fails closed before action conversion on
  provider refusals, content filtering, context truncation, unknown completion
  reasons, multiple choices, legacy `function_call` deltas, malformed choices,
  and invalid tool-call types.
- The high-level Anthropic/OpenAI `AgentStream` now requires a supported
  completion reason before the final marker (the low-level `StreamParser`
  event contract remains compatible), and non-streaming OpenAI/Ollama replies
  reject explicitly non-function tool-call types.
- Anthropic/OpenAI completion reasons must agree with whether a tool payload is
  present, so tool-shaped endings cannot promote valid-looking text and
  ordinary endings cannot authorize contradictory tool calls.

## [0.7.0] - 2026-08-10

### Added

- Added `AgentRequestSpec` and `prepare_agent_request`, the recommended
  protocol-aware request path with matching built-in prompts, declarative
  streaming, default high-confidence secret redaction, and a history
  preparation report.
- Added `PreparedAgentRequest::parse_response` and `response_stream` so the
  response provider, protocol, and delivery mode stay bound to the request;
  using the wrong delivery path is rejected.
- Added `HistoryReport`, `PreparedHistory`, and report-bearing history helpers
  that distinguish changed, middle-elided, and fully omitted turns.
- Added `AgentResponse` to unify bounded Text and NativeTools response parsing,
  metadata access, and action conversion. Text mode now rejects native tool
  calls explicitly.
- Added `AgentStream`, which folds low-level events into an `AgentResponse` and
  succeeds only after a complete, protocol-consistent stream.
- Added NativeTools request, non-streaming response, and streaming support for
  Ollama `/api/chat` using its OpenAI-shaped tool-call wire format.
- Added `AgentSession::accept_agent_response` as the high-level response
  ingestion point.
- Added `pending_proposal`, `transcript_truncated`, rejection feedback, and
  explicit command-execution failure transitions to the session API.
- Added an integration-first, no-I/O quickstart example and complete MIT and
  Apache-2.0 license texts.

### Changed

- Aligned local development and CI with the terminal-family Rust `stable`
  toolchain convention (`minimal` profile with rustfmt and Clippy).
- High-level agent request preparation now redacts high-confidence secrets by
  default before history budgeting. Opting out is explicit.
- `HttpRequest` debug output now reports only URL/body lengths and header count
  instead of emitting AI-bound context or caller-controlled header names and
  values.
- Plain HTTP loopback endpoints are accepted for every provider so local
  OpenAI-compatible servers and proxies can be configured without weakening
  the HTTPS requirement for remote hosts.
- Base URL validation now rejects surrounding whitespace and ports outside the
  `u16` range.
- Failed provider, transport, and command turns are framed as general bounded
  diagnostics rather than incorrectly describing each as a JSON violation.
- In-place proposal lifecycle changes re-run transcript compaction before the
  session can be snapshotted.

### Compatibility

- Existing provider request builders, response parsers, native tool helpers,
  `StreamParser`, and protocol-specific session ingestion methods remain
  available as lower-level APIs.
- Existing `bound_history` and `bound_history_with` tuple results are retained;
  the new report-bearing variants are additive.
- The validated snapshot wire version remains version 1.

### Security

- Default history preparation now avoids cloning oversized clean turns before
  bounding them, preventing attacker-controlled transient memory growth.
- Anthropic streaming now validates the complete content-block lifecycle and
  rejects index reuse or block-kind confusion before publishing tool calls.
- Token-limited Text responses cannot become actions even when their partial
  contents happen to be syntactically valid JSON.
- Text-mode non-streaming and streaming paths fail closed when native tool
  calls appear.
- Custom environment tag names are syntax- and length-checked before they are
  used as untrusted-data framing delimiters.
- A complete streamed tool call still becomes only a proposal and never an
  execution authorization.

## [0.6.0] - 2026-08-08

### Added

- Added bounded snapshot decoding with allocation budgets and semantic restore
  validation.
- Added bounded byte-oriented parsing for non-streaming text and native tool
  responses.
- Added raw-response and decoded-frame ceilings to the streaming parser.
- Added reported request builders, provider configuration validation, and
  request-side history/system-prompt budgets.

[Unreleased]: https://github.com/beamiter/jagent/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/beamiter/jagent/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/beamiter/jagent/releases/tag/v0.6.0
