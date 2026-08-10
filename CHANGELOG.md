# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- `HttpRequest` debug output now reports only request-body length instead of
  emitting AI-bound user context.
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
