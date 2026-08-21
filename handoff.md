# Engineering handoff

Updated: 2026-08-21
Baseline: 0.7.0
Release target: Unreleased

The 0.7 baseline adds an integration-first path over the hardened 0.6 primitives.
The low-level provider, tool, stream, and session APIs remain available, while
new integrations can bind request preparation, response decoding, and session
ingestion to one `AgentProtocol` and receive machine-readable history
transformation reports.

## Current 0.7 surface

### Capability discovery

- `agent_capabilities(provider)` returns the same provider protocol/delivery
  matrix that `prepare_agent_request` now checks before request construction.
- `AgentCapabilities::{to_wire,from_wire}` provides a strict, versioned,
  256-byte ASCII contract for environment or IPC discovery. Version 1 can
  encode subsets, but requires canonical field/list order and rejects unknown,
  duplicate, empty, overlong, or future-version values.
- `negotiate_with(peer, preferred, delivery)` selects only the first mutually
  supported protocol; it never invents a fallback. Capability agreement
  changes encoding only and cannot authorize a command.

### Recommended agent loop

- `AgentRequestSpec::new(history, protocol)` selects the protocol-matched
  built-in system prompt, non-streaming delivery, and high-confidence secret
  redaction by default.
- `prepare_agent_request` returns `PreparedAgentRequest { request, report }`.
  The report distinguishes changed, middle-elided, and fully omitted history
  turns without retaining sensitive contents. The prepared value also retains
  the provider, protocol, and delivery mode used to build the wire request.
- `PreparedAgentRequest::parse_response` performs one bounded JSON decode and
  selects `ChatResponse` or `ToolResponse` according to the bound protocol.
  It requires one complete, protocol-consistent provider envelope; Text mode
  rejects native calls instead of discarding them. The low-level native-tool
  parser retains sparse-fixture compatibility but rejects explicit failures,
  ambiguous choices, and completion metadata inconsistent with a present
  call; the legacy low-level chat parser remains a tolerant text extractor.
- `AgentSession::accept_agent_response` is the paired high-level ingestion
  point. Token-limited output and protocol mismatches never become actions.
- `PreparedAgentRequest::response_stream` creates the correctly paired
  `AgentStream`, which transparently returns low-level `StreamEvent`s while
  folding a response internally. Conversion succeeds only after `Done`;
  refusal/filter/error states, unknown completion, premature EOF, and Text-mode
  tool calls fail closed. `protocol()` and `is_complete()` expose diagnostics;
  `into_response` remains authoritative.
- Delivery mismatch is rejected: non-streaming requests use `parse_response`,
  while streaming requests use `response_stream`.
- Anthropic, OpenAI-compatible Chat Completions, and Ollama `/api/chat` all
  support Text, NativeTools, and streaming through the high-level path.
  Ollama uses OpenAI-shaped tools without a `tool_choice` field.

The executable `examples/quickstart.rs` exercises the complete path without
network or process I/O. It uses an OpenAI-compatible loopback endpoint, proves
default redaction and report accounting, decodes a local response fixture,
and explicitly walks proposal, approval, and observation transitions.

`examples/streaming.rs` covers the complementary NativeTools streaming path.
It feeds a local OpenAI-compatible SSE fixture across arbitrary byte
boundaries, completes the prepared accumulator, and demonstrates that a raw
tool event is not approval.

### Session review lifecycle

- `pending_proposal()` exposes a borrowed, state-bound view of the proposal a
  review UI should currently display.
- `reject_with_feedback` validates feedback atomically and records it as the
  next untrusted user turn.
- `observe_execution_failure` records failed start, timeout, or cancellation
  without inventing an exit code; diagnostic/partial output is sampled under
  the observation budget.
- `transcript_truncated()` exposes whether in-memory compaction removed older
  activity. In-place approval, edit, rejection, and manual-review mutations
  recompact before a snapshot can be captured.
- The persisted `ProtocolError` variant retains snapshot compatibility but is
  now framed as a general failed-turn diagnostic rather than falsely claiming
  every transport or execution failure violated JSON.

### Provider and context preparation

- `HistoryReport` and `PreparedHistory` expose input, sent, omitted, changed,
  elided, and retained-text-byte counts.
- `redact_secrets_cow` borrows clean input; `redact_secrets` remains the owned
  compatibility wrapper. Detection covers established provider/package token
  formats, named secrets and headers, private-key blocks, and signed/OAuth URL
  parameters without treating every opaque identifier as a secret.
- `HttpRequest::Debug` reports body length instead of logging AI-bound user
  context. Credential headers remain redacted.
- Plain HTTP is accepted for syntactic loopback endpoints for every provider,
  enabling local OpenAI-compatible servers and proxies. Remote endpoints still
  require HTTPS and an ASCII DNS name or canonical IP literal; URL whitespace,
  invalid ports, credentials, query, fragment, encoded/ambiguous hosts,
  backslash, control, and ambiguous display characters are rejected.
- Block and environment context budgets are enforced after JSON encoding,
  local output elision is disclosed, and untrusted values cannot close their
  raw prompt envelope.

## Compatibility contract

The high-level path is additive:

- Existing `build_chat_request*`, `build_agent_chat_request*`,
  `parse_chat_response*`, `parse_tool_response*`, and `StreamParser` APIs stay
  available.
- `AgentResponse::parse_bytes` and `AgentStream::new` remain low-level options
  for integrations that intentionally pair provider, protocol, and delivery
  mode themselves.
- Existing request builders continue to apply their own safety bounds.
  Compatibility builders returning `HttpRequest` still discard their limited
  omission report; integrations wanting history preparation diagnostics
  should move to `prepare_agent_request`.
- `bound_history` and `bound_history_with` retain their tuple results. New
  report-bearing variants expose preparation and elision separately.
- `accept_model_reply` and `accept_model_tool_reply` remain valid for callers
  that intentionally coordinate the protocol themselves.
- Snapshot version 1 and its bounded decoder remain unchanged. New session
  behavior is represented within the existing validated schema.

See [docs/jterm4-migration.md](docs/jterm4-migration.md) for ownership and
consumer migration details.

## Security properties to preserve

1. A model response can create only a proposal. Execution requires the exact
   `ApprovedCommand` returned for the currently pending proposal ID.
2. Text, native-tool, non-streaming, and streaming response paths all reject
   truncation or protocol mismatch before action conversion.
3. Secret redaction is a conservative outbound safeguard, not a substitute
   for transport security or integration-owned DLP policy.
4. Debug formatting must not create a second sink for API credentials or
   request bodies.
5. Provider response bytes, streaming bodies/frames, histories, prompts,
   transcript fields, observations, tool identifiers, and arguments remain
   independently bounded.
6. Restore continues to validate proposal ordering, lifecycle, observation
   binding, active state, and transcript budgets before making a session live.

## Remaining integration boundaries

`jagent` remains sans-IO. Consumers own:

- HTTP header/count limits, TLS policy, redirects, deadlines, cancellation,
  and response status handling;
- API-key acquisition and storage;
- PTY/process creation, shell semantics, execution timeout, and output capture;
- the approval UI and deliberate execution of `ApprovedCommand`;
- snapshot storage permissions, replacement, and lifecycle.

Do not process a raw streamed `ToolCall` as authorization. Fold the complete
stream, ingest the resulting `AgentResponse`, display the resulting proposal,
and require explicit approval.

## Development and release contract

- `Cargo.toml` declares Rust 1.86 as the MSRV; `rust-toolchain.toml` keeps local
  development on current stable with rustfmt and Clippy.
- CI checks stable formatting, linting, rustdoc warnings, tests, executable
  examples, the packaged crate, and a dedicated Rust 1.86 build/test lane.
- Third-party GitHub Actions are pinned to full commit SHAs and Dependabot
  watches both Cargo and workflow dependencies.
- The generic transport/review lifecycle is documented in
  `docs/integration-guide.md`; historical consumer-specific notes remain in
  `docs/jterm4-migration.md`.

## Release checklist

Before tagging the next release:

1. Set the intended package version and ensure the root entry in `Cargo.lock`
   matches it. Move user-visible entries out of `Unreleased` only as part of
   that release change.
2. Run the no-I/O example and the locked stable-toolchain gates.
3. Run the Rust 1.86 check/test lane.
4. Verify the packaged crate contains README, changelog, contribution and
   security policies, both license files, both guides, and both examples.

```text
cargo fmt --all -- --check
cargo run --locked --example quickstart
cargo run --locked --example streaming
cargo check --locked --all-targets --all-features
cargo test --locked --all-targets --all-features --no-fail-fast
cargo test --locked --all-features --doc
cargo clippy --locked --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --all-features --no-deps
cargo +1.86.0 check --locked --all-targets --all-features
cargo +1.86.0 test --locked --all-targets --all-features --no-fail-fast
cargo package --locked --allow-dirty
```

The examples intentionally send and execute nothing. Successful runs prove
request construction, secure-default reporting, bounded local response
parsing, and explicit review transitions only.
