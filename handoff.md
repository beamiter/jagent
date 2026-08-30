# Engineering handoff

Updated: 2026-08-25
Baseline: 0.7.0
Release target: Unreleased

The 0.7 baseline adds an integration-first path over the hardened 0.6 primitives.
The low-level provider, tool, stream, and session APIs remain available, while
new integrations can bind request preparation, response decoding, and session
ingestion to one `AgentProtocol` and receive machine-readable history
transformation reports.

The high-level history window is now a release-mode postcondition as well as a
debug invariant: if a provider builder omits any turn after
`bound_history_*_with_report` has already produced the canonical window,
`prepare_agent_request` fails closed. This keeps the preparation report and the
encoded request truthful if the two JSON-wire budgets ever drift.

Snapshot restoration now owns the complete retained lifecycle rather than
validating its atoms independently. Resumable approval and execution states
must name the final proposal, turn counters and terminal states must agree with
the retained history, observations must immediately follow their approved
proposal, and every approved proposal must retain either that observation or
the exact proposal-bound execution-failure/unknown-result diagnostic. This
centralizes the invariant previously audited again by terminal consumers and
prevents persisted text from covering the command an approval UI may act on.

Untrusted action/response JSON now has one interpretation across the complete
and streaming paths. A shared allocation-light preflight rejects duplicate
object members recursively before the ordinary `Value` decoder retains a
response tree or any response delta/action is promoted; the same rule covers
JSON-in-text actions and string-valued native-tool arguments. This closes the
previous first-value/last-value ambiguity without a second decoded response
tree or a public API change. The compatibility `Value`-based response APIs
still require already-decoded trusted input, where duplicate spelling is
necessarily no longer observable.

The same recursive uniqueness rule now closes the remaining outbound gap.
`HttpRequest::validate_transport` rejects duplicate members inside nested
messages, tools, and provider options as well as at the root, without retaining
a decoded request tree. Directly constructed or mutated request bodies can no
longer pass validation while different transports/providers choose different
values.
The visitor is exposed narrowly as `validate_no_duplicate_members(&[u8])`, so
`jterm_core::bounded_json` can re-export one semantic implementation for
credential, IPC, and persistence boundaries instead of copying the decoder.
The development graph forces serde_json's map-backed `arbitrary_precision`
number representation and its `raw_value` Value-decoder escape path. The
preflight rejects the private RawValue sentinel in every object position, so a
feature-unified decoder cannot reinterpret a string as a second unchecked JSON
document; genuine large numbers and ordinary near-miss member names remain
accepted.

## 2026-08-22 ten-round provider hardening

1. `ChatConfig` and `HttpRequest` Debug report only transport metadata, counts,
   and byte lengths; they do not echo credentials, model/URL text, header
   names or values, or request bodies.
2. Provider parsing bounds the raw spelling, rejects invisible formatting,
   and never reflects an unknown value into diagnostics.
3. `ChatConfig::endpoint` makes configuration validation part of endpoint
   resolution while retaining the unchecked `Provider::endpoint` compatibility API.
4. Every built request now obtains its URL through that validated endpoint.
5. Provider extension fields cannot replace core `model`, `messages`,
   `system`, `max_tokens`, `temperature`, or `options` fields.
6. Extensions likewise cannot replace `stream` or `stream_options`, so a
   protocol-bound delivery mode cannot be silently changed.
7. Extension count, name size/visibility, and uniqueness are checked while
   ordinary future provider fields remain available.
8. The complete encoded request body has a final 4 MiB ceiling after provider
   fields and JSON escaping are applied.
9. History retention budgets exact JSON string escaping and message framing;
   escape-heavy input is sampled to the same ceiling as ordinary text.
10. `HistoryReport::sent_history_json_bytes` exposes that non-sensitive wire
    cost, with a serde-based oracle and cross-provider failure-path tests.

## 2026-08-22 twenty-round request transport hardening

1. Completed endpoint URLs have an explicit byte ceiling independent of the
   configured base-URL ceiling.
2. A request may contain at most 16 headers before entering an HTTP stack.
3. Header names and values share a checked 64 KiB aggregate byte budget.
4. Header names must use the RFC token alphabet.
5. Generated and validated header names use one canonical lowercase spelling.
6. Header values must be printable ASCII, excluding control and newline
   injection.
7. Duplicate header names are rejected instead of relying on transport-specific
   merge rules.
8. Exactly one `content-type: application/json` header is required.
9. The complete serialized body retains its 4 MiB post-assembly ceiling.
10. The transport validator requires one syntactically complete top-level JSON
    object with unique member names recursively, without allocating a second
    `serde_json::Value` tree.
11. `HttpRequest::transport_metrics` uses checked arithmetic and exposes only
    byte/count metadata.
12. `HttpRequest` Debug now derives its URL/header/body accounting from that
    same content-free metrics contract.
13. Every built request passes the public transport validator as its final
    postcondition.
14. Each provider-extension value is measured by the real JSON serializer
    before it is cloned into the body.
15. Extensions also share a 2 MiB encoded aggregate budget, preserving room
    for the bounded core request fields.
16. Extension serialization and budget errors are generic and never echo a
    caller-controlled field name or value.
17. The extension preflight runs before history cloning, body construction,
    or extension cloning.
18. `max_tokens` is limited to the documented positive range ending at one
    million instead of accepting an arbitrary `u32` from corrupt settings.
19. Port zero is rejected for both HTTPS and local-loopback HTTP authorities.
20. Cross-provider and adversarial direct-request tests pin metrics, headers,
    body shape, extension escaping, aggregate budgets, port, and token edges.

## Current 0.7 surface

### Capability discovery

- `agent_capabilities(provider)` returns the same provider protocol/delivery
  matrix that `prepare_agent_request` checks, but deliberately emits v1 for
  safe first contact with older 0.7 peers. `agent_capabilities_for_peer`
  mirrors a decoded peer's schema version; `agent_capabilities_v2` is the
  explicit out-of-band opt-in.
- `AgentCapabilities::{to_wire,from_wire}` provides a strict, versioned,
  256-byte ASCII contract for environment or IPC discovery. The opt-in
  version 2 form uses `modes` as exact protocol/delivery pairs rather than a
  Cartesian product. Strict version-1 tokens remain readable and are the
  default outbound form during rolling upgrades. Its downgrade selects only a
  Cartesian subset of the exact matrix and can never overclaim a crossed mode;
  both versions reject unknown, duplicate, non-canonical, empty, overlong, or
  future-version values.
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
- `CommandExecutionOutcome` and `observe_execution` form the preferred typed
  executor-to-session handoff: `Exited` requires a real status, while `Failed`
  records failed start, timeout, or cancellation without inventing one.
  Diagnostic/partial output is sampled under the observation budget; the two
  older observation methods remain compatible.
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
- `HttpRequest::Debug` reports only URL/body lengths and the header count.
  Caller-controlled header names and values are omitted wholesale rather than
  relying on a finite credential-name list.
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
6. Encoded response frames and action objects reject duplicate JSON members
   recursively; no parser-specific first/last-value choice may select an
   action or completion state.
7. Restore validates proposal ordering, adjacent execution outcomes, model-turn
   accounting, final-turn state binding, and transcript budgets before making
   a session live.

## Remaining integration boundaries

`jagent` remains sans-IO. It now validates the request value's URL/body/header
shape and budgets; consumers still own:

- HTTP framing overhead, TLS policy, redirects, deadlines, cancellation, and
  response status handling;
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
