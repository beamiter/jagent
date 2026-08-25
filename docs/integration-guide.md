# Integration guide

This guide describes the recommended `jagent` 0.7 integration path. The crate
is intentionally sans-IO: it prepares bounded HTTP requests and validates
model responses, while the embedding application owns networking, command
execution, storage, cancellation, and every user-facing decision.

For a runnable non-streaming loop, start with
[`examples/quickstart.rs`](../examples/quickstart.rs). The
[`examples/streaming.rs`](../examples/streaming.rs) companion covers streaming
native tool calls. Both examples use local fixtures and perform no network or
process I/O.

## End-to-end flow

```text
user input
    │
    ▼
AgentSession::submit_user
    │  build_user_prompt_with(protocol)
    ▼
Message history + untrusted environment context
    │
    ▼
prepare_agent_request ──► preparation report
    │                     (redaction / elision / omission)
    ▼
integration-owned HTTP transport
    │
    ├─ complete body ──► prepared.parse_response(...)
    │
    └─ chunks ─────────► prepared.response_stream(...)
                          push → finish → into_response
    │
    ▼
AgentSession::accept_agent_response
    │
    ├─ Said / Completed ──► render
    │
    └─ Proposal ──────────► display exact command and warning
                              │
                              ├─ reject / manual review
                              │
                              └─ approve ──► ApprovedCommand
                                               │
                                               ▼
                                      integration-owned executor
                                               │
                                               ▼
                                         observe_execution
```

Keep one `AgentProtocol` value for the full turn. It determines the session's
closing instruction, built-in system prompt, provider schema, and response
decoder. `PreparedAgentRequest` retains the provider, protocol, and delivery
mode so the integration does not have to reconstruct those choices later.

## Discover and negotiate capabilities

When the transport/UI and executor/shell are separate processes, exchange a
capability token before selecting `AgentProtocol`. The token is bounded ASCII
and contains only a schema version plus supported protocol/delivery modes;
it is safe to carry in an environment variable or IPC metadata field and must
not be confused with user context.

1. For first contact, advertise `agent_capabilities(provider)`. It deliberately
   emits version 1 so pre-v2 0.7 peers can decode it.
2. Decode the peer with
   `AgentCapabilities::from_wire`.
3. Use `agent_capabilities_for_peer(provider, peer)` when replying, or the
   explicit `agent_capabilities_v2(provider)` only when peer support is already
   known out of band.
4. Call `local.negotiate_with(peer, preferences, delivery)` with the local
   preference order and required `AgentDelivery`.
5. If it returns `None`, report the incompatibility. Do not silently switch
   protocols after a request has been prepared.

`prepare_agent_request` independently checks the same capability table before
building provider bytes, so discovery cannot drift from enforcement. Version
2 advertises exact pairs and therefore does not infer unsupported crossed
combinations. Strict version-1 tokens remain accepted with their original
Cartesian-product semantics, and remain the default outbound form during a
rolling upgrade. A non-Cartesian provider matrix is safely reduced to a
representable v1 subset rather than overclaimed. Malformed, duplicate,
non-canonical, empty, unknown-version, or overlong tokens fail closed.
Capability agreement does not change the review contract: native tool calls
remain proposals until the session returns an `ApprovedCommand`.

## Prepare a turn

1. Call `AgentSession::submit_user`.
2. Build the session instruction with `build_user_prompt_with(protocol)`.
3. If terminal or repository context is needed, wrap it with
   `agent_user_prompt`; `EnvironmentMeta` and `BlockContext` are encoded as
   explicitly untrusted user-role data.
4. Put that text in the provider `Message` history.
5. Call `prepare_agent_request` with `AgentRequestSpec::new`.

The high-level specification uses the protocol-matched built-in prompt and
high-confidence history redaction by default. Inspect
`PreparedAgentRequest::report` before sending:

- `changed_history_turns` reports retained turns changed by redaction;
- `elided_history_turns` reports retained turns shortened to a byte budget;
- `omitted_history_turns` reports older turns dropped from the window;
- `request_body_bytes` lets the transport enforce an equal or tighter body
  policy without logging the body.

The report contains counts and sizes, never the affected text. Disabling
redaction or replacing the built-in system prompt is an explicit advanced
choice on `AgentRequestSpec`. Low-level provider builders do not apply the
high-level redaction default. Context builders independently enforce their
budget after JSON escaping, mark locally shortened block output as truncated,
and escape any raw closing-tag prefix inside an untrusted value so it cannot
terminate the surrounding prompt envelope.

Preparation also checks that the provider builder omits no additional turn
from this already-bounded history. That is a release-mode failure, not a debug
assertion: a successful `PreparedAgentRequest` therefore has one truthful
history-loss report even if internal wire budgets change in a later release.

## Perform the HTTP request

`prepared.request` contains a URL, lowercase headers, and a serialized JSON
body for an HTTP `POST`. Builders have already called
`HttpRequest::validate_transport`; an integration that mutates or constructs
the public value directly should call it again immediately before transport.
The validator requires one complete JSON object whose member names are unique
at every depth, so nested messages, tools, and provider options cannot acquire
transport-dependent first-value/last-value meanings.
Its `HttpRequestMetrics` result contains only URL/header/body sizes and counts.
The embedding transport remains responsible for:

- TLS policy, DNS resolution, proxy behavior, and redirect policy;
- connection, response, and overall deadlines plus cancellation;
- HTTP status validation and bounded header ingestion;
- keeping credentials out of logs, command-line arguments, and error text;
- bounding response bytes before retaining or retrying them.

Remote base URLs must use HTTPS. The host must be an ASCII DNS name or a
canonical IP literal; ambiguous numeric, percent-encoded, Unicode, empty-label,
and otherwise transport-dependent spellings are rejected. Plain HTTP is
accepted only for syntactic loopback hosts so local Ollama and
OpenAI-compatible servers remain usable. `HttpRequest` debug formatting emits
only the same content-free sizes/counts: it omits the URL, every header name
and value, and the body. Applications should still treat the value itself as
sensitive.

If the transport fails before a response can be decoded, record a bounded,
non-secret diagnostic with `AgentSession::model_failed`. A retry can then use
`can_retry_model` and `retry_model`; do not feed an HTTP error page to a model
response parser. `SessionError::Protocol` exposes its `ParseError` through the
standard `Error::source` chain when an integration needs typed diagnostics.

## Decode a response

For a non-streaming request, pass the complete response bytes to
`prepared.parse_response`. The decoder enforces the provider envelope bound,
uses the bound action protocol, and rejects a delivery-mode mismatch. The
high-level path also requires exactly one completed, protocol-consistent
provider envelope and rejects declared refusals, filtering, pauses, errors, and
unknown completion states before action parsing. The byte-oriented decoder
also rejects duplicate JSON object members at every depth. This prevents the
same provider bytes from selecting different completion states, tool calls, or
commands under first-value-wins and last-value-wins JSON implementations. The
same preflight rejects serde_json's private RawValue sentinel before a
feature-unified `Value` decoder can reinterpret its string as unchecked JSON.
`AgentResponse::protocol` exposes the retained wire protocol for diagnostics.

The lower-level chat and native-tool parsers retain compatibility with sparse
fixtures used by existing integrations. The native-tool parser still accepts
missing completion metadata and harmless sparse shapes that resolve to
`NoToolCall`, but rejects explicit failure states, ambiguous choices, and
completion metadata inconsistent with a present call. The legacy low-level
chat parser remains fully tolerant and only extracts text/metadata;
network-facing agent loops should use the prepared high-level path unless they
deliberately own those validation distinctions.

For a streaming request:

1. Create the accumulator with `prepared.response_stream()`.
2. Feed every transport chunk to `push` and render returned text or usage
   events as desired.
3. At transport EOF, call `finish` and handle its returned events too.
4. Continue only after a `StreamEvent::Done` and no `StreamEvent::Protocol`;
   `is_complete` exposes whether that low-level `Done` marker was observed.
5. Consume the accumulator with `into_response`.

Chunk boundaries need not align with UTF-8 characters, SSE frames, or NDJSON
lines. `AgentStream` reassembles them under byte and frame limits and applies
the same duplicate-member rejection to every decoded frame. A returned
`StreamEvent::ToolCall` is not execution permission: streamed calls remain
non-actionable until the enclosing response completes, is converted into an
`AgentResponse`, and is accepted by the session. `AgentStream::protocol`
reports the selected wire protocol. A generation-bound event—including an
Anthropic context-window limit—marks the response incomplete and prevents its
partial contents from becoming an action.

## Review and execution

Pass the decoded value to `AgentSession::accept_agent_response`. A `run`
action yields `ModelOutcome::Proposal`; show its exact `id`, `command`, and
optional `danger` warning to the user.

- `approve(id)` returns an `ApprovedCommand` and moves the session to
  `AwaitingObservation`.
- `edit_and_approve(id, command)` authorizes the validated edited command.
- `reject_with_feedback(id, feedback)` requests another model turn.
- `edit_for_manual_review(id, command)` returns text for a normal editor but
  does not authorize execution.

Only an `ApprovedCommand` should cross into the executor. Keep its
`proposal_id` attached to a `CommandExecutionOutcome`, using `Exited` only for
a real status and `Failed` for setup, timeout, or cancellation paths, then pass
it to `observe_execution`. The older `observe` and
`observe_execution_failure` entry points remain source-compatible shorthands.
Warnings from `is_dangerous` are review hints, never authorization or proof of
safety. They recognize common destructive shell, Git, service,
infrastructure, storage, wrapper, and review-smuggling forms, but intentionally
remain conservative heuristics rather than a shell policy engine.

## Persistence and shutdown

When `session.snapshot()` returns a value, serialize it with `to_json` and
store it with integration-owned permissions and atomic replacement. Restore
untrusted bytes only through `AgentSessionSnapshot::from_json`, followed by
`AgentSession::restore`; both encoded and semantic bounds are checked.

Use the session's `cancellation_token` to let long-running transport and
executor work observe cancellation. Calling `session.cancel()` changes the
state machine immediately, but terminating sockets, child processes, and PTYs
is still the integration's responsibility.

## Production checklist

- Keep provider, protocol, and delivery mode bound through the prepared path.
- Surface history omission and elision instead of silently implying complete
  context.
- Never place terminal output or environment metadata in system instructions.
- Never log request bodies, credential headers, raw secrets, or unrestricted
  provider error bodies.
- Require an explicit review action for every proposal ID.
- Apply transport and executor timeouts independently of model turn limits.
- Test malformed, token-limited, interrupted, oversized, and multi-tool-call
  responses as well as the happy path.
- Exercise snapshot restore with corrupted and stale data.

Existing integrations migrating from low-level APIs should also read
[`jterm4-migration.md`](jterm4-migration.md).
