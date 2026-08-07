# jagent

Review-first terminal AI agent core shared across the current terminal stack.
[jsh](https://github.com/beamiter/jsh) and
[forge](https://github.com/beamiter/forge) depend on it directly, while
[jterm_core](https://github.com/beamiter/jterm_core) carries the common
terminal integration used by [anvil](https://github.com/beamiter/anvil),
[ember](https://github.com/beamiter/ember), forge, and
[frost](https://github.com/beamiter/frost).

The crate is deliberately **sans-IO**: no HTTP client, no PTY, no process
spawning, no UI. Integrations provide transport and execution; jagent provides
the parts that must behave identically everywhere.

## Modules

| Module     | Contents |
|------------|----------|
| `session`  | Pure agent state machine: strict, pre-parse-bounded `run`/`say`/`done` protocol (`parse_action` for JSON-in-text, `parse_tool_action` for native tool calls — both yielding the same `ParsedAction`), proposal approval/rejection/manual-review transitions, validated restart snapshots, bounded transcript, turn budget, cancellation. |
| `safety`   | `is_dangerous` destructive-pattern warnings and the compatibility `is_auto_approvable` hook, which now always fails closed because command text alone cannot prove what a configured shell will execute. |
| `provider` | Anthropic / OpenAI-compatible / Ollama chat request construction returning plain `HttpRequest { url, headers, body }` data (`build_agent_chat_request` adds an `AgentProtocol` selector), plus strict response text extraction (structured variant with token usage via `parse_chat_response_full`) and history bounding with an optional per-turn preparation hook (`bound_history_with`). |
| `prompt`   | Agent system prompts — `build_agent_system_prompt` (JSON protocol) and `build_agent_tool_system_prompt` (schema-carried protocol) — and user-role context encoding (`BlockContext`, `EnvironmentMeta`) with explicit untrusted-data framing. |
| `stream`   | Sans-IO streaming-response parser: push raw body bytes into a `StreamParser` (Anthropic SSE / OpenAI-compatible SSE / Ollama NDJSON) and receive `TextDelta` / `ToolCall` / `ReachedTokenLimit` / `Usage` / `Done` events. Tool calls have per-call and whole-response bounds and are published only after the enclosing response completes; truncation, payload after an end signal, malformed indexes/frames, and empty text responses fail closed. Pair with `build_chat_request_streaming`. |
| `tools`    | The same three actions carried by the providers' **native** tool-calling: provider-correct schemas (Anthropic `input_schema`, OpenAI `function`/`parameters`; Ollama returns `InvalidConfiguration`), plus `parse_tool_response` → `ToolResponse::to_action` ingestion into the identical `ParsedAction` values. Token-limited output never becomes an action. Fully additive: `AgentProtocol::Text` reproduces 0.4 byte-for-byte. |
| `redact`   | High-confidence scrubbing for AI-bound text: common provider and service tokens, private-key blocks, JWTs, explicit bearer credentials, and passwords embedded in URL userinfo. Non-secret URL and authentication framing is preserved for context. |

## Invariants

1. Generated commands are never executed by this crate. Approval returns an
   `ApprovedCommand` value; the caller must deliberately hand it to a shell.
2. Malformed model replies fail closed; parse failure never degrades into a
   command proposal.
3. Transcripts, observations, and context payloads are byte-bounded.
4. Terminal output and environment metadata travel as untrusted user-role
   data, never inside system instructions.
5. Every command proposal requires explicit approval. The former read-only
   auto-approval classifier is retained as a compatibility API but always
   returns `false`: aliases, functions, Git helpers, write-capable flags, and
   sensitive file reads cannot be ruled out from command text alone.
6. Restored snapshots revalidate transcript bounds, command shape, active
   state, and strictly increasing proposal IDs. The ID on an approval card is
   a unique binding to its pending command, never a best-effort lookup key,
   and is not reused when the same session starts a fresh task.
7. Native tool calls are withheld until their whole streaming response is
   known complete. A later protocol error cannot leave an actionable call
   behind, and token-limited output is never promoted to an action.

## Wire decoding

Version 0.6 makes the allocation boundary explicit. `session::Turn`,
`provider::Message`, and `prompt::BlockContext` own attacker-influenced text
but no longer implement `Deserialize`; they are in-memory construction and
serialization values, not standalone persistence formats. The allocation-free
schema atoms (`Role`, `ProposalId`, `ProposalStatus`, and `AgentState`) retain
Serde decoding, but decoding one does not validate a conversation or session.

Persisted Agent state must enter through `AgentSessionSnapshot::from_json` and
then `AgentSession::restore`. Embeddings that need to audit a decoded snapshot
without serializing it again can use its read-only field accessors. For
non-streaming provider traffic, use `parse_chat_response_bytes`,
`parse_chat_response_full_bytes`, or `tools::parse_tool_response_bytes`; each
checks `MAX_RESPONSE_JSON_BYTES` before allocating a `serde_json::Value`. The
corresponding `Value` APIs are only for trusted or already-bounded values.

## Sketch

```rust
use jagent::{provider, AgentSession, ModelOutcome};

let mut session = AgentSession::new(16);
session.submit_user("free up disk space in this repo")?;

let config = provider::ChatConfig::new(jagent::Provider::Anthropic);
let request = provider::build_chat_request(
    &config,
    Some(&jagent::build_agent_system_prompt()),
    &[provider::Message { role: jagent::Role::User, text: session.build_user_prompt() }],
)?;
// ... perform `request` with your own HTTP stack ...
// let reply = provider::parse_chat_response(config.provider, &response_json)?;
// match session.accept_model_reply(&reply)? {
//     ModelOutcome::Proposal { id, command, danger } => { /* show review UI */ }
//     ModelOutcome::Said(text) | ModelOutcome::Completed(text) => { /* render */ }
// }
```

The same loop over the providers' native tool-calling — the provider enforces
the action schema, so no JSON-in-text parsing is involved:

```rust
use jagent::{provider, tools, AgentProtocol, AgentSession, ModelOutcome};

let protocol = AgentProtocol::NativeTools;
let request = provider::build_agent_chat_request(
    &config,
    Some(&jagent::build_agent_tool_system_prompt()),
    &[provider::Message {
        role: jagent::Role::User,
        text: session.build_user_prompt_with(protocol),
    }],
    protocol,
)?;
// ... perform `request` with your own HTTP stack ...
// let reply = tools::parse_tool_response(config.provider, &response_json)?;
// match session.accept_model_tool_reply(&reply)? {
//     ModelOutcome::Proposal { id, command, danger } => { /* same review UI */ }
//     ModelOutcome::Said(text) | ModelOutcome::Completed(text) => { /* render */ }
// }
```

A tool call is still only a proposal: it becomes the same `ParsedAction`, walks
the same state machine, and approval still returns an `ApprovedCommand` that
only the caller can act on. A reply must carry **exactly one** tool call — zero
fails closed even when prose is present, and prose alongside a call is kept as
the action's visible thought rather than dropped.

## Development

The minimum supported Rust version (MSRV) is **1.86**. CI runs the check and
test suites against that compiler and keeps formatting, Clippy, and rustdoc on
the current stable toolchain. The MSRV gate uses the committed lockfile:

```text
cargo +1.86.0 check --locked --all-targets
cargo +1.86.0 test --locked --all-targets
cargo +1.86.0 test --locked --doc
```

See [the integration migration notes](docs/jterm4-migration.md) for the current
ownership boundary between this crate, `jterm_core`, and its consumers.

## License

MIT OR Apache-2.0
