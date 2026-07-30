# jagent

Review-first terminal AI agent core, extracted from
[jterm4](https://github.com/beamiter/jterm4)'s native Shell Agent so the same
agent behavior can be embedded in [jsh](https://github.com/beamiter/jsh) and
other terminals (jterm1/2/3).

The crate is deliberately **sans-IO**: no HTTP client, no PTY, no process
spawning, no UI. Integrations provide transport and execution; jagent provides
the parts that must behave identically everywhere.

## Modules

| Module     | Contents |
|------------|----------|
| `session`  | Pure agent state machine: strict `run`/`say`/`done` protocol (`parse_action` for JSON-in-text, `parse_tool_action` for native tool calls — both yielding the same `ParsedAction`), proposal approval/rejection/manual-review transitions, bounded transcript, turn budget, cancellation. |
| `safety`   | `is_dangerous` destructive-pattern warnings and the fail-closed `is_auto_approvable` read-only allowlist. |
| `provider` | Anthropic / OpenAI-compatible / Ollama chat request construction returning plain `HttpRequest { url, headers, body }` data (`build_agent_chat_request` adds an `AgentProtocol` selector), plus strict response text extraction (structured variant with token usage via `parse_chat_response_full`) and history bounding with an optional per-turn preparation hook (`bound_history_with`). |
| `prompt`   | Agent system prompts — `build_agent_system_prompt` (JSON protocol) and `build_agent_tool_system_prompt` (schema-carried protocol) — and user-role context encoding (`BlockContext`, `EnvironmentMeta`) with explicit untrusted-data framing. |
| `stream`   | Sans-IO streaming-response parser: push raw body bytes into a `StreamParser` (Anthropic SSE / OpenAI-compatible SSE / Ollama NDJSON) and receive `TextDelta` / `ToolCall` / `ReachedTokenLimit` / `Usage` / `Done` events; tool-call argument deltas accumulate into one complete call, and malformed frames fail closed. Pair with `build_chat_request_streaming`. |
| `tools`    | The same three actions carried by the providers' **native** tool-calling: provider-correct schemas (Anthropic `input_schema`, OpenAI `function`/`parameters`; Ollama returns `InvalidConfiguration`), plus `parse_tool_response` → `ToolResponse::to_action` ingestion into the identical `ParsedAction` values. Fully additive: `AgentProtocol::Text` reproduces 0.4 byte-for-byte. |

## Invariants

1. Generated commands are never executed by this crate. Approval returns an
   `ApprovedCommand` value; the caller must deliberately hand it to a shell.
2. Malformed model replies fail closed; parse failure never degrades into a
   command proposal.
3. Transcripts, observations, and context payloads are byte-bounded.
4. Terminal output and environment metadata travel as untrusted user-role
   data, never inside system instructions.

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

## License

MIT OR Apache-2.0
