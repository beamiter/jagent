# jagent

Review-first terminal AI agent core, extracted from
[jterm4](https://github.com/beamiter/jterm4)'s native Shell Agent so the same
agent behavior can be embedded in [rsh](https://github.com/beamiter/rsh) and
other terminals (jterm1/2/3).

The crate is deliberately **sans-IO**: no HTTP client, no PTY, no process
spawning, no UI. Integrations provide transport and execution; jagent provides
the parts that must behave identically everywhere.

## Modules

| Module     | Contents |
|------------|----------|
| `session`  | Pure agent state machine: strict JSON `run`/`say`/`done` protocol, proposal approval/rejection/manual-review transitions, bounded transcript, turn budget, cancellation. |
| `safety`   | `is_dangerous` destructive-pattern warnings and the fail-closed `is_auto_approvable` read-only allowlist. |
| `provider` | Anthropic / OpenAI-compatible / Ollama chat request construction returning plain `HttpRequest { url, headers, body }` data, plus strict response text extraction and history bounding. |
| `prompt`   | Agent system prompt and user-role context encoding (`BlockContext`, `EnvironmentMeta`) with explicit untrusted-data framing. |

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

## License

MIT OR Apache-2.0
