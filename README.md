# jagent

[![CI](https://github.com/beamiter/jagent/actions/workflows/ci.yml/badge.svg)](https://github.com/beamiter/jagent/actions/workflows/ci.yml)
![MSRV](https://img.shields.io/badge/MSRV-1.86-dea584.svg)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

`jagent` is a review-first, sans-IO Rust core for terminal AI agents. It owns
the bounded session state machine, provider wire formats, protocol parsing,
request preparation, streaming accumulation, and command-review invariants.
Your integration owns HTTP, process execution, durable storage, and UI.

[jsh](https://github.com/beamiter/jsh) and
[forge](https://github.com/beamiter/forge) use it directly.
[jterm_core](https://github.com/beamiter/jterm_core) carries the shared
integration used by anvil, ember, forge, and frost.

| Capability | Contract |
|---|---|
| Providers | Anthropic Messages, OpenAI-compatible Chat Completions, and Ollama `/api/chat` |
| Action protocols | Strict JSON-in-text and provider-native tools |
| Delivery | Bounded complete-response decoding and incremental SSE/NDJSON streaming |
| Execution | Never performed by this crate; every command becomes a proposal requiring explicit approval |
| Runtime coupling | None: no HTTP client, async runtime, process, PTY, storage, or UI dependency |
| Rust baseline | Rust 1.86 (edition 2021) |

Documentation:

- [Integration guide](docs/integration-guide.md) — transport, streaming,
  review, failure, and persistence ownership.
- [Quickstart example](examples/quickstart.rs) — a complete non-streaming Text
  turn with a local response fixture.
- [Streaming example](examples/streaming.rs) — chunked native-tool streaming
  through the same review state machine.
- [Migration notes](docs/jterm4-migration.md) — compatibility guidance for
  `jterm_core` and existing terminal consumers.
- [Changelog](CHANGELOG.md) — release history and compatibility notes.

## Quick start

The recommended 0.7 path keeps the request protocol, matching system prompt,
redaction policy, response decoder, and session ingestion together:

1. Submit user input to an `AgentSession`.
2. Encode terminal/environment context as untrusted user-role data.
3. Call `prepare_agent_request`; high-confidence secret redaction is on by
   default, with counts for retained-turn redaction, elision, and omission.
4. Perform the returned HTTP request with your own transport.
5. Decode bytes with `prepared.parse_response`; the prepared request keeps the
   provider, protocol, and non-streaming delivery mode paired for you.
6. Display every proposed command. Only `approve` returns an
   `ApprovedCommand`, and the integration must still execute it deliberately.

Run the repository example:

```text
cargo run --example quickstart
```

It performs no network or process I/O. It builds a request for a loopback
OpenAI-compatible endpoint, checks the preparation report, then feeds a local
response fixture through the review state machine.

```rust
use jagent::{
    agent_user_prompt, prepare_agent_request, AgentProtocol, AgentRequestSpec,
    AgentSession, ChatConfig, EnvironmentMeta, Message, ModelOutcome, Provider,
    Role,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protocol = AgentProtocol::Text;
    let mut session = AgentSession::new(8);

    // Deliberately resembles a leaked token so the secure default is visible.
    const SECRET: &str = "ghp_1234567890abcdefghijABCDEFGHIJ123456";
    session.submit_user(format!(
        "show the current directory; an accidental token was pasted: {SECRET}"
    ))?;

    let environment = EnvironmentMeta {
        cwd: "/workspace/jagent".into(),
        shell: "bash".into(),
        os: "linux".into(),
        git: None,
    };
    let history = [Message {
        role: Role::User,
        text: agent_user_prompt(
            &session.build_user_prompt_with(protocol),
            &environment,
            None,
        ),
    }];

    let config = ChatConfig {
        provider: Provider::OpenAiCompatible,
        api_key: None,
        model: "local-agent".into(),
        base_url: "http://127.0.0.1:1234".into(),
        max_tokens: 512,
        temperature: Some(0.0),
    };
    let prepared = prepare_agent_request(
        &config,
        AgentRequestSpec::new(&history, protocol),
    )?;

    assert_eq!(
        prepared.request.url,
        "http://127.0.0.1:1234/chat/completions"
    );
    assert!(prepared.report.redaction_enabled);
    assert_eq!(prepared.report.history.changed_history_turns, 1);
    assert_eq!(prepared.report.history.omitted_history_turns, 0);
    assert!(!prepared.request.body.contains(SECRET));

    // A real integration sends `prepared.request` and supplies the bounded
    // response bytes. This local fixture keeps the example completely sans-IO.
    let response_bytes = br#"{
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "{\"action\":\"run\",\"command\":\"pwd\"}"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8}
    }"#;
    let response = prepared.parse_response(response_bytes)?;
    let outcome = session.accept_agent_response(&response)?;

    let ModelOutcome::Proposal {
        id,
        command,
        danger,
    } = outcome
    else {
        panic!("fixture should produce a command proposal");
    };
    assert_eq!(command, "pwd");
    assert!(danger.is_none());

    // Approval only yields a value. jagent never starts the command.
    let approved = session.approve(id)?;
    assert_eq!(approved.command, "pwd");

    // After the integration deliberately executes the approved value, it
    // records the real exit status and output. Here both are simulated.
    session.observe(approved.proposal_id, 0, "/workspace/jagent\n")?;
    Ok(())
}
```

`EnvironmentMeta` and optional `BlockContext` values are JSON-encoded inside
the user role. They are never interpolated into system instructions. Default
request preparation redacts the outbound `Message`; it does not mutate the
session transcript or claim to be a complete DLP system.

## Streaming

Select streaming declaratively on the same request specification. The prepared
request creates the matching `AgentStream`, so callers do not re-enter its
provider or protocol:

```rust
# use jagent::{prepare_agent_request, AgentProtocol, AgentRequestSpec, ChatConfig, Message, Provider};
# fn example(history: &[Message]) -> Result<(), Box<dyn std::error::Error>> {
# let config = ChatConfig::new(Provider::OpenAiCompatible);
let protocol = AgentProtocol::Text;
let prepared = prepare_agent_request(
    &config,
    AgentRequestSpec::new(history, protocol).streaming(true),
)?;
let mut response_stream = prepared.response_stream()?;

// Send `prepared.request` with your transport. For each received chunk:
// for event in response_stream.push(chunk) { render(event); }
// At transport EOF:
// for event in response_stream.finish() { render(event); }
// let response = response_stream.into_response()?;
// let outcome = session.accept_agent_response(&response)?;
# let _ = (&prepared.request, &mut response_stream);
# Ok(())
# }
```

`push` and `finish` return the underlying `StreamEvent`s unchanged while the
wrapper accumulates a high-level response. `into_response` succeeds only after
`Done`. Protocol failure, premature EOF, or a native tool call in Text mode
fails closed. A streamed `ToolCall` event is not execution authorization;
integrations should act only on the proposal produced after successful
response conversion and session ingestion. `parse_response` rejects a request
prepared for streaming, and `response_stream` rejects one prepared for a
complete response body. `AgentStream::protocol` reports the bound protocol and
`is_complete` reports whether the low-level `Done` marker was observed;
`into_response` remains the authoritative final validation step.

Run the no-I/O native-tools companion to exercise arbitrary chunk boundaries,
stream completion, response conversion, and proposal approval:

```text
cargo run --example streaming
```

## Native tools

Set `AgentProtocol::NativeTools` on `AgentRequestSpec`.
`prepare_agent_request` selects the matching tool-aware system prompt and
provider schema, and its prepared response path retains that protocol.
`AgentResponse` then contains a `ToolResponse`, but the session still converts
`run` into the same reviewed proposal as Text mode.

All built-in providers support the two agent protocols and their streaming
forms:

| Provider | Text | NativeTools | Streaming |
|---|---:|---:|---:|
| Anthropic Messages | Yes | Yes | Yes |
| OpenAI-compatible Chat Completions | Yes | Yes | Yes |
| Ollama `/api/chat` | Yes | Yes | Yes |

Ollama uses the function schema documented by its
[`/api/chat` API](https://docs.ollama.com/api/chat) and
[tool-calling guide](https://docs.ollama.com/capabilities/tool-calling), while
omitting the undocumented `tool_choice` field. Exactly one tool call is
required; zero, multiple, malformed, token-limited, or protocol-mismatched
calls fail closed.

## Capability discovery

Split integrations can exchange one bounded, non-sensitive capability token
before choosing a wire protocol. `agent_capabilities(provider)` emits the
version-1 compatibility form that existing 0.7 peers understand. After a peer
token has been decoded, `agent_capabilities_for_peer` emits the same provider
matrix in that peer's schema version. `negotiate` selects the first mutually
supported protocol from the caller's preference order and never guesses a
fallback:

```rust
use jagent::{
    agent_capabilities, agent_capabilities_for_peer, AgentCapabilities,
    AgentDelivery, AgentProtocol, Provider,
};

let first_contact = agent_capabilities(Provider::Ollama);
assert_eq!(first_contact.version(), 1); // safe for an unprobed 0.7 peer
let peer = AgentCapabilities::from_wire(
    "jagent-agent/2;modes=text+complete,native-tools+streaming",
)?;
let local = agent_capabilities_for_peer(Provider::Ollama, peer);
assert_eq!(local.version(), peer.version());
let protocol = local
    .negotiate_with(
        peer,
        &[AgentProtocol::NativeTools, AgentProtocol::Text],
        AgentDelivery::Complete,
    )
    .ok_or("no mutually supported Agent protocol")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Version 2 tokens list exact protocol/delivery pairs, so a peer can advertise
`jagent-agent/2;modes=text+complete,native-tools+streaming` without falsely
claiming either crossed combination. Strict version-1 tokens such as
`jagent-agent/1;protocols=text;delivery=complete` remain accepted and retain
their historical Cartesian-product meaning. Names, duplicates, non-canonical
ordering, empty sets, unknown fields/versions, whitespace, and values over 256
bytes are rejected. Tokens contain no endpoint, credential, model, transcript,
or terminal context. Capability agreement selects only an encoding; it never
authorizes a tool call or command. Do not send the opt-in
`agent_capabilities_v2` value to an unprobed peer: an older peer correctly
rejects it as an unsupported version. If a future provider matrix is not a
Cartesian product, compatibility-first v1 emission chooses a representable
subset and may omit a usable mode; it never invents a crossed combination.

## Safety invariants

1. Generated commands are never executed by this crate. Approval returns an
   `ApprovedCommand`; the caller must deliberately hand it to an executor.
2. Ambiguous JSON fails closed. Outbound request validation and inbound
   complete responses, streaming frames, text actions, and native-tool
   arguments reject duplicate object members at every depth instead of
   inheriting a parser's first/last-value rule. The private serde_json RawValue
   escape key is rejected before a feature-unified decoder can reparse its
   string; parse failure never becomes a command proposal.
3. Transcripts, observations, request history, encoded prompt contexts,
   response envelopes, streamed frames, model text, and tool arguments are
   byte-bounded.
4. Terminal output and environment metadata travel as explicitly untrusted
   user-role data, never as system instructions. Context budgets are enforced
   after JSON encoding, and untrusted values cannot spell their raw enclosing
   closing tag.
5. Every proposal requires explicit approval. `is_auto_approvable` remains as
   a compatibility hook and always returns `false`.
6. Snapshot restore revalidates transcript bounds, command shape, active
   state, strictly increasing proposal IDs, and pending approval bindings.
7. Native tool calls are withheld by the low-level stream parser until their
   enclosing response completes. Token-limited output is never promoted to an
   action.

`is_dangerous` provides review warnings only. It neither authorizes nor blocks
execution, and command-text heuristics cannot prove what a configured shell,
alias, function, or helper will do.

## High-level modules

| Module | Responsibility |
|---|---|
| `agent` | `AgentRequestSpec` and `prepare_agent_request`: protocol-matched prompt/schema, secure history preparation, streaming selection, and a history preparation report. |
| `capabilities` | Versioned protocol/delivery discovery, bounded wire tokens, and deterministic preference negotiation. |
| `response` | `AgentResponse` and `AgentStream`: protocol-aware bounded decoding, response metadata, action conversion, and streaming accumulation. |
| `session` | Pure proposal/review/observation state machine, bounded transcript, turn budget, cancellation, and validated snapshots. |
| `prompt` | Fixed system prompts plus untrusted user-role environment and selected-block framing. |
| `provider` | Provider configuration and the lower-level Anthropic, OpenAI-compatible, and Ollama request/response codecs. |
| `tools` | Native tool schemas and low-level `ToolResponse` parsing. |
| `stream` | Low-level SSE/NDJSON `StreamParser` and `StreamEvent`s. |
| `redact` | Conservative high-confidence secret scrubbing; the borrowing API is `redact_secrets_cow`. |
| `safety` | Non-authorizing destructive-command warnings. |

`validate_no_duplicate_members` exposes the same allocation-light recursive
JSON preflight used by these wire decoders. Integrations can apply it to other
already byte-bounded JSON trust boundaries before their own typed decode,
without constructing a second `Value` tree or learning a hostile field name
from the error. It also rejects serde_json's private RawValue sentinel, which a
feature-unified `Value` decoder would otherwise reinterpret as a second,
unchecked JSON document.

## Low-level compatibility APIs

The 0.7 high-level path is additive. Existing integrations can migrate in
stages:

- `provider::build_chat_request*` and `provider::build_agent_chat_request*`
  remain the low-level request builders. Compatibility variants return only
  `HttpRequest`; `*_with_report` variants return `BuiltRequest` with the
  omissions introduced by that build.
- `provider::bound_history` and `bound_history_with` keep their established
  tuple return types. New `*_with_report` preparation functions distinguish
  changed, elided, and fully omitted turns.
- `provider::parse_chat_response*_bytes` and
  `tools::parse_tool_response_bytes` remain bounded byte entry points.
  Their `serde_json::Value` counterparts are only for trusted or already
  transport-bounded values. These lower-level parsers preserve compatibility
  with sparse legacy fixtures; high-level `AgentResponse` decoding requires a
  complete, unambiguous provider envelope before action parsing.
- `session::accept_model_reply` and `accept_model_tool_reply` remain available
  when an integration deliberately manages protocol pairing itself.
- `StreamParser` remains the raw event parser for integrations that need to
  own accumulation. `AgentStream` is the recommended protocol-aware wrapper.

`AgentResponse::parse_bytes` and `AgentStream::new` also remain available when
an integration deliberately pairs the provider, protocol, and delivery mode
itself. New code should normally decode through its `PreparedAgentRequest` so
those choices stay bound to the request that was sent.

Text-mode request builders retain their compatibility wire behavior. The
high-level path adds secure defaults and fuller reporting; disabling redaction
or replacing the built-in system prompt is explicit on `AgentRequestSpec`.

## Persistence and wire bounds

`session::Turn`, `provider::Message`, and `prompt::BlockContext` are
serialize-only in-memory values, not standalone unbounded decoding formats.
Persisted state must enter through `AgentSessionSnapshot::from_json` and then
`AgentSession::restore`. Allocation-free schema atoms such as `Role`,
`ProposalId`, `ProposalStatus`, and `AgentState` retain Serde decoding, but a
decoded atom does not validate a session.

`PreparedAgentRequest::parse_response` delegates to bounded
`AgentResponse::parse_bytes`, which enforces the shared non-streaming envelope
limit before one JSON decode. A prepared response stream inherits
`StreamParser`'s raw-response, frame-count, per-frame, text, tool-argument, and
call limits. HTTP headers, redirect policy, deadlines, socket cancellation,
TLS policy, and process execution remain integration responsibilities.

See [the integration migration notes](docs/jterm4-migration.md) for the
ownership boundary with `jterm_core` and existing consumers. Release changes
are recorded in [CHANGELOG.md](CHANGELOG.md).

## Development

The repository follows the terminal-family Rust toolchain convention:
`rust-toolchain.toml` selects stable with the minimal profile plus rustfmt and
Clippy. The crate's declared MSRV is Rust 1.86, which CI checks separately.
The full local release gate is:

```text
cargo fmt --all -- --check
cargo run --locked --example quickstart
cargo run --locked --example streaming
cargo check --locked --all-targets --all-features
cargo test --locked --all-targets --all-features --no-fail-fast
cargo test --locked --all-features --doc
cargo clippy --locked --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --all-features --no-deps
cargo package --locked --allow-dirty
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for compatibility and test expectations.
Potential vulnerabilities should be reported through [SECURITY.md](SECURITY.md).

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE), or
- [MIT License](LICENSE-MIT)

at your option.
