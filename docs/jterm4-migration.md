# Migrating terminal integrations onto jagent

`jagent` was originally extracted from the codebase now named `forge`
(historical `jterm4` commit `81070b8`). The current dependency shape is:

- `jsh` embeds `jagent` directly.
- `jterm_core` supplies the shared integration used by `anvil`, `ember`,
  `forge`, and `frost`.
- `forge` also uses `jagent` directly for APIs outside the `jterm_core`
  wrapper.

The old project name below is retained only where it identifies historical
prompt tags or extraction state.

## Ownership boundary

| Concern | Owner | Integration notes |
|---|---|---|
| Protocol-matched request preparation and response decoding | `jagent::agent`, `jagent::response` | `prepare_agent_request` binds prompt, schema, history policy, delivery mode, and decoder to one `AgentProtocol`. This is the recommended 0.7 path. |
| Agent state, proposal review, snapshots, and text-protocol parsing | `jagent::session` | Approval produces an `ApprovedCommand`; execution remains a caller action. |
| Native tool schemas and response parsing | `jagent::tools` | Text and native-tool protocols converge on the same `ParsedAction` values. |
| Low-level provider encoding and stream events | `jagent::provider`, `jagent::stream` | Compatibility surface for integrations that deliberately coordinate prompt, protocol, delivery, and parsing themselves. |
| System prompts and untrusted user-role context | `jagent::prompt` | `BlockContext` and `EnvironmentMeta` are shared wire-facing shapes. |
| History bounding and secret preparation | `jagent::agent`, or `jagent::provider` plus `jagent::redact` | The high-level path redacts high-confidence secrets by default and reports changes, elision, and omission. Low-level builders bound history but do not implicitly redact it. |
| Dangerous-command warnings | `jagent::safety` | `is_auto_approvable` is retained for compatibility but always returns `false`. |
| HTTP/process/PTY execution, configuration, UI, and durable storage | Consumer or `jterm_core` | `jagent` exposes plain request and validated snapshot values but performs no IO. |

## Recommended 0.7 migration

Move one complete request/response path at a time so protocol choices cannot
drift between layers:

1. Keep a single `AgentProtocol` for the turn and use it with
   `AgentSession::build_user_prompt_with`.
2. Construct untrusted user-role context with `agent_user_prompt`, then call
   `prepare_agent_request(config, AgentRequestSpec::new(history, protocol))`.
3. Surface the returned history report. Redaction is enabled by default;
   disabling it with `AgentRequestSpec::redact_secrets(false)` is an explicit
   integration policy decision.
4. Perform the returned request with the consumer's bounded HTTP transport.
5. Decode a complete response with `prepared.parse_response`, or create the
   streaming accumulator with `prepared.response_stream`, feed every chunk,
   call `finish` at EOF, and then call `into_response`.
6. Pass the completed `AgentResponse` to
   `AgentSession::accept_agent_response`. Never interpret a raw streamed
   `ToolCall` as an approval or execution token.
7. Preserve the existing review UI boundary: display the exact proposal and
   execute only the `ApprovedCommand` returned for its ID.

This high-level route is additive. Consumers may retain low-level provider or
stream APIs where they need byte-stable compatibility, but they then own the
prompt/protocol/delivery pairing and secret-preparation policy explicitly.
See the generic [integration guide](integration-guide.md) for transport,
failure, persistence, and production guidance.

## Low-level compatibility and 0.6 wire decoding

The string-owning public shapes `session::Turn`, `provider::Message`, and
`prompt::BlockContext` became serialize-only in 0.6. They are values an
integration constructs after applying its own input policy, not generic JSON
storage schemas. `Role`, `ProposalId`, `ProposalStatus`, and `AgentState` keep
`Deserialize` because they are allocation-free schema atoms; their surrounding
conversation or session still needs contextual validation.

Code that previously re-serialized an `AgentSessionSnapshot` into a local
`#[derive(Deserialize)]` audit struct should instead inspect the bounded value
directly through `version()`, `transcript()`, `transcript_truncated()`,
`state()`, `turns_used()`, `max_turns()`, and `next_proposal_id()`. Persistent
session bytes still enter only through `AgentSessionSnapshot::from_json`, and a
usable session still enters only through `AgentSession::restore`.

For non-streaming HTTP replies, prefer the byte APIs so the shared 1 MiB
envelope ceiling runs before JSON allocation:

- `provider::parse_chat_response_bytes` for display text;
- `provider::parse_chat_response_full_bytes` for text plus token-limit/usage
  metadata;
- `tools::parse_tool_response_bytes` for native tool replies.

The older `Value` APIs remain available for integrations that already decoded
the response under an equivalent transport limit. They are not raw-network
entry points.

## Historical compatibility notes

1. The former provider `Turn { role, text }` became
   `jagent::provider::Message { role, text }`; `session::Turn` is a separate
   state-machine type.
2. The prompt tags were made application-neutral:
   `<jterm4_selected_block_context>` became `<selected_block_context>`, and
   `<jterm4_agent_environment>` became `<agent_environment>`.
3. `BlockContext` keeps the fields `cmd`, `output`, `cwd`, `exit_code`, and
   `truncated`, preserving the migrated payload shape.
4. `Provider::default_model` is only a crate default. Integrations that need a
   stable model choice should continue to supply it through `ChatConfig`.
5. `bound_history` and the `*_with_report` request builders expose an
   omitted-turn count. A builder reports only omissions introduced by that
   invocation; the integration must carry forward any count from earlier
   preparation and owns the user-visible explanation.

## Integration checklist

1. Prefer `prepare_agent_request` and `accept_agent_response` for new or
   migrated agent loops; import low-level modules only where the consumer needs
   their additional control.
2. Treat the high-level preparation report as part of the UI contract and
   surface changed, elided, or omitted context without logging its contents.
3. For a deliberately low-level path, prepare AI-bound history with
   `jagent::provider::bound_history_cow_with_report(history,
   jagent::redact_secrets_cow)`, then carry its report alongside any omissions
   from a later request builder.
4. Keep transport, response status handling, cancellation, credentials,
   process execution, and storage in the integration layer.
5. Present every command proposal for review and execute only the
   `ApprovedCommand` returned by an explicit approval action.
6. Exercise both the integration's tests and jagent's locked stable-toolchain
   check/test gate before updating a pinned dependency revision.
