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
| Agent state, proposal review, snapshots, and text-protocol parsing | `jagent::session` | Approval produces an `ApprovedCommand`; execution remains a caller action. |
| Native tool schemas and response parsing | `jagent::tools` | Text and native-tool protocols converge on the same `ParsedAction` values. |
| Provider request/response encoding and streaming parsing | `jagent::provider`, `jagent::stream` | Sans-IO: callers provide HTTP transport, request permits, and cancellation. |
| System prompts and untrusted user-role context | `jagent::prompt` | `BlockContext` and `EnvironmentMeta` are shared wire-facing shapes. |
| History bounding and secret preparation | `jagent::provider`, `jagent::redact` | Use `bound_history_with(history, jagent::redact_secrets)` so redaction happens before byte budgeting. Use a `*_with_report` request builder and surface its `omitted_history_turns`; redaction is opt-in and integrations decide which AI-bound funnels enable it. |
| Dangerous-command warnings | `jagent::safety` | `is_auto_approvable` is retained for compatibility but always returns `false`. |
| HTTP/process/PTY execution, configuration, UI, and durable storage | Consumer or `jterm_core` | `jagent` exposes plain request and validated snapshot values but performs no IO. |

## 0.6 wire-decoding migration

The string-owning public shapes `session::Turn`, `provider::Message`, and
`prompt::BlockContext` are serialize-only in 0.6. They are values an
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

1. Import the shared `session`, `safety`, `provider`, `prompt`, `stream`, and
   `tools` APIs needed by the consumer instead of copying their
   implementations.
2. Prepare AI-bound history with
   `bound_history_with(history, jagent::redact_secrets)` when shared secret
   scrubbing is enabled, retaining the returned omission count.
3. Build provider traffic with the applicable `*_with_report` function,
   combine its omission count with any earlier preparation count, and surface
   incomplete context to the user. Keep transport, cancellation, and
   credentials in the integration layer, and parse provider traffic with
   `jagent`.
4. Present every command proposal for review and execute only the
   `ApprovedCommand` returned by an explicit approval action.
5. Exercise both the integration's tests and jagent's Rust 1.86 locked
   check/test gate before updating a pinned dependency revision.
