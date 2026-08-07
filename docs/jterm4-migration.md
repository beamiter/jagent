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
| History bounding and secret preparation | `jagent::provider`, `jagent::redact` | Use `bound_history_with(history, jagent::redact_secrets)` so redaction happens before byte budgeting. Redaction is opt-in; integrations decide which AI-bound funnels enable it. |
| Dangerous-command warnings | `jagent::safety` | `is_auto_approvable` is retained for compatibility but always returns `false`. |
| HTTP/process/PTY execution, configuration, UI, and durable storage | Consumer or `jterm_core` | `jagent` exposes plain request and validated snapshot values but performs no IO. |

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
5. `bound_history` returns an omitted-turn count. The integration owns any
   user-visible note that explains those omissions.

## Integration checklist

1. Import the shared `session`, `safety`, `provider`, `prompt`, `stream`, and
   `tools` APIs needed by the consumer instead of copying their
   implementations.
2. Prepare AI-bound history with
   `bound_history_with(history, jagent::redact_secrets)` when shared secret
   scrubbing is enabled.
3. Build and parse provider traffic with `jagent`, while keeping transport,
   cancellation, and credentials in the integration layer.
4. Present every command proposal for review and execute only the
   `ApprovedCommand` returned by an explicit approval action.
5. Exercise both the integration's tests and jagent's Rust 1.86 locked
   check/test gate before updating a pinned dependency revision.
