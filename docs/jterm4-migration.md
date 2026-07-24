# Migrating jterm4 onto jagent

jagent was extracted from jterm4 (state as of jterm4 commit `81070b8`), so the
migration is mostly deleting jterm4's copies and re-importing. This file maps
every moved item and lists the deliberate behavior differences.

## Mapping

| jterm4 today | jagent | Notes |
|---|---|---|
| `src/agent.rs` — everything except `is_dangerous`/`is_auto_approvable` + helpers | `jagent::session` | API-identical port (types, methods, tests). One doc-comment now says "surface" instead of "pinned pane". |
| `src/agent.rs::is_dangerous`, `is_auto_approvable`, `strip_shell_prefixes`, `git_subcommand`, `has_rm_rf_dangerous_target`, `is_shell_assignment` | `jagent::safety` | Verbatim, including tests. |
| `src/ai/mod.rs::Provider` (enum, `as_config_value`, `display_name`, `default_model`, `default_base_url`, `endpoint`, `FromStr`) | `jagent::provider::Provider` | `endpoint` is now `pub`. `FromStr` error type is `ProviderError`, not `AiError`. |
| `src/ai/mod.rs::{Role, Turn}` | `jagent::provider::{Role, Message}` | `Turn` renamed to `Message` (avoids clashing with `session::Turn`); fields unchanged (`role`, `text`). |
| `AiClient::request_body` + header assembly in `send_turns_blocking_cancellable` | `jagent::provider::build_chat_request(&ChatConfig, system, history) -> HttpRequest` | Sans-IO: returns `{url, headers, body}`; jterm4 keeps its curl transport, permits, cancellation, and passes the returned headers/body via stdin config exactly as today. |
| `AiClient::parse_response` + `content_text` | `jagent::provider::parse_chat_response(provider, &Value)` | Same extraction and token-limit note; errors map `AiError::Empty → ProviderError::EmptyResponse`, `AiError::ResponseTooLarge → ProviderError::ResponseTooLarge`. |
| `AiClient::prepare_request_history` | `jagent::provider::bound_history` | Same budgets (40 turns / 256 KiB / 192 KiB per turn). Redaction is NOT in the crate: apply `crate::redact` to each `Message` before calling. |
| `validate_client_values` | `jagent::provider::ChatConfig::validate` | Same rules plus `max_tokens > 0`. |
| `src/ai/mod.rs::BlockContext` | `jagent::prompt::BlockContext` | Field-identical (`cmd`, `output`, `cwd`, `exit_code`, `truncated`) → serde-compatible with persisted payloads. |
| `build_agent_system_prompt` | `jagent::prompt::build_agent_system_prompt` | Text change: "selected Block context" → "selected block context", "Pane environment metadata" → "Environment metadata". |
| `agent_user_prompt(prompt, cwd, shell, os, git, block)` | `jagent::prompt::agent_user_prompt(prompt, &EnvironmentMeta, block)` | Loose args become `EnvironmentMeta { cwd, shell, os, git: Option<GitMeta> }`; `git_meta::RepoMeta` maps 1:1 onto `GitMeta { branch, dirty, ahead, behind }`. |
| `user_prompt_with_block_context` | `jagent::prompt::user_prompt_with_block_context` | See tag rename below. |
| `sample_output` / `elide_middle` | internal `jagent::text::elide_middle` | Not public; `session::sample_observation` is the public sampler. jterm4 keeps its own `sample_output` for non-agent call sites. |

Stays in jterm4 (deliberately not extracted): curl transport + request permits +
`AiCancellationToken`, secret redaction, `AiClient`/config plumbing,
`build_explain_prompt`, `build_nl_to_cmd_*`, `build_session_prompt`,
`build_system_prompt`, conversation snapshots, and all GTK review/panel UI.

## Behavior differences to account for

1. **Context tag rename** — the crate emits neutral tags:
   `<jterm4_selected_block_context>` → `<selected_block_context>`,
   `<jterm4_agent_environment>` → `<agent_environment>`. Purely prompt-visible;
   update the jterm4 tests that assert the old tag names.
2. **Default Anthropic model** — crate default is `claude-sonnet-5` (jterm4's
   constant was `claude-sonnet-4-6`). jterm4's config layer supplies its own
   model value, so this only matters where jterm4 called
   `Provider::default_model` directly.
3. **Transcript-omission note** — `bound_history` returns the omitted count;
   jterm4's wording ("…omitted by jterm4's request safety budget") stays in
   jterm4 when it appends the note to the system prompt.

## Suggested steps

1. Add `jagent = { path = "../jagent" }` (or a published version) to jterm4.
2. Replace `src/agent.rs` with `pub use jagent::session::*; pub use jagent::safety::*;`
   shims, run the test suite, then inline the new imports at call sites and
   delete the shim.
3. Switch `AiClient` internals to `build_chat_request`/`parse_chat_response`/
   `bound_history`, keeping redaction and the curl/permit layer wrapped around
   them.
4. Replace the prompt builders and `BlockContext` import; adapt the two tag
   assertions and the `agent_user_prompt` signature at call sites.
5. Delete the now-duplicated code; `cargo test` + clippy + the Agent acceptance
   checklist (`docs/AI_AGENT_CHAT_ACCEPTANCE.md`) close the loop.
