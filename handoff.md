# Engineering handoff

Updated: 2026-08-08

The current baseline hardens provider parsing, streamed responses, native tool calls,
approval binding, task resets, transcript retention, and snapshot validation. It now
also decodes agent snapshots under their own allocation budgets and enforces provider
configuration and request budgets at the public API boundary. The full all-feature
test suite and strict Clippy gate pass at this handoff.

## Completed since the previous handoff

- `AgentSessionSnapshot` no longer implements `Deserialize`. `from_json` decodes
  through `DeserializeSeed`/`Visitor` implementations that refuse the 129th
  transcript entry before constructing it, reject unknown and duplicate fields,
  charge per-field and cumulative text budgets while decoding, and reject trailing
  content. The schema and the `AgentSnapshotError` categories are unchanged, and
  `restore` keeps its semantic audit for hand-built snapshots.
- `ChatConfig::validate` bounds model and base-URL lengths and rejects URL userinfo,
  query, fragment, backslash, whitespace, control, and visually ambiguous characters.
  HTTPS is required except for a loopback Ollama endpoint.
- The public request builders bound history themselves (`bound_history` is
  idempotent, so a caller that already prepared its history sends the same bytes)
  and reject an over-budget system prompt rather than eliding safety instructions.
- `parse_chat_response_bytes` is a bounded, byte-oriented response entry point.
- The 0.6 public wire boundary is explicit: the string-owning `Turn`, `Message`,
  and `BlockContext` values are serialize-only, while allocation-free scalar
  schema atoms retain `Deserialize`. Read-only `AgentSessionSnapshot` accessors
  let integrations audit a bounded decoded snapshot without re-serializing and
  decoding its transcript through an ordinary `Vec<Turn>`.
- `parse_chat_response_full_bytes` and `parse_tool_response_bytes` extend the
  shared 1 MiB pre-allocation envelope gate to structured chat metadata and
  native tool replies. Their `Value` counterparts remain available only for
  trusted or already-bounded caller-owned values.
- `StreamParser` now enforces an 8 MiB raw-response ceiling and a 4,096-frame
  ceiling itself, alongside its existing line/frame, delivered-text,
  tool-argument, and call budgets. Tiny ignored frames and keep-alive floods
  therefore cannot turn the public sans-IO parser into an unbounded CPU or
  temporary-allocation path.

## Remaining boundaries

### Report what the request builders dropped

`build_chat_request*` now bounds history internally and discards the omitted-turn
count. A caller cannot distinguish "sent everything" from "silently sent the newest
40 turns". Surface the omission (a returned count, or a builder that fails closed on
an unbounded input) so an integration can tell the user their context was trimmed.

### Keep transport metadata bounded around streaming

`StreamParser` now bounds the raw body and decoded frame count itself, but it is
sans-IO: response header counts, cumulative header bytes, connection deadlines,
redirect policy, and socket cancellation remain the integration's responsibility.
Keep those transport limits aligned across jsh, jterm_core, and Forge.

## Release checks

Run before the next release:

```text
cargo fmt --all -- --check
cargo test --locked --all-targets --all-features --no-fail-fast
cargo test --locked --doc
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo doc --locked --all-features --no-deps
```
