# Engineering handoff

Updated: 2026-08-01

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

## Remaining boundaries

### Extend bounded decoding to the public transcript types

`Turn`, `AgentState`, and `Message` still derive `Deserialize`, so an embedding
application that decodes them directly — rather than through
`AgentSessionSnapshot::from_json` — gets ordinary Serde collection behavior with no
entry, field, or cumulative budget. Either expose the bounded seeds for these types
or make the constraint explicit in their documentation so integrations do not build
a second, unbounded wire path around the hardened one.

### Report what the request builders dropped

`build_chat_request*` now bounds history internally and discards the omitted-turn
count. A caller cannot distinguish "sent everything" from "silently sent the newest
40 turns". Surface the omission (a returned count, or a builder that fails closed on
an unbounded input) so an integration can tell the user their context was trimmed.

### Keep the response envelope limit honest for streaming

`parse_chat_response_bytes` bounds one non-streaming envelope, but streamed bodies
are fed to `stream::StreamParser` by the integration, so response header counts,
cumulative header bytes, and socket-level limits remain the integration's
responsibility. Document that boundary in `stream` or add a bounded ingest wrapper.

## Release checks

Run before the next release:

```text
cargo fmt --all -- --check
cargo test --locked --all-targets --all-features --no-fail-fast
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo doc --locked --all-features --no-deps
```
