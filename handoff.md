# Engineering handoff

Updated: 2026-08-01

The current baseline hardens provider parsing, streamed responses, native tool calls,
approval binding, task resets, transcript retention, and snapshot validation. The
full all-feature test suite and strict Clippy gate pass at this handoff.

## Remaining boundary

### Allocation-aware snapshot decoding

`AgentSessionSnapshot::from_json` rejects inputs larger than 256 KiB and
`AgentSession::restore` applies strict semantic validation, but the JSON layer still
uses ordinary Serde collection deserialization inside that byte envelope. Replace it
with bounded `DeserializeSeed`/`Visitor` implementations so transcript entry counts,
per-field lengths, cumulative text, duplicate fields, and nesting are rejected while
decoding rather than after allocation.

Acceptance criteria:

- Preserve the current snapshot schema and error categories.
- Reject unknown and duplicate fields before constructing a session.
- Stop wide transcript arrays before allocating entry 129.
- Charge per-field and cumulative byte budgets during decoding.
- Add adversarial wide-array, oversized-field, duplicate-field, and cumulative-text
  tests while retaining all existing round-trip tests.

## Release checks

Run before the next release:

```text
cargo fmt --all -- --check
cargo test --locked --all-targets --all-features --no-fail-fast
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo doc --locked --all-features --no-deps
```
