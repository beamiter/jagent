# Contributing to jagent

Thanks for helping improve `jagent`. The crate is a small, security-sensitive
core shared by terminal integrations, so changes should keep behavior explicit,
bounded, and independent of any particular HTTP or process runtime.

## Toolchain

The repository's `rust-toolchain.toml` selects the current stable compiler
with rustfmt and Clippy. Rust 1.86 is the minimum supported Rust version
(MSRV); avoid APIs stabilized after it unless the MSRV is intentionally raised
in `Cargo.toml`, CI, the README, and the changelog together.

## Local checks

Run the same gates as CI before opening a pull request:

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

CI also checks and tests the crate with Rust 1.86. If that toolchain is
installed locally, the equivalent commands are:

```text
cargo +1.86.0 check --locked --all-targets --all-features
cargo +1.86.0 test --locked --all-targets --all-features --no-fail-fast
```

## Change checklist

- Keep `jagent` sans-IO. Networking, PTYs, process execution, durable storage,
  and approval UI belong to consumers.
- Preserve the review boundary: model output may create a proposal, never an
  executable authorization.
- Put a byte or item budget in front of every attacker-controlled allocation.
- Treat terminal output and environment metadata as untrusted user-role data.
- Add regression tests for malformed, truncated, oversized, and state-mismatch
  cases as well as the happy path.
- Keep high-level and low-level compatibility APIs clearly distinguished.
- Update public rustdoc, examples, README, migration notes, and `CHANGELOG.md`
  when behavior or API changes.
- Do not include credentials, provider bodies, or real user transcripts in
  fixtures, snapshots, logs, issues, or commits.

Use an entry under `CHANGELOG.md`'s `Unreleased` section for user-visible
changes. Version bumps and release dates are separate release-maintainer work.

Security-sensitive findings should follow [SECURITY.md](SECURITY.md) instead
of being disclosed in a public issue.
