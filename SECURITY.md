# Security policy

`jagent` sits on the boundary between untrusted model output and terminal
execution. Please report vulnerabilities privately, especially issues that
could bypass proposal review, confuse proposal IDs, expose credentials, evade
wire bounds, or turn malformed responses into actions.

## Reporting

Use GitHub's private vulnerability reporting form for this repository:

<https://github.com/beamiter/jagent/security/advisories/new>

Include the affected revision or version, the provider and protocol involved,
the expected invariant, and a minimal reproducer using synthetic data. Do not
include real API keys, user transcripts, or other sensitive material. If the
private form is unavailable, open a public issue that asks for a private
contact channel without publishing exploit details.

Please allow maintainers time to reproduce, fix, and coordinate disclosure
before sharing the report publicly.

## Supported code

Security fixes are developed against the default branch. Release support is
documented when a release is published; consumers using an unreleased Git
revision should pin an audited commit and update deliberately.

## Security boundary

The crate validates and bounds session state, provider wire formats, streamed
frames, model actions, and snapshots. It never performs network or process I/O.
The embedding application remains responsible for credentials, TLS, redirects,
HTTP limits, cancellation, filesystem and process isolation, the approval UI,
and executing only the exact `ApprovedCommand` produced for the proposal the
user reviewed.

Command danger heuristics are warnings only. A command that receives no
warning is not thereby safe or suitable for automatic execution.
