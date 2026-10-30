# Research proposal boundary

`jagent::research` adds a sans-IO research interface for `jsh research`.
It does not spawn processes, provide HTTP transport, load Flint, implement code
changes, grant approvals or submit DLC jobs. Existing terminal-agent APIs and
review-first command semantics are unchanged.

## Contract

`ResearchContext::from_json` accepts at most 256 KiB of encoded JSON using
`jsh.research.bridge/v1`. It rejects duplicate members, unknown fields, malformed
identities, inconsistent trial accounting and an impossible ready/budget state.
The memory window contains at most 16 trials, with 2048-byte hypotheses,
plus a separate best-candidate identity and an explicit omitted-trial count.
The jsh host retains an older incumbent's scored record alongside the latest 15
trials when needed. Scores must match the recorded outcome: kept/baseline/archived
trials have both stages, screened-out trials have only screening, and queued
trials have no scores. Confirmation always requires screening evidence.
`needs_proposal` is rejected when any visible trial is queued or running. A
nonready context need not include the pending trial within its bounded window;
absence from that window does not prove that the campaign has no pending work.
When no trials are omitted, a declared best candidate must occur in the complete
history. A visible `baseline` trial must match the frozen `baseline_commit`.
Omitted trials remain opaque; their identities are not fabricated or inferred.

`prepare_request` returns provider-neutral messages and a JSON response schema.
The system instructions are constant. All observations and hypotheses remain
untrusted user-role data. The result is NOT an HTTP request; the integration
chooses a model/provider and implements transport. Prompt separation reduces
instruction confusion but is not a proof of immunity to prompt injection.

`ResearchProposal::from_json` accepts at most 32 KiB, requires schema version 2,
and binds the candidate to campaign ID, manifest digest, state revision and the
expected current-best commit. Commit IDs must be complete lowercase Git hashes;
actual existence and path checks belong to jsh. The expected incumbent is not an
assertion about Git ancestry: candidates may build on another recorded branch.
Parents must name completed trials visible in the context (or the retained best).
Unknown command, score, budget or executor fields are rejected. Hypotheses are
byte-bounded even though JSON Schema maxLength counts Unicode characters.

A model cannot invent a usable Git commit by describing one. The host's approved
code-authoring step must first implement a candidate in a development worktree.
The completed candidate is then submitted as a bound proposal.

## Host sequence

1. Obtain a fresh `jsh research --root CAMPAIGN bridge` snapshot.
2. Only for `needs_proposal`, build messages/schema with `prepare_request`.
3. Have an approved agent implement and commit a bounded candidate.
4. Parse its proposal against the original snapshot.
5. Submit the exact proposal to `jsh research --root CAMPAIGN submit FILE`.
6. Execute only through the controller's explicit `run --execute` authorization.
7. Read a new snapshot; never reuse a stale one for a different candidate.

Snapshot validation is not execution authorization. Between steps 4 and 5 the
campaign may stop, advance or exhaust its budget. jsh must therefore validate
again under its live writer lock. Exact request replays are idempotent; the same
ID with different content is an error. A stale response must not be silently
rebound to new state. Preserve its candidate and ask the host to reassess it.

This phase establishes the model/control-plane boundary, not a functioning
Flint/DLC service. Remote job leases, authenticated callbacks, GPU-hour budgets,
container isolation and independent result validation remain executor work.

## Remote execution boundary (2026-10-30)

`research::remote::{RemoteBinding, RemoteObservation}` validates the shared
`jsh.research.remote/v1` response shape, fresh exchange nonce, request digest,
provider job identity, status, lease and result binding. Expectations must come
from the host's durable ledger, never from model output. Parse with
`from_authenticated_json` only after the host verifies gateway authentication.
The validator cannot verify HMAC, compare metrics_sha256 to payload bytes, apply
campaign guardrails, reserve GPUs, issue requests or authorize execution.

The matching jsh remote staging client persists submit/cancel intent and full
GPU-time reservations before IO, recovers by lookup rather than blind resubmission,
and archives authenticated results without promoting the local incumbent.
See [jsh remote contract](https://github.com/beamiter/jsh/blob/master/docs/autoresearch-remote.md).
Actual Alibaba DLC SDK/gateway deployment and GPU integration remain separate;
mock tests must never be described as a completed cluster run.
