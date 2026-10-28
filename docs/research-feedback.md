# Research author feedback v2

EditRequest accepts input schema versions 1 and 2. V1 remains unchanged: at most
four outcome category strings. V2 requires at most four exact objects with
`attempt`, `status`, `stage`, `failure_code`, `trial_id`, and `scores`.
Attempt indices are strictly increasing integers in 1..1000. Stages and failure
codes are closed enums; raw logs, messages, commands and arbitrary extra fields
are not accepted. Scores always have screen/confirm members, each finite or null.
Unbound observations cannot contain measurements. Successful outcomes require
completed screen evidence; kept/archived require confirmation, screened_out must
not claim confirmation. The host supplies only complete-stage measurements.

V2 also requires failure codes to match the authoring stage, and a trial ID
cannot appear before submission or repeat within the four-record window.
Evaluation failures require a finished/recovered stage; confirmation measurements
require screening measurements. These checks reject inconsistent host evidence
without adding fields or changing V1 category-only inputs.
Older ledgers without stage metadata can still use `legacy_unknown` with the
host's `prepare` placeholder; this observation must remain unbound and scoreless.

When a V2 observation names a trial visible in the same context, its status and
screen/confirm scores must match that trial. Missing context stages correspond to
null feedback stages; equivalent numbers such as `8` and `8.0` compare equally.
An observation may refer to a trial outside the bounded window when history was
omitted, but not to a nonexistent trial in a complete history. Unbound failure
observations and legacy V1 categories keep their existing behavior. A completed
but unreconciled outcome must be reconciled before preparing another author call.

This is input validation, not remote attestation or scientific proof. All feedback
remains untrusted user-role data. A host must associate observations with its own
campaign records; the model may not supply results for promotion. The protocol
still uses the same constant system prompt, request carrier and schema-v1 edit
plan. Existing v1 clients and proposal APIs are unaffected.

Provider error envelopes now fail before looking at choices. A valid-looking
edit plan cannot override an error returned in the same response. Errors never
echo provider payloads or private source text. No model/cluster IO is added.
