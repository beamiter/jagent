# Source-authoring integration

`jagent::research::author::{EditRequest, EditPlan}` is the shared, sans-IO
validation boundary used by the model-driven jsh research loop. It accepts
host-selected tracked UTF-8 files and exact campaign/revision/blob identities,
then returns messages/schema or validates complete chat-completion responses.
It does not perform network requests, write files, create commits or run code.

The model returns a hypothesis and complete replacement contents for a subset
of supplied editable files. The host creates the real candidate commit and
submits it through the existing context-bound admission protocol. Truncation,
refusals, tool calls, foreign bindings, duplicate/unlisted edits and unchanged
plans are rejected. A valid plan remains untrusted code, not an approval.

Limits: at most eight input files, 64 KiB total input source, 128 KiB total
replacement source, 512 KiB encoded request, 256 KiB encoded plan and 1 MiB
encoded completion envelope. The envelope bound includes ignored reasoning
metadata. JSON Schema character limits are supplemented by Rust byte limits.

Only already-supplied files can be edited: creation/deletion, arbitrary commands,
model-written evaluator metrics and native tool dispatch are deliberately absent.
A host must recheck live campaign state, Git blob identity, path permissions and
execution authorization before creating/evaluating a candidate. Fixed system
instructions and untrusted user-role observations do not constitute a sandbox.

The Research protocol workflow now selects the MSRV using explicit
`cargo +1.86.0`; merely installing it does not override rust-toolchain.toml.
No research workflow writes to repository branches or uses model credentials.
