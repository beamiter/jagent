//! Review-first terminal AI agent core, extracted from jterm4.
//!
//! This crate is deliberately **sans-IO**: it has no HTTP client, no PTY, no
//! process spawning, and no UI. Integrations (terminal emulators, shells)
//! provide transport and execution; this crate provides the parts that must
//! behave identically everywhere:
//!
//! - [`session`] — the pure agent state machine. The model may only *propose*
//!   commands; approval returns an [`session::ApprovedCommand`] value to the
//!   caller, and nothing in this crate can execute it.
//! - [`safety`] — recognizable-danger warnings and a retired auto-approval
//!   compatibility hook that always fails closed.
//! - [`provider`] — provider-neutral chat request construction (Anthropic /
//!   OpenAI-compatible / Ollama) returning plain [`provider::HttpRequest`]
//!   data, plus strict response text extraction.
//! - [`prompt`] — the agent system prompt and user-role context encoding.
//!   Terminal bytes and environment metadata are always carried as explicitly
//!   untrusted user-role data, never interpolated into system instructions.
//! - [`stream`] — sans-IO incremental parsing of streaming chat response
//!   bodies (Anthropic SSE, OpenAI-compatible SSE, Ollama NDJSON) into
//!   byte-bounded [`stream::StreamEvent`]s.
//! - [`tools`] — the same `run`/`say`/`done` actions carried by the
//!   providers' *native* tool-calling instead of JSON-in-text, so the
//!   provider enforces the schema. Additive: [`tools::AgentProtocol::Text`]
//!   keeps today's behavior byte-for-byte, and tool calls are converted into
//!   the same [`session::ParsedAction`] values, so the state machine and
//!   every invariant below apply unchanged.
//!
//! # Invariants (inherited from jterm4)
//!
//! 1. Generated commands are never executed without an explicit caller action
//!    on an [`session::ApprovedCommand`].
//! 2. Malformed model replies fail closed; parse failure never degrades into a
//!    command proposal.
//! 3. All transcripts, observations, and context payloads are byte-bounded.
//! 4. Terminal output and environment metadata are untrusted user-role data.
//! 5. Persisted sessions are revalidated on restore; proposal ids uniquely
//!    bind an approval action to the pending command the caller displayed.

pub mod prompt;
pub mod provider;
pub mod redact;
pub mod safety;
pub mod session;
pub mod stream;
mod text;
pub mod tools;

pub use prompt::{
    agent_user_prompt, build_agent_system_prompt, build_agent_tool_system_prompt, BlockContext,
    EnvironmentMeta, GitMeta,
};
pub use provider::{
    ChatConfig, ChatResponse, HttpRequest, Message, Provider, ProviderError, Role, Usage,
};
pub use redact::redact_secrets;
pub use safety::{is_auto_approvable, is_dangerous};
pub use session::{
    AgentSession, AgentSessionSnapshot, AgentSnapshotError, AgentState, ApprovedCommand,
    ModelOutcome, ParseError, ProposalId, ProposalStatus, SessionError, Turn,
    MAX_ACTION_JSON_BYTES, MAX_AGENT_SNAPSHOT_JSON_BYTES, MAX_SESSION_TURNS,
};
pub use stream::{StreamEvent, StreamParser};
pub use tools::{AgentProtocol, ToolCall, ToolResponse};
