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
//! - [`safety`] — recognizable-danger warnings and the fail-closed read-only
//!   auto-approval allowlist.
//! - [`provider`] — provider-neutral chat request construction (Anthropic /
//!   OpenAI-compatible / Ollama) returning plain [`provider::HttpRequest`]
//!   data, plus strict response text extraction.
//! - [`prompt`] — the agent system prompt and user-role context encoding.
//!   Terminal bytes and environment metadata are always carried as explicitly
//!   untrusted user-role data, never interpolated into system instructions.
//!
//! # Invariants (inherited from jterm4)
//!
//! 1. Generated commands are never executed without an explicit caller action
//!    on an [`session::ApprovedCommand`].
//! 2. Malformed model replies fail closed; parse failure never degrades into a
//!    command proposal.
//! 3. All transcripts, observations, and context payloads are byte-bounded.
//! 4. Terminal output and environment metadata are untrusted user-role data.

pub mod prompt;
pub mod provider;
pub mod redact;
pub mod safety;
pub mod session;
mod text;

pub use prompt::{
    agent_user_prompt, build_agent_system_prompt, BlockContext, EnvironmentMeta, GitMeta,
};
pub use provider::{
    ChatConfig, ChatResponse, HttpRequest, Message, Provider, ProviderError, Role, Usage,
};
pub use redact::redact_secrets;
pub use safety::{is_auto_approvable, is_dangerous};
pub use session::{
    AgentSession, AgentSessionSnapshot, AgentSnapshotError, AgentState, ApprovedCommand,
    ModelOutcome, ParseError, ProposalId, ProposalStatus, SessionError, Turn,
    MAX_AGENT_SNAPSHOT_JSON_BYTES,
};
