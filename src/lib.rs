//! Review-first terminal AI agent core shared by `jsh` and the
//! `jterm_core`-based terminal family.
//!
//! This crate is deliberately **sans-IO**: it has no HTTP client, no PTY, no
//! process spawning, and no UI. Integrations (terminal emulators, shells)
//! provide transport and execution; this crate provides the parts that must
//! behave identically everywhere:
//!
//! - [`agent`] — the recommended request path. It binds one action protocol
//!   to the matching prompt, provider schema, delivery mode, history policy,
//!   and response decoder.
//! - [`response`] — protocol-aware non-streaming response decoding and a
//!   streaming accumulator that both produce the same high-level response.
//! - [`session`] — the pure agent state machine. The model may only *propose*
//!   commands; approval returns an [`session::ApprovedCommand`] value to the
//!   caller, and nothing in this crate can execute it.
//! - [`safety`] — recognizable-danger warnings and a retired auto-approval
//!   compatibility hook that always fails closed.
//! - [`provider`] — provider-neutral chat request construction (Anthropic /
//!   OpenAI-compatible / Ollama) returning [`provider::BuiltRequest`] data
//!   that keeps its plain [`provider::HttpRequest`] and history diagnostics,
//!   plus strict response extraction with byte-oriented entry points that
//!   bound the encoded envelope before JSON allocation.
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
//! String-owning public transcript/request/context shapes are deliberately
//! serialize-only. Persisted Agent bytes enter through
//! [`session::AgentSessionSnapshot::from_json`]; high-level agent response
//! bytes enter through [`agent::PreparedAgentRequest::parse_response`]. The
//! lower-level alternatives are [`provider::parse_chat_response_bytes`],
//! [`provider::parse_chat_response_full_bytes`], or
//! [`tools::parse_tool_response_bytes`]. The corresponding `serde_json::Value`
//! APIs require trusted or already-bounded caller-owned values.
//!
//! # Invariants
//!
//! 1. Generated commands are never executed without an explicit caller action
//!    on an [`session::ApprovedCommand`].
//! 2. Malformed model replies fail closed; parse failure never degrades into a
//!    command proposal.
//! 3. All transcripts, observations, and context payloads are byte-bounded.
//! 4. Terminal output and environment metadata are untrusted user-role data.
//! 5. Persisted sessions are revalidated on restore; proposal ids uniquely
//!    bind an approval action to the pending command the caller displayed.

pub mod agent;
pub mod prompt;
pub mod provider;
pub mod redact;
pub mod response;
pub mod safety;
pub mod session;
pub mod stream;
mod text;
pub mod tools;

pub use agent::{
    prepare_agent_request, AgentRequestReport, AgentRequestSpec, PreparedAgentRequest,
};
pub use prompt::{
    agent_user_prompt, build_agent_system_prompt, build_agent_tool_system_prompt, BlockContext,
    EnvironmentMeta, GitMeta,
};
pub use provider::{
    BuiltRequest, ChatConfig, ChatResponse, HistoryReport, HttpRequest, Message, PreparedHistory,
    Provider, ProviderError, Role, Usage,
};
pub use redact::{redact_secrets, redact_secrets_cow};
pub use response::{AgentResponse, AgentStream};
pub use safety::{is_auto_approvable, is_dangerous};
pub use session::{
    AgentSession, AgentSessionSnapshot, AgentSnapshotError, AgentState, ApprovedCommand,
    CommandExecutionFailure, ModelOutcome, ParseError, PendingProposal, ProposalId, ProposalStatus,
    SessionError, Turn, MAX_ACTION_JSON_BYTES, MAX_AGENT_SNAPSHOT_JSON_BYTES, MAX_SESSION_TURNS,
};
pub use stream::{StreamEvent, StreamParser};
pub use tools::{AgentProtocol, ToolCall, ToolResponse};
