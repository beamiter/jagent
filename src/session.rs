//! Pure Agent-session protocol and state machine.
//!
//! The model may only propose commands. Approval returns an ApprovedCommand
//! value to the caller; this module has no PTY, shell, process, or UI access and
//! therefore cannot execute a command by itself.

use crate::safety::{is_dangerous, validate_command_text, CommandTextError, MAX_COMMAND_BYTES};
use crate::text::elide_middle;
use crate::tools::{
    AgentProtocol, ToolResponse, MAX_TOOL_ARGUMENTS_BYTES, MAX_TOOL_NAME_BYTES, TOOL_DONE,
    TOOL_RUN, TOOL_SAY,
};
use serde::de::{self, DeserializeSeed, Deserializer, VariantAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const MAX_TRANSCRIPT_BYTES: usize = 32 * 1024;
const MAX_STORED_TRANSCRIPT_BYTES: usize = 128 * 1024;
const MAX_STORED_TRANSCRIPT_ENTRIES: usize = 128;
const MAX_OBSERVATION_BYTES: usize = 4 * 1024;
const MAX_MESSAGE_BYTES: usize = 16 * 1024;
pub(crate) const MAX_THOUGHT_BYTES: usize = 4 * 1024;
const COMMAND_EXECUTION_DIAGNOSTIC_PREFIX: &str = "command execution for proposal #";
/// Maximum model turns in one task. Construction and snapshot restoration
/// share this bound so every session created through the public API remains
/// restorable while persisted data cannot inject an effectively unbounded
/// agent loop.
pub const MAX_SESSION_TURNS: u32 = 1_000;
/// Byte cap applied before parsing one JSON-in-text model action. This is
/// large enough for every bounded decoded field even under JSON escaping,
/// while preventing an unknown field from making `serde_json` allocate an
/// arbitrarily large value before the action schema rejects it.
pub const MAX_ACTION_JSON_BYTES: usize = 128 * 1024;

/// Allocation-free identifier atom used by the session snapshot schema.
///
/// Deserialization only reconstructs the numeric atom. It does not prove that
/// an id belongs to a live pending proposal; approval APIs and snapshot restore
/// enforce that contextual binding.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Hash, Serialize)]
pub struct ProposalId(u64);

impl ProposalId {
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Allocation-free proposal-status atom used by the bounded snapshot decoder.
/// A decoded status is not a validated proposal lifecycle by itself.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
    ManualReview,
}

/// One retained turn in an [`AgentSession`] transcript.
///
/// This type is serialize-only. It owns attacker-influenced strings and is not
/// an independent wire format; persisted transcripts must enter through
/// [`AgentSessionSnapshot::from_json`], whose private seeds enforce entry,
/// per-field, and cumulative budgets before allocating retained turns.
///
/// ```compile_fail
/// let _: jagent::Turn = serde_json::from_str(r#"{"User":"hello"}"#).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Turn {
    User(String),
    AssistantThought(String),
    AssistantSay(String),
    AssistantProposed {
        id: ProposalId,
        command: String,
        status: ProposalStatus,
    },
    Observation {
        proposal_id: ProposalId,
        exit_code: i32,
        output_sample: String,
    },
    /// A bounded diagnostic from a failed protocol, provider, model,
    /// transport, or command-execution turn.
    ///
    /// The variant name is retained for snapshot compatibility; the contents
    /// are deliberately broader than JSON parser failures.
    ProtocolError(String),
}

impl Turn {
    fn to_prompt(&self) -> String {
        match self {
            Self::User(message) => format!("User: {message}"),
            Self::AssistantThought(thought) => format!("Assistant (thought): {thought}"),
            Self::AssistantSay(message) => format!(
                "Assistant: {}",
                serde_json::json!({"action": "say", "message": message})
            ),
            Self::AssistantProposed {
                command, status, ..
            } => {
                let action = serde_json::json!({"action": "run", "command": command});
                let verdict = match status {
                    ProposalStatus::Pending => "[awaiting user approval]",
                    ProposalStatus::Approved => "[user approved; awaiting/received output]",
                    ProposalStatus::Rejected => "[user rejected this proposal]",
                    ProposalStatus::ManualReview => {
                        "[user moved this command to the prompt for manual review; it was not executed]"
                    }
                };
                format!("Assistant: {action}\n{verdict}")
            }
            Self::Observation {
                exit_code,
                output_sample,
                ..
            } => format!("Output (exit={exit_code}):\n{output_sample}"),
            Self::ProtocolError(message) => {
                format!("[previous Agent turn failed: {message}]")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedAction {
    Run {
        thought: Option<String>,
        command: String,
    },
    Say {
        thought: Option<String>,
        message: String,
    },
    Done {
        thought: Option<String>,
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Empty,
    InvalidFence,
    InvalidJson(String),
    ExpectedObject,
    MissingField(&'static str),
    InvalidFieldType(&'static str),
    EmptyField(&'static str),
    FieldTooLarge(&'static str),
    UnknownAction(String),
    UnexpectedField(String),
    InvalidCommand(String),
    /// Native tool mode: the reply carried no tool call. Accompanying prose is
    /// never promoted to an action, so this fails closed even with text
    /// present. See [`crate::tools`].
    NoToolCall,
    /// Native tool mode: the reply carried more than one tool call. The state
    /// machine advances one action per turn; choosing one would be a guess.
    MultipleToolCalls(usize),
    /// The provider reported that generation stopped at a token or context
    /// bound. Even syntactically complete-looking output is treated as
    /// partial and cannot become an action.
    TruncatedResponse,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty reply"),
            Self::InvalidFence => write!(f, "invalid or unterminated JSON code fence"),
            Self::InvalidJson(error) => write!(f, "invalid JSON: {error}"),
            Self::ExpectedObject => write!(f, "top-level JSON value must be an object"),
            Self::MissingField(field) => write!(f, "missing required field '{field}'"),
            Self::InvalidFieldType(field) => write!(f, "field '{field}' must be a string"),
            Self::EmptyField(field) => write!(f, "field '{field}' must not be empty"),
            Self::FieldTooLarge(field) => write!(f, "field '{field}' exceeds its size limit"),
            Self::UnknownAction(action) => write!(f, "unknown action '{action}'"),
            Self::UnexpectedField(field) => write!(f, "unexpected field '{field}'"),
            Self::InvalidCommand(message) => write!(f, "invalid command: {message}"),
            Self::NoToolCall => write!(f, "reply contained no tool call"),
            Self::MultipleToolCalls(count) => {
                write!(
                    f,
                    "reply contained {count} tool calls; exactly one is required"
                )
            }
            Self::TruncatedResponse => {
                write!(
                    f,
                    "reply reached a provider generation limit and may be truncated"
                )
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Strictly parse one action. A single json code fence is tolerated, but
/// prose, duplicate object members, unknown actions/keys, wrong types, and
/// empty required fields fail. Parse failure never degrades into a command
/// proposal.
pub fn parse_action(raw: &str) -> Result<ParsedAction, ParseError> {
    if raw.len() > MAX_ACTION_JSON_BYTES {
        return Err(ParseError::FieldTooLarge("reply"));
    }
    let payload = strip_json_fence(raw.trim())?;
    if payload.is_empty() {
        return Err(ParseError::Empty);
    }
    let value = crate::json::from_str(payload)
        .map_err(|error| ParseError::InvalidJson(error.to_string()))?;
    let object = value.as_object().ok_or(ParseError::ExpectedObject)?;
    let action = required_string(object, "action", 32)?;
    action_from_object(&action, object, &["action"])
}

/// Map one *native* tool call onto the same [`ParsedAction`] values
/// [`parse_action`] produces, so the state machine and every invariant apply
/// unchanged whichever protocol carried the reply.
///
/// `arguments` is the raw JSON object text (empty means "no arguments" and is
/// read as `{}`). The rules are identical to the text protocol minus the
/// `action` field, which the tool name supplies: duplicate object members,
/// unknown names, wrong types, missing or empty required fields, extra keys,
/// and multi-line commands all fail closed and never degrade into a proposal.
pub fn parse_tool_action(name: &str, arguments: &str) -> Result<ParsedAction, ParseError> {
    let name = name.trim();
    if arguments.len() > MAX_TOOL_ARGUMENTS_BYTES {
        return Err(ParseError::FieldTooLarge("arguments"));
    }
    let payload = arguments.trim();
    let value: Value = if payload.is_empty() {
        Value::Object(Map::new())
    } else {
        crate::json::from_str(payload)
            .map_err(|error| ParseError::InvalidJson(error.to_string()))?
    };
    let object = value.as_object().ok_or(ParseError::ExpectedObject)?;
    action_from_object(name, object, &[])
}

/// Shared tail of both protocols: `action` names the shape, `object` carries
/// the fields, and `also_allowed` lists keys the *carrier* contributes (the
/// text protocol's own `action` key; nothing in tool mode).
fn action_from_object(
    action: &str,
    object: &Map<String, Value>,
    also_allowed: &[&str],
) -> Result<ParsedAction, ParseError> {
    let thought = optional_string(object, "thought", MAX_THOUGHT_BYTES)?;
    match action {
        TOOL_RUN => {
            reject_unexpected(object, also_allowed, &["thought", "command"])?;
            let command = required_string(object, "command", MAX_COMMAND_BYTES)?;
            validate_command(&command)?;
            Ok(ParsedAction::Run { thought, command })
        }
        TOOL_SAY => {
            reject_unexpected(object, also_allowed, &["thought", "message"])?;
            let message = required_string(object, "message", MAX_MESSAGE_BYTES)?;
            Ok(ParsedAction::Say { thought, message })
        }
        TOOL_DONE => {
            reject_unexpected(object, also_allowed, &["thought", "message"])?;
            let message = required_string(object, "message", MAX_MESSAGE_BYTES)?;
            Ok(ParsedAction::Done { thought, message })
        }
        other => Err(ParseError::UnknownAction(elide_middle(
            other,
            MAX_TOOL_NAME_BYTES,
        ))),
    }
}

fn strip_json_fence(raw: &str) -> Result<&str, ParseError> {
    if !raw.starts_with("```") {
        return Ok(raw);
    }
    let newline = raw.find('\n').ok_or(ParseError::InvalidFence)?;
    let language = raw[3..newline].trim();
    if !language.is_empty() && !language.eq_ignore_ascii_case("json") {
        return Err(ParseError::InvalidFence);
    }
    raw[newline + 1..]
        .strip_suffix("```")
        .map(str::trim)
        .ok_or(ParseError::InvalidFence)
}

fn required_string(
    object: &Map<String, Value>,
    field: &'static str,
    max_bytes: usize,
) -> Result<String, ParseError> {
    let value = object.get(field).ok_or(ParseError::MissingField(field))?;
    let value = value
        .as_str()
        .ok_or(ParseError::InvalidFieldType(field))?
        .trim();
    if value.is_empty() {
        return Err(ParseError::EmptyField(field));
    }
    if value.len() > max_bytes {
        return Err(ParseError::FieldTooLarge(field));
    }
    Ok(value.to_string())
}

fn optional_string(
    object: &Map<String, Value>,
    field: &'static str,
    max_bytes: usize,
) -> Result<Option<String>, ParseError> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    let value = value
        .as_str()
        .ok_or(ParseError::InvalidFieldType(field))?
        .trim();
    if value.is_empty() {
        return Ok(None);
    }
    if value.len() > max_bytes {
        return Err(ParseError::FieldTooLarge(field));
    }
    Ok(Some(value.to_string()))
}

fn reject_unexpected(
    object: &Map<String, Value>,
    also_allowed: &[&str],
    allowed: &[&str],
) -> Result<(), ParseError> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()) && !also_allowed.contains(&field.as_str()))
    {
        return Err(ParseError::UnexpectedField(elide_middle(
            field,
            MAX_TOOL_NAME_BYTES,
        )));
    }
    Ok(())
}

fn validate_command(command: &str) -> Result<(), ParseError> {
    validate_command_text(command)
        .map(|_| ())
        .map_err(|error| match error {
            CommandTextError::TooLarge => ParseError::FieldTooLarge("command"),
            CommandTextError::Empty => ParseError::EmptyField("command"),
            CommandTextError::LineBreak => {
                ParseError::InvalidCommand("must be exactly one visible line".into())
            }
            CommandTextError::ControlCharacter => {
                ParseError::InvalidCommand("contains a control character".into())
            }
            CommandTextError::VisualSpoof => ParseError::InvalidCommand(
                "contains an invisible or bidirectional formatting character".into(),
            ),
        })
}

/// Allocation-free state atom used by the bounded snapshot schema.
///
/// Direct deserialization checks only this enum's local shape. It does not
/// establish that the state agrees with a transcript, proposal lifecycle, or
/// turn counter; [`AgentSession::restore`] performs those checks.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub enum AgentState {
    Ready,
    AwaitingModel,
    AwaitingApproval { proposal_id: ProposalId },
    AwaitingObservation { proposal_id: ProposalId },
    Completed,
    Cancelled,
    TurnLimitReached,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    EmptyUserMessage,
    UserMessageTooLarge,
    InvalidTransition {
        operation: &'static str,
        state: AgentState,
    },
    Protocol(ParseError),
    StaleProposal {
        expected: ProposalId,
        received: ProposalId,
    },
    ProposalNotFound(ProposalId),
    TurnLimitReached,
    Cancelled,
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyUserMessage => write!(f, "user message must not be empty"),
            Self::UserMessageTooLarge => write!(
                f,
                "user message exceeds the {} byte Agent limit",
                MAX_MESSAGE_BYTES
            ),
            Self::InvalidTransition { operation, state } => {
                write!(f, "cannot {operation} while session is {state:?}")
            }
            Self::Protocol(error) => write!(f, "model protocol error: {error}"),
            Self::StaleProposal { expected, received } => write!(
                f,
                "proposal id {} is stale; expected {}",
                received.get(),
                expected.get()
            ),
            Self::ProposalNotFound(id) => write!(f, "proposal {} is not in transcript", id.get()),
            Self::TurnLimitReached => write!(f, "agent turn limit reached"),
            Self::Cancelled => write!(f, "agent session cancelled"),
        }
    }
}

impl std::error::Error for SessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelOutcome {
    Proposal {
        id: ProposalId,
        command: String,
        danger: Option<&'static str>,
    },
    Said(String),
    Completed(String),
}

/// Explicit authorization token returned after approval. The integration
/// layer may choose to type this into a terminal; constructing a session or
/// receiving a proposal never performs that action.
#[must_use = "approval only yields a command; the caller must deliberately handle it"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedCommand {
    pub proposal_id: ProposalId,
    pub command: String,
    pub danger: Option<&'static str>,
}

/// Borrowed view of the command proposal currently awaiting user approval.
///
/// The view is available only while the session is in
/// [`AgentState::AwaitingApproval`] and its identifier still binds to the
/// retained pending transcript entry. Holding the view prevents mutable
/// session transitions until it is no longer used, so the returned command
/// cannot become stale through the same borrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingProposal<'a> {
    /// Identifier the caller must pass back to an approval or rejection API.
    pub id: ProposalId,
    /// Exact command shown on the pending approval card.
    pub command: &'a str,
}

/// Why an approved command did not produce a normal process exit status.
///
/// This is a command-level outcome, distinct from cancelling the entire
/// [`AgentSession`] with [`AgentSession::cancel`]. Diagnostic text or partial
/// output is supplied separately to
/// [`AgentSession::observe_execution_failure`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommandExecutionFailure {
    /// The integration could not start the process at all.
    FailedToStart,
    /// The integration stopped the command after its execution deadline.
    TimedOut,
    /// The command was cancelled without cancelling the enclosing Agent task.
    Cancelled,
}

impl CommandExecutionFailure {
    fn prompt_description(self) -> &'static str {
        match self {
            Self::FailedToStart => "failed to start",
            Self::TimedOut => "timed out",
            Self::Cancelled => "was cancelled",
        }
    }
}

/// Typed result of deliberately handling one [`ApprovedCommand`].
///
/// Integrations should carry this value from their process boundary into
/// [`AgentSession::observe_execution`] instead of mapping setup, timeout, or
/// cancellation failures onto a made-up exit code. The evidence string is
/// still untrusted and is sampled only when the session ingests the outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CommandExecutionOutcome {
    /// The command process produced a normal shell exit status.
    Exited { exit_code: i32, output: String },
    /// The integration could not obtain a normal exit status.
    Failed {
        failure: CommandExecutionFailure,
        detail: String,
    },
}

impl CommandExecutionOutcome {
    /// Construct a normal command exit with its captured output.
    pub fn exited(exit_code: i32, output: impl Into<String>) -> Self {
        Self::Exited {
            exit_code,
            output: output.into(),
        }
    }

    /// Construct an execution failure without inventing an exit status.
    pub fn failed(failure: CommandExecutionFailure, detail: impl Into<String>) -> Self {
        Self::Failed {
            failure,
            detail: detail.into(),
        }
    }

    /// Normal exit status, or `None` when execution did not produce one.
    pub const fn exit_code(&self) -> Option<i32> {
        match self {
            Self::Exited { exit_code, .. } => Some(*exit_code),
            Self::Failed { .. } => None,
        }
    }

    /// Failure category, or `None` after a normal exit.
    pub const fn failure(&self) -> Option<CommandExecutionFailure> {
        match self {
            Self::Exited { .. } => None,
            Self::Failed { failure, .. } => Some(*failure),
        }
    }

    /// Captured command output or untrusted failure detail.
    pub fn evidence(&self) -> &str {
        match self {
            Self::Exited { output, .. } => output,
            Self::Failed { detail, .. } => detail,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[derive(Debug)]
pub struct AgentSession {
    transcript: Vec<Turn>,
    transcript_truncated: bool,
    state: AgentState,
    turns_used: u32,
    max_turns: u32,
    next_proposal_id: u64,
    cancelled: CancellationToken,
}

impl AgentSession {
    pub fn new(max_turns: u32) -> Self {
        Self {
            transcript: Vec::new(),
            transcript_truncated: false,
            state: AgentState::Ready,
            turns_used: 0,
            max_turns: max_turns.clamp(1, MAX_SESSION_TURNS),
            next_proposal_id: 1,
            cancelled: CancellationToken(Arc::new(AtomicBool::new(false))),
        }
    }

    pub fn transcript(&self) -> &[Turn] {
        &self.transcript
    }

    /// Whether older entries have been removed from this live session by its
    /// retained-transcript entry or byte budget.
    ///
    /// This reports stored transcript compaction. Individual observations and
    /// the final provider prompt have their own sampling bounds, so callers
    /// should not interpret `false` as a promise that every original output
    /// byte will be sent to the model.
    pub fn transcript_truncated(&self) -> bool {
        self.transcript_truncated
    }

    /// Return the exact proposal currently awaiting approval, if any.
    ///
    /// The lookup checks both [`AgentState::AwaitingApproval`] and the retained
    /// proposal's [`ProposalStatus::Pending`] status. It therefore fails closed
    /// with `None` if an internally inconsistent session could ever reach this
    /// read-only API. This is useful after restoring a snapshot and redrawing
    /// the approval card without scanning [`Self::transcript`] manually.
    ///
    /// ```
    /// use jagent::{AgentSession, ModelOutcome};
    ///
    /// let mut session = AgentSession::new(4);
    /// session.submit_user("show files").unwrap();
    /// let ModelOutcome::Proposal { id, .. } = session
    ///     .accept_model_reply(r#"{"action":"run","command":"ls"}"#)
    ///     .unwrap()
    /// else {
    ///     panic!("expected a proposal");
    /// };
    /// let pending = session.pending_proposal().unwrap();
    /// assert_eq!(pending.id, id);
    /// assert_eq!(pending.command, "ls");
    /// ```
    pub fn pending_proposal(&self) -> Option<PendingProposal<'_>> {
        let AgentState::AwaitingApproval { proposal_id } = self.state else {
            return None;
        };
        self.transcript.iter().rev().find_map(|turn| match turn {
            Turn::AssistantProposed {
                id,
                command,
                status: ProposalStatus::Pending,
            } if *id == proposal_id => Some(PendingProposal { id: *id, command }),
            _ => None,
        })
    }

    pub fn state(&self) -> AgentState {
        self.state
    }

    pub fn turns_used(&self) -> u32 {
        self.turns_used
    }

    pub fn max_turns(&self) -> u32 {
        self.max_turns
    }

    /// A `done` reply closes one task, but the user may still ask a follow-up
    /// while this session has model-turn budget left. The transcript is kept so
    /// the follow-up retains the completed task's context.
    pub fn can_continue_after_completion(&self) -> bool {
        self.state == AgentState::Completed
            && self.turns_used < self.max_turns
            && !self.cancelled.is_cancelled()
    }

    pub fn continue_after_completion(&mut self) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        if !self.can_continue_after_completion() {
            return Err(self.invalid_transition("continue a completed task"));
        }
        self.state = AgentState::Ready;
        Ok(())
    }

    /// Completed or exhausted sessions can start a fresh task in the same
    /// surface without closing and rebuilding the Agent UI. This explicitly
    /// drops the old model transcript and restores the configured turn budget.
    /// Proposal ids remain monotonic across the reset so delayed actions from
    /// the previous task stay stale.
    pub fn start_new_task(&mut self) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        if !matches!(
            self.state,
            AgentState::Completed | AgentState::TurnLimitReached
        ) {
            return Err(self.invalid_transition("start a new task"));
        }
        let max_turns = self.max_turns;
        // Proposal ids are authorization bindings, not transcript-local row
        // numbers. Preserve the counter across task resets so a delayed click
        // or callback from an old approval card can never match a new task's
        // proposal merely because both would otherwise be numbered `1`.
        let next_proposal_id = self.next_proposal_id;
        let mut fresh = Self::new(max_turns);
        fresh.next_proposal_id = next_proposal_id;
        *self = fresh;
        Ok(())
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancelled.clone()
    }

    pub fn submit_user(&mut self, message: impl Into<String>) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        if self.turns_used >= self.max_turns {
            self.state = AgentState::TurnLimitReached;
            return Err(SessionError::TurnLimitReached);
        }
        if self.state != AgentState::Ready {
            return Err(self.invalid_transition("submit user input"));
        }
        let message = message.into();
        let message = message.trim();
        if message.is_empty() {
            return Err(SessionError::EmptyUserMessage);
        }
        if message.len() > MAX_MESSAGE_BYTES {
            return Err(SessionError::UserMessageTooLarge);
        }
        self.push_turn(Turn::User(message.to_string()));
        self.state = AgentState::AwaitingModel;
        Ok(())
    }

    /// Ingest a reply carried by the JSON-in-text protocol.
    pub fn accept_model_reply(&mut self, raw: &str) -> Result<ModelOutcome, SessionError> {
        self.accept_parsed_action(|| parse_action(raw))
    }

    /// Ingest a reply carried by the provider's *native* tool-calling, parsed
    /// with [`crate::tools::parse_tool_response`].
    ///
    /// The reply is resolved to exactly one [`ParsedAction`] by
    /// [`ToolResponse::to_action`] — including the mixed text-and-tool rule —
    /// and then runs through the identical state machine as
    /// [`Self::accept_model_reply`]: the same [`ModelOutcome`]s, the same
    /// transcript turns, and the same fail-closed protocol errors. A `run`
    /// tool call is still only a proposal; approval still returns an
    /// [`ApprovedCommand`] and this crate still cannot execute anything.
    pub fn accept_model_tool_reply(
        &mut self,
        reply: &ToolResponse,
    ) -> Result<ModelOutcome, SessionError> {
        self.accept_parsed_action(|| reply.to_action())
    }

    /// Ingest a complete, protocol-aware provider response.
    ///
    /// This is the preferred high-level entry point for responses produced by
    /// [`crate::response::AgentResponse::parse_bytes`] or
    /// [`crate::response::AgentStream`]. It resolves both text and native-tool
    /// replies through [`crate::response::AgentResponse::to_action`] and then
    /// uses the same state transition path as [`Self::accept_model_reply`] and
    /// [`Self::accept_model_tool_reply`]. In particular, a provider-reported
    /// generation bound fails with
    /// [`ParseError::TruncatedResponse`] before even a syntactically
    /// complete-looking command can become a proposal.
    pub fn accept_agent_response(
        &mut self,
        response: &crate::response::AgentResponse,
    ) -> Result<ModelOutcome, SessionError> {
        self.accept_parsed_action(|| response.to_action())
    }

    fn accept_parsed_action(
        &mut self,
        parse: impl FnOnce() -> Result<ParsedAction, ParseError>,
    ) -> Result<ModelOutcome, SessionError> {
        self.check_not_cancelled()?;
        if self.state != AgentState::AwaitingModel {
            return Err(self.invalid_transition("accept a model reply"));
        }
        if self.turns_used >= self.max_turns {
            self.state = AgentState::TurnLimitReached;
            return Err(SessionError::TurnLimitReached);
        }
        self.turns_used = self.turns_used.saturating_add(1);
        let action = match parse() {
            Ok(action) => action,
            Err(error) => {
                self.push_turn(Turn::ProtocolError(elide_middle(
                    &error.to_string(),
                    MAX_MESSAGE_BYTES,
                )));
                self.state = self.ready_or_limited();
                return Err(SessionError::Protocol(error));
            }
        };
        match action {
            ParsedAction::Run { thought, command } => {
                let Some(next_proposal_id) = self.next_proposal_id.checked_add(1) else {
                    self.push_turn(Turn::ProtocolError(
                        "proposal identifier space is exhausted".into(),
                    ));
                    self.state = AgentState::TurnLimitReached;
                    return Err(SessionError::TurnLimitReached);
                };
                self.push_thought(thought);
                let id = ProposalId(self.next_proposal_id);
                self.next_proposal_id = next_proposal_id;
                self.push_turn(Turn::AssistantProposed {
                    id,
                    command: command.clone(),
                    status: ProposalStatus::Pending,
                });
                self.state = AgentState::AwaitingApproval { proposal_id: id };
                Ok(ModelOutcome::Proposal {
                    id,
                    danger: is_dangerous(&command),
                    command,
                })
            }
            ParsedAction::Say { thought, message } => {
                self.push_thought(thought);
                self.push_turn(Turn::AssistantSay(message.clone()));
                self.state = self.ready_or_limited();
                Ok(ModelOutcome::Said(message))
            }
            ParsedAction::Done { thought, message } => {
                self.push_thought(thought);
                self.push_turn(Turn::AssistantSay(message.clone()));
                self.state = AgentState::Completed;
                Ok(ModelOutcome::Completed(message))
            }
        }
    }

    /// Record a provider/transport failure without interpreting it as model
    /// output. The user can retry or revise their request when turns remain.
    pub fn model_failed(&mut self, message: impl Into<String>) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        if self.state != AgentState::AwaitingModel {
            return Err(self.invalid_transition("record a model failure"));
        }
        let message = elide_middle(&message.into(), MAX_MESSAGE_BYTES);
        // Repeated provider attempts replace one another, but a command
        // execution diagnostic is model context that the next successful
        // attempt still needs to see.
        let replace_previous = matches!(
            self.transcript.last(),
            Some(Turn::ProtocolError(previous))
                if !previous.starts_with(COMMAND_EXECUTION_DIAGNOSTIC_PREFIX)
        );
        if replace_previous {
            let Some(Turn::ProtocolError(previous)) = self.transcript.last_mut() else {
                unreachable!("replace_previous only matches a diagnostic turn")
            };
            *previous = message;
            self.compact_transcript();
        } else {
            self.push_turn(Turn::ProtocolError(message));
        }
        self.state = self.ready_or_limited();
        Ok(())
    }

    /// Re-run the most recent failed model turn without appending a duplicate
    /// user instruction. The recorded protocol/transport error remains in the
    /// prompt so the provider can correct its next reply.
    pub fn retry_model(&mut self) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        if !self.can_retry_model() {
            return Err(self.invalid_transition("retry the last model request"));
        }
        self.state = AgentState::AwaitingModel;
        Ok(())
    }

    pub fn can_retry_model(&self) -> bool {
        self.state == AgentState::Ready
            && self.turns_used < self.max_turns
            && matches!(self.transcript.last(), Some(Turn::ProtocolError(_)))
    }

    pub fn approve(&mut self, id: ProposalId) -> Result<ApprovedCommand, SessionError> {
        self.approve_inner(id, None)
    }

    pub fn edit_and_approve(
        &mut self,
        id: ProposalId,
        edited_command: impl Into<String>,
    ) -> Result<ApprovedCommand, SessionError> {
        let command = edited_command.into();
        validate_command(&command).map_err(SessionError::Protocol)?;
        self.approve_inner(id, Some(command))
    }

    fn approve_inner(
        &mut self,
        id: ProposalId,
        edited_command: Option<String>,
    ) -> Result<ApprovedCommand, SessionError> {
        self.check_not_cancelled()?;
        self.expect_pending_proposal(id, "approve a proposal")?;
        let turn = self.pending_proposal_mut(id)?;
        let Turn::AssistantProposed {
            command, status, ..
        } = turn
        else {
            unreachable!("proposal_mut only returns proposal turns")
        };
        if let Some(edited) = edited_command {
            *command = edited;
        }
        *status = ProposalStatus::Approved;
        let approved = ApprovedCommand {
            proposal_id: id,
            danger: is_dangerous(command),
            command: command.clone(),
        };
        // Editing the command and lengthening the rendered status can push a
        // previously in-budget transcript over its retained byte ceiling.
        // Compact before exposing a snapshot-capable post-approval state.
        self.compact_transcript();
        self.state = AgentState::AwaitingObservation { proposal_id: id };
        Ok(approved)
    }

    /// Reject the pending proposal and immediately request an alternative from
    /// the model. The transcript records the rejection but no additional user
    /// feedback, preserving the original rejection behavior.
    pub fn reject(&mut self, id: ProposalId) -> Result<(), SessionError> {
        self.reject_inner(id, None)
    }

    /// Reject the pending proposal and record bounded user feedback for the
    /// model's next turn.
    ///
    /// Feedback is trimmed and fully validated before the proposal status,
    /// transcript, or session state is changed. Empty feedback returns
    /// [`SessionError::EmptyUserMessage`], and feedback above the same bound as
    /// ordinary user input returns [`SessionError::UserMessageTooLarge`]. On
    /// either error the pending proposal remains untouched. Accepted feedback
    /// is retained as a [`Turn::User`] immediately after the rejected proposal
    /// and does not consume an additional model turn.
    pub fn reject_with_feedback(
        &mut self,
        id: ProposalId,
        feedback: impl Into<String>,
    ) -> Result<(), SessionError> {
        let feedback = feedback.into();
        let feedback = feedback.trim();
        if feedback.is_empty() {
            return Err(SessionError::EmptyUserMessage);
        }
        if feedback.len() > MAX_MESSAGE_BYTES {
            return Err(SessionError::UserMessageTooLarge);
        }
        self.reject_inner(id, Some(feedback.to_string()))
    }

    fn reject_inner(
        &mut self,
        id: ProposalId,
        feedback: Option<String>,
    ) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        self.expect_pending_proposal(id, "reject a proposal")?;
        {
            let turn = self.pending_proposal_mut(id)?;
            if let Turn::AssistantProposed { status, .. } = turn {
                *status = ProposalStatus::Rejected;
            }
        }
        // Even a status-only transition changes the rendered transcript size.
        self.compact_transcript();
        if let Some(feedback) = feedback {
            self.push_turn(Turn::User(feedback));
        }
        // The rejection status itself asks the next model call for an
        // alternative. Optional feedback follows it as an ordinary user turn.
        self.state = if self.turns_used >= self.max_turns {
            AgentState::TurnLimitReached
        } else {
            AgentState::AwaitingModel
        };
        Ok(())
    }

    /// Move an edited proposal into the shell's normal line editor without
    /// authorizing execution. The UI owns the actual review-only insertion;
    /// this transition merely records that the Agent must not expect output or
    /// assume the command ran.
    pub fn edit_for_manual_review(
        &mut self,
        id: ProposalId,
        edited_command: impl Into<String>,
    ) -> Result<String, SessionError> {
        self.check_not_cancelled()?;
        self.expect_pending_proposal(id, "move a proposal to manual review")?;
        let edited_command = edited_command.into();
        validate_command(&edited_command).map_err(SessionError::Protocol)?;
        let turn = self.pending_proposal_mut(id)?;
        let Turn::AssistantProposed {
            command, status, ..
        } = turn
        else {
            unreachable!("proposal_mut only returns proposal turns")
        };
        *command = edited_command;
        *status = ProposalStatus::ManualReview;
        let command = command.clone();
        // Both the edited command and manual-review verdict can grow the
        // retained representation without appending a new turn.
        self.compact_transcript();
        self.state = self.ready_or_limited();
        Ok(command)
    }

    pub fn observe(
        &mut self,
        id: ProposalId,
        exit_code: i32,
        output: &str,
    ) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        self.expect_awaiting_observation(id, "record command output")?;
        self.push_turn(Turn::Observation {
            proposal_id: id,
            exit_code,
            output_sample: sample_observation(output),
        });
        self.finish_observation();
        Ok(())
    }

    /// Record one typed execution outcome against the approved proposal.
    ///
    /// This is the preferred integration entry point because its enum makes
    /// the presence or absence of a normal exit status explicit. The legacy
    /// [`Self::observe`] and [`Self::observe_execution_failure`] methods remain
    /// available for source compatibility and implement the same transitions.
    pub fn observe_execution(
        &mut self,
        id: ProposalId,
        outcome: CommandExecutionOutcome,
    ) -> Result<(), SessionError> {
        match outcome {
            CommandExecutionOutcome::Exited { exit_code, output } => {
                self.observe(id, exit_code, &output)
            }
            CommandExecutionOutcome::Failed { failure, detail } => {
                self.observe_execution_failure(id, failure, &detail)
            }
        }
    }

    /// Record that an approved command produced no normal process exit status.
    ///
    /// `detail` may contain a spawn error or partial command output. It is
    /// sampled with the same UTF-8-safe byte bound as a normal observation and
    /// framed as untrusted diagnostic data in the next model prompt. The
    /// transition is proposal-id-bound exactly like [`Self::observe`], then
    /// advances to [`AgentState::AwaitingModel`] or
    /// [`AgentState::TurnLimitReached`]. No synthetic exit code is invented.
    ///
    /// The failure is stored in the existing bounded diagnostic transcript
    /// representation, preserving the version-1 snapshot schema.
    ///
    /// ```
    /// use jagent::session::CommandExecutionFailure;
    /// use jagent::{AgentSession, AgentState, ModelOutcome};
    ///
    /// let mut session = AgentSession::new(4);
    /// session.submit_user("run the check").unwrap();
    /// let ModelOutcome::Proposal { id, .. } = session
    ///     .accept_model_reply(r#"{"action":"run","command":"check"}"#)
    ///     .unwrap()
    /// else {
    ///     panic!("expected a proposal");
    /// };
    /// session.approve(id).unwrap();
    /// session
    ///     .observe_execution_failure(
    ///         id,
    ///         CommandExecutionFailure::FailedToStart,
    ///         "executable not found",
    ///     )
    ///     .unwrap();
    /// assert_eq!(session.state(), AgentState::AwaitingModel);
    /// ```
    pub fn observe_execution_failure(
        &mut self,
        id: ProposalId,
        failure: CommandExecutionFailure,
        detail: &str,
    ) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        self.expect_awaiting_observation(id, "record a command execution failure")?;

        let mut message = format!(
            "{COMMAND_EXECUTION_DIAGNOSTIC_PREFIX}{} {}; no normal exit status was available",
            id.get(),
            failure.prompt_description()
        );
        let detail = sample_observation(detail);
        if detail.trim().is_empty() {
            message.push('.');
        } else {
            message.push_str(". Untrusted diagnostic or partial output:\n");
            message.push_str(&detail);
        }
        self.push_turn(Turn::ProtocolError(message));
        self.finish_observation();
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.cancelled.0.store(true, Ordering::SeqCst);
        self.state = AgentState::Cancelled;
    }

    pub fn build_user_prompt(&self) -> String {
        self.build_user_prompt_with(AgentProtocol::Text)
    }

    /// [`Self::build_user_prompt`] with the closing instruction matched to the
    /// protocol in use, so native mode does not ask for a JSON object the
    /// schema has already replaced. [`AgentProtocol::Text`] is byte-identical
    /// to [`Self::build_user_prompt`].
    ///
    /// The transcript body itself is protocol-independent: past actions are
    /// always rendered as the canonical JSON shapes, which stay an unambiguous
    /// record of what was proposed and how the user ruled on it.
    pub fn build_user_prompt_with(&self, protocol: AgentProtocol) -> String {
        let mut entries: Vec<String> = self.transcript.iter().map(Turn::to_prompt).collect();
        if self.transcript_truncated {
            entries.insert(
                0,
                "[older Agent activity was omitted by the in-memory safety budget]".to_string(),
            );
        }
        entries.push(
            match protocol {
                AgentProtocol::Text => {
                    "Reply with exactly one JSON object from the protocol; no markdown."
                }
                AgentProtocol::NativeTools => {
                    "Continue by calling exactly one tool: run, say, or done."
                }
            }
            .to_string(),
        );
        elide_middle(&entries.join("\n\n"), MAX_TRANSCRIPT_BYTES)
    }

    /// Resolve the exact pending card the user is acting on. Matching status
    /// as well as id is defense in depth for sessions restored from storage:
    /// an old approved/rejected turn must never be mistaken for the visible
    /// pending proposal even if persistence was corrupted.
    fn pending_proposal_mut(&mut self, id: ProposalId) -> Result<&mut Turn, SessionError> {
        self.transcript
            .iter_mut()
            .find(|turn| {
                matches!(
                    turn,
                    Turn::AssistantProposed {
                        id: candidate,
                        status: ProposalStatus::Pending,
                        ..
                    } if *candidate == id
                )
            })
            .ok_or(SessionError::ProposalNotFound(id))
    }

    fn expect_pending_proposal(
        &self,
        id: ProposalId,
        operation: &'static str,
    ) -> Result<(), SessionError> {
        match self.state {
            AgentState::AwaitingApproval { proposal_id } if proposal_id == id => Ok(()),
            AgentState::AwaitingApproval { proposal_id } => Err(SessionError::StaleProposal {
                expected: proposal_id,
                received: id,
            }),
            _ => Err(self.invalid_transition(operation)),
        }
    }

    fn expect_awaiting_observation(
        &self,
        id: ProposalId,
        operation: &'static str,
    ) -> Result<(), SessionError> {
        match self.state {
            AgentState::AwaitingObservation { proposal_id } if proposal_id == id => Ok(()),
            AgentState::AwaitingObservation { proposal_id } => Err(SessionError::StaleProposal {
                expected: proposal_id,
                received: id,
            }),
            _ => Err(self.invalid_transition(operation)),
        }
    }

    fn finish_observation(&mut self) {
        self.state = if self.turns_used >= self.max_turns {
            AgentState::TurnLimitReached
        } else {
            AgentState::AwaitingModel
        };
    }

    fn push_thought(&mut self, thought: Option<String>) {
        if let Some(thought) = thought {
            self.push_turn(Turn::AssistantThought(thought));
        }
    }

    fn push_turn(&mut self, turn: Turn) {
        self.transcript.push(turn);
        self.compact_transcript();
    }

    fn compact_transcript(&mut self) {
        while self.transcript.len() > 1
            && (self.transcript.len() > MAX_STORED_TRANSCRIPT_ENTRIES
                || stored_transcript_bytes(&self.transcript) > MAX_STORED_TRANSCRIPT_BYTES)
        {
            // An observation is meaningful only together with the approved
            // proposal it reports on. Treat that adjacent pair as one oldest
            // history unit: removing just the proposal would leave a live
            // session able to snapshot an orphan observation that strict
            // restoration must reject. Other turns remain independently
            // trimmable so one large message does not evict extra context.
            let remove_count = match self.transcript.as_slice() {
                [Turn::AssistantProposed { id, .. }, Turn::Observation { proposal_id, .. }, ..]
                    if id == proposal_id =>
                {
                    2
                }
                _ => 1,
            };
            self.transcript.drain(..remove_count);
            self.transcript_truncated = true;
        }
    }

    fn ready_or_limited(&self) -> AgentState {
        if self.turns_used >= self.max_turns {
            AgentState::TurnLimitReached
        } else {
            AgentState::Ready
        }
    }

    fn check_not_cancelled(&self) -> Result<(), SessionError> {
        if self.cancelled.is_cancelled() || self.state == AgentState::Cancelled {
            Err(SessionError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn invalid_transition(&self, operation: &'static str) -> SessionError {
        SessionError::InvalidTransition {
            operation,
            state: self.state,
        }
    }
}

impl AgentSession {
    /// Capture a serializable snapshot for cross-restart persistence.
    ///
    /// Cancelled and empty sessions return `None`: there is nothing worth
    /// resuming and a cancelled token cannot be revived meaningfully.
    pub fn snapshot(&self) -> Option<AgentSessionSnapshot> {
        if self.cancelled.is_cancelled()
            || self.state == AgentState::Cancelled
            || self.transcript.is_empty()
        {
            return None;
        }
        Some(AgentSessionSnapshot {
            version: AGENT_SNAPSHOT_VERSION,
            transcript: self.transcript.clone(),
            transcript_truncated: self.transcript_truncated,
            state: self.state,
            turns_used: self.turns_used,
            max_turns: self.max_turns,
            next_proposal_id: self.next_proposal_id,
        })
    }

    /// Rebuild a session from a snapshot taken by [`Self::snapshot`].
    ///
    /// In-flight states are normalized for a world where the process died:
    /// `AwaitingModel` and `AwaitingApproval` survive as-is (the caller can
    /// re-request the model / re-render the approval card), while
    /// `AwaitingObservation` becomes a `ProtocolError` note plus
    /// Ready/TurnLimitReached — the approved command's output is gone.
    pub fn restore(snapshot: AgentSessionSnapshot) -> Result<Self, AgentSnapshotError> {
        if snapshot.version != AGENT_SNAPSHOT_VERSION {
            return Err(AgentSnapshotError::UnsupportedVersion(snapshot.version));
        }
        if snapshot.max_turns == 0 || snapshot.max_turns > MAX_SESSION_TURNS {
            return Err(AgentSnapshotError::Invalid("max_turns out of range"));
        }
        if snapshot.turns_used > snapshot.max_turns {
            return Err(AgentSnapshotError::Invalid("turns_used exceeds max_turns"));
        }
        if snapshot.transcript.is_empty() {
            return Err(AgentSnapshotError::Invalid("empty transcript"));
        }
        if snapshot.state == AgentState::Cancelled {
            return Err(AgentSnapshotError::Invalid(
                "cancelled sessions are not restorable",
            ));
        }
        let facts = validate_snapshot_transcript(&snapshot.transcript)?;
        validate_snapshot_lifecycle(&snapshot, &facts)?;
        let minimum_next_id =
            facts
                .highest_proposal_id
                .checked_add(1)
                .ok_or(AgentSnapshotError::Invalid(
                    "proposal identifier space is exhausted",
                ))?;
        let next_proposal_id = snapshot.next_proposal_id.max(minimum_next_id).max(1);
        if next_proposal_id == u64::MAX {
            return Err(AgentSnapshotError::Invalid(
                "proposal identifier space is exhausted",
            ));
        }
        let mut session = Self {
            transcript: snapshot.transcript,
            transcript_truncated: snapshot.transcript_truncated,
            state: snapshot.state,
            turns_used: snapshot.turns_used,
            max_turns: snapshot.max_turns,
            next_proposal_id,
            cancelled: CancellationToken(Arc::new(AtomicBool::new(false))),
        };
        session.compact_transcript();
        if let AgentState::AwaitingObservation { proposal_id } = session.state {
            session.push_turn(Turn::ProtocolError(format!(
                "{COMMAND_EXECUTION_DIAGNOSTIC_PREFIX}{} has an unknown result: the application \
                 exited before its output was observed; no normal exit status was available",
                proposal_id.get()
            )));
            session.state = session.ready_or_limited();
        }
        Ok(session)
    }
}

#[derive(Debug, Default)]
struct SnapshotTranscriptFacts {
    highest_proposal_id: u64,
    pending_proposal: Option<(ProposalId, usize)>,
    proposal_statuses: HashMap<ProposalId, (ProposalStatus, usize)>,
    observed_proposals: HashSet<ProposalId>,
    model_actions: u32,
    protocol_errors: u32,
}

/// Validate every invariant that public session transitions establish before
/// trusting persisted state. In particular, proposal ids are a security
/// binding between the approval card and the command returned to the caller;
/// duplicates or reordering must not be repaired heuristically.
fn validate_snapshot_transcript(
    transcript: &[Turn],
) -> Result<SnapshotTranscriptFacts, AgentSnapshotError> {
    if transcript.len() > MAX_STORED_TRANSCRIPT_ENTRIES {
        return Err(AgentSnapshotError::Invalid(
            "transcript exceeds its entry limit",
        ));
    }

    let mut facts = SnapshotTranscriptFacts::default();
    for (index, turn) in transcript.iter().enumerate() {
        match turn {
            Turn::User(message) => {
                validate_snapshot_text(message, MAX_MESSAGE_BYTES, true, "invalid user turn")?;
            }
            Turn::AssistantThought(thought) => {
                validate_snapshot_text(
                    thought,
                    MAX_THOUGHT_BYTES,
                    true,
                    "invalid assistant thought",
                )?;
            }
            Turn::AssistantSay(message) => {
                validate_snapshot_text(
                    message,
                    MAX_MESSAGE_BYTES,
                    true,
                    "invalid assistant message",
                )?;
                facts.model_actions = facts.model_actions.saturating_add(1);
            }
            Turn::AssistantProposed {
                id,
                command,
                status,
            } => {
                facts.model_actions = facts.model_actions.saturating_add(1);
                if id.get() == 0 || id.get() <= facts.highest_proposal_id {
                    return Err(AgentSnapshotError::Invalid(
                        "proposal ids are zero, duplicated, or out of order",
                    ));
                }
                if validate_command(command).is_err() {
                    return Err(AgentSnapshotError::Invalid(
                        "proposal command violates its safety bounds",
                    ));
                }
                facts.highest_proposal_id = id.get();
                facts.proposal_statuses.insert(*id, (*status, index));
                if *status == ProposalStatus::Pending
                    && facts.pending_proposal.replace((*id, index)).is_some()
                {
                    return Err(AgentSnapshotError::Invalid(
                        "snapshot contains multiple pending proposals",
                    ));
                }
            }
            Turn::Observation {
                proposal_id,
                output_sample,
                ..
            } => {
                if proposal_id.get() == 0 || output_sample.len() > MAX_OBSERVATION_BYTES {
                    return Err(AgentSnapshotError::Invalid(
                        "observation violates its safety bounds",
                    ));
                }
                let approved_immediately_before = facts
                    .proposal_statuses
                    .get(proposal_id)
                    .is_some_and(|(status, proposal_index)| {
                        *status == ProposalStatus::Approved
                            && proposal_index.checked_add(1) == Some(index)
                    });
                if !approved_immediately_before || !facts.observed_proposals.insert(*proposal_id) {
                    return Err(AgentSnapshotError::Invalid(
                        "observation does not immediately follow one previously unobserved approved proposal",
                    ));
                }
            }
            Turn::ProtocolError(message) => {
                validate_snapshot_text(
                    message,
                    MAX_MESSAGE_BYTES,
                    false,
                    "protocol error exceeds its safety bound",
                )?;
                facts.protocol_errors = facts.protocol_errors.saturating_add(1);
            }
        }
    }
    if stored_transcript_bytes(transcript) > MAX_STORED_TRANSCRIPT_BYTES {
        return Err(AgentSnapshotError::Invalid(
            "transcript exceeds its byte limit",
        ));
    }
    Ok(facts)
}

/// Revalidate the cross-field state machine invariants that no standalone
/// decoded [`AgentState`], [`ProposalStatus`], or transcript turn can prove.
///
/// In particular, the final retained turn must be the one the resumable state
/// names, the model-turn counter must be possible for the retained history,
/// and every approved proposal must have an adjacent recorded outcome. Without
/// these checks a syntactically valid persisted document could cover an older
/// approval card with newer text or silently erase what happened after a
/// reviewed command was handed to the integration.
fn validate_snapshot_lifecycle(
    snapshot: &AgentSessionSnapshot,
    facts: &SnapshotTranscriptFacts,
) -> Result<(), AgentSnapshotError> {
    if snapshot.turns_used < facts.model_actions
        || (!snapshot.transcript_truncated
            && snapshot.turns_used > facts.model_actions.saturating_add(facts.protocol_errors))
    {
        return Err(AgentSnapshotError::Invalid(
            "turn counter is inconsistent with the transcript",
        ));
    }

    let final_index = snapshot
        .transcript
        .len()
        .checked_sub(1)
        .ok_or(AgentSnapshotError::Invalid("empty transcript"))?;
    let final_turn = &snapshot.transcript[final_index];

    // A resumable approval/execution state must bind to the final card. Merely
    // finding the same id somewhere in history is insufficient: newer turns
    // would make the visible transcript disagree with the action being
    // authorized or normalized.
    match snapshot.state {
        AgentState::AwaitingApproval { proposal_id }
            if facts.pending_proposal == Some((proposal_id, final_index)) => {}
        AgentState::AwaitingApproval { .. } => {
            return Err(AgentSnapshotError::Invalid(
                "approval state does not identify the final pending proposal",
            ));
        }
        AgentState::AwaitingObservation { proposal_id }
            if facts.pending_proposal.is_none()
                && facts.proposal_statuses.get(&proposal_id)
                    == Some(&(ProposalStatus::Approved, final_index))
                && !facts.observed_proposals.contains(&proposal_id) => {}
        AgentState::AwaitingObservation { .. } => {
            return Err(AgentSnapshotError::Invalid(
                "observation state does not identify the final unobserved approved proposal",
            ));
        }
        _ if facts.pending_proposal.is_some() => {
            return Err(AgentSnapshotError::Invalid(
                "pending proposal exists outside approval state",
            ));
        }
        _ => {}
    }

    let final_state_is_valid = match snapshot.state {
        AgentState::Ready => {
            snapshot.turns_used < snapshot.max_turns
                && matches!(
                    final_turn,
                    Turn::AssistantSay(_)
                        | Turn::ProtocolError(_)
                        | Turn::AssistantProposed {
                            status: ProposalStatus::ManualReview,
                            ..
                        }
                )
        }
        AgentState::AwaitingModel => {
            snapshot.turns_used < snapshot.max_turns
                && matches!(
                    final_turn,
                    Turn::User(_)
                        | Turn::ProtocolError(_)
                        | Turn::Observation { .. }
                        | Turn::AssistantProposed {
                            status: ProposalStatus::Rejected,
                            ..
                        }
                )
        }
        // Both in-flight states were bound to the final turn above.
        AgentState::AwaitingApproval { .. } | AgentState::AwaitingObservation { .. } => true,
        AgentState::Completed => matches!(final_turn, Turn::AssistantSay(_)),
        AgentState::TurnLimitReached => {
            snapshot.turns_used == snapshot.max_turns
                && matches!(
                    final_turn,
                    Turn::AssistantSay(_)
                        | Turn::ProtocolError(_)
                        | Turn::Observation { .. }
                        | Turn::AssistantProposed {
                            status: ProposalStatus::Rejected | ProposalStatus::ManualReview,
                            ..
                        }
                )
        }
        AgentState::Cancelled => false,
    };
    if !final_state_is_valid {
        return Err(AgentSnapshotError::Invalid(
            "session state does not match the final transcript turn or turn budget",
        ));
    }

    for (proposal_id, (status, index)) in &facts.proposal_statuses {
        if *status != ProposalStatus::Approved || facts.observed_proposals.contains(proposal_id) {
            continue;
        }
        let is_current_execution = matches!(
            snapshot.state,
            AgentState::AwaitingObservation {
                proposal_id: current
            } if current == *proposal_id
        );
        if !is_current_execution
            && !is_documented_execution_result(snapshot.transcript.get(index + 1), *proposal_id)
        {
            return Err(AgentSnapshotError::Invalid(
                "approved proposal execution lifecycle is inconsistent",
            ));
        }
    }
    Ok(())
}

fn is_documented_execution_result(turn: Option<&Turn>, proposal_id: ProposalId) -> bool {
    let Some(Turn::ProtocolError(message)) = turn else {
        return false;
    };
    let proposal_id = proposal_id.get();

    // Snapshot schema version 1 spans both spellings. The older form is kept
    // for compatibility; new sessions emit the proposal-bound diagnostic form.
    if message
        == &format!(
            "the application exited before proposal #{proposal_id}'s output was observed; its \
             result is unknown"
        )
        || message
            == &format!(
                "{COMMAND_EXECUTION_DIAGNOSTIC_PREFIX}{proposal_id} has an unknown result: the \
                 application exited before its output was observed; no normal exit status was \
                 available"
            )
    {
        return true;
    }

    ["failed to start", "timed out", "was cancelled"]
        .into_iter()
        .any(|failure| {
            let prefix = format!(
                "{COMMAND_EXECUTION_DIAGNOSTIC_PREFIX}{proposal_id} {failure}; no normal exit \
                 status was available"
            );
            if message == &format!("{prefix}.") {
                return true;
            }
            let detail_prefix = format!("{prefix}. Untrusted diagnostic or partial output:\n");
            message.strip_prefix(&detail_prefix).is_some_and(|detail| {
                !detail.trim().is_empty() && detail.len() <= MAX_OBSERVATION_BYTES
            })
        })
}

fn validate_snapshot_text(
    value: &str,
    max_bytes: usize,
    require_nonempty: bool,
    reason: &'static str,
) -> Result<(), AgentSnapshotError> {
    if value.len() > max_bytes || (require_nonempty && value.trim().is_empty()) {
        return Err(AgentSnapshotError::Invalid(reason));
    }
    Ok(())
}

const AGENT_SNAPSHOT_VERSION: u32 = 1;

/// Byte cap for one encoded snapshot; larger inputs are refused rather than
/// truncated so a corrupt or hostile file cannot balloon memory.
pub const MAX_AGENT_SNAPSHOT_JSON_BYTES: usize = 256 * 1024;

/// Serializable capture of an [`AgentSession`] for cross-restart persistence.
/// Serialization is pure; where and how the JSON is stored stays with the
/// embedding application.
///
/// The type deliberately does **not** implement [`serde::Deserialize`].
/// [`Self::from_json`] is the only wire path, so no caller can reach the
/// unbounded Serde collection decoding this schema previously used.
#[derive(Debug, Clone, Serialize)]
pub struct AgentSessionSnapshot {
    version: u32,
    transcript: Vec<Turn>,
    transcript_truncated: bool,
    state: AgentState,
    turns_used: u32,
    max_turns: u32,
    next_proposal_id: u64,
}

impl AgentSessionSnapshot {
    /// Snapshot schema version carried by this decoded value.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Bounded, unmodified transcript decoded from the snapshot.
    pub fn transcript(&self) -> &[Turn] {
        &self.transcript
    }

    /// Whether older transcript entries had already been compacted away.
    pub fn transcript_truncated(&self) -> bool {
        self.transcript_truncated
    }

    /// Persisted state before [`AgentSession::restore`] normalizes any
    /// interrupted observation.
    pub fn state(&self) -> AgentState {
        self.state
    }

    /// Number of model turns charged to the persisted task.
    pub fn turns_used(&self) -> u32 {
        self.turns_used
    }

    /// Persisted model-turn ceiling.
    pub fn max_turns(&self) -> u32 {
        self.max_turns
    }

    /// Next proposal identifier recorded by the snapshot.
    pub fn next_proposal_id(&self) -> u64 {
        self.next_proposal_id
    }

    pub fn to_json(&self) -> Result<String, AgentSnapshotError> {
        let encoded = serde_json::to_string(self)
            .map_err(|error| AgentSnapshotError::Encode(error.to_string()))?;
        if encoded.len() > MAX_AGENT_SNAPSHOT_JSON_BYTES {
            return Err(AgentSnapshotError::TooLarge {
                limit: MAX_AGENT_SNAPSHOT_JSON_BYTES,
            });
        }
        Ok(encoded)
    }

    /// Decode a snapshot under the schema's own allocation budgets.
    ///
    /// The encoded envelope is bounded first, then decoding itself refuses the
    /// 129th transcript entry before constructing it, rejects unknown and
    /// duplicate fields, and charges each retained string against both its own
    /// field limit and the cumulative transcript budget. Trailing content after
    /// the snapshot object is rejected as well.
    pub fn from_json(encoded: &str) -> Result<Self, AgentSnapshotError> {
        if encoded.len() > MAX_AGENT_SNAPSHOT_JSON_BYTES {
            return Err(AgentSnapshotError::TooLarge {
                limit: MAX_AGENT_SNAPSHOT_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_str(encoded);
        let snapshot = SnapshotSeed
            .deserialize(&mut deserializer)
            .map_err(|error| AgentSnapshotError::Decode(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| AgentSnapshotError::Decode(error.to_string()))?;
        Ok(snapshot)
    }
}

/// Allocation budget shared by every seed decoding one snapshot.
///
/// [`AgentSessionSnapshot::from_json`] already refuses an oversized encoded
/// envelope, but a 256 KiB document can still describe tens of thousands of
/// tiny transcript entries or a handful of near-envelope-sized strings.
/// Ordinary Serde collection deserialization would allocate all of them and
/// only meet the transcript bounds afterwards, so the seeds below charge as
/// they decode and fail closed at the first entry that does not fit.
struct SnapshotBudget {
    remaining_text: usize,
}

impl SnapshotBudget {
    fn new() -> Self {
        Self {
            remaining_text: MAX_STORED_TRANSCRIPT_BYTES,
        }
    }

    /// Validate one string against its own field limit and the cumulative
    /// transcript budget *before* it is retained in a [`Turn`].
    fn charge<E: de::Error>(
        &mut self,
        field: &'static str,
        text: &str,
        max_bytes: usize,
    ) -> Result<(), E> {
        if text.len() > max_bytes {
            return Err(de::Error::custom(format_args!(
                "transcript field '{field}' exceeds its {max_bytes}-byte limit"
            )));
        }
        self.remaining_text = self.remaining_text.checked_sub(text.len()).ok_or_else(|| {
            de::Error::custom(format_args!(
                "transcript exceeds its {MAX_STORED_TRANSCRIPT_BYTES}-byte cumulative limit"
            ))
        })?;
        Ok(())
    }
}

/// Decode one string, rejecting it before it is owned if it does not fit.
struct BoundedText<'a> {
    budget: &'a mut SnapshotBudget,
    field: &'static str,
    max_bytes: usize,
}

impl<'de> DeserializeSeed<'de> for BoundedText<'_> {
    type Value = String;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for BoundedText<'_> {
    type Value = String;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "a '{}' string of at most {} bytes",
            self.field, self.max_bytes
        )
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        self.budget.charge(self.field, value, self.max_bytes)?;
        Ok(value.to_owned())
    }
}

#[derive(Deserialize)]
#[serde(variant_identifier)]
enum TurnTag {
    User,
    AssistantThought,
    AssistantSay,
    AssistantProposed,
    Observation,
    ProtocolError,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum ProposedField {
    Id,
    Command,
    Status,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum ObservationField {
    ProposalId,
    ExitCode,
    OutputSample,
}

#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum SnapshotField {
    Version,
    Transcript,
    TranscriptTruncated,
    State,
    TurnsUsed,
    MaxTurns,
    NextProposalId,
}

const TURN_VARIANTS: &[&str] = &[
    "User",
    "AssistantThought",
    "AssistantSay",
    "AssistantProposed",
    "Observation",
    "ProtocolError",
];
const PROPOSED_FIELDS: &[&str] = &["id", "command", "status"];
const OBSERVATION_FIELDS: &[&str] = &["proposal_id", "exit_code", "output_sample"];
const SNAPSHOT_FIELDS: &[&str] = &[
    "version",
    "transcript",
    "transcript_truncated",
    "state",
    "turns_used",
    "max_turns",
    "next_proposal_id",
];

struct TurnSeed<'a> {
    budget: &'a mut SnapshotBudget,
}

impl<'de> DeserializeSeed<'de> for TurnSeed<'_> {
    type Value = Turn;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_enum("Turn", TURN_VARIANTS, self)
    }
}

impl<'de> Visitor<'de> for TurnSeed<'_> {
    type Value = Turn;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an agent transcript turn")
    }

    fn visit_enum<A: de::EnumAccess<'de>>(self, data: A) -> Result<Self::Value, A::Error> {
        let budget = self.budget;
        let (tag, variant) = data.variant::<TurnTag>()?;
        let text = |budget, field, max_bytes| BoundedText {
            budget,
            field,
            max_bytes,
        };
        match tag {
            TurnTag::User => Ok(Turn::User(variant.newtype_variant_seed(text(
                budget,
                "User",
                MAX_MESSAGE_BYTES,
            ))?)),
            TurnTag::AssistantThought => {
                Ok(Turn::AssistantThought(variant.newtype_variant_seed(
                    text(budget, "AssistantThought", MAX_THOUGHT_BYTES),
                )?))
            }
            TurnTag::AssistantSay => Ok(Turn::AssistantSay(variant.newtype_variant_seed(text(
                budget,
                "AssistantSay",
                MAX_MESSAGE_BYTES,
            ))?)),
            TurnTag::ProtocolError => Ok(Turn::ProtocolError(
                variant.newtype_variant_seed(text(budget, "ProtocolError", MAX_MESSAGE_BYTES))?,
            )),
            TurnTag::AssistantProposed => {
                variant.struct_variant(PROPOSED_FIELDS, ProposedSeed { budget })
            }
            TurnTag::Observation => {
                variant.struct_variant(OBSERVATION_FIELDS, ObservationSeed { budget })
            }
        }
    }
}

struct ProposedSeed<'a> {
    budget: &'a mut SnapshotBudget,
}

impl<'de> Visitor<'de> for ProposedSeed<'_> {
    type Value = Turn;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a proposed-command turn")
    }

    fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let budget = self.budget;
        let mut id = None;
        let mut command = None;
        let mut status = None;
        while let Some(key) = map.next_key::<ProposedField>()? {
            match key {
                ProposedField::Id => {
                    if id.is_some() {
                        return Err(de::Error::duplicate_field("id"));
                    }
                    id = Some(map.next_value::<ProposalId>()?);
                }
                ProposedField::Command => {
                    if command.is_some() {
                        return Err(de::Error::duplicate_field("command"));
                    }
                    command = Some(map.next_value_seed(BoundedText {
                        budget: &mut *budget,
                        field: "command",
                        max_bytes: MAX_COMMAND_BYTES,
                    })?);
                }
                ProposedField::Status => {
                    if status.is_some() {
                        return Err(de::Error::duplicate_field("status"));
                    }
                    status = Some(map.next_value::<ProposalStatus>()?);
                }
            }
        }
        Ok(Turn::AssistantProposed {
            id: id.ok_or_else(|| de::Error::missing_field("id"))?,
            command: command.ok_or_else(|| de::Error::missing_field("command"))?,
            status: status.ok_or_else(|| de::Error::missing_field("status"))?,
        })
    }
}

struct ObservationSeed<'a> {
    budget: &'a mut SnapshotBudget,
}

impl<'de> Visitor<'de> for ObservationSeed<'_> {
    type Value = Turn;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an observation turn")
    }

    fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let budget = self.budget;
        let mut proposal_id = None;
        let mut exit_code = None;
        let mut output_sample = None;
        while let Some(key) = map.next_key::<ObservationField>()? {
            match key {
                ObservationField::ProposalId => {
                    if proposal_id.is_some() {
                        return Err(de::Error::duplicate_field("proposal_id"));
                    }
                    proposal_id = Some(map.next_value::<ProposalId>()?);
                }
                ObservationField::ExitCode => {
                    if exit_code.is_some() {
                        return Err(de::Error::duplicate_field("exit_code"));
                    }
                    exit_code = Some(map.next_value::<i32>()?);
                }
                ObservationField::OutputSample => {
                    if output_sample.is_some() {
                        return Err(de::Error::duplicate_field("output_sample"));
                    }
                    output_sample = Some(map.next_value_seed(BoundedText {
                        budget: &mut *budget,
                        field: "output_sample",
                        max_bytes: MAX_OBSERVATION_BYTES,
                    })?);
                }
            }
        }
        Ok(Turn::Observation {
            proposal_id: proposal_id.ok_or_else(|| de::Error::missing_field("proposal_id"))?,
            exit_code: exit_code.ok_or_else(|| de::Error::missing_field("exit_code"))?,
            output_sample: output_sample
                .ok_or_else(|| de::Error::missing_field("output_sample"))?,
        })
    }
}

struct TranscriptSeed<'a> {
    budget: &'a mut SnapshotBudget,
}

impl<'de> DeserializeSeed<'de> for TranscriptSeed<'_> {
    type Value = Vec<Turn>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for TranscriptSeed<'_> {
    type Value = Vec<Turn>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "at most {MAX_STORED_TRANSCRIPT_ENTRIES} agent transcript turns"
        )
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let budget = self.budget;
        // A hostile size hint must not preallocate on its own.
        let mut turns = Vec::with_capacity(
            seq.size_hint()
                .unwrap_or(0)
                .min(MAX_STORED_TRANSCRIPT_ENTRIES),
        );
        while turns.len() < MAX_STORED_TRANSCRIPT_ENTRIES {
            let Some(turn) = seq.next_element_seed(TurnSeed {
                budget: &mut *budget,
            })?
            else {
                return Ok(turns);
            };
            turns.push(turn);
        }
        // Detect a 129th entry without building it: `IgnoredAny` proves the
        // array is over-wide while allocating no turn at all.
        if seq.next_element::<de::IgnoredAny>()?.is_some() {
            return Err(de::Error::custom(format_args!(
                "transcript exceeds its {MAX_STORED_TRANSCRIPT_ENTRIES}-entry limit"
            )));
        }
        Ok(turns)
    }
}

/// The bounded wire decoder for [`AgentSessionSnapshot`].
struct SnapshotSeed;

impl<'de> DeserializeSeed<'de> for SnapshotSeed {
    type Value = AgentSessionSnapshot;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_struct("AgentSessionSnapshot", SNAPSHOT_FIELDS, self)
    }
}

impl<'de> Visitor<'de> for SnapshotSeed {
    type Value = AgentSessionSnapshot;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an agent session snapshot object")
    }

    fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut budget = SnapshotBudget::new();
        let mut version = None;
        let mut transcript = None;
        let mut transcript_truncated = None;
        let mut state = None;
        let mut turns_used = None;
        let mut max_turns = None;
        let mut next_proposal_id = None;
        while let Some(key) = map.next_key::<SnapshotField>()? {
            match key {
                SnapshotField::Version => {
                    if version.is_some() {
                        return Err(de::Error::duplicate_field("version"));
                    }
                    version = Some(map.next_value::<u32>()?);
                }
                SnapshotField::Transcript => {
                    if transcript.is_some() {
                        return Err(de::Error::duplicate_field("transcript"));
                    }
                    transcript = Some(map.next_value_seed(TranscriptSeed {
                        budget: &mut budget,
                    })?);
                }
                SnapshotField::TranscriptTruncated => {
                    if transcript_truncated.is_some() {
                        return Err(de::Error::duplicate_field("transcript_truncated"));
                    }
                    transcript_truncated = Some(map.next_value::<bool>()?);
                }
                SnapshotField::State => {
                    if state.is_some() {
                        return Err(de::Error::duplicate_field("state"));
                    }
                    state = Some(map.next_value::<AgentState>()?);
                }
                SnapshotField::TurnsUsed => {
                    if turns_used.is_some() {
                        return Err(de::Error::duplicate_field("turns_used"));
                    }
                    turns_used = Some(map.next_value::<u32>()?);
                }
                SnapshotField::MaxTurns => {
                    if max_turns.is_some() {
                        return Err(de::Error::duplicate_field("max_turns"));
                    }
                    max_turns = Some(map.next_value::<u32>()?);
                }
                SnapshotField::NextProposalId => {
                    if next_proposal_id.is_some() {
                        return Err(de::Error::duplicate_field("next_proposal_id"));
                    }
                    next_proposal_id = Some(map.next_value::<u64>()?);
                }
            }
        }
        Ok(AgentSessionSnapshot {
            version: version.ok_or_else(|| de::Error::missing_field("version"))?,
            transcript: transcript.ok_or_else(|| de::Error::missing_field("transcript"))?,
            transcript_truncated: transcript_truncated
                .ok_or_else(|| de::Error::missing_field("transcript_truncated"))?,
            state: state.ok_or_else(|| de::Error::missing_field("state"))?,
            turns_used: turns_used.ok_or_else(|| de::Error::missing_field("turns_used"))?,
            max_turns: max_turns.ok_or_else(|| de::Error::missing_field("max_turns"))?,
            next_proposal_id: next_proposal_id
                .ok_or_else(|| de::Error::missing_field("next_proposal_id"))?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSnapshotError {
    Encode(String),
    Decode(String),
    TooLarge { limit: usize },
    UnsupportedVersion(u32),
    Invalid(&'static str),
}

impl std::fmt::Display for AgentSnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(message) => write!(f, "encode agent snapshot: {message}"),
            Self::Decode(message) => write!(f, "decode agent snapshot: {message}"),
            Self::TooLarge { limit } => {
                write!(f, "agent snapshot exceeds the {limit}-byte safety limit")
            }
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported agent snapshot version {version}")
            }
            Self::Invalid(reason) => write!(f, "invalid agent snapshot: {reason}"),
        }
    }
}

impl std::error::Error for AgentSnapshotError {}

impl Drop for AgentSession {
    fn drop(&mut self) {
        self.cancelled.0.store(true, Ordering::SeqCst);
    }
}

fn stored_transcript_bytes(transcript: &[Turn]) -> usize {
    transcript.iter().fold(0_usize, |total, turn| {
        total.saturating_add(turn.to_prompt().len())
    })
}

pub fn sample_observation(output: &str) -> String {
    elide_middle(output, MAX_OBSERVATION_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_roundtrip_preserves_resumable_states() {
        let mut session = AgentSession::new(10);
        session.submit_user("list files").unwrap();
        session
            .accept_model_reply(r#"{"action":"run","command":"ls"}"#)
            .unwrap();
        // AwaitingApproval survives a roundtrip with the card still pending.
        let snapshot = session.snapshot().expect("live session snapshots");
        let json = snapshot.to_json().unwrap();
        let restored =
            AgentSession::restore(AgentSessionSnapshot::from_json(&json).unwrap()).unwrap();
        assert_eq!(restored.transcript(), session.transcript());
        assert_eq!(restored.turns_used(), session.turns_used());
        assert!(matches!(
            restored.state(),
            AgentState::AwaitingApproval { .. }
        ));
        // The restored session accepts the approval and continues.
        let AgentState::AwaitingApproval { proposal_id } = restored.state() else {
            unreachable!();
        };
        let mut restored = restored;
        let approved = restored.approve(proposal_id).unwrap();
        assert!(!approved.command.is_empty());
    }

    #[test]
    fn snapshot_accessors_expose_the_bounded_roundtrip_without_redecoding() {
        let mut session = AgentSession::new(7);
        session.submit_user("inspect the tree").unwrap();
        let ModelOutcome::Proposal { id, .. } = session
            .accept_model_reply(r#"{"action":"run","command":"find . -maxdepth 1"}"#)
            .unwrap()
        else {
            panic!("expected a proposal");
        };

        let encoded = session.snapshot().unwrap().to_json().unwrap();
        let snapshot = AgentSessionSnapshot::from_json(&encoded).unwrap();
        assert_eq!(snapshot.version(), AGENT_SNAPSHOT_VERSION);
        assert_eq!(snapshot.transcript(), session.transcript());
        assert!(!snapshot.transcript_truncated());
        assert_eq!(
            snapshot.state(),
            AgentState::AwaitingApproval { proposal_id: id }
        );
        assert_eq!(snapshot.turns_used(), 1);
        assert_eq!(snapshot.max_turns(), 7);
        assert_eq!(snapshot.next_proposal_id(), id.get() + 1);
    }

    #[test]
    fn allocation_free_schema_atoms_keep_their_serde_contract() {
        let role: crate::provider::Role = serde_json::from_str(r#""assistant""#).unwrap();
        assert_eq!(role, crate::provider::Role::Assistant);

        let proposal_id: ProposalId = serde_json::from_str("7").unwrap();
        assert_eq!(proposal_id.get(), 7);
        let status: ProposalStatus = serde_json::from_str(r#""ManualReview""#).unwrap();
        assert_eq!(status, ProposalStatus::ManualReview);
        let state: AgentState =
            serde_json::from_str(r#"{"AwaitingApproval":{"proposal_id":7}}"#).unwrap();
        assert_eq!(state, AgentState::AwaitingApproval { proposal_id });

        assert!(serde_json::from_str::<AgentState>(r#"{"Unknown":{}}"#).is_err());
    }

    #[test]
    fn constructor_and_restore_share_the_turn_budget_bound() {
        let mut session = AgentSession::new(u32::MAX);
        assert_eq!(session.max_turns(), MAX_SESSION_TURNS);
        session.submit_user("hello").unwrap();
        let restored = AgentSession::restore(session.snapshot().unwrap()).unwrap();
        assert_eq!(restored.max_turns(), MAX_SESSION_TURNS);

        assert_eq!(AgentSession::new(0).max_turns(), 1);
    }

    #[test]
    fn restore_normalizes_lost_observation_to_a_protocol_note() {
        let mut session = AgentSession::new(10);
        session.submit_user("run it").unwrap();
        session
            .accept_model_reply(r#"{"action":"run","command":"make"}"#)
            .unwrap();
        let AgentState::AwaitingApproval { proposal_id } = session.state() else {
            unreachable!();
        };
        let approved = session.approve(proposal_id).unwrap();
        assert_eq!(approved.command, "make");
        assert!(matches!(
            session.state(),
            AgentState::AwaitingObservation { .. }
        ));

        let snapshot = session.snapshot().unwrap();
        let restored = AgentSession::restore(snapshot).unwrap();
        assert_eq!(restored.state(), AgentState::Ready);
        assert!(matches!(
            restored.transcript().last(),
            Some(Turn::ProtocolError(note)) if note.contains("output was")
        ));
    }

    #[test]
    fn cancelled_and_empty_sessions_do_not_snapshot() {
        let session = AgentSession::new(5);
        assert!(session.snapshot().is_none(), "empty transcript");
        let mut session = AgentSession::new(5);
        session.submit_user("hello").unwrap();
        session.cancel();
        assert!(session.snapshot().is_none(), "cancelled");
    }

    #[test]
    fn restore_rejects_corrupt_snapshots() {
        assert!(matches!(
            AgentSessionSnapshot::from_json("not json"),
            Err(AgentSnapshotError::Decode(_))
        ));
        let huge = "x".repeat(MAX_AGENT_SNAPSHOT_JSON_BYTES + 1);
        assert!(matches!(
            AgentSessionSnapshot::from_json(&huge),
            Err(AgentSnapshotError::TooLarge { .. })
        ));

        let mut session = AgentSession::new(10);
        session.submit_user("hi").unwrap();
        let mut snapshot = session.snapshot().unwrap();
        snapshot.version = 99;
        assert!(matches!(
            AgentSession::restore(snapshot),
            Err(AgentSnapshotError::UnsupportedVersion(99))
        ));
        let mut snapshot = session.snapshot().unwrap();
        snapshot.turns_used = snapshot.max_turns + 1;
        assert!(matches!(
            AgentSession::restore(snapshot),
            Err(AgentSnapshotError::Invalid(_))
        ));
    }

    /// Build a syntactically valid snapshot document around `transcript`.
    fn snapshot_json(transcript: &str) -> String {
        format!(
            concat!(
                r#"{{"version":1,"transcript":[{}],"transcript_truncated":false,"#,
                r#""state":"Ready","turns_used":1,"max_turns":10,"next_proposal_id":1}}"#
            ),
            transcript
        )
    }

    fn decode_error(json: &str) -> String {
        match AgentSessionSnapshot::from_json(json) {
            Err(AgentSnapshotError::Decode(message)) => message,
            other => panic!("expected a decode error, got {other:?}"),
        }
    }

    #[test]
    fn decoding_stops_before_building_the_129th_transcript_entry() {
        let entry = r#"{"User":"x"}"#;

        let widest_accepted = vec![entry; MAX_STORED_TRANSCRIPT_ENTRIES].join(",");
        let snapshot = AgentSessionSnapshot::from_json(&snapshot_json(&widest_accepted))
            .expect("the entry limit itself still decodes");
        assert_eq!(snapshot.transcript.len(), MAX_STORED_TRANSCRIPT_ENTRIES);

        // Far more entries than the limit still fit inside the encoded
        // envelope, which is exactly the amplification the visitor prevents:
        // decoding fails without ever allocating the extra turns.
        let over_wide = vec![entry; 8 * MAX_STORED_TRANSCRIPT_ENTRIES].join(",");
        let json = snapshot_json(&over_wide);
        assert!(json.len() < MAX_AGENT_SNAPSHOT_JSON_BYTES);
        assert!(
            decode_error(&json).contains("128-entry limit"),
            "wide transcripts must be refused while decoding"
        );
    }

    #[test]
    fn decoding_charges_per_field_and_cumulative_text_budgets() {
        let oversized = "t".repeat(MAX_THOUGHT_BYTES + 1);
        let json = snapshot_json(&format!(r#"{{"AssistantThought":"{oversized}"}}"#));
        assert!(decode_error(&json).contains("'AssistantThought' exceeds"));

        let oversized = "o".repeat(MAX_OBSERVATION_BYTES + 1);
        let json = snapshot_json(&format!(
            r#"{{"Observation":{{"proposal_id":1,"exit_code":0,"output_sample":"{oversized}"}}}}"#
        ));
        assert!(decode_error(&json).contains("'output_sample' exceeds"));

        // Every individual message fits, but together they exceed what a live
        // session could ever have retained.
        let message = "m".repeat(MAX_MESSAGE_BYTES);
        let entry = format!(r#"{{"User":"{message}"}}"#);
        let entries = MAX_STORED_TRANSCRIPT_BYTES / MAX_MESSAGE_BYTES + 1;
        let json = snapshot_json(&vec![entry; entries].join(","));
        assert!(json.len() < MAX_AGENT_SNAPSHOT_JSON_BYTES);
        assert!(decode_error(&json).contains("cumulative limit"));
    }

    #[test]
    fn decoding_rejects_duplicate_unknown_and_trailing_content() {
        let valid = snapshot_json(r#"{"User":"hi"}"#);
        assert!(AgentSessionSnapshot::from_json(&valid).is_ok());

        let duplicated = valid.replace(r#""turns_used":1"#, r#""turns_used":1,"turns_used":2"#);
        assert!(decode_error(&duplicated).contains("duplicate field `turns_used`"));

        let duplicated_turn_field = snapshot_json(
            r#"{"AssistantProposed":{"id":1,"command":"ls","command":"rm -rf /","status":"Pending"}}"#,
        );
        assert!(decode_error(&duplicated_turn_field).contains("duplicate field `command`"));

        let unknown = valid.replace(r#""version":1"#, r#""version":1,"trailer":"x""#);
        assert!(decode_error(&unknown).contains("trailer"));

        let unknown_turn_field = snapshot_json(
            r#"{"Observation":{"proposal_id":1,"exit_code":0,"output_sample":"ok","extra":1}}"#,
        );
        assert!(decode_error(&unknown_turn_field).contains("extra"));

        let missing = valid.replace(r#""max_turns":10,"#, "");
        assert!(decode_error(&missing).contains("max_turns"));

        assert!(decode_error(&format!("{valid}{valid}")).contains("trailing"));
    }

    #[test]
    fn restore_repairs_a_stale_proposal_id_counter() {
        let mut session = AgentSession::new(10);
        session.submit_user("run").unwrap();
        session
            .accept_model_reply(r#"{"action":"run","command":"true"}"#)
            .unwrap();
        let mut snapshot = session.snapshot().unwrap();
        snapshot.next_proposal_id = 0;
        let mut restored = AgentSession::restore(snapshot).unwrap();
        let AgentState::AwaitingApproval { proposal_id } = restored.state() else {
            unreachable!();
        };
        restored.reject(proposal_id).unwrap();
        let outcome = restored
            .accept_model_reply(r#"{"action":"run","command":"false"}"#)
            .unwrap();
        let ModelOutcome::Proposal { id, .. } = outcome else {
            panic!("expected proposal");
        };
        assert!(id.get() > proposal_id.get(), "ids must stay unique");
    }

    fn persisted_snapshot(
        transcript: Vec<Turn>,
        state: AgentState,
        next_proposal_id: u64,
    ) -> AgentSessionSnapshot {
        let snapshot = AgentSessionSnapshot {
            version: AGENT_SNAPSHOT_VERSION,
            transcript,
            transcript_truncated: false,
            state,
            turns_used: 1,
            max_turns: 10,
            next_proposal_id,
        };
        let encoded = snapshot.to_json().expect("encode hostile fixture");
        AgentSessionSnapshot::from_json(&encoded).expect("decode hostile fixture under budgets")
    }

    #[test]
    fn restore_rejects_unknown_nonapproved_and_duplicate_observations() {
        let approved = ProposalId(1);
        let unknown = ProposalId(2);
        let proposal = Turn::AssistantProposed {
            id: approved,
            command: "printf reviewed".into(),
            status: ProposalStatus::Approved,
        };
        let observation = |proposal_id| Turn::Observation {
            proposal_id,
            exit_code: 0,
            output_sample: "ok".into(),
        };

        let unknown_observation = persisted_snapshot(
            vec![
                Turn::User("inspect".into()),
                proposal.clone(),
                observation(unknown),
            ],
            AgentState::Ready,
            3,
        );
        assert!(matches!(
            AgentSession::restore(unknown_observation),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("previously unobserved approved proposal")
        ));

        let rejected_observation = persisted_snapshot(
            vec![
                Turn::User("inspect".into()),
                Turn::AssistantProposed {
                    id: approved,
                    command: "printf rejected".into(),
                    status: ProposalStatus::Rejected,
                },
                observation(approved),
            ],
            AgentState::Ready,
            2,
        );
        assert!(matches!(
            AgentSession::restore(rejected_observation),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("previously unobserved approved proposal")
        ));

        let duplicate_observation = persisted_snapshot(
            vec![
                Turn::User("inspect".into()),
                proposal,
                observation(approved),
                observation(approved),
            ],
            AgentState::Ready,
            2,
        );
        assert!(matches!(
            AgentSession::restore(duplicate_observation),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("previously unobserved approved proposal")
        ));
    }

    #[test]
    fn restore_rejects_awaiting_observation_after_output_was_recorded() {
        let id = ProposalId(1);
        let snapshot = persisted_snapshot(
            vec![
                Turn::User("inspect".into()),
                Turn::AssistantProposed {
                    id,
                    command: "printf reviewed".into(),
                    status: ProposalStatus::Approved,
                },
                Turn::Observation {
                    proposal_id: id,
                    exit_code: 0,
                    output_sample: "done".into(),
                },
            ],
            AgentState::AwaitingObservation { proposal_id: id },
            2,
        );

        assert!(matches!(
            AgentSession::restore(snapshot),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("unobserved approved proposal")
        ));
    }

    #[test]
    fn restore_rejects_awaiting_observation_with_a_pending_proposal() {
        let approved = ProposalId(1);
        let pending = ProposalId(2);
        let mut snapshot = persisted_snapshot(
            vec![
                Turn::User("inspect".into()),
                Turn::AssistantProposed {
                    id: approved,
                    command: "printf approved".into(),
                    status: ProposalStatus::Approved,
                },
                Turn::AssistantProposed {
                    id: pending,
                    command: "printf pending".into(),
                    status: ProposalStatus::Pending,
                },
            ],
            AgentState::AwaitingObservation {
                proposal_id: approved,
            },
            3,
        );
        // Keep the independent model-turn counter valid so this fixture
        // isolates the conflicting active proposal lifecycles.
        snapshot.turns_used = 2;

        assert!(matches!(
            AgentSession::restore(snapshot),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("unobserved approved proposal")
        ));
    }

    #[test]
    fn restore_binds_resumable_state_to_the_final_transcript_turn() {
        let mut session = AgentSession::new(10);
        session.submit_user("inspect").unwrap();
        let _pending = accept_run_proposal(&mut session, "printf reviewed");
        let mut covered = session.snapshot().unwrap();

        // The id and Pending status still agree with AwaitingApproval, but a
        // newer turn covers the card. Restoring this shape would let an
        // integration authorize history rather than the visible final turn.
        covered
            .transcript
            .push(Turn::ProtocolError("cover the approval card".into()));
        assert!(matches!(
            AgentSession::restore(covered),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("final pending proposal")
        ));

        let mut completed = AgentSession::new(10);
        completed.submit_user("run").unwrap();
        let id = accept_run_proposal(&mut completed, "true");
        let _approved = completed.approve(id).unwrap();
        completed.observe(id, 0, "ok").unwrap();
        let mut completed = completed.snapshot().unwrap();
        completed.state = AgentState::Completed;
        assert!(matches!(
            AgentSession::restore(completed),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("final transcript turn")
        ));
    }

    #[test]
    fn restore_rejects_erased_or_reordered_approved_command_outcomes() {
        let id = ProposalId(1);
        let erased = persisted_snapshot(
            vec![
                Turn::User("inspect".into()),
                Turn::AssistantProposed {
                    id,
                    command: "printf reviewed".into(),
                    status: ProposalStatus::Approved,
                },
                Turn::AssistantSay("nothing happened".into()),
            ],
            AgentState::Ready,
            2,
        );
        let mut erased = erased;
        erased.turns_used = 2;
        assert!(matches!(
            AgentSession::restore(erased),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("execution lifecycle")
        ));

        let reordered = persisted_snapshot(
            vec![
                Turn::User("inspect".into()),
                Turn::AssistantProposed {
                    id,
                    command: "printf reviewed".into(),
                    status: ProposalStatus::Approved,
                },
                Turn::AssistantSay("cover the result".into()),
                Turn::Observation {
                    proposal_id: id,
                    exit_code: 0,
                    output_sample: "ok".into(),
                },
            ],
            AgentState::AwaitingModel,
            2,
        );
        let mut reordered = reordered;
        reordered.turns_used = 2;
        assert!(matches!(
            AgentSession::restore(reordered),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("immediately follow")
        ));
    }

    #[test]
    fn restore_rejects_an_impossible_model_turn_counter() {
        let mut session = AgentSession::new(10);
        session.submit_user("inspect").unwrap();
        let id = accept_run_proposal(&mut session, "true");
        let _approved = session.approve(id).unwrap();
        session.observe(id, 0, "ok").unwrap();
        let mut snapshot = session.snapshot().unwrap();
        snapshot.turns_used = 0;

        assert!(matches!(
            AgentSession::restore(snapshot),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("turn counter")
        ));
    }

    #[test]
    fn entry_compaction_keeps_proposal_observation_pairs_restorable() {
        let mut session = AgentSession::new(MAX_SESSION_TURNS);
        assert!(!session.transcript_truncated());
        session.submit_user("run repeatedly").unwrap();

        // One initial user turn plus 64 proposal/observation pairs crosses the
        // 128-entry ceiling. The next proposal used to trim only proposal #1,
        // leaving observation #1 at the front of a snapshot produced entirely
        // through the public state-machine API.
        for _ in 0..MAX_STORED_TRANSCRIPT_ENTRIES / 2 {
            let id = accept_run_proposal(&mut session, "true");
            let _approved = session.approve(id).unwrap();
            session.observe(id, 0, "ok").unwrap();
        }
        assert!(session.transcript_truncated());
        assert!(session.snapshot().unwrap().transcript_truncated);

        let _pending = accept_run_proposal(&mut session, "true");
        let snapshot = session.snapshot().unwrap();
        assert!(session.transcript_truncated());
        assert!(snapshot.transcript_truncated);
        assert!(snapshot.transcript.len() <= MAX_STORED_TRANSCRIPT_ENTRIES);
        assert!(!matches!(
            snapshot.transcript.first(),
            Some(Turn::Observation { .. })
        ));
        assert_snapshot_roundtrips(&session);
    }

    #[test]
    fn byte_compaction_keeps_public_session_snapshots_restorable() {
        let mut session = AgentSession::new(MAX_SESSION_TURNS);
        session.submit_user("run large commands").unwrap();
        let command = "x".repeat(MAX_COMMAND_BYTES);
        let output = "o".repeat(MAX_OBSERVATION_BYTES);

        // This stays far below the entry ceiling while repeatedly crossing the
        // aggregate byte ceiling. Check every state at which an embedding app
        // may persist, not only the tidy boundary after an observation arrives.
        for _ in 0..10 {
            let id = accept_run_proposal(&mut session, &command);
            assert_snapshot_roundtrips(&session);
            let _approved = session.approve(id).unwrap();
            assert_snapshot_roundtrips(&session);
            session.observe(id, 0, &output).unwrap();
            assert_snapshot_roundtrips(&session);
        }

        let snapshot = session.snapshot().unwrap();
        assert!(snapshot.transcript_truncated);
        assert!(stored_transcript_bytes(&snapshot.transcript) <= MAX_STORED_TRANSCRIPT_BYTES);
    }

    #[test]
    fn restore_rejects_duplicate_ids_that_could_misbind_approval() {
        let id = ProposalId(7);
        let snapshot = AgentSessionSnapshot {
            version: AGENT_SNAPSHOT_VERSION,
            transcript: vec![
                Turn::User("inspect".into()),
                // A corrupt snapshot could put a previously approved command
                // before the benign pending card carrying the same id. A
                // first-id lookup must never return the hidden command.
                Turn::AssistantProposed {
                    id,
                    command: "rm -rf important-data".into(),
                    status: ProposalStatus::Approved,
                },
                Turn::AssistantProposed {
                    id,
                    command: "printf reviewed".into(),
                    status: ProposalStatus::Pending,
                },
            ],
            transcript_truncated: false,
            state: AgentState::AwaitingApproval { proposal_id: id },
            turns_used: 1,
            max_turns: 10,
            next_proposal_id: 8,
        };

        assert!(matches!(
            AgentSession::restore(snapshot),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("duplicated")
        ));
    }

    #[test]
    fn approval_lookup_requires_the_pending_status_as_well_as_the_id() {
        let id = ProposalId(3);
        let mut session = AgentSession {
            transcript: vec![
                Turn::AssistantProposed {
                    id,
                    command: "rm -rf important-data".into(),
                    status: ProposalStatus::Approved,
                },
                Turn::AssistantProposed {
                    id,
                    command: "printf reviewed".into(),
                    status: ProposalStatus::Pending,
                },
            ],
            transcript_truncated: false,
            state: AgentState::AwaitingApproval { proposal_id: id },
            turns_used: 1,
            max_turns: 10,
            next_proposal_id: 4,
            cancelled: CancellationToken(Arc::new(AtomicBool::new(false))),
        };

        let approved = session.approve(id).unwrap();
        assert_eq!(approved.command, "printf reviewed");
    }

    #[test]
    fn restore_revalidates_active_state_commands_and_identifier_space() {
        let mut session = AgentSession::new(10);
        session.submit_user("run").unwrap();
        session
            .accept_model_reply(r#"{"action":"run","command":"true"}"#)
            .unwrap();

        let mut wrong_status = session.snapshot().unwrap();
        let Turn::AssistantProposed { status, .. } = wrong_status.transcript.last_mut().unwrap()
        else {
            unreachable!();
        };
        *status = ProposalStatus::Approved;
        assert!(matches!(
            AgentSession::restore(wrong_status),
            Err(AgentSnapshotError::Invalid(_))
        ));

        let mut hidden_input = session.snapshot().unwrap();
        let Turn::AssistantProposed { command, .. } = hidden_input.transcript.last_mut().unwrap()
        else {
            unreachable!();
        };
        *command = "true\nrm -rf important-data".into();
        assert!(matches!(
            AgentSession::restore(hidden_input),
            Err(AgentSnapshotError::Invalid(_))
        ));

        let mut visually_spoofed = session.snapshot().unwrap();
        let Turn::AssistantProposed { command, .. } =
            visually_spoofed.transcript.last_mut().unwrap()
        else {
            unreachable!();
        };
        *command = "printf safe\u{202e}; rm -rf important".into();
        assert!(matches!(
            AgentSession::restore(visually_spoofed),
            Err(AgentSnapshotError::Invalid(_))
        ));

        let mut exhausted = session.snapshot().unwrap();
        let Turn::AssistantProposed { id, .. } = exhausted.transcript.last_mut().unwrap() else {
            unreachable!();
        };
        *id = ProposalId(u64::MAX);
        exhausted.state = AgentState::AwaitingApproval { proposal_id: *id };
        assert!(matches!(
            AgentSession::restore(exhausted),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("identifier space")
        ));

        let mut exhausted_counter = session.snapshot().unwrap();
        exhausted_counter.next_proposal_id = u64::MAX;
        assert!(matches!(
            AgentSession::restore(exhausted_counter),
            Err(AgentSnapshotError::Invalid(reason))
                if reason.contains("identifier space")
        ));
    }

    fn run_reply(command: &str) -> String {
        serde_json::json!({"action":"run", "command": command}).to_string()
    }

    fn accept_run_proposal(session: &mut AgentSession, command: &str) -> ProposalId {
        let ModelOutcome::Proposal { id, .. } = session
            .accept_model_reply(&run_reply(command))
            .expect("accept a valid run proposal")
        else {
            panic!("valid run action must produce a proposal");
        };
        id
    }

    fn assert_snapshot_roundtrips(session: &AgentSession) {
        let encoded = session
            .snapshot()
            .expect("live session has a snapshot")
            .to_json()
            .expect("encode live snapshot");
        let decoded = AgentSessionSnapshot::from_json(&encoded).expect("decode live snapshot");
        AgentSession::restore(decoded).expect("a live session's snapshot must remain restorable");
    }

    fn protocol_error_padding_with_prompt_bytes(mut bytes: usize) -> Vec<Turn> {
        let overhead = Turn::ProtocolError(String::new()).to_prompt().len();
        let mut turns = Vec::new();
        while bytes > 0 {
            assert!(bytes >= overhead, "padding target must fit one turn");
            let mut payload_bytes = bytes.saturating_sub(overhead).min(MAX_MESSAGE_BYTES);
            let remainder = bytes - overhead - payload_bytes;
            if remainder > 0 && remainder < overhead {
                // Leave exactly enough for one final diagnostic wrapper rather
                // than an unrepresentable fragment smaller than its framing.
                let shifted = overhead - remainder;
                assert!(payload_bytes >= shifted);
                payload_bytes -= shifted;
            }
            let turn = Turn::ProtocolError("x".repeat(payload_bytes));
            let cost = turn.to_prompt().len();
            assert!(cost <= bytes);
            turns.push(turn);
            bytes -= cost;
        }
        turns
    }

    fn full_budget_pending_session(command: &str) -> (AgentSession, ProposalId) {
        let id = ProposalId(1);
        let proposal = Turn::AssistantProposed {
            id,
            command: command.into(),
            status: ProposalStatus::Pending,
        };
        let proposal_bytes = proposal.to_prompt().len();
        let mut transcript =
            protocol_error_padding_with_prompt_bytes(MAX_STORED_TRANSCRIPT_BYTES - proposal_bytes);
        transcript.push(proposal);
        assert_eq!(
            stored_transcript_bytes(&transcript),
            MAX_STORED_TRANSCRIPT_BYTES
        );
        let snapshot = AgentSessionSnapshot {
            version: AGENT_SNAPSHOT_VERSION,
            transcript,
            transcript_truncated: false,
            state: AgentState::AwaitingApproval { proposal_id: id },
            turns_used: 1,
            max_turns: 10,
            next_proposal_id: 2,
        };
        let session = AgentSession::restore(snapshot).expect("full-budget fixture is valid");
        assert!(!session.transcript_truncated());
        (session, id)
    }

    #[test]
    fn strict_parser_accepts_only_action_specific_schema() {
        assert_eq!(
            parse_action(r#"{"action":"say","message":"which repo?"}"#).unwrap(),
            ParsedAction::Say {
                thought: None,
                message: "which repo?".into()
            }
        );
        assert!(matches!(
            parse_action(r#"{"action":"run","command":"ls","message":"extra"}"#),
            Err(ParseError::UnexpectedField(_))
        ));
        assert!(matches!(
            parse_action("not json"),
            Err(ParseError::InvalidJson(_))
        ));
        assert!(matches!(
            parse_action(r#"{"action":"run","command":""}"#),
            Err(ParseError::EmptyField("command"))
        ));
        assert!(matches!(
            parse_action("{\"action\":\"run\",\"command\":\"printf ok\\nwhoami\"}"),
            Err(ParseError::InvalidCommand(_))
        ));
        assert!(matches!(
            parse_action("{\"action\":\"run\",\"command\":\"printf\\tok\"}"),
            Err(ParseError::InvalidCommand(_))
        ));
        for hidden in [
            '\u{202e}',
            '\u{2066}',
            '\u{200b}',
            '\u{2028}',
            '\u{feff}',
            '\u{00a0}',
            '\u{2003}',
            '\u{034f}',
            '\u{fe0f}',
            '\u{e0020}',
        ] {
            let reply = serde_json::json!({
                "action": "run",
                "command": format!("printf safe{hidden}; rm -rf important"),
            })
            .to_string();
            assert!(matches!(
                parse_action(&reply),
                Err(ParseError::InvalidCommand(message))
                    if message.contains("invisible or bidirectional")
            ));
        }
        assert!(parse_action(
            &serde_json::json!({"action": "run", "command": "printf '编译🙂'"}).to_string()
        )
        .is_ok());
    }

    #[test]
    fn run_action_rejects_unassigned_tag_plane_and_bidi_escapes() {
        // `validate_command` is the only gate between a model reply and an
        // approval card, so it has to refuse code points that are invisible
        // without being controls. U+E0000 and U+E0080 are unassigned, which
        // means `char::is_control` is false and they are not whitespace: an
        // enumeration of only the assigned tag characters let
        // `{"run": "ls -la /etc\u{E0000}"}` through.
        for hidden in [
            '\u{e0000}',
            '\u{e0002}',
            '\u{e001f}',
            '\u{e0080}',
            '\u{e00ff}',
            '\u{e01f0}',
            '\u{e0fff}',
            '\u{fff0}',
            '\u{fff8}',
            '\u{202a}',
            '\u{202c}',
            '\u{202d}',
            '\u{202e}',
            '\u{2068}',
            '\u{2069}',
        ] {
            let command = format!("ls -la /etc{hidden}");
            assert!(
                matches!(
                    validate_command(&command),
                    Err(ParseError::InvalidCommand(ref message))
                        if message.contains("invisible or bidirectional")
                ),
                "validate_command accepted U+{:04X}",
                hidden as u32
            );
            let reply = serde_json::json!({"action": "run", "command": command}).to_string();
            assert!(
                matches!(
                    parse_action(&reply),
                    Err(ParseError::InvalidCommand(ref message))
                        if message.contains("invisible or bidirectional")
                ),
                "parse_action accepted U+{:04X}",
                hidden as u32
            );
        }
        // The widened ranges must not swallow assigned neighbours: a command
        // that legitimately spells them still parses.
        for visible in ['\u{fff9}', '\u{fffb}', '\u{13430}'] {
            let reply = serde_json::json!({
                "action": "run",
                "command": format!("printf '%s' '{visible}'"),
            })
            .to_string();
            assert!(
                parse_action(&reply).is_ok(),
                "parse_action rejected legitimate U+{:04X}",
                visible as u32
            );
        }
    }

    #[test]
    fn action_protocol_rejects_duplicate_json_members_in_both_carriers() {
        let text = r#"{"action":"run","command":"printf safe","command":"rm -rf important"}"#;
        let Err(ParseError::InvalidJson(message)) = parse_action(text) else {
            panic!("duplicate text-protocol command must fail as invalid JSON");
        };
        assert!(message.contains("duplicate JSON object member"));
        assert!(!message.contains("command"));

        // JSON escape decoding happens before member comparison, so two wire
        // spellings cannot bypass the same rule in native tool arguments.
        let native = r#"{"command":"printf safe","\u0063ommand":"rm -rf important"}"#;
        let Err(ParseError::InvalidJson(message)) = parse_tool_action("run", native) else {
            panic!("duplicate native-tool command must fail as invalid JSON");
        };
        assert!(message.contains("duplicate JSON object member"));
        assert!(!message.contains("command"));
    }

    #[test]
    fn parser_tolerates_one_json_fence_but_no_prose() {
        let parsed =
            parse_action("```json\n{\"action\":\"done\",\"message\":\"ok\"}\n```").unwrap();
        assert!(matches!(parsed, ParsedAction::Done { .. }));
        assert!(parse_action("result: {\"action\":\"done\",\"message\":\"ok\"}").is_err());
        assert!(parse_action("```text\n{}\n```").is_err());
    }

    #[test]
    fn action_parser_bounds_raw_json_and_reported_unknown_keys() {
        let oversized = format!(
            r#"{{"action":"say","message":"ok","padding":"{}"}}"#,
            "x".repeat(MAX_ACTION_JSON_BYTES)
        );
        assert_eq!(
            parse_action(&oversized),
            Err(ParseError::FieldTooLarge("reply"))
        );

        let unknown = "x".repeat(MAX_TOOL_NAME_BYTES * 4);
        let reply = format!(r#"{{"action":"say","message":"ok","{unknown}":true}}"#);
        let Err(ParseError::UnexpectedField(reported)) = parse_action(&reply) else {
            panic!("expected the unknown field to fail closed");
        };
        assert!(reported.len() <= MAX_TOOL_NAME_BYTES);
        assert!(reported.contains("bytes elided"));
    }

    #[test]
    fn approval_is_explicit_and_observation_advances_session() {
        let mut session = AgentSession::new(4);
        session.submit_user("show files").unwrap();
        let outcome = session.accept_model_reply(&run_reply("ls -la")).unwrap();
        let ModelOutcome::Proposal { id, .. } = outcome else {
            panic!("expected proposal")
        };
        assert_eq!(
            session.state(),
            AgentState::AwaitingApproval { proposal_id: id }
        );
        let approved = session.approve(id).unwrap();
        assert_eq!(approved.command, "ls -la");
        assert_eq!(
            session.state(),
            AgentState::AwaitingObservation { proposal_id: id }
        );
        session.observe(id, 0, "a\nb").unwrap();
        assert_eq!(session.state(), AgentState::AwaitingModel);
        assert!(matches!(
            session.transcript().last(),
            Some(Turn::Observation { .. })
        ));
    }

    #[test]
    fn execution_failures_advance_without_inventing_an_exit_status() {
        for (failure, description) in [
            (CommandExecutionFailure::FailedToStart, "failed to start"),
            (CommandExecutionFailure::TimedOut, "timed out"),
            (CommandExecutionFailure::Cancelled, "was cancelled"),
        ] {
            let mut session = AgentSession::new(4);
            session.submit_user("inspect").unwrap();
            let id = accept_run_proposal(&mut session, "long-running-command");
            let _approved = session.approve(id).unwrap();

            let outcome = CommandExecutionOutcome::failed(failure, "partial output");
            assert_eq!(outcome.exit_code(), None);
            assert_eq!(outcome.failure(), Some(failure));
            assert_eq!(outcome.evidence(), "partial output");
            session.observe_execution(id, outcome).unwrap();
            assert_eq!(session.state(), AgentState::AwaitingModel);
            assert!(matches!(
                session.transcript().last(),
                Some(Turn::ProtocolError(message))
                    if message.contains(description)
                        && message.contains("no normal exit status")
                        && message.contains("partial output")
            ));
            let prompt = session.build_user_prompt();
            assert!(prompt.contains("previous Agent turn failed"));
            assert!(prompt.contains("no normal exit status was available"));
            assert!(!prompt.contains("Output (exit="));
            assert_snapshot_roundtrips(&session);
        }
    }

    #[test]
    fn typed_normal_exit_preserves_real_status_and_restores() {
        let mut session = AgentSession::new(4);
        session.submit_user("inspect").unwrap();
        let id = accept_run_proposal(&mut session, "false");
        let _approved = session.approve(id).unwrap();

        let outcome = CommandExecutionOutcome::exited(23, "checked output");
        assert_eq!(outcome.exit_code(), Some(23));
        assert_eq!(outcome.failure(), None);
        assert_eq!(outcome.evidence(), "checked output");
        session.observe_execution(id, outcome).unwrap();

        assert!(matches!(
            session.transcript().last(),
            Some(Turn::Observation {
                proposal_id,
                exit_code: 23,
                output_sample,
            }) if *proposal_id == id && output_sample == "checked output"
        ));
        assert_snapshot_roundtrips(&session);
    }

    #[test]
    fn execution_failure_detail_is_bounded_utf8_safe_and_untrusted() {
        let mut session = AgentSession::new(4);
        session.submit_user("inspect").unwrap();
        let id = accept_run_proposal(&mut session, "slow-command");
        let _approved = session.approve(id).unwrap();

        let detail = "部分输出🙂".repeat(MAX_OBSERVATION_BYTES);
        session
            .observe_execution_failure(id, CommandExecutionFailure::TimedOut, &detail)
            .unwrap();
        let Some(Turn::ProtocolError(message)) = session.transcript().last() else {
            panic!("execution failure must be retained as a bounded diagnostic");
        };
        assert!(message.contains("Untrusted diagnostic or partial output"));
        assert!(message.contains("bytes elided"));
        assert!(message.len() <= MAX_MESSAGE_BYTES);
        assert_snapshot_roundtrips(&session);
    }

    #[test]
    fn execution_failure_is_proposal_bound_and_atomic_on_error() {
        let mut session = AgentSession::new(4);
        session.submit_user("inspect").unwrap();
        let id = accept_run_proposal(&mut session, "pwd");
        let _approved = session.approve(id).unwrap();
        let state_before = session.state();
        let transcript_before = session.transcript().to_vec();
        let stale = ProposalId(id.get() + 1);

        assert_eq!(
            session.observe_execution(
                stale,
                CommandExecutionOutcome::failed(
                    CommandExecutionFailure::FailedToStart,
                    "not found",
                ),
            ),
            Err(SessionError::StaleProposal {
                expected: id,
                received: stale,
            })
        );
        assert_eq!(session.state(), state_before);
        assert_eq!(session.transcript(), transcript_before);

        session
            .observe_execution_failure(id, CommandExecutionFailure::FailedToStart, "not found")
            .unwrap();
        assert!(matches!(
            session.observe_execution_failure(id, CommandExecutionFailure::Cancelled, ""),
            Err(SessionError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn model_transport_failure_does_not_erase_execution_failure_context() {
        let mut session = AgentSession::new(4);
        session.submit_user("inspect").unwrap();
        let id = accept_run_proposal(&mut session, "missing-command");
        let _approved = session.approve(id).unwrap();
        session
            .observe_execution_failure(
                id,
                CommandExecutionFailure::FailedToStart,
                "executable not found",
            )
            .unwrap();

        session.model_failed("provider unavailable").unwrap();
        assert_eq!(session.state(), AgentState::Ready);
        let prompt = session.build_user_prompt();
        assert!(prompt.contains("no normal exit status was available"));
        assert!(prompt.contains("provider unavailable"));
        assert!(session.can_retry_model());
        assert_snapshot_roundtrips(&session);
    }

    #[test]
    fn pending_proposal_is_a_state_bound_borrowed_view() {
        let mut session = AgentSession::new(4);
        assert_eq!(session.pending_proposal(), None);

        session.submit_user("show files").unwrap();
        assert_eq!(session.pending_proposal(), None);
        let id = accept_run_proposal(&mut session, "ls -la");
        assert_eq!(
            session.pending_proposal(),
            Some(PendingProposal {
                id,
                command: "ls -la",
            })
        );

        let snapshot = session.snapshot().unwrap();
        let restored = AgentSession::restore(snapshot).unwrap();
        assert_eq!(
            restored.pending_proposal(),
            Some(PendingProposal {
                id,
                command: "ls -la",
            })
        );

        session.reject(id).unwrap();
        assert_eq!(session.pending_proposal(), None);
    }

    #[test]
    fn edit_and_approve_returns_only_edited_command() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let ModelOutcome::Proposal { id, .. } =
            session.accept_model_reply(&run_reply("rm -rf /")).unwrap()
        else {
            panic!("expected proposal")
        };
        let approved = session.edit_and_approve(id, "  ls /  ").unwrap();
        assert_eq!(approved.command, "  ls /  ");
        assert!(approved.danger.is_none());
    }

    #[test]
    fn edited_proposal_cannot_hide_additional_pty_input() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let ModelOutcome::Proposal { id, .. } =
            session.accept_model_reply(&run_reply("pwd")).unwrap()
        else {
            panic!("expected proposal")
        };

        assert!(matches!(
            session.edit_and_approve(id, "pwd\nwhoami"),
            Err(SessionError::Protocol(ParseError::InvalidCommand(_)))
        ));
        assert!(matches!(
            session.edit_and_approve(id, "pwd\t--help"),
            Err(SessionError::Protocol(ParseError::InvalidCommand(_)))
        ));
        assert_eq!(
            session.state(),
            AgentState::AwaitingApproval { proposal_id: id }
        );
    }

    #[test]
    fn edited_command_recomputes_risk_before_execution_handoff() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let ModelOutcome::Proposal { id, danger, .. } = session
            .accept_model_reply(&run_reply("git status"))
            .unwrap()
        else {
            panic!("expected proposal")
        };
        assert!(danger.is_none());

        let approved = session
            .edit_and_approve(id, "git reset --hard HEAD~1")
            .unwrap();
        assert!(approved.danger.is_some());
    }

    #[test]
    fn in_place_proposal_transitions_recompact_before_snapshotting() {
        let (mut approved, approved_id) = full_budget_pending_session("true");
        let _approved_command = approved.approve(approved_id).unwrap();
        assert!(approved.transcript_truncated());
        assert!(stored_transcript_bytes(approved.transcript()) <= MAX_STORED_TRANSCRIPT_BYTES);
        assert_snapshot_roundtrips(&approved);

        let (mut edited, edited_id) = full_budget_pending_session("true");
        let _edited_command = edited
            .edit_and_approve(edited_id, "x".repeat(MAX_COMMAND_BYTES))
            .unwrap();
        assert!(edited.transcript_truncated());
        assert!(stored_transcript_bytes(edited.transcript()) <= MAX_STORED_TRANSCRIPT_BYTES);
        assert_snapshot_roundtrips(&edited);

        let (mut rejected, rejected_id) = full_budget_pending_session("true");
        rejected.reject(rejected_id).unwrap();
        assert!(rejected.transcript_truncated());
        assert!(stored_transcript_bytes(rejected.transcript()) <= MAX_STORED_TRANSCRIPT_BYTES);
        assert_snapshot_roundtrips(&rejected);

        let (mut manual, manual_id) = full_budget_pending_session("true");
        manual
            .edit_for_manual_review(manual_id, "x".repeat(MAX_COMMAND_BYTES))
            .unwrap();
        assert!(manual.transcript_truncated());
        assert!(stored_transcript_bytes(manual.transcript()) <= MAX_STORED_TRANSCRIPT_BYTES);
        assert_snapshot_roundtrips(&manual);
    }

    #[test]
    fn rejection_is_recorded_and_requests_an_alternative() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let ModelOutcome::Proposal { id, .. } =
            session.accept_model_reply(&run_reply("find /")).unwrap()
        else {
            panic!("expected proposal")
        };
        session.reject(id).unwrap();
        assert_eq!(session.state(), AgentState::AwaitingModel);
        assert!(session.build_user_prompt().contains("user rejected"));
        assert!(matches!(
            session.transcript().last(),
            Some(Turn::AssistantProposed {
                status: ProposalStatus::Rejected,
                ..
            })
        ));
        assert_eq!(session.pending_proposal(), None);
        assert!(session.approve(id).is_err());
    }

    #[test]
    fn rejection_feedback_is_validated_atomically_and_sent_to_the_model() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let id = accept_run_proposal(&mut session, "find /");
        let state_before = session.state();
        let transcript_before = session.transcript().to_vec();

        assert_eq!(
            session.reject_with_feedback(id, " \n\t "),
            Err(SessionError::EmptyUserMessage)
        );
        assert_eq!(session.state(), state_before);
        assert_eq!(session.transcript(), transcript_before);

        assert_eq!(
            session.reject_with_feedback(id, "x".repeat(MAX_MESSAGE_BYTES + 1)),
            Err(SessionError::UserMessageTooLarge)
        );
        assert_eq!(session.state(), state_before);
        assert_eq!(session.transcript(), transcript_before);
        assert_eq!(
            session.pending_proposal(),
            Some(PendingProposal {
                id,
                command: "find /",
            })
        );

        session
            .reject_with_feedback(id, "  stay inside the repository  ")
            .unwrap();
        assert_eq!(session.state(), AgentState::AwaitingModel);
        assert!(matches!(
            session.transcript().get(session.transcript().len() - 2),
            Some(Turn::AssistantProposed {
                status: ProposalStatus::Rejected,
                ..
            })
        ));
        assert_eq!(
            session.transcript().last(),
            Some(&Turn::User("stay inside the repository".into()))
        );
        let prompt = session.build_user_prompt();
        assert!(prompt.contains("user rejected"));
        assert!(prompt.contains("User: stay inside the repository"));
        assert_snapshot_roundtrips(&session);
    }

    #[test]
    fn invalid_rejection_feedback_is_checked_before_proposal_identity() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let id = accept_run_proposal(&mut session, "pwd");
        let stale = ProposalId(id.get() + 1);

        assert_eq!(
            session.reject_with_feedback(stale, ""),
            Err(SessionError::EmptyUserMessage)
        );
        assert_eq!(
            session.pending_proposal(),
            Some(PendingProposal { id, command: "pwd" })
        );
    }

    #[test]
    fn manual_review_records_non_execution_and_returns_to_user_control() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let ModelOutcome::Proposal { id, .. } = session
            .accept_model_reply(&run_reply("find . -maxdepth 1"))
            .unwrap()
        else {
            panic!("expected proposal")
        };

        let command = session
            .edit_for_manual_review(id, "  find . -maxdepth 2  ")
            .unwrap();
        assert_eq!(command, "  find . -maxdepth 2  ");
        assert_eq!(session.state(), AgentState::Ready);
        let prompt = session.build_user_prompt();
        assert!(prompt.contains("manual review"));
        assert!(prompt.contains("it was not executed"));
        assert!(session.approve(id).is_err());
    }

    #[test]
    fn manual_review_rejects_hidden_submission_bytes() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let ModelOutcome::Proposal { id, .. } =
            session.accept_model_reply(&run_reply("pwd")).unwrap()
        else {
            panic!("expected proposal")
        };

        assert!(matches!(
            session.edit_for_manual_review(id, "pwd\rwhoami"),
            Err(SessionError::Protocol(ParseError::InvalidCommand(_)))
        ));
        assert_eq!(
            session.state(),
            AgentState::AwaitingApproval { proposal_id: id }
        );
    }

    #[test]
    fn stale_observation_and_out_of_order_actions_fail() {
        let mut session = AgentSession::new(3);
        assert!(session.approve(ProposalId(1)).is_err());
        session.submit_user("inspect").unwrap();
        let ModelOutcome::Proposal { id, .. } =
            session.accept_model_reply(&run_reply("pwd")).unwrap()
        else {
            panic!("expected proposal")
        };
        let _approved = session.approve(id).unwrap();
        assert!(matches!(
            session.observe(ProposalId(id.get() + 1), 0, "wrong"),
            Err(SessionError::StaleProposal { .. })
        ));
    }

    #[test]
    fn malformed_reply_never_becomes_a_proposal() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        assert!(matches!(
            session.accept_model_reply("run: rm -rf /"),
            Err(SessionError::Protocol(_))
        ));
        assert_eq!(session.state(), AgentState::Ready);
        assert!(!session
            .transcript()
            .iter()
            .any(|turn| matches!(turn, Turn::AssistantProposed { .. })));
        let prompt = session.build_user_prompt();
        assert!(prompt.contains("previous Agent turn failed"));
        assert!(!prompt.contains("violated the JSON protocol"));
    }

    #[test]
    fn failed_model_turn_can_retry_without_duplicate_user_input() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        session.model_failed("temporary network error").unwrap();
        assert!(session.can_retry_model());
        let prompt = session.build_user_prompt();
        assert!(prompt.contains("previous Agent turn failed"));
        assert!(prompt.contains("temporary network error"));
        assert!(!prompt.contains("violated the JSON protocol"));
        let transcript_len = session.transcript().len();

        session.retry_model().unwrap();
        assert_eq!(session.state(), AgentState::AwaitingModel);
        assert_eq!(session.transcript().len(), transcript_len);
        assert!(!session.can_retry_model());
    }

    #[test]
    fn repeated_transport_retries_replace_the_previous_failure() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        for index in 0..100 {
            session
                .model_failed(format!(
                    "temporary network error {index} {}",
                    "x".repeat(32 * 1024)
                ))
                .unwrap();
            assert!(session.can_retry_model());
            if index < 99 {
                session.retry_model().unwrap();
            }
        }

        assert_eq!(session.transcript().len(), 2);
        assert!(session.build_user_prompt().len() <= MAX_TRANSCRIPT_BYTES + 128);
        assert!(session.build_user_prompt().contains("network error 99"));
    }

    #[test]
    fn revised_instructions_cannot_grow_failed_session_without_bound() {
        let mut session = AgentSession::new(3);
        for index in 0..300 {
            session
                .submit_user(format!("revision {index} {}", "x".repeat(1024)))
                .unwrap();
            session
                .model_failed(format!("provider unavailable {index}"))
                .unwrap();
        }

        assert!(session.transcript().len() <= MAX_STORED_TRANSCRIPT_ENTRIES);
        assert!(
            stored_transcript_bytes(session.transcript()) <= MAX_STORED_TRANSCRIPT_BYTES,
            "stored Agent transcript exceeded its byte budget"
        );
        let prompt = session.build_user_prompt();
        assert!(prompt.contains("older Agent activity was omitted"));
        assert!(prompt.contains("provider unavailable 299"));
    }

    #[test]
    fn oversized_user_message_is_rejected_without_starting_a_turn() {
        let mut session = AgentSession::new(3);
        assert_eq!(
            session.submit_user("界".repeat(MAX_MESSAGE_BYTES)),
            Err(SessionError::UserMessageTooLarge)
        );
        assert_eq!(session.state(), AgentState::Ready);
        assert!(session.transcript().is_empty());
    }

    #[test]
    fn successful_turn_is_not_retryable_as_a_failure() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        assert!(matches!(
            session
                .accept_model_reply(r#"{"action":"say","message":"ready"}"#)
                .unwrap(),
            ModelOutcome::Said(_)
        ));
        assert!(!session.can_retry_model());
        assert!(session.retry_model().is_err());
    }

    #[test]
    fn completed_task_can_reopen_for_a_context_preserving_follow_up() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        assert!(matches!(
            session
                .accept_model_reply(r#"{"action":"done","message":"inspection complete"}"#)
                .unwrap(),
            ModelOutcome::Completed(_)
        ));
        let transcript_len = session.transcript().len();
        assert!(session.can_continue_after_completion());

        session.continue_after_completion().unwrap();
        assert_eq!(session.state(), AgentState::Ready);
        assert_eq!(session.transcript().len(), transcript_len);
        session.submit_user("now show a concise summary").unwrap();
        let prompt = session.build_user_prompt();
        assert!(prompt.contains("inspection complete"));
        assert!(prompt.contains("now show a concise summary"));
    }

    #[test]
    fn terminal_session_can_start_a_fresh_task_with_a_reset_budget() {
        let mut session = AgentSession::new(1);
        session.submit_user("inspect").unwrap();
        session
            .accept_model_reply(r#"{"action":"say","message":"one turn used"}"#)
            .unwrap();
        assert_eq!(session.state(), AgentState::TurnLimitReached);
        assert_eq!(session.turns_used(), 1);
        assert!(!session.transcript().is_empty());
        let old_task_token = session.cancellation_token();

        session.start_new_task().unwrap();
        assert!(old_task_token.is_cancelled());
        assert!(!session.cancellation_token().is_cancelled());
        assert_eq!(session.state(), AgentState::Ready);
        assert_eq!(session.turns_used(), 0);
        assert_eq!(session.max_turns(), 1);
        assert!(session.transcript().is_empty());
        session.submit_user("fresh task").unwrap();
    }

    #[test]
    fn task_reset_never_reuses_an_old_approval_binding() {
        let mut session = AgentSession::new(1);
        session.submit_user("first task").unwrap();
        let ModelOutcome::Proposal { id: old_id, .. } = session
            .accept_model_reply(&run_reply("printf old"))
            .unwrap()
        else {
            panic!("expected first proposal")
        };
        session.reject(old_id).unwrap();
        assert_eq!(session.state(), AgentState::TurnLimitReached);

        session.start_new_task().unwrap();
        session.submit_user("second task").unwrap();
        let ModelOutcome::Proposal { id: new_id, .. } = session
            .accept_model_reply(&run_reply("rm -rf important-data"))
            .unwrap()
        else {
            panic!("expected second proposal")
        };
        assert!(new_id.get() > old_id.get());

        assert_eq!(
            session.approve(old_id),
            Err(SessionError::StaleProposal {
                expected: new_id,
                received: old_id,
            })
        );
        assert_eq!(
            session.state(),
            AgentState::AwaitingApproval {
                proposal_id: new_id
            }
        );
    }

    #[test]
    fn active_task_cannot_be_reset_or_reopened_accidentally() {
        let mut session = AgentSession::new(3);
        assert!(session.start_new_task().is_err());
        assert!(session.continue_after_completion().is_err());
        session.submit_user("inspect").unwrap();
        assert!(session.start_new_task().is_err());
    }

    #[test]
    fn turn_cap_allows_final_observation_then_seals() {
        let mut session = AgentSession::new(1);
        session.submit_user("pwd").unwrap();
        let ModelOutcome::Proposal { id, .. } =
            session.accept_model_reply(&run_reply("pwd")).unwrap()
        else {
            panic!("expected proposal")
        };
        let _approved = session.approve(id).unwrap();
        session.observe(id, 0, "/tmp").unwrap();
        assert_eq!(session.state(), AgentState::TurnLimitReached);
        assert!(matches!(
            session.submit_user("again"),
            Err(SessionError::TurnLimitReached)
        ));
    }

    #[test]
    fn turn_cap_allows_final_execution_failure_then_seals() {
        let mut session = AgentSession::new(1);
        session.submit_user("run once").unwrap();
        let id = accept_run_proposal(&mut session, "missing-command");
        let _approved = session.approve(id).unwrap();
        session
            .observe_execution_failure(
                id,
                CommandExecutionFailure::FailedToStart,
                "executable was not found",
            )
            .unwrap();

        assert_eq!(session.state(), AgentState::TurnLimitReached);
        assert!(session
            .build_user_prompt()
            .contains("no normal exit status was available"));
        assert_snapshot_roundtrips(&session);
    }

    #[test]
    fn cancellation_token_and_state_are_immediate() {
        let mut session = AgentSession::new(3);
        let token = session.cancellation_token();
        session.submit_user("inspect").unwrap();
        session.cancel();
        assert!(token.is_cancelled());
        assert_eq!(session.state(), AgentState::Cancelled);
        assert!(matches!(
            session.accept_model_reply(&run_reply("pwd")),
            Err(SessionError::Cancelled)
        ));
    }

    fn tool_reply(name: &str, arguments: &str) -> ToolResponse {
        ToolResponse::new(
            "",
            vec![crate::tools::ToolCall {
                id: "call_1".into(),
                name: name.into(),
                arguments: arguments.into(),
            }],
        )
    }

    #[test]
    fn protocol_aware_text_token_limit_never_becomes_a_proposal() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        let response = crate::response::AgentResponse::Text(crate::provider::ChatResponse {
            // This deliberately looks complete: completeness comes from the
            // provider stop reason, not from successfully parsing the bytes.
            text: run_reply("rm -rf important-data"),
            reached_token_limit: true,
            usage: Some(crate::provider::Usage {
                input_tokens: Some(10),
                output_tokens: Some(20),
            }),
        });

        assert_eq!(
            session.accept_agent_response(&response),
            Err(SessionError::Protocol(ParseError::TruncatedResponse))
        );
        assert_eq!(session.state(), AgentState::Ready);
        assert!(session.can_retry_model());
        assert!(!session
            .transcript()
            .iter()
            .any(|turn| matches!(turn, Turn::AssistantProposed { .. })));
    }

    #[test]
    fn protocol_aware_native_response_matches_the_legacy_ingestion_path() {
        let reply = ToolResponse::new(
            "Inspecting first.",
            vec![crate::tools::ToolCall {
                id: "toolu_1".into(),
                name: "run".into(),
                arguments: r#"{"command":"ls -la"}"#.into(),
            }],
        );
        let response = crate::response::AgentResponse::NativeTools(reply.clone());
        let mut direct = AgentSession::new(4);
        let mut protocol_aware = AgentSession::new(4);
        direct.submit_user("show files").unwrap();
        protocol_aware.submit_user("show files").unwrap();

        let direct_outcome = direct.accept_model_tool_reply(&reply).unwrap();
        let protocol_aware_outcome = protocol_aware.accept_agent_response(&response).unwrap();
        assert_eq!(protocol_aware_outcome, direct_outcome);
        assert_eq!(protocol_aware.state(), direct.state());
        assert_eq!(protocol_aware.turns_used(), direct.turns_used());
        assert_eq!(protocol_aware.transcript(), direct.transcript());
    }

    #[test]
    fn native_tool_reply_walks_the_identical_state_machine() {
        let mut session = AgentSession::new(4);
        session.submit_user("show files").unwrap();

        let reply = ToolResponse::new(
            "Listing the directory first.",
            vec![crate::tools::ToolCall {
                id: "toolu_1".into(),
                name: "run".into(),
                arguments: r#"{"command":"ls -la"}"#.into(),
            }],
        );
        let outcome = session.accept_model_tool_reply(&reply).unwrap();
        let ModelOutcome::Proposal { id, command, .. } = outcome else {
            panic!("expected proposal")
        };
        assert_eq!(command, "ls -la");
        // A tool call is a proposal and nothing more: the session parks in
        // AwaitingApproval exactly as the text protocol does.
        assert_eq!(
            session.state(),
            AgentState::AwaitingApproval { proposal_id: id }
        );
        // The accompanying prose is preserved as a visible thought.
        assert!(session
            .transcript()
            .iter()
            .any(|turn| matches!(turn, Turn::AssistantThought(text)
                if text == "Listing the directory first.")));

        let approved = session.approve(id).unwrap();
        assert_eq!(approved.command, "ls -la");
        assert_eq!(approved.proposal_id, id);
        assert_eq!(
            session.state(),
            AgentState::AwaitingObservation { proposal_id: id }
        );

        session.observe(id, 0, "a\nb").unwrap();
        assert_eq!(session.state(), AgentState::AwaitingModel);

        // The turn after can complete via the done tool.
        let outcome = session
            .accept_model_tool_reply(&tool_reply("done", r#"{"message":"listed"}"#))
            .unwrap();
        assert_eq!(outcome, ModelOutcome::Completed("listed".into()));
        assert_eq!(session.state(), AgentState::Completed);
        assert_eq!(session.turns_used(), 2);

        // The transcript renders identically to a text-protocol run.
        let prompt = session.build_user_prompt();
        assert!(prompt.contains(r#"{"action":"run","command":"ls -la"}"#));
        assert!(prompt.contains("user approved"));
    }

    #[test]
    fn malformed_tool_reply_never_becomes_a_proposal() {
        for reply in [
            // No tool call, only prose that describes a command.
            ToolResponse::new("I will run rm -rf / now", Vec::new()),
            // Unknown tool name.
            tool_reply("exec", r#"{"command":"rm -rf /"}"#),
            // Malformed arguments.
            tool_reply("run", "{\"command\":"),
            // Multi-line command smuggling extra PTY input.
            tool_reply("run", "{\"command\":\"ls\\nrm -rf /\"}"),
            // Two calls at once.
            ToolResponse::new(
                "",
                vec![
                    crate::tools::ToolCall {
                        id: "a".into(),
                        name: "run".into(),
                        arguments: r#"{"command":"ls"}"#.into(),
                    },
                    crate::tools::ToolCall {
                        id: "b".into(),
                        name: "run".into(),
                        arguments: r#"{"command":"rm -rf /"}"#.into(),
                    },
                ],
            ),
        ] {
            let mut session = AgentSession::new(3);
            session.submit_user("inspect").unwrap();
            assert!(
                matches!(
                    session.accept_model_tool_reply(&reply),
                    Err(SessionError::Protocol(_))
                ),
                "{reply:?}"
            );
            assert_eq!(session.state(), AgentState::Ready);
            assert!(
                !session
                    .transcript()
                    .iter()
                    .any(|turn| matches!(turn, Turn::AssistantProposed { .. })),
                "{reply:?}"
            );
            // The failed turn is retryable, exactly as a bad JSON reply is.
            assert!(session.can_retry_model());
        }
    }

    #[test]
    fn tool_actions_equal_the_text_protocol_actions() {
        let cases = [
            (
                "run",
                r#"{"command":"ls"}"#,
                r#"{"action":"run","command":"ls"}"#,
            ),
            (
                "say",
                r#"{"message":"which repo?"}"#,
                r#"{"action":"say","message":"which repo?"}"#,
            ),
            (
                "done",
                r#"{"message":"finished","thought":"all green"}"#,
                r#"{"action":"done","message":"finished","thought":"all green"}"#,
            ),
        ];
        for (name, arguments, json) in cases {
            assert_eq!(
                parse_tool_action(name, arguments).unwrap(),
                parse_action(json).unwrap(),
                "{name}"
            );
        }

        // The tool carries the action, so an `action` key in the arguments is
        // an unexpected field rather than a second source of truth.
        assert!(matches!(
            parse_tool_action("run", r#"{"action":"run","command":"ls"}"#),
            Err(ParseError::UnexpectedField(field)) if field == "action"
        ));
        // Absent arguments read as an empty object and fail on the required
        // field rather than inventing one.
        assert_eq!(
            parse_tool_action("run", ""),
            Err(ParseError::MissingField("command"))
        );
    }

    #[test]
    fn user_prompt_tail_follows_the_protocol_without_changing_text_mode() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        assert_eq!(
            session.build_user_prompt_with(AgentProtocol::Text),
            session.build_user_prompt()
        );
        assert!(session
            .build_user_prompt()
            .ends_with("Reply with exactly one JSON object from the protocol; no markdown."));

        let native = session.build_user_prompt_with(AgentProtocol::NativeTools);
        assert!(native.ends_with("Continue by calling exactly one tool: run, say, or done."));
        assert!(!native.contains("JSON object"));
        assert!(native.contains("User: inspect"));
    }

    #[test]
    fn observation_sampling_is_bounded_and_utf8_safe() {
        let output = "编译失败🙂".repeat(2_000);
        let sample = sample_observation(&output);
        assert!(sample.contains("bytes elided"));
        assert!(sample.starts_with('编'));
        assert!(sample.ends_with('🙂'));
        assert!(sample.len() <= MAX_OBSERVATION_BYTES);
    }
}
