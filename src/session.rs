//! Pure Agent-session protocol and state machine.
//!
//! The model may only propose commands. Approval returns an ApprovedCommand
//! value to the caller; this module has no PTY, shell, process, or UI access and
//! therefore cannot execute a command by itself.

use crate::safety::{is_dangerous, MAX_COMMAND_BYTES};
use crate::text::elide_middle;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const MAX_TRANSCRIPT_BYTES: usize = 32 * 1024;
const MAX_STORED_TRANSCRIPT_BYTES: usize = 128 * 1024;
const MAX_STORED_TRANSCRIPT_ENTRIES: usize = 128;
const MAX_OBSERVATION_BYTES: usize = 4 * 1024;
const MAX_MESSAGE_BYTES: usize = 16 * 1024;
const MAX_THOUGHT_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Hash, Serialize)]
pub struct ProposalId(u64);

impl ProposalId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
    ManualReview,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Serialize)]
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
                format!("[previous model reply violated the JSON protocol: {message}]")
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
        }
    }
}

impl std::error::Error for ParseError {}

/// Strictly parse one action. A single json code fence is tolerated, but
/// prose, unknown actions/keys, wrong types, and empty required fields fail.
/// Parse failure never degrades into a command proposal.
pub fn parse_action(raw: &str) -> Result<ParsedAction, ParseError> {
    let payload = strip_json_fence(raw.trim())?;
    if payload.is_empty() {
        return Err(ParseError::Empty);
    }
    let value: Value = serde_json::from_str(payload)
        .map_err(|error| ParseError::InvalidJson(error.to_string()))?;
    let object = value.as_object().ok_or(ParseError::ExpectedObject)?;
    let action = required_string(object, "action", 32)?;
    let thought = optional_string(object, "thought", MAX_THOUGHT_BYTES)?;
    match action.as_str() {
        "run" => {
            reject_unexpected(object, &["action", "thought", "command"])?;
            let command = required_string(object, "command", MAX_COMMAND_BYTES)?;
            validate_command(&command)?;
            Ok(ParsedAction::Run { thought, command })
        }
        "say" => {
            reject_unexpected(object, &["action", "thought", "message"])?;
            let message = required_string(object, "message", MAX_MESSAGE_BYTES)?;
            Ok(ParsedAction::Say { thought, message })
        }
        "done" => {
            reject_unexpected(object, &["action", "thought", "message"])?;
            let message = required_string(object, "message", MAX_MESSAGE_BYTES)?;
            Ok(ParsedAction::Done { thought, message })
        }
        other => Err(ParseError::UnknownAction(other.to_string())),
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

fn reject_unexpected(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), ParseError> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(ParseError::UnexpectedField(field.clone()));
    }
    Ok(())
}

fn validate_command(command: &str) -> Result<(), ParseError> {
    if command.len() > MAX_COMMAND_BYTES {
        return Err(ParseError::FieldTooLarge("command"));
    }
    if command.contains(['\r', '\n']) {
        return Err(ParseError::InvalidCommand(
            "must be exactly one visible line".into(),
        ));
    }
    if command.chars().any(char::is_control) {
        return Err(ParseError::InvalidCommand(
            "contains a control character".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
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

impl std::error::Error for SessionError {}

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
            max_turns: max_turns.max(1),
            next_proposal_id: 1,
            cancelled: CancellationToken(Arc::new(AtomicBool::new(false))),
        }
    }

    pub fn transcript(&self) -> &[Turn] {
        &self.transcript
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
    pub fn start_new_task(&mut self) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        if !matches!(
            self.state,
            AgentState::Completed | AgentState::TurnLimitReached
        ) {
            return Err(self.invalid_transition("start a new task"));
        }
        let max_turns = self.max_turns;
        *self = Self::new(max_turns);
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

    pub fn accept_model_reply(&mut self, raw: &str) -> Result<ModelOutcome, SessionError> {
        self.check_not_cancelled()?;
        if self.state != AgentState::AwaitingModel {
            return Err(self.invalid_transition("accept a model reply"));
        }
        if self.turns_used >= self.max_turns {
            self.state = AgentState::TurnLimitReached;
            return Err(SessionError::TurnLimitReached);
        }
        self.turns_used = self.turns_used.saturating_add(1);
        let action = match parse_action(raw) {
            Ok(action) => action,
            Err(error) => {
                self.push_turn(Turn::ProtocolError(error.to_string()));
                self.state = self.ready_or_limited();
                return Err(SessionError::Protocol(error));
            }
        };
        match action {
            ParsedAction::Run { thought, command } => {
                self.push_thought(thought);
                let id = ProposalId(self.next_proposal_id);
                self.next_proposal_id = self.next_proposal_id.saturating_add(1);
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
        if let Some(Turn::ProtocolError(previous)) = self.transcript.last_mut() {
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
        if command.trim().is_empty() {
            return Err(SessionError::Protocol(ParseError::EmptyField("command")));
        }
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
        let turn = self.proposal_mut(id)?;
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
        self.state = AgentState::AwaitingObservation { proposal_id: id };
        Ok(approved)
    }

    pub fn reject(&mut self, id: ProposalId) -> Result<(), SessionError> {
        self.check_not_cancelled()?;
        self.expect_pending_proposal(id, "reject a proposal")?;
        let turn = self.proposal_mut(id)?;
        if let Turn::AssistantProposed { status, .. } = turn {
            *status = ProposalStatus::Rejected;
        }
        // Rejection is part of the transcript, so the next model call can
        // propose an alternative without requiring synthetic user text.
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
        if edited_command.trim().is_empty() {
            return Err(SessionError::Protocol(ParseError::EmptyField("command")));
        }
        validate_command(&edited_command).map_err(SessionError::Protocol)?;
        let turn = self.proposal_mut(id)?;
        let Turn::AssistantProposed {
            command, status, ..
        } = turn
        else {
            unreachable!("proposal_mut only returns proposal turns")
        };
        *command = edited_command;
        *status = ProposalStatus::ManualReview;
        let command = command.clone();
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
        match self.state {
            AgentState::AwaitingObservation { proposal_id } if proposal_id == id => {}
            AgentState::AwaitingObservation { proposal_id } => {
                return Err(SessionError::StaleProposal {
                    expected: proposal_id,
                    received: id,
                });
            }
            _ => return Err(self.invalid_transition("record command output")),
        }
        self.push_turn(Turn::Observation {
            proposal_id: id,
            exit_code,
            output_sample: sample_observation(output),
        });
        self.state = if self.turns_used >= self.max_turns {
            AgentState::TurnLimitReached
        } else {
            AgentState::AwaitingModel
        };
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.cancelled.0.store(true, Ordering::SeqCst);
        self.state = AgentState::Cancelled;
    }

    pub fn build_user_prompt(&self) -> String {
        let mut entries: Vec<String> = self.transcript.iter().map(Turn::to_prompt).collect();
        if self.transcript_truncated {
            entries.insert(
                0,
                "[older Agent activity was omitted by the in-memory safety budget]".to_string(),
            );
        }
        entries
            .push("Reply with exactly one JSON object from the protocol; no markdown.".to_string());
        elide_middle(&entries.join("\n\n"), MAX_TRANSCRIPT_BYTES)
    }

    fn proposal_mut(&mut self, id: ProposalId) -> Result<&mut Turn, SessionError> {
        self.transcript
            .iter_mut()
            .find(|turn| {
                matches!(turn, Turn::AssistantProposed { id: candidate, .. } if *candidate == id)
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
            self.transcript.remove(0);
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
        if snapshot.max_turns == 0 || snapshot.max_turns > 1_000 {
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
        let highest_proposal_id = snapshot
            .transcript
            .iter()
            .filter_map(|turn| match turn {
                Turn::AssistantProposed { id, .. } => Some(id.get()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let mut session = Self {
            transcript: snapshot.transcript,
            transcript_truncated: snapshot.transcript_truncated,
            state: snapshot.state,
            turns_used: snapshot.turns_used,
            max_turns: snapshot.max_turns,
            next_proposal_id: snapshot
                .next_proposal_id
                .max(highest_proposal_id.saturating_add(1)),
            cancelled: CancellationToken(Arc::new(AtomicBool::new(false))),
        };
        session.compact_transcript();
        match session.state {
            AgentState::AwaitingApproval { proposal_id } => {
                let pending = session.transcript.iter().any(|turn| {
                    matches!(
                        turn,
                        Turn::AssistantProposed { id, status: ProposalStatus::Pending, .. }
                            if *id == proposal_id
                    )
                });
                if !pending {
                    session.push_turn(Turn::ProtocolError(
                        "the pending proposal was lost across a restart".into(),
                    ));
                    session.state = session.ready_or_limited();
                }
            }
            AgentState::AwaitingObservation { proposal_id } => {
                session.push_turn(Turn::ProtocolError(format!(
                    "the application exited before proposal #{}'s output was \
                     observed; its result is unknown",
                    proposal_id.get()
                )));
                session.state = session.ready_or_limited();
            }
            _ => {}
        }
        Ok(session)
    }
}

const AGENT_SNAPSHOT_VERSION: u32 = 1;

/// Byte cap for one encoded snapshot; larger inputs are refused rather than
/// truncated so a corrupt or hostile file cannot balloon memory.
pub const MAX_AGENT_SNAPSHOT_JSON_BYTES: usize = 256 * 1024;

/// Serializable capture of an [`AgentSession`] for cross-restart persistence.
/// Serialization is pure; where and how the JSON is stored stays with the
/// embedding application.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
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

    pub fn from_json(encoded: &str) -> Result<Self, AgentSnapshotError> {
        if encoded.len() > MAX_AGENT_SNAPSHOT_JSON_BYTES {
            return Err(AgentSnapshotError::TooLarge {
                limit: MAX_AGENT_SNAPSHOT_JSON_BYTES,
            });
        }
        serde_json::from_str(encoded).map_err(|error| AgentSnapshotError::Decode(error.to_string()))
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
        restored.approve(proposal_id).unwrap();
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
        session.approve(proposal_id).unwrap();
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

    fn run_reply(command: &str) -> String {
        serde_json::json!({"action":"run", "command": command}).to_string()
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
        assert!(session.approve(id).is_err());
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
    }

    #[test]
    fn failed_model_turn_can_retry_without_duplicate_user_input() {
        let mut session = AgentSession::new(3);
        session.submit_user("inspect").unwrap();
        session.model_failed("temporary network error").unwrap();
        assert!(session.can_retry_model());
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

        session.start_new_task().unwrap();
        assert_eq!(session.state(), AgentState::Ready);
        assert_eq!(session.turns_used(), 0);
        assert_eq!(session.max_turns(), 1);
        assert!(session.transcript().is_empty());
        session.submit_user("fresh task").unwrap();
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
