//! Protocol-aware request preparation for the safe, high-level agent path.
//!
//! The lower-level [`crate::provider`] builders intentionally remain useful
//! for ordinary chat traffic and compatibility-sensitive integrations. Agent
//! loops have a stricter coordination problem: the system prompt, provider
//! tool schema, streaming flag, response decoder, and session ingestion path
//! must all agree on one [`AgentProtocol`]. This module makes the request half
//! of that contract a single operation and reports every history
//! transformation it performs.

use crate::prompt::{build_agent_system_prompt, build_agent_tool_system_prompt};
use crate::provider::{
    bound_history_cow_with_report, bound_history_with_report,
    build_agent_chat_request_streaming_with_report, build_agent_chat_request_with_report,
    ChatConfig, HistoryReport, HttpRequest, Message, Provider, ProviderError,
};
use crate::redact::redact_secrets_cow;
use crate::response::{AgentResponse, AgentStream};
use crate::tools::AgentProtocol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SystemPrompt<'a> {
    BuiltIn,
    Custom(Option<&'a str>),
}

/// Declarative settings for one protocol-aware agent request.
///
/// The recommended defaults use jagent's matching built-in system prompt,
/// produce a non-streaming request, and redact high-confidence secrets from
/// every history turn before byte budgeting. Custom or absent system prompts
/// are available for advanced integrations, but the caller then owns their
/// protocol and safety framing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AgentRequestSpec<'a> {
    history: &'a [Message],
    protocol: AgentProtocol,
    streaming: bool,
    redact_secrets: bool,
    system_prompt: SystemPrompt<'a>,
}

impl std::fmt::Debug for AgentRequestSpec<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let system_prompt = match self.system_prompt {
            SystemPrompt::BuiltIn => "built-in",
            SystemPrompt::Custom(Some(_)) => "custom",
            SystemPrompt::Custom(None) => "none",
        };
        formatter
            .debug_struct("AgentRequestSpec")
            .field("history_turns", &self.history.len())
            .field("protocol", &self.protocol)
            .field("streaming", &self.streaming)
            .field("redact_secrets", &self.redact_secrets)
            .field("system_prompt", &system_prompt)
            .finish()
    }
}

impl<'a> AgentRequestSpec<'a> {
    /// Start a request using the protocol-matched built-in prompt and secret
    /// redaction.
    pub fn new(history: &'a [Message], protocol: AgentProtocol) -> Self {
        Self {
            history,
            protocol,
            streaming: false,
            redact_secrets: true,
            system_prompt: SystemPrompt::BuiltIn,
        }
    }

    /// Select a streaming response body when `enabled` is true.
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Enable or disable high-confidence secret redaction over history.
    /// Redaction is enabled by default on this high-level path.
    pub fn redact_secrets(mut self, enabled: bool) -> Self {
        self.redact_secrets = enabled;
        self
    }

    /// Replace the built-in system prompt. `None` deliberately sends no
    /// system prompt. Custom system text is not secret-redacted because it is
    /// trusted application policy rather than user context.
    pub fn system_prompt(mut self, system: Option<&'a str>) -> Self {
        self.system_prompt = SystemPrompt::Custom(system);
        self
    }

    /// The action wire protocol bound to this request.
    pub fn protocol(&self) -> AgentProtocol {
        self.protocol
    }

    /// Whether the provider response is expected to stream.
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }
}

/// Non-sensitive diagnostics for one prepared agent request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentRequestReport {
    /// History window transformations, including redaction changes, turn
    /// elision, and whole-turn omission.
    pub history: HistoryReport,
    /// Whether the high-level redaction policy was enabled. When it is,
    /// `history.changed_history_turns` is the number of retained turns whose
    /// contents were redacted.
    pub redaction_enabled: bool,
    /// Size of the final provider JSON body.
    pub request_body_bytes: usize,
}

/// A transport-ready HTTP request and its history-preparation diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "inspect report before sending the prepared request"]
pub struct PreparedAgentRequest {
    /// The bounded provider-shaped request for the integration's transport.
    pub request: HttpRequest,
    /// Non-sensitive preparation diagnostics for UI or telemetry.
    pub report: AgentRequestReport,
    provider: Provider,
    protocol: AgentProtocol,
    streaming: bool,
}

impl PreparedAgentRequest {
    /// Provider whose wire format the request and response use.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Agent action protocol bound to the matching prompt and tool schema.
    pub fn protocol(&self) -> AgentProtocol {
        self.protocol
    }

    /// Whether this request selected streaming delivery.
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Parse a complete non-streaming response with the provider and protocol
    /// already bound to this request.
    ///
    /// Calling this on a streaming request is a configuration error; use
    /// [`Self::response_stream`] for that delivery mode.
    pub fn parse_response(&self, body: &[u8]) -> Result<AgentResponse, ProviderError> {
        if self.streaming {
            return Err(ProviderError::InvalidConfiguration(
                "a streaming agent request must be decoded with response_stream".into(),
            ));
        }
        AgentResponse::parse_bytes(self.provider, self.protocol, body)
    }

    /// Create the correctly configured accumulator for this streaming
    /// request.
    ///
    /// Calling this on a non-streaming request is a configuration error; use
    /// [`Self::parse_response`] after receiving its complete body.
    pub fn response_stream(&self) -> Result<AgentStream, ProviderError> {
        if !self.streaming {
            return Err(ProviderError::InvalidConfiguration(
                "a non-streaming agent request must be decoded with parse_response".into(),
            ));
        }
        Ok(AgentStream::new(self.provider, self.protocol))
    }
}

/// Prepare one provider request whose prompt, tool schema, delivery mode, and
/// history policy are bound to the same [`AgentRequestSpec`].
pub fn prepare_agent_request(
    config: &ChatConfig,
    spec: AgentRequestSpec<'_>,
) -> Result<PreparedAgentRequest, ProviderError> {
    let prepared_history = if spec.redact_secrets {
        bound_history_cow_with_report(spec.history, redact_secrets_cow)
    } else {
        bound_history_with_report(spec.history)
    };

    let built_in_system = if spec.system_prompt == SystemPrompt::BuiltIn {
        Some(match spec.protocol {
            AgentProtocol::Text => build_agent_system_prompt(),
            AgentProtocol::NativeTools => build_agent_tool_system_prompt(),
        })
    } else {
        None
    };
    let system = match spec.system_prompt {
        SystemPrompt::BuiltIn => built_in_system.as_deref(),
        SystemPrompt::Custom(system) => system,
    };

    let built = if spec.streaming {
        build_agent_chat_request_streaming_with_report(
            config,
            system,
            &prepared_history.messages,
            spec.protocol,
        )?
    } else {
        build_agent_chat_request_with_report(
            config,
            system,
            &prepared_history.messages,
            spec.protocol,
        )?
    };
    // Preparation already produced an idempotent history window. Keeping the
    // assertion here protects that contract if either layer's budgets change.
    debug_assert_eq!(built.omitted_history_turns, 0);

    let request_body_bytes = built.request.body.len();
    Ok(PreparedAgentRequest {
        request: built.request,
        report: AgentRequestReport {
            history: prepared_history.report,
            redaction_enabled: spec.redact_secrets,
            request_body_bytes,
        },
        provider: config.provider,
        protocol: spec.protocol,
        streaming: spec.streaming,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Provider, Role, MAX_REQUEST_HISTORY_TURNS, MAX_REQUEST_TURN_BYTES};
    use serde_json::Value;

    fn config(provider: Provider) -> ChatConfig {
        ChatConfig {
            provider,
            api_key: Some("transport-key".into()),
            model: "test-model".into(),
            base_url: provider.default_base_url().into(),
            max_tokens: 512,
            temperature: None,
        }
    }

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            text: text.into(),
        }
    }

    #[test]
    fn built_in_prompt_and_tool_schema_are_bound_to_one_protocol() {
        let history = [user("inspect")];

        let text = prepare_agent_request(
            &config(Provider::Anthropic),
            AgentRequestSpec::new(&history, AgentProtocol::Text),
        )
        .unwrap();
        let text_body: Value = serde_json::from_str(&text.request.body).unwrap();
        assert_eq!(text_body["system"], build_agent_system_prompt());
        assert!(text_body.get("tools").is_none());
        assert_eq!(text.provider(), Provider::Anthropic);
        assert_eq!(text.protocol(), AgentProtocol::Text);
        assert!(!text.is_streaming());
        assert!(matches!(
            text.response_stream(),
            Err(ProviderError::InvalidConfiguration(_))
        ));

        let native = prepare_agent_request(
            &config(Provider::Anthropic),
            AgentRequestSpec::new(&history, AgentProtocol::NativeTools),
        )
        .unwrap();
        let native_body: Value = serde_json::from_str(&native.request.body).unwrap();
        assert_eq!(native_body["system"], build_agent_tool_system_prompt());
        assert_eq!(native_body["tools"].as_array().map(Vec::len), Some(3));
    }

    #[test]
    fn secure_defaults_redact_and_report_without_leaking_through_debug() {
        let secret = "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ";
        let history = [user(&format!("token={secret}"))];
        let spec = AgentRequestSpec::new(&history, AgentProtocol::Text);
        assert!(!format!("{spec:?}").contains(secret));
        let prepared = prepare_agent_request(&config(Provider::OpenAiCompatible), spec).unwrap();

        assert!(!prepared.request.body.contains(secret));
        assert!(prepared.request.body.contains("[REDACTED:github-token]"));
        assert!(prepared.report.redaction_enabled);
        assert_eq!(prepared.report.history.changed_history_turns, 1);
        assert_eq!(prepared.report.history.omitted_history_turns, 0);
        assert!(!format!("{prepared:?}").contains(secret));

        let unredacted = prepare_agent_request(
            &config(Provider::OpenAiCompatible),
            AgentRequestSpec::new(&history, AgentProtocol::Text).redact_secrets(false),
        )
        .unwrap();
        assert!(unredacted.request.body.contains(secret));
        assert!(!unredacted.report.redaction_enabled);
        assert_eq!(unredacted.report.history.changed_history_turns, 0);
    }

    #[test]
    fn report_exposes_elision_and_omission_and_streaming_is_declarative() {
        let mut history: Vec<Message> = (0..MAX_REQUEST_HISTORY_TURNS + 1)
            .map(|index| user(&format!("turn {index}")))
            .collect();
        history.push(user(&"x".repeat(MAX_REQUEST_TURN_BYTES + 1)));

        let prepared = prepare_agent_request(
            &config(Provider::OpenAiCompatible),
            AgentRequestSpec::new(&history, AgentProtocol::Text)
                .streaming(true)
                .redact_secrets(false),
        )
        .unwrap();
        let body: Value = serde_json::from_str(&prepared.request.body).unwrap();

        assert_eq!(body["stream"], true);
        assert!(prepared.is_streaming());
        assert!(prepared.response_stream().is_ok());
        assert!(matches!(
            prepared.parse_response(b"{}"),
            Err(ProviderError::InvalidConfiguration(_))
        ));
        assert!(prepared.report.history.omitted_history_turns > 0);
        assert_eq!(prepared.report.history.elided_history_turns, 1);
        assert_eq!(
            prepared.report.request_body_bytes,
            prepared.request.body.len()
        );
    }

    #[test]
    fn custom_or_absent_system_prompt_is_explicit() {
        let history = [user("hello")];
        let custom = prepare_agent_request(
            &config(Provider::Anthropic),
            AgentRequestSpec::new(&history, AgentProtocol::Text)
                .system_prompt(Some("custom policy")),
        )
        .unwrap();
        let body: Value = serde_json::from_str(&custom.request.body).unwrap();
        assert_eq!(body["system"], "custom policy");

        let absent = prepare_agent_request(
            &config(Provider::Anthropic),
            AgentRequestSpec::new(&history, AgentProtocol::Text).system_prompt(None),
        )
        .unwrap();
        let body: Value = serde_json::from_str(&absent.request.body).unwrap();
        assert!(body.get("system").is_none());
    }

    #[test]
    fn prepared_request_decodes_with_its_bound_provider_and_protocol() {
        let history = [user("hello")];
        let prepared = prepare_agent_request(
            &config(Provider::OpenAiCompatible),
            AgentRequestSpec::new(&history, AgentProtocol::Text),
        )
        .unwrap();
        let response = prepared
            .parse_response(
                br#"{"choices":[{"message":{"content":"{\"action\":\"done\",\"message\":\"ok\"}"},"finish_reason":"stop"}]}"#,
            )
            .unwrap();
        assert_eq!(
            response.to_action().unwrap(),
            crate::session::ParsedAction::Done {
                thought: None,
                message: "ok".into(),
            }
        );

        let ollama = prepare_agent_request(
            &config(Provider::Ollama),
            AgentRequestSpec::new(&history, AgentProtocol::NativeTools),
        )
        .unwrap();
        let response = ollama
            .parse_response(
                br#"{"message":{"content":"","tool_calls":[{"function":{"name":"run","arguments":{"command":"pwd"}}}]},"done":true,"done_reason":"stop"}"#,
            )
            .unwrap();
        assert_eq!(
            response.to_action().unwrap(),
            crate::session::ParsedAction::Run {
                thought: None,
                command: "pwd".into(),
            }
        );
    }
}
