//! Provider-neutral chat request construction and response parsing.
//!
//! Sans-IO: [`build_chat_request_with_report`] returns a [`BuiltRequest`]
//! describing exactly one POST and how many history turns this build omitted;
//! the integration performs its [`HttpRequest`] with whatever HTTP stack it
//! already trusts (curl child process, ureq, …) and hands the response JSON to
//! [`parse_chat_response`]. Nothing in this module opens a socket. The shorter
//! [`build_chat_request`] compatibility entry point returns only the request.

use crate::safety::is_unsafe_invisible_char;
use crate::text::elide_middle;
use crate::tools::{agent_body_fields, AgentProtocol};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;

pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
pub const MAX_MODEL_TEXT_BYTES: usize = 256 * 1024;
pub const MAX_REQUEST_HISTORY_TURNS: usize = 40;
pub const MAX_REQUEST_HISTORY_BYTES: usize = 256 * 1024;
pub const MAX_REQUEST_TURN_BYTES: usize = 192 * 1024;
/// Generous ceiling for a credential copied into one HTTP header. This bounds
/// request construction and rejects configuration mistakes without exposing
/// or echoing the credential in an error.
pub const MAX_API_KEY_BYTES: usize = 16 * 1024;
/// Byte ceiling for a configured model identifier.
pub const MAX_MODEL_BYTES: usize = 1024;
/// Byte ceiling for a configured base URL.
pub const MAX_BASE_URL_BYTES: usize = 4 * 1024;
/// Byte ceiling for one request's system prompt. A system prompt carries the
/// protocol and safety instructions, so an over-budget one is rejected rather
/// than elided: silently truncating it could drop the rules the reply is
/// parsed against.
pub const MAX_REQUEST_SYSTEM_BYTES: usize = 64 * 1024;
/// Byte ceiling for one encoded non-streaming response envelope, applied by
/// [`parse_chat_response_bytes`]. Integrations that decode the body themselves
/// must apply an equivalent transport limit before calling the `Value`-based
/// entry points.
pub const MAX_RESPONSE_JSON_BYTES: usize = 1024 * 1024;

/// Supported wire protocols. OpenAI-compatible intentionally includes local
/// and hosted services which implement the Chat Completions endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAiCompatible,
    Ollama,
}

impl Provider {
    pub fn as_config_value(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAiCompatible => "openai-compatible",
            Self::Ollama => "ollama",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::OpenAiCompatible => "OpenAI-compatible",
            Self::Ollama => "Ollama",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Self::Anthropic => "claude-sonnet-5",
            Self::OpenAiCompatible => "gpt-4o-mini",
            Self::Ollama => "codellama:7b",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::Anthropic => "https://api.anthropic.com",
            Self::OpenAiCompatible => "https://api.openai.com/v1",
            Self::Ollama => "http://localhost:11434",
        }
    }

    pub fn endpoint(self, base_url: &str) -> String {
        let base = base_url.trim_end_matches('/');
        match self {
            Self::Anthropic if base.ends_with("/v1/messages") => base.to_string(),
            Self::Anthropic if base.ends_with("/v1") => format!("{base}/messages"),
            Self::Anthropic => format!("{base}/v1/messages"),
            Self::OpenAiCompatible if base.ends_with("/chat/completions") => base.to_string(),
            Self::OpenAiCompatible => format!("{base}/chat/completions"),
            Self::Ollama if base.ends_with("/api/chat") => base.to_string(),
            Self::Ollama if base.ends_with("/api") => format!("{base}/chat"),
            Self::Ollama => format!("{base}/api/chat"),
        }
    }
}

impl FromStr for Provider {
    type Err = ProviderError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "openai" | "openai-compatible" | "openai_compatible" => Ok(Self::OpenAiCompatible),
            "ollama" => Ok(Self::Ollama),
            other => Err(ProviderError::InvalidConfiguration(format!(
                "unknown AI provider '{other}'"
            ))),
        }
    }
}

/// Allocation-free role atom used by the provider request schema.
///
/// `Deserialize` is intentionally available for this scalar enum. Decoding a
/// [`Role`] does not decode or bound a conversation; persisted or network
/// history needs an embedding-owned envelope and collection budget.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One in-memory turn used to construct a provider-neutral chat request.
///
/// This type is serialize-only by design: it is an input value, not a
/// persistence or network decoder. Request builders bound values that already
/// exist in memory, but an embedding that stores history must bound its encoded
/// envelope, entry count, and cumulative text while decoding.
///
/// ```compile_fail
/// let _: jagent::provider::Message =
///     serde_json::from_str(r#"{"role":"user","text":"hello"}"#).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Message {
    pub role: Role,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderError {
    InvalidConfiguration(String),
    MissingApiKey(Provider),
    EmptyResponse,
    ResponseTooLarge {
        limit: usize,
    },
    /// The reply's shape violates the provider's own wire format — for
    /// example a tool call with no name. Fail closed rather than guess.
    MalformedResponse(String),
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => write!(f, "invalid configuration: {message}"),
            Self::MissingApiKey(provider) => {
                write!(f, "{} requires an API key", provider.display_name())
            }
            Self::EmptyResponse => write!(f, "the model returned an empty response"),
            Self::ResponseTooLarge { limit } => {
                write!(f, "model response exceeds the {limit} byte limit")
            }
            Self::MalformedResponse(detail) => write!(f, "malformed model response: {detail}"),
        }
    }
}

impl std::error::Error for ProviderError {}

/// A fully described HTTP POST for the integration's transport to perform.
/// Headers already include content-type and any credential the provider needs.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub url: String,
    /// Lowercase header names. Credentials appear here; integrations must keep
    /// headers out of argv (pass via stdin config for curl-style transports).
    pub headers: Vec<(String, String)>,
    /// JSON body, already serialized.
    pub body: String,
}

impl std::fmt::Debug for HttpRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HttpRequest")
            .field("url", &self.url)
            .field("headers", &RedactedHeaders(&self.headers))
            .field("body", &self.body)
            .finish()
    }
}

/// One bounded provider request together with the history loss introduced by
/// this build operation.
///
/// `omitted_history_turns` counts only turns that these request builders
/// removed. If an integration bounds history before calling a builder, it
/// remains responsible for carrying that earlier omission count forward.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "inspect omitted_history_turns before sending the request"]
pub struct BuiltRequest {
    pub request: HttpRequest,
    pub omitted_history_turns: usize,
}

struct RedactedHeaders<'a>(&'a [(String, String)]);

impl std::fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = formatter.debug_list();
        for (name, value) in self.0 {
            let value = if name.eq_ignore_ascii_case("authorization")
                || name.eq_ignore_ascii_case("x-api-key")
            {
                "[REDACTED]"
            } else {
                value
            };
            list.entry(&(name, value));
        }
        list.finish()
    }
}

/// Chat client configuration owned by the integration.
#[derive(Clone, PartialEq)]
pub struct ChatConfig {
    pub provider: Provider,
    pub api_key: Option<String>,
    pub model: String,
    pub base_url: String,
    pub max_tokens: u32,
    /// Sampling temperature. `None` keeps the provider default; agent loops
    /// that require strict JSON replies benefit from a low value such as 0.0.
    pub temperature: Option<f32>,
}

impl std::fmt::Debug for ChatConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatConfig")
            .field("provider", &self.provider)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .finish()
    }
}

impl ChatConfig {
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            api_key: None,
            model: provider.default_model().to_string(),
            base_url: provider.default_base_url().to_string(),
            max_tokens: 1024,
            temperature: None,
        }
    }

    /// Reject any configuration this crate would otherwise turn into a
    /// request. Every bound here is the library's own: an integration may add
    /// stricter policy, but it can no longer be the only thing standing
    /// between a hostile settings file and an outbound HTTP request.
    pub fn validate(&self) -> Result<(), ProviderError> {
        validate_model(&self.model)?;
        validate_base_url(self.provider, self.base_url.trim())?;
        if self.max_tokens == 0 {
            return Err(ProviderError::InvalidConfiguration(
                "max_tokens must be positive".into(),
            ));
        }
        if let Some(api_key) = self.api_key.as_deref() {
            let api_key = api_key.trim();
            if api_key.len() > MAX_API_KEY_BYTES {
                return Err(ProviderError::InvalidConfiguration(
                    "API key exceeds its byte limit".into(),
                ));
            }
            if api_key.chars().any(char::is_control) {
                return Err(ProviderError::InvalidConfiguration(
                    "API key contains a control character".into(),
                ));
            }
        }
        if let Some(temperature) = self.temperature {
            if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
                return Err(ProviderError::InvalidConfiguration(
                    "temperature must be a finite value in 0.0..=2.0".into(),
                ));
            }
        }
        Ok(())
    }
}

fn validate_model(model: &str) -> Result<(), ProviderError> {
    if model.trim().is_empty()
        || model.len() > MAX_MODEL_BYTES
        || model.chars().any(char::is_control)
        || model.chars().any(is_unsafe_invisible_char)
    {
        return Err(ProviderError::InvalidConfiguration(format!(
            "model must be visible, non-empty text of at most {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(())
}

/// Accept only endpoints this crate is willing to send a credential to.
///
/// A base URL usually arrives from a settings file or an environment variable,
/// so it is attacker-reachable in exactly the situations where the request
/// carries an API key. Userinfo would smuggle credentials into a persisted
/// URL, a query or fragment is not a base-URL component (the provider endpoint
/// is appended after this string), and visually ambiguous characters would let
/// a configured host read as one origin while resolving as another.
fn validate_base_url(provider: Provider, base_url: &str) -> Result<(), ProviderError> {
    let invalid = || {
        ProviderError::InvalidConfiguration(format!(
            "base URL must be an absolute HTTPS URL of at most {MAX_BASE_URL_BYTES} bytes with \
             no credentials, query, fragment, backslash, control, or visually ambiguous \
             characters (plain HTTP is accepted only for a loopback Ollama endpoint)"
        ))
    };
    if base_url.is_empty()
        || base_url.len() > MAX_BASE_URL_BYTES
        || base_url.contains([' ', '?', '#', '\\'])
        || base_url.chars().any(char::is_control)
        || base_url.chars().any(is_unsafe_invisible_char)
    {
        return Err(invalid());
    }
    let (scheme, rest) = base_url.split_once("://").ok_or_else(invalid)?;
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return Err(invalid());
    }
    match scheme {
        "https" => Ok(()),
        "http" if provider == Provider::Ollama && is_loopback_authority(authority) => Ok(()),
        _ => Err(invalid()),
    }
}

/// True for `localhost`, an IPv4 loopback literal, or a bracketed IPv6
/// loopback literal, each with an optional numeric port.
fn is_loopback_authority(authority: &str) -> bool {
    let host = if let Some(rest) = authority.strip_prefix('[') {
        let Some((literal, port)) = rest.split_once(']') else {
            return false;
        };
        if !port.is_empty() && !port.strip_prefix(':').is_some_and(is_port) {
            return false;
        }
        literal
    } else {
        match authority.split_once(':') {
            Some((host, port)) if is_port(port) => host,
            Some(_) => return false,
            None => authority,
        }
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::Ipv4Addr>()
            .is_ok_and(|address| address.is_loopback())
        || host
            .parse::<std::net::Ipv6Addr>()
            .is_ok_and(|address| address.is_loopback())
}

fn is_port(port: &str) -> bool {
    !port.is_empty() && port.chars().all(|digit| digit.is_ascii_digit())
}

/// Bound a conversation history to the request budgets, newest-first. Leading
/// assistant turns are removed while at least one later turn can remain; a
/// singleton assistant turn is retained for compatibility with existing wire
/// requests.
/// Returns the retained history and how many older turns were omitted.
pub fn bound_history(history: &[Message]) -> (Vec<Message>, usize) {
    bound_history_with(history, str::to_string)
}

/// [`bound_history`] with a per-turn preparation hook (redaction, normalization)
/// applied to each turn's text *before* the byte budget elides it, so the
/// budget is measured against what will actually be sent.
pub fn bound_history_with(
    history: &[Message],
    prepare: impl Fn(&str) -> String,
) -> (Vec<Message>, usize) {
    let mut retained_reversed = Vec::new();
    let mut retained_bytes = 0_usize;
    for turn in history.iter().rev() {
        if retained_reversed.len() >= MAX_REQUEST_HISTORY_TURNS {
            break;
        }
        let text = elide_middle(&prepare(&turn.text), MAX_REQUEST_TURN_BYTES);
        let cost = text.len().saturating_add(32);
        if !retained_reversed.is_empty()
            && retained_bytes.saturating_add(cost) > MAX_REQUEST_HISTORY_BYTES
        {
            break;
        }
        retained_bytes = retained_bytes.saturating_add(cost);
        retained_reversed.push(Message {
            role: turn.role,
            text,
        });
    }
    retained_reversed.reverse();
    let mut omitted = history.len().saturating_sub(retained_reversed.len());
    while retained_reversed.len() > 1
        && retained_reversed
            .first()
            .is_some_and(|turn| turn.role == Role::Assistant)
    {
        retained_reversed.remove(0);
        omitted = omitted.saturating_add(1);
    }
    (retained_reversed, omitted)
}

/// Build one bounded chat POST, discarding the number of history turns this
/// operation omitted. New integrations should prefer
/// [`build_chat_request_with_report`]; this compatibility entry point keeps
/// its established return type and exact request bytes.
pub fn build_chat_request(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
) -> Result<HttpRequest, ProviderError> {
    build_chat_request_with_report(config, system, history).map(|built| built.request)
}

/// Build one bounded chat POST while preserving the number of history turns
/// this operation omitted. The request bytes are identical to
/// [`build_chat_request`].
pub fn build_chat_request_with_report(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
) -> Result<BuiltRequest, ProviderError> {
    build_request(config, system, history, false)
}

/// [`build_chat_request`] with the provider's streaming flag set
/// (`"stream": true` for all three providers). The response body then
/// arrives as SSE (Anthropic, OpenAI-compatible) or NDJSON (Ollama) and is
/// parsed incrementally with [`crate::stream::StreamParser`].
pub fn build_chat_request_streaming(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
) -> Result<HttpRequest, ProviderError> {
    build_chat_request_streaming_with_report(config, system, history).map(|built| built.request)
}

/// [`build_chat_request_streaming`] while preserving the number of history
/// turns this build omitted. The request bytes are identical to the
/// compatibility entry point.
pub fn build_chat_request_streaming_with_report(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
) -> Result<BuiltRequest, ProviderError> {
    build_request(config, system, history, true)
}

/// Build one agent-loop chat POST in the requested protocol.
///
/// [`AgentProtocol::Text`] is byte-identical to [`build_chat_request`] — the
/// JSON-in-text protocol parsed by [`crate::session::parse_action`].
/// [`AgentProtocol::NativeTools`] adds the provider-correct `tools` and
/// `tool_choice` fields (see [`crate::tools`]) and changes nothing else;
/// replies are then ingested with [`crate::tools::parse_tool_response`].
///
/// Ollama has no standard tool-calling in the chat endpoint this crate
/// targets, so native mode returns [`ProviderError::InvalidConfiguration`]
/// there instead of emitting a request that would degrade to prose.
pub fn build_agent_chat_request(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    protocol: AgentProtocol,
) -> Result<HttpRequest, ProviderError> {
    build_agent_chat_request_with_report(config, system, history, protocol)
        .map(|built| built.request)
}

/// [`build_agent_chat_request`] while preserving the number of history turns
/// this build omitted. The request bytes are identical to the compatibility
/// entry point.
pub fn build_agent_chat_request_with_report(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    protocol: AgentProtocol,
) -> Result<BuiltRequest, ProviderError> {
    build_agent_request(config, system, history, protocol, false)
}

/// [`build_agent_chat_request`] with the provider's streaming flag set, as
/// [`build_chat_request_streaming`] does. Tool-call deltas in the response are
/// accumulated by [`crate::stream::StreamParser`].
pub fn build_agent_chat_request_streaming(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    protocol: AgentProtocol,
) -> Result<HttpRequest, ProviderError> {
    build_agent_chat_request_streaming_with_report(config, system, history, protocol)
        .map(|built| built.request)
}

/// [`build_agent_chat_request_streaming`] while preserving the number of
/// history turns this build omitted. The request bytes are identical to the
/// compatibility entry point.
pub fn build_agent_chat_request_streaming_with_report(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    protocol: AgentProtocol,
) -> Result<BuiltRequest, ProviderError> {
    build_agent_request(config, system, history, protocol, true)
}

fn build_agent_request(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    protocol: AgentProtocol,
    stream: bool,
) -> Result<BuiltRequest, ProviderError> {
    match protocol {
        AgentProtocol::Text => build_request(config, system, history, stream),
        AgentProtocol::NativeTools => {
            // Validate the configuration before reporting protocol support so
            // the error the caller sees is the first thing actually wrong.
            config.validate()?;
            let extra = agent_body_fields(config.provider)?;
            build_request_with(config, system, history, stream, &extra)
        }
    }
}

fn build_request(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    stream: bool,
) -> Result<BuiltRequest, ProviderError> {
    build_request_with(config, system, history, stream, &[])
}

fn build_request_with(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    stream: bool,
    extra_body_fields: &[(&'static str, Value)],
) -> Result<BuiltRequest, ProviderError> {
    config.validate()?;
    if let Some(system) = system {
        if system.len() > MAX_REQUEST_SYSTEM_BYTES {
            return Err(ProviderError::InvalidConfiguration(format!(
                "system prompt exceeds the {MAX_REQUEST_SYSTEM_BYTES}-byte request limit"
            )));
        }
    }
    // `bound_history` is idempotent, so a caller that already bounded (or
    // redacted and bounded) its history sends exactly what it prepared, while
    // a caller that forgot cannot make this crate emit an unbounded body.
    let (history, omitted_history_turns) = bound_history(history);
    let history = &history[..];
    let api_key = config
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty());
    let mut headers = vec![("content-type".to_string(), "application/json".to_string())];
    match config.provider {
        Provider::Anthropic => {
            let key = api_key.ok_or(ProviderError::MissingApiKey(config.provider))?;
            headers.push(("x-api-key".to_string(), key.to_string()));
            headers.push((
                "anthropic-version".to_string(),
                ANTHROPIC_API_VERSION.to_string(),
            ));
        }
        Provider::OpenAiCompatible | Provider::Ollama => {
            if let Some(key) = api_key {
                headers.push(("authorization".to_string(), format!("Bearer {key}")));
            }
        }
    }
    let mut body = request_body(config, system, history, stream);
    for (field, value) in extra_body_fields {
        body[*field] = value.clone();
    }
    Ok(BuiltRequest {
        request: HttpRequest {
            url: config.provider.endpoint(&config.base_url),
            headers,
            body: body.to_string(),
        },
        omitted_history_turns,
    })
}

fn request_body(
    config: &ChatConfig,
    system: Option<&str>,
    history: &[Message],
    stream: bool,
) -> Value {
    let mut messages: Vec<Value> = history
        .iter()
        .map(|turn| json!({"role": turn.role.as_str(), "content": turn.text}))
        .collect();
    match config.provider {
        Provider::Anthropic => {
            let mut body = json!({
                "model": config.model,
                "max_tokens": config.max_tokens,
                "messages": messages,
            });
            if let Some(system) = system {
                body["system"] = Value::String(system.to_string());
            }
            if let Some(temperature) = config.temperature {
                body["temperature"] = json!(temperature);
            }
            if stream {
                body["stream"] = json!(true);
            }
            body
        }
        Provider::OpenAiCompatible => {
            if let Some(system) = system {
                messages.insert(0, json!({"role": "system", "content": system}));
            }
            let mut body = json!({
                "model": config.model,
                "max_tokens": config.max_tokens,
                "messages": messages,
            });
            if let Some(temperature) = config.temperature {
                body["temperature"] = json!(temperature);
            }
            if stream {
                body["stream"] = json!(true);
                // Most Chat Completions servers only report token usage during
                // streaming when asked; the parser treats the frame as optional
                // so servers that ignore this stay compatible.
                body["stream_options"] = json!({"include_usage": true});
            }
            body
        }
        Provider::Ollama => {
            if let Some(system) = system {
                messages.insert(0, json!({"role": "system", "content": system}));
            }
            let mut body = json!({
                "model": config.model,
                "messages": messages,
                "stream": stream,
                "options": {"num_predict": config.max_tokens},
            });
            if let Some(temperature) = config.temperature {
                body["options"]["temperature"] = json!(temperature);
            }
            body
        }
    }
}

/// Token usage reported by the provider, when the response carries it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

/// Structured result of parsing one non-streaming chat response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    /// Assistant text exactly as extracted — no advisory notes appended.
    pub text: String,
    /// The provider stopped at the output-token limit; the integration should
    /// present the text as partial rather than complete.
    pub reached_token_limit: bool,
    pub usage: Option<Usage>,
}

/// Extract the assistant text from one non-streaming chat response body.
///
/// This is the bounded entry point: the encoded envelope is refused above
/// [`MAX_RESPONSE_JSON_BYTES`] before `serde_json` allocates anything from it.
/// The [`Value`]-based functions below stay available for integrations that
/// already decode the body, but those make the transport's own envelope limit
/// an integration requirement.
pub fn parse_chat_response_bytes(provider: Provider, body: &[u8]) -> Result<String, ProviderError> {
    let parsed = parse_chat_response_full_bytes(provider, body)?;
    Ok(chat_response_text(parsed))
}

/// Parse one non-streaming chat response into its structured parts from a
/// bounded encoded envelope.
///
/// This is the canonical byte-oriented entry point for callers that need
/// token-limit and usage metadata. The size check happens before
/// `serde_json` constructs a [`Value`].
pub fn parse_chat_response_full_bytes(
    provider: Provider,
    body: &[u8],
) -> Result<ChatResponse, ProviderError> {
    let response = decode_response_value(body)?;
    parse_chat_response_full(provider, &response)
}

/// Extract the assistant text from one non-streaming chat response.
/// A reached token limit is surfaced as a visible trailing note, never as an
/// error, so partial answers stay reviewable.
///
/// `response` is already decoded. This API is only for trusted or
/// already-bounded caller-owned values: the caller's transport must have
/// applied an encoded-envelope limit before allocating it.
/// [`parse_chat_response_bytes`] does that for integrations that do not need
/// structured metadata.
pub fn parse_chat_response(provider: Provider, response: &Value) -> Result<String, ProviderError> {
    let parsed = parse_chat_response_full(provider, response)?;
    Ok(chat_response_text(parsed))
}

fn chat_response_text(parsed: ChatResponse) -> String {
    let mut text = parsed.text;
    if parsed.reached_token_limit {
        text.push_str(
            "\n\n[Response reached the configured output limit. Ask to continue or \
             increase max_tokens.]",
        );
    }
    text
}

/// [`parse_chat_response`] returning the structured parts: raw text, the
/// token-limit flag (so integrations word their own advisory note), and any
/// token usage the provider reported.
///
/// `response` must be trusted or already bounded. Network bytes should enter
/// through [`parse_chat_response_full_bytes`] so the encoded envelope is
/// checked before a [`Value`] is allocated.
pub fn parse_chat_response_full(
    provider: Provider,
    response: &Value,
) -> Result<ChatResponse, ProviderError> {
    let reached_token_limit = reached_token_limit(provider, response);
    let text = match provider {
        Provider::Anthropic => response
            .get("content")
            .and_then(Value::as_array)
            .map(|parts| {
                join_model_text(
                    parts
                        .iter()
                        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                        .filter_map(|part| part.get("text").and_then(Value::as_str)),
                )
            })
            .transpose()?
            .unwrap_or_default(),
        Provider::OpenAiCompatible => match response.pointer("/choices/0/message/content") {
            Some(content) => content_text(content)?.unwrap_or_default(),
            None => String::new(),
        },
        Provider::Ollama => bounded_model_text(
            response
                .pointer("/message/content")
                .and_then(Value::as_str)
                .or_else(|| response.get("response").and_then(Value::as_str))
                .unwrap_or_default(),
        )?,
    };
    if text.trim().is_empty() {
        return Err(ProviderError::EmptyResponse);
    }
    if text.len() > MAX_MODEL_TEXT_BYTES {
        return Err(ProviderError::ResponseTooLarge {
            limit: MAX_MODEL_TEXT_BYTES,
        });
    }
    Ok(ChatResponse {
        text,
        reached_token_limit,
        usage: parse_usage(provider, response),
    })
}

/// Decode a provider response only after enforcing the shared encoded-body
/// ceiling. Kept crate-visible so native-tool parsing uses the identical gate.
pub(crate) fn decode_response_value(body: &[u8]) -> Result<Value, ProviderError> {
    if body.len() > MAX_RESPONSE_JSON_BYTES {
        return Err(ProviderError::ResponseTooLarge {
            limit: MAX_RESPONSE_JSON_BYTES,
        });
    }
    serde_json::from_slice(body)
        .map_err(|error| ProviderError::MalformedResponse(error.to_string()))
}

/// Did the provider stop at the output-token limit? Shared by the text and
/// native-tool ingestion paths so both report the same condition.
pub(crate) fn reached_token_limit(provider: Provider, response: &Value) -> bool {
    match provider {
        Provider::Anthropic => {
            response.get("stop_reason").and_then(Value::as_str) == Some("max_tokens")
        }
        Provider::OpenAiCompatible => {
            response
                .pointer("/choices/0/finish_reason")
                .and_then(Value::as_str)
                == Some("length")
        }
        Provider::Ollama => response.get("done_reason").and_then(Value::as_str) == Some("length"),
    }
}

/// Best-effort usage extraction; providers that omit the fields yield `None`.
pub(crate) fn parse_usage(provider: Provider, response: &Value) -> Option<Usage> {
    let (input, output) = match provider {
        Provider::Anthropic => (
            response.pointer("/usage/input_tokens"),
            response.pointer("/usage/output_tokens"),
        ),
        Provider::OpenAiCompatible => (
            response.pointer("/usage/prompt_tokens"),
            response.pointer("/usage/completion_tokens"),
        ),
        Provider::Ollama => (
            response.get("prompt_eval_count"),
            response.get("eval_count"),
        ),
    };
    let usage = Usage {
        input_tokens: input.and_then(Value::as_u64),
        output_tokens: output.and_then(Value::as_u64),
    };
    (usage.input_tokens.is_some() || usage.output_tokens.is_some()).then_some(usage)
}

pub(crate) fn content_text(value: &Value) -> Result<Option<String>, ProviderError> {
    if let Some(text) = value.as_str() {
        return bounded_model_text(text).map(Some);
    }
    value
        .as_array()
        .map(|parts| {
            join_model_text(
                parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str)),
            )
            .map(Some)
        })
        .unwrap_or(Ok(None))
}

/// Join provider text blocks without first collecting or allocating an
/// unbounded intermediate string. The separator counts toward the same
/// cumulative model-text budget as the block contents.
pub(crate) fn join_model_text<'a>(
    parts: impl IntoIterator<Item = &'a str>,
) -> Result<String, ProviderError> {
    let mut joined = String::new();
    let mut first = true;
    for part in parts {
        let separator_bytes = usize::from(!first);
        let total = joined
            .len()
            .checked_add(separator_bytes)
            .and_then(|length| length.checked_add(part.len()))
            .ok_or(ProviderError::ResponseTooLarge {
                limit: MAX_MODEL_TEXT_BYTES,
            })?;
        if total > MAX_MODEL_TEXT_BYTES {
            return Err(ProviderError::ResponseTooLarge {
                limit: MAX_MODEL_TEXT_BYTES,
            });
        }
        if !first {
            joined.push('\n');
        }
        joined.push_str(part);
        first = false;
    }
    Ok(joined)
}

fn bounded_model_text(text: &str) -> Result<String, ProviderError> {
    if text.len() > MAX_MODEL_TEXT_BYTES {
        return Err(ProviderError::ResponseTooLarge {
            limit: MAX_MODEL_TEXT_BYTES,
        });
    }
    Ok(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(provider: Provider) -> ChatConfig {
        ChatConfig {
            provider,
            api_key: Some("test-key".into()),
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

    fn assistant(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            text: text.into(),
        }
    }

    #[test]
    fn request_shapes_match_each_provider() {
        let history = [user("hello")];

        let anthropic = build_chat_request(&config(Provider::Anthropic), Some("sys"), &history)
            .expect("anthropic request");
        assert_eq!(anthropic.url, "https://api.anthropic.com/v1/messages");
        assert!(anthropic
            .headers
            .iter()
            .any(|(name, value)| name == "x-api-key" && value == "test-key"));
        let body: Value = serde_json::from_str(&anthropic.body).unwrap();
        assert_eq!(body["system"], "sys");
        assert_eq!(body["messages"][0]["role"], "user");

        let openai = build_chat_request(&config(Provider::OpenAiCompatible), Some("sys"), &history)
            .expect("openai request");
        assert_eq!(openai.url, "https://api.openai.com/v1/chat/completions");
        let body: Value = serde_json::from_str(&openai.body).unwrap();
        assert_eq!(body["messages"][0]["role"], "system");

        let ollama =
            build_chat_request(&config(Provider::Ollama), None, &history).expect("ollama request");
        assert_eq!(ollama.url, "http://localhost:11434/api/chat");
        let body: Value = serde_json::from_str(&ollama.body).unwrap();
        assert_eq!(body["stream"], false);
        assert_eq!(body["options"]["num_predict"], 512);
    }

    #[test]
    fn debug_output_never_exposes_api_credentials() {
        let secret = "sk-test-never-log-this-value";
        let mut chat = config(Provider::Anthropic);
        chat.api_key = Some(secret.into());
        let config_debug = format!("{chat:?}");
        assert!(config_debug.contains("[REDACTED]"));
        assert!(!config_debug.contains(secret));

        let request = build_chat_request(&chat, None, &[user("hello")]).unwrap();
        let request_debug = format!("{request:?}");
        assert!(request_debug.contains("[REDACTED]"));
        assert!(!request_debug.contains(secret));
    }

    #[test]
    fn streaming_requests_only_add_the_streaming_fields() {
        let history = [user("hello")];
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let plain = build_chat_request(&config(provider), Some("sys"), &history).unwrap();
            let streaming =
                build_chat_request_streaming(&config(provider), Some("sys"), &history).unwrap();
            assert_eq!(streaming.url, plain.url);
            assert_eq!(streaming.headers, plain.headers);
            let mut streaming_body: Value = serde_json::from_str(&streaming.body).unwrap();
            assert_eq!(streaming_body["stream"], true, "{provider:?}");
            let plain_body: Value = serde_json::from_str(&plain.body).unwrap();
            // Normalizing the streaming-only fields away recovers the
            // non-streaming body exactly; nothing else may differ.
            match provider {
                Provider::Ollama => {
                    assert_eq!(plain_body["stream"], false);
                    streaming_body["stream"] = json!(false);
                }
                Provider::OpenAiCompatible => {
                    assert!(plain_body.get("stream").is_none());
                    assert_eq!(
                        streaming_body["stream_options"],
                        json!({"include_usage": true}),
                    );
                    let body = streaming_body.as_object_mut().unwrap();
                    body.remove("stream");
                    body.remove("stream_options");
                }
                Provider::Anthropic => {
                    assert!(plain_body.get("stream").is_none());
                    streaming_body.as_object_mut().unwrap().remove("stream");
                }
            }
            assert_eq!(streaming_body, plain_body, "{provider:?}");
        }
    }

    #[test]
    fn anthropic_requires_an_api_key() {
        let mut config = config(Provider::Anthropic);
        config.api_key = None;
        assert!(matches!(
            build_chat_request(&config, None, &[user("hi")]),
            Err(ProviderError::MissingApiKey(Provider::Anthropic))
        ));
    }

    #[test]
    fn endpoint_normalization_tolerates_existing_paths() {
        assert_eq!(
            Provider::Anthropic.endpoint("https://proxy.example/v1"),
            "https://proxy.example/v1/messages"
        );
        assert_eq!(
            Provider::OpenAiCompatible.endpoint("https://a.example/v1/chat/completions"),
            "https://a.example/v1/chat/completions"
        );
        assert_eq!(
            Provider::Ollama.endpoint("http://127.0.0.1:11434/api/"),
            "http://127.0.0.1:11434/api/chat"
        );
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let mut bad = config(Provider::Ollama);
        bad.base_url = "localhost:11434".into();
        assert!(bad.validate().is_err());
        let mut bad = config(Provider::Ollama);
        bad.model = "  ".into();
        assert!(bad.validate().is_err());
        let mut bad = config(Provider::Ollama);
        bad.temperature = Some(f32::NAN);
        assert!(bad.validate().is_err());
        let mut bad = config(Provider::Ollama);
        bad.temperature = Some(2.5);
        assert!(bad.validate().is_err());
        let mut bad = config(Provider::OpenAiCompatible);
        bad.api_key = Some("safe-prefix\r\nx-injected: yes".into());
        let error = bad.validate().unwrap_err().to_string();
        assert!(error.contains("control character"));
        assert!(!error.contains("safe-prefix"));
        let mut bad = config(Provider::OpenAiCompatible);
        bad.api_key = Some("x".repeat(MAX_API_KEY_BYTES + 1));
        assert!(bad.validate().is_err());
    }

    #[test]
    fn base_urls_must_be_https_unless_ollama_stays_on_loopback() {
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let mut config = config(provider);
            config.base_url = provider.default_base_url().into();
            assert!(
                config.validate().is_ok(),
                "{provider:?} must accept its own default endpoint"
            );
        }

        // Plain HTTP is a loopback Ollama concession, not a general one.
        for loopback in [
            "http://localhost:11434",
            "http://LOCALHOST",
            "http://127.0.0.1:11434",
            "http://127.6.6.6",
            "http://[::1]:11434",
        ] {
            let mut ollama = config(Provider::Ollama);
            ollama.base_url = loopback.into();
            assert!(ollama.validate().is_ok(), "{loopback} is loopback");

            let mut cloud = config(Provider::OpenAiCompatible);
            cloud.base_url = loopback.into();
            assert!(
                cloud.validate().is_err(),
                "{loopback} must not downgrade a cloud provider to HTTP"
            );
        }
        for remote in [
            "http://example.com",
            "http://localhost.example.com",
            "http://127.0.0.1.example.com",
            "http://[::2]:11434",
            "http://127.0.0.1:not-a-port",
        ] {
            let mut ollama = config(Provider::Ollama);
            ollama.base_url = remote.into();
            assert!(
                ollama.validate().is_err(),
                "{remote} must not pass as loopback"
            );
        }
    }

    #[test]
    fn base_urls_and_models_reject_smuggled_components() {
        for hostile in [
            "https:///missing-authority",
            "https://user:secret@example.com/v1",
            "https://example.com/v1?api-key=secret",
            "https://example.com/v1#fragment",
            "https://example.com\\v1",
            "https://exam\u{200b}ple.com",
            "https://example.com/\u{202e}v1",
            "https://example.com/v 1",
            "ftp://example.com",
            "example.com",
            "",
        ] {
            let mut bad = config(Provider::OpenAiCompatible);
            bad.base_url = hostile.into();
            assert!(bad.validate().is_err(), "{hostile:?} must be rejected");
        }

        let mut long = config(Provider::OpenAiCompatible);
        long.base_url = format!("https://example.com/{}", "x".repeat(MAX_BASE_URL_BYTES));
        assert!(long.validate().is_err());

        for hostile in ["gpt\u{202e}4o", "gpt\u{200b}4o", "gpt\t4o", "gpt\n4o"] {
            let mut bad = config(Provider::OpenAiCompatible);
            bad.model = hostile.into();
            assert!(bad.validate().is_err(), "{hostile:?} must be rejected");
        }
        let mut long = config(Provider::OpenAiCompatible);
        long.model = "m".repeat(MAX_MODEL_BYTES + 1);
        assert!(long.validate().is_err());
    }

    #[test]
    fn builders_bound_history_and_system_prompts_without_the_caller() {
        let history: Vec<Message> = (0..MAX_REQUEST_HISTORY_TURNS * 3)
            .map(|index| user(&format!("turn {index}")))
            .collect();
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let config = config(provider);
            let built = build_chat_request_with_report(&config, Some("sys"), &history).unwrap();
            assert_eq!(
                built.omitted_history_turns,
                MAX_REQUEST_HISTORY_TURNS * 2,
                "{provider:?}"
            );
            assert_eq!(
                build_chat_request(&config, Some("sys"), &history).unwrap(),
                built.request,
                "legacy and reported builders must be byte-identical for {provider:?}"
            );

            let body: Value = serde_json::from_str(&built.request.body).unwrap();
            let messages = body["messages"].as_array().unwrap();
            let messages = if provider == Provider::Anthropic {
                messages.as_slice()
            } else {
                // OpenAI-compatible and Ollama carry `system` as their first
                // message; the history window follows it.
                &messages[1..]
            };
            assert_eq!(messages.len(), MAX_REQUEST_HISTORY_TURNS);
            // The newest turns are the ones kept.
            assert_eq!(
                messages.last().unwrap()["content"],
                format!("turn {}", MAX_REQUEST_HISTORY_TURNS * 3 - 1)
            );

            let streaming =
                build_chat_request_streaming_with_report(&config, Some("sys"), &history).unwrap();
            assert_eq!(
                streaming.omitted_history_turns, built.omitted_history_turns,
                "{provider:?}"
            );
            assert_eq!(
                build_chat_request_streaming(&config, Some("sys"), &history).unwrap(),
                streaming.request,
                "streaming legacy and reported builders must agree for {provider:?}"
            );

            // Bounding is idempotent. The second invocation reports only the
            // loss it introduced, not the 80 turns its caller already removed.
            let (bounded, omitted) = bound_history(&history);
            assert_eq!(omitted, MAX_REQUEST_HISTORY_TURNS * 2);
            let prepared = build_chat_request_with_report(&config, Some("sys"), &bounded).unwrap();
            assert_eq!(prepared.omitted_history_turns, 0, "{provider:?}");
            assert_eq!(prepared.request, built.request, "{provider:?}");
        }

        let oversized = "s".repeat(MAX_REQUEST_SYSTEM_BYTES + 1);
        assert!(matches!(
            build_chat_request(&config(Provider::Anthropic), Some(&oversized), &history),
            Err(ProviderError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn reported_builder_counts_byte_budget_omissions() {
        let history = vec![
            user(&"a".repeat(MAX_REQUEST_TURN_BYTES)),
            user(&"b".repeat(MAX_REQUEST_TURN_BYTES)),
            user(&"c".repeat(MAX_REQUEST_TURN_BYTES)),
        ];
        let built =
            build_chat_request_with_report(&config(Provider::OpenAiCompatible), None, &history)
                .unwrap();
        assert_eq!(built.omitted_history_turns, 2);
        let body: Value = serde_json::from_str(&built.request.body).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert!(messages[0]["content"].as_str().unwrap().starts_with('c'));
    }

    #[test]
    fn singleton_assistant_preserves_the_legacy_wire_and_reports_no_omission() {
        let (bounded, omitted) = bound_history(&[assistant("orphan")]);
        assert_eq!(bounded, vec![assistant("orphan")]);
        assert_eq!(omitted, 0);

        let golden = [
            (
                Provider::Anthropic,
                r#"{"max_tokens":512,"messages":[{"content":"orphan","role":"assistant"}],"model":"test-model","system":"sys"}"#,
            ),
            (
                Provider::OpenAiCompatible,
                r#"{"max_tokens":512,"messages":[{"content":"sys","role":"system"},{"content":"orphan","role":"assistant"}],"model":"test-model"}"#,
            ),
            (
                Provider::Ollama,
                r#"{"messages":[{"content":"sys","role":"system"},{"content":"orphan","role":"assistant"}],"model":"test-model","options":{"num_predict":512},"stream":false}"#,
            ),
        ];
        for (provider, expected_body) in golden {
            let config = config(provider);
            let history = [assistant("orphan")];
            let built = build_chat_request_with_report(&config, Some("sys"), &history).unwrap();
            assert_eq!(built.omitted_history_turns, 0, "{provider:?}");
            assert_eq!(built.request.body, expected_body, "{provider:?}");
            assert_eq!(
                build_chat_request(&config, Some("sys"), &history)
                    .unwrap()
                    .body,
                expected_body,
                "legacy builder drifted for {provider:?}"
            );
        }

        let (bounded, omitted) = bound_history(&[assistant("old"), user("current")]);
        assert_eq!(bounded, vec![user("current")]);
        assert_eq!(omitted, 1);
    }

    #[test]
    fn oversized_single_turn_is_elided_but_not_reported_as_omitted() {
        let oversized = "界".repeat(MAX_REQUEST_TURN_BYTES / "界".len() + 1);
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let built =
                build_chat_request_with_report(&config(provider), None, &[user(&oversized)])
                    .unwrap();
            assert_eq!(built.omitted_history_turns, 0, "{provider:?}");
            assert!(built.request.body.contains("bytes elided"), "{provider:?}");
            assert!(!built.request.body.contains(&oversized), "{provider:?}");
        }
    }

    #[test]
    fn byte_oriented_response_parsing_bounds_the_envelope() {
        let body = serde_json::json!({"content": [{"type": "text", "text": "hi"}]}).to_string();
        assert_eq!(
            parse_chat_response_bytes(Provider::Anthropic, body.as_bytes()).unwrap(),
            "hi"
        );

        let structured = serde_json::json!({
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 17, "output_tokens": 23},
        })
        .to_string();
        assert_eq!(
            parse_chat_response_full_bytes(Provider::Anthropic, structured.as_bytes()).unwrap(),
            ChatResponse {
                text: "partial".into(),
                reached_token_limit: true,
                usage: Some(Usage {
                    input_tokens: Some(17),
                    output_tokens: Some(23),
                }),
            }
        );
        let rendered =
            parse_chat_response_bytes(Provider::Anthropic, structured.as_bytes()).unwrap();
        assert!(rendered.starts_with("partial\n\n[Response reached"));

        assert!(matches!(
            parse_chat_response_bytes(Provider::Anthropic, b"{not json"),
            Err(ProviderError::MalformedResponse(_))
        ));
        assert!(matches!(
            parse_chat_response_full_bytes(Provider::Anthropic, b"{not json"),
            Err(ProviderError::MalformedResponse(_))
        ));
        let huge = vec![b' '; MAX_RESPONSE_JSON_BYTES + 1];
        assert!(matches!(
            parse_chat_response_full_bytes(Provider::Anthropic, &huge),
            Err(ProviderError::ResponseTooLarge {
                limit: MAX_RESPONSE_JSON_BYTES
            })
        ));
    }

    #[test]
    fn temperature_is_optional_and_provider_shaped() {
        let history = [user("hi")];
        let mut with_temperature = config(Provider::Anthropic);
        with_temperature.temperature = Some(0.0);
        let request = build_chat_request(&with_temperature, None, &history).unwrap();
        let body: Value = serde_json::from_str(&request.body).unwrap();
        assert_eq!(body["temperature"], 0.0);

        let mut ollama = config(Provider::Ollama);
        ollama.temperature = Some(0.2);
        let request = build_chat_request(&ollama, None, &history).unwrap();
        let body: Value = serde_json::from_str(&request.body).unwrap();
        assert!((body["options"]["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);

        let request =
            build_chat_request(&config(Provider::OpenAiCompatible), None, &history).unwrap();
        let body: Value = serde_json::from_str(&request.body).unwrap();
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn history_bounding_keeps_recent_complete_context() {
        let mut history = Vec::new();
        for index in 0..100 {
            history.push(Message {
                role: if index % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                text: format!("turn {index}"),
            });
        }
        let (retained, omitted) = bound_history(&history);
        assert_eq!(retained.len() + omitted, 100);
        assert!(retained.len() <= MAX_REQUEST_HISTORY_TURNS);
        assert_eq!(retained.first().map(|turn| turn.role), Some(Role::User));
        assert_eq!(retained.last().unwrap().text, "turn 99");
    }

    #[test]
    fn responses_are_extracted_per_provider() {
        let anthropic = serde_json::json!({
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        });
        assert_eq!(
            parse_chat_response(Provider::Anthropic, &anthropic).unwrap(),
            "hi"
        );

        let openai = serde_json::json!({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        });
        assert_eq!(
            parse_chat_response(Provider::OpenAiCompatible, &openai).unwrap(),
            "hello"
        );

        let ollama = serde_json::json!({"message": {"content": "yo"}});
        assert_eq!(
            parse_chat_response(Provider::Ollama, &ollama).unwrap(),
            "yo"
        );

        let truncated = serde_json::json!({
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
        });
        let text = parse_chat_response(Provider::Anthropic, &truncated).unwrap();
        assert!(text.starts_with("partial"));
        assert!(text.contains("output limit"));

        let empty = serde_json::json!({"content": []});
        assert!(matches!(
            parse_chat_response(Provider::Anthropic, &empty),
            Err(ProviderError::EmptyResponse)
        ));
    }

    #[test]
    fn response_text_budget_is_enforced_while_blocks_are_joined() {
        // Each block fits independently, but the joined value (including its
        // separator) does not. Parsing must reject before constructing an
        // oversized aggregate string.
        let half = "x".repeat(MAX_MODEL_TEXT_BYTES / 2);
        let anthropic = json!({
            "content": [
                {"type": "text", "text": half},
                {"type": "text", "text": half},
            ]
        });
        assert!(matches!(
            parse_chat_response_full(Provider::Anthropic, &anthropic),
            Err(ProviderError::ResponseTooLarge {
                limit: MAX_MODEL_TEXT_BYTES
            })
        ));

        let openai = json!({
            "choices": [{"message": {"content": [
                {"type": "text", "text": half},
                {"type": "text", "text": half},
            ]}}]
        });
        assert!(matches!(
            parse_chat_response_full(Provider::OpenAiCompatible, &openai),
            Err(ProviderError::ResponseTooLarge {
                limit: MAX_MODEL_TEXT_BYTES
            })
        ));

        let ollama = json!({
            "message": {"content": "x".repeat(MAX_MODEL_TEXT_BYTES + 1)}
        });
        assert!(matches!(
            parse_chat_response_full(Provider::Ollama, &ollama),
            Err(ProviderError::ResponseTooLarge {
                limit: MAX_MODEL_TEXT_BYTES
            })
        ));
    }

    #[test]
    fn full_parse_returns_raw_text_flag_and_usage() {
        let anthropic = serde_json::json!({
            "content": [{"type": "text", "text": "partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 12, "output_tokens": 34},
        });
        let parsed = parse_chat_response_full(Provider::Anthropic, &anthropic).unwrap();
        assert_eq!(parsed.text, "partial");
        assert!(parsed.reached_token_limit);
        assert_eq!(
            parsed.usage,
            Some(Usage {
                input_tokens: Some(12),
                output_tokens: Some(34),
            })
        );

        let openai = serde_json::json!({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        });
        let parsed = parse_chat_response_full(Provider::OpenAiCompatible, &openai).unwrap();
        assert!(!parsed.reached_token_limit);
        assert_eq!(
            parsed.usage,
            Some(Usage {
                input_tokens: Some(5),
                output_tokens: Some(7),
            })
        );

        let ollama = serde_json::json!({
            "message": {"content": "yo"},
            "prompt_eval_count": 3,
            "eval_count": 9,
        });
        let parsed = parse_chat_response_full(Provider::Ollama, &ollama).unwrap();
        assert_eq!(
            parsed.usage,
            Some(Usage {
                input_tokens: Some(3),
                output_tokens: Some(9),
            })
        );

        // Responses without usage fields yield None, not zeros.
        let bare = serde_json::json!({"message": {"content": "yo"}});
        let parsed = parse_chat_response_full(Provider::Ollama, &bare).unwrap();
        assert_eq!(parsed.usage, None);
    }

    #[test]
    fn history_bounding_prepare_hook_runs_before_the_byte_budget() {
        // The hook's output, not the raw text, must be what the budget
        // measures: a hook that shrinks an oversized turn keeps it intact.
        let oversized = "x".repeat(MAX_REQUEST_TURN_BYTES * 2);
        let history = [user(&oversized)];
        let (retained, _) = bound_history_with(&history, |text| text[..8].to_string());
        assert_eq!(retained[0].text, "xxxxxxxx");

        let (retained, _) = bound_history_with(&[user("secret")], |text| {
            text.replace("secret", "[redacted]")
        });
        assert_eq!(retained[0].text, "[redacted]");
    }
}
