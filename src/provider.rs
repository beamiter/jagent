//! Provider-neutral chat request construction and response parsing.
//!
//! Sans-IO: [`build_chat_request_with_report`] returns a [`BuiltRequest`]
//! describing exactly one POST and how many history turns this build omitted;
//! the integration performs its [`HttpRequest`] with whatever HTTP stack it
//! already trusts (curl child process, ureq, …) and hands the response JSON to
//! [`parse_chat_response`]. Nothing in this module opens a socket. The shorter
//! [`build_chat_request`] compatibility entry point returns only the request.

use crate::safety::is_unsafe_invisible_char;
use crate::text::{ceil_char_boundary, elide_middle, floor_char_boundary};
use crate::tools::{agent_body_fields, AgentProtocol};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{self, Write};
use std::str::FromStr;

pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";
pub const MAX_MODEL_TEXT_BYTES: usize = 256 * 1024;
pub const MAX_REQUEST_HISTORY_TURNS: usize = 40;
/// Ceiling for encoded retained-history message objects (including JSON
/// escaping and separators, excluding only the surrounding array brackets).
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
/// Byte ceiling for the complete encoded JSON request body. The history and
/// system budgets keep ordinary requests well below this; this final guard
/// also covers JSON escaping and provider-specific extension fields.
pub const MAX_REQUEST_JSON_BYTES: usize = 4 * 1024 * 1024;
/// Byte ceiling for a completed provider endpoint after its fixed path is
/// appended to a validated base URL.
pub const MAX_REQUEST_URL_BYTES: usize = MAX_BASE_URL_BYTES + 64;
/// Maximum number of headers in one sans-I/O request.
pub const MAX_REQUEST_HEADERS: usize = 16;
/// Aggregate header-name and header-value bytes in one request. Framing added
/// by an HTTP implementation is intentionally excluded.
pub const MAX_REQUEST_HEADER_BYTES: usize = 64 * 1024;
/// Per-extension encoded JSON ceiling, measured before cloning its [`Value`].
pub const MAX_REQUEST_EXTENSION_JSON_BYTES: usize = 1024 * 1024;
/// Aggregate encoded JSON ceiling for provider extensions.
pub const MAX_REQUEST_EXTENSIONS_JSON_BYTES: usize = 2 * 1024 * 1024;
/// Generous library-side guard against a corrupt setting becoming an
/// implausibly large provider generation request.
pub const MAX_REQUEST_MAX_TOKENS: u32 = 1_000_000;
/// Byte ceiling for one encoded non-streaming response envelope, applied by
/// [`parse_chat_response_bytes`]. Integrations that decode the body themselves
/// must apply an equivalent transport limit before calling the `Value`-based
/// entry points.
pub const MAX_RESPONSE_JSON_BYTES: usize = 1024 * 1024;
/// Byte ceiling for one provider name parsed from configuration.
pub const MAX_PROVIDER_NAME_BYTES: usize = 64;

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

    /// Append this provider's path to an already trusted base URL.
    ///
    /// This compatibility helper does not validate `base_url`; outbound
    /// request code should prefer [`ChatConfig::endpoint`].
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
        if value.len() > MAX_PROVIDER_NAME_BYTES
            || value.chars().any(char::is_control)
            || value.chars().any(is_unsafe_invisible_char)
        {
            return Err(ProviderError::InvalidConfiguration(
                "AI provider name is invalid or exceeds its byte limit".into(),
            ));
        }
        let value = value.trim();
        if value.eq_ignore_ascii_case("anthropic") || value.eq_ignore_ascii_case("claude") {
            Ok(Self::Anthropic)
        } else if value.eq_ignore_ascii_case("openai")
            || value.eq_ignore_ascii_case("openai-compatible")
            || value.eq_ignore_ascii_case("openai_compatible")
        {
            Ok(Self::OpenAiCompatible)
        } else if value.eq_ignore_ascii_case("ollama") {
            Ok(Self::Ollama)
        } else {
            Err(ProviderError::InvalidConfiguration(
                "unknown AI provider".into(),
            ))
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

/// Non-sensitive accounting for one [`HttpRequest`].
///
/// Counts deliberately omit every header name/value and all URL/body text, so
/// callers may report this value without creating a credential or prompt sink.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HttpRequestMetrics {
    pub url_bytes: usize,
    pub header_count: usize,
    pub header_bytes: usize,
    pub body_bytes: usize,
}

impl HttpRequest {
    /// Return checked, content-free wire-size accounting.
    pub fn transport_metrics(&self) -> Result<HttpRequestMetrics, ProviderError> {
        let header_bytes = self
            .headers
            .iter()
            .try_fold(0_usize, |total, (name, value)| {
                total.checked_add(name.len())?.checked_add(value.len())
            });
        let Some(header_bytes) = header_bytes else {
            return Err(ProviderError::InvalidConfiguration(
                "request header byte accounting overflowed".into(),
            ));
        };
        Ok(HttpRequestMetrics {
            url_bytes: self.url.len(),
            header_count: self.headers.len(),
            header_bytes,
            body_bytes: self.body.len(),
        })
    }

    /// Validate the complete sans-I/O value immediately before transport.
    ///
    /// Builders call this themselves. It is public because `HttpRequest` has
    /// public fields for compatibility and an integration may mutate or build
    /// one directly before handing it to an HTTP stack.
    pub fn validate_transport(&self) -> Result<HttpRequestMetrics, ProviderError> {
        let metrics = self.transport_metrics()?;
        if !transport_url_is_valid(&self.url, MAX_REQUEST_URL_BYTES) {
            return Err(ProviderError::InvalidConfiguration(
                "request URL is not a bounded absolute HTTPS or loopback HTTP URL".into(),
            ));
        }
        if metrics.header_count > MAX_REQUEST_HEADERS
            || metrics.header_bytes > MAX_REQUEST_HEADER_BYTES
        {
            return Err(ProviderError::InvalidConfiguration(
                "request headers exceed their count or byte limit".into(),
            ));
        }

        let mut content_type_count = 0_usize;
        for (index, (name, value)) in self.headers.iter().enumerate() {
            if !is_lowercase_header_name(name) {
                return Err(ProviderError::InvalidConfiguration(
                    "request header name is not canonical lowercase HTTP token text".into(),
                ));
            }
            if !value.bytes().all(|byte| matches!(byte, 0x20..=0x7e)) {
                return Err(ProviderError::InvalidConfiguration(
                    "request header value is not printable ASCII".into(),
                ));
            }
            if self.headers[..index]
                .iter()
                .any(|(previous, _)| previous == name)
            {
                return Err(ProviderError::InvalidConfiguration(
                    "request contains a duplicate header name".into(),
                ));
            }
            if name == "content-type" {
                content_type_count = content_type_count.saturating_add(1);
                if value != "application/json" {
                    return Err(ProviderError::InvalidConfiguration(
                        "request content-type must be application/json".into(),
                    ));
                }
            }
        }
        if content_type_count != 1 {
            return Err(ProviderError::InvalidConfiguration(
                "request must contain exactly one JSON content-type header".into(),
            ));
        }
        if metrics.body_bytes > MAX_REQUEST_JSON_BYTES || !is_json_object(&self.body) {
            return Err(ProviderError::InvalidConfiguration(
                "request body must be one JSON object within its byte limit".into(),
            ));
        }
        Ok(metrics)
    }
}

fn is_lowercase_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn is_json_object(body: &str) -> bool {
    struct ObjectOnly;

    impl<'de> Deserialize<'de> for ObjectOnly {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            struct Visitor;

            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = ObjectOnly;

                fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    formatter.write_str("a JSON object")
                }

                fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
                where
                    M: serde::de::MapAccess<'de>,
                {
                    let mut fields = HashSet::new();
                    while let Some(field) = map.next_key::<String>()? {
                        if !fields.insert(field) {
                            return Err(serde::de::Error::custom(
                                "duplicate top-level request field",
                            ));
                        }
                        map.next_value::<serde::de::IgnoredAny>()?;
                    }
                    Ok(ObjectOnly)
                }
            }

            deserializer.deserialize_map(Visitor)
        }
    }

    let mut deserializer = serde_json::Deserializer::from_str(body);
    ObjectOnly::deserialize(&mut deserializer).is_ok() && deserializer.end().is_ok()
}

impl std::fmt::Debug for HttpRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let metrics = self.transport_metrics().unwrap_or(HttpRequestMetrics {
            url_bytes: self.url.len(),
            header_count: self.headers.len(),
            header_bytes: usize::MAX,
            body_bytes: self.body.len(),
        });
        formatter
            .debug_struct("HttpRequest")
            // A caller can construct this public sans-I/O value directly;
            // avoid echoing URL userinfo/query credentials in diagnostics.
            .field("url_bytes", &metrics.url_bytes)
            // Header names and values are both caller-controlled. A finite
            // sensitive-name list cannot cover cookies, proxy credentials,
            // provider extensions, or application-specific secret headers,
            // while hostile names can also forge log structure. Keep only
            // the count in Debug; the transport still receives exact bytes.
            .field("header_count", &metrics.header_count)
            .field("header_bytes", &metrics.header_bytes)
            // Request bodies contain user context and can therefore contain
            // credentials that no finite pattern list will reliably catch.
            // Keep Debug useful for transport diagnostics without turning an
            // innocent tracing statement into a second secret sink.
            .field("body_bytes", &metrics.body_bytes)
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

/// Chat client configuration owned by the integration.
#[derive(Clone, PartialEq)]
pub struct ChatConfig {
    pub provider: Provider,
    pub api_key: Option<String>,
    pub model: String,
    pub base_url: String,
    pub max_tokens: u32,
    /// Sampling temperature. `None` keeps the provider/model default and is
    /// the portable choice: some reasoning-capable models reject an explicit
    /// temperature. Set a value only when the selected endpoint documents
    /// support for it.
    pub temperature: Option<f32>,
}

impl std::fmt::Debug for ChatConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatConfig")
            .field("provider", &self.provider)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("model_bytes", &self.model.len())
            .field("base_url_bytes", &self.base_url.len())
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
        // Validate the exact bytes that request construction will use. The
        // previous `trim()` accepted surrounding whitespace here but later
        // built an unusable, transport-dependent URL from the original value.
        validate_base_url(&self.base_url)?;
        if !(1..=MAX_REQUEST_MAX_TOKENS).contains(&self.max_tokens) {
            return Err(ProviderError::InvalidConfiguration(format!(
                "max_tokens must be between 1 and {MAX_REQUEST_MAX_TOKENS}"
            )));
        }
        if let Some(api_key) = self.api_key.as_deref() {
            validate_api_key(api_key)?;
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

    /// Return the exact provider endpoint only after validating every
    /// transport-relevant field in this configuration.
    pub fn endpoint(&self) -> Result<String, ProviderError> {
        self.validate()?;
        Ok(self.provider.endpoint(&self.base_url))
    }
}

/// Validate the exact credential bytes request construction will place in an
/// HTTP header. Silently trimming a settings value makes configuration
/// diagnostics disagree with the request, while non-ASCII or whitespace
/// bytes are accepted inconsistently by HTTP client implementations.
fn validate_api_key(api_key: &str) -> Result<(), ProviderError> {
    if api_key.is_empty() {
        return Err(ProviderError::InvalidConfiguration(
            "API key must not be empty".into(),
        ));
    }
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
    if !api_key.bytes().all(|byte| matches!(byte, 0x21..=0x7e)) {
        return Err(ProviderError::InvalidConfiguration(
            "API key must contain only visible ASCII characters with no whitespace".into(),
        ));
    }
    Ok(())
}

fn validate_model(model: &str) -> Result<(), ProviderError> {
    if model.trim().is_empty()
        || model.trim() != model
        || model.len() > MAX_MODEL_BYTES
        || model.chars().any(char::is_control)
        || model.chars().any(is_unsafe_invisible_char)
    {
        return Err(ProviderError::InvalidConfiguration(format!(
            "model must be visible, non-empty text of at most {MAX_MODEL_BYTES} bytes with no \
             surrounding whitespace"
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
/// a configured host read as one origin while resolving as another. Plain
/// HTTP is accepted only for a syntactic loopback host. This applies to every
/// provider because OpenAI-compatible local servers and local provider
/// proxies are common, while a remote clear-text endpoint would expose both
/// prompts and credentials.
fn validate_base_url(base_url: &str) -> Result<(), ProviderError> {
    let invalid = || {
        ProviderError::InvalidConfiguration(format!(
            "base URL must be an absolute HTTPS URL of at most {MAX_BASE_URL_BYTES} bytes with \
             an ASCII DNS name or canonical IP literal and no surrounding whitespace, \
             credentials, query, fragment, backslash, control, or visually ambiguous characters \
             (plain HTTP is accepted only for a loopback endpoint)"
        ))
    };
    transport_url_is_valid(base_url, MAX_BASE_URL_BYTES)
        .then_some(())
        .ok_or_else(invalid)
}

fn transport_url_is_valid(url: &str, max_bytes: usize) -> bool {
    if url.is_empty()
        || url.trim() != url
        || url.len() > max_bytes
        || url.contains(['?', '#', '\\'])
        || url.chars().any(char::is_whitespace)
        || url.chars().any(char::is_control)
        || url.chars().any(is_unsafe_invisible_char)
    {
        return false;
    }
    let Some((scheme, rest)) = url.split_once("://") else {
        return false;
    };
    let authority = rest.split('/').next().unwrap_or_default();
    let Some(host) = authority_host(authority) else {
        return false;
    };
    if authority.contains('@') || !is_valid_url_host(host) {
        return false;
    }
    match scheme {
        "https" => true,
        "http" if is_loopback_authority(authority) => true,
        _ => false,
    }
}

/// Accept a canonical IP literal or an ASCII DNS hostname.
///
/// URL parsers disagree in particularly dangerous ways around percent-encoded
/// hosts, Unicode/IDNA input, empty labels, and legacy numeric IPv4 spellings.
/// Requiring callers to supply an already-ASCII hostname (punycode for an IDN)
/// keeps the authority the validator reviews identical to the authority a
/// transport resolves. Canonical IPv4 and bracketed IPv6 literals remain
/// available for local endpoints.
fn is_valid_url_host(host: &str) -> bool {
    if host.parse::<std::net::Ipv4Addr>().is_ok() || host.parse::<std::net::Ipv6Addr>().is_ok() {
        return true;
    }
    if host.is_empty() || host.len() > 253 || !host.is_ascii() {
        return false;
    }

    let mut labels = host.split('.').peekable();
    while let Some(label) = labels.next() {
        if label.is_empty()
            || label.len() > 63
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || !label
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphanumeric)
            || !label
                .as_bytes()
                .last()
                .is_some_and(u8::is_ascii_alphanumeric)
        {
            return false;
        }

        // WHATWG-style URL parsers treat a host whose final label is a
        // decimal/octal/hex number as an IPv4 address, including spellings
        // such as 2130706433 and 0x7f000001. `Ipv4Addr` above accepts only the
        // canonical dotted form; reject every other numeric-final spelling so
        // a transport cannot resolve a different host than this validator saw.
        if labels.peek().is_none() && is_url_ipv4_number(label) {
            return false;
        }
    }
    true
}

fn is_url_ipv4_number(label: &str) -> bool {
    if let Some(hex) = label
        .strip_prefix("0x")
        .or_else(|| label.strip_prefix("0X"))
    {
        return !hex.is_empty() && hex.bytes().all(|byte| byte.is_ascii_hexdigit());
    }
    label.bytes().all(|byte| byte.is_ascii_digit())
}

/// True for `localhost`, an IPv4 loopback literal, or a bracketed IPv6
/// loopback literal, each with an optional numeric port.
fn is_loopback_authority(authority: &str) -> bool {
    let Some(host) = authority_host(authority) else {
        return false;
    };
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::Ipv4Addr>()
            .is_ok_and(|address| address.is_loopback())
        || host
            .parse::<std::net::Ipv6Addr>()
            .is_ok_and(|address| address.is_loopback())
}

/// Resolve the host slice only when the complete authority has a syntactically
/// valid bracket/port shape. This deliberately does not perform DNS or IDNA
/// processing, which belongs to the transport, but it prevents validation
/// from accepting a malformed authority that a transport might reinterpret.
fn authority_host(authority: &str) -> Option<&str> {
    if authority.is_empty() {
        return None;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let (literal, suffix) = rest.split_once(']')?;
        if literal.parse::<std::net::Ipv6Addr>().is_err() {
            return None;
        }
        if !suffix.is_empty() && !suffix.strip_prefix(':').is_some_and(is_port) {
            return None;
        }
        return Some(literal);
    }
    if authority.contains(['[', ']']) {
        return None;
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if !host.contains(':') && !host.is_empty() && is_port(port) => {
            Some(host)
        }
        Some(_) => None,
        None => Some(authority),
    }
}

fn is_port(port: &str) -> bool {
    !port.is_empty()
        && port.chars().all(|digit| digit.is_ascii_digit())
        && port.parse::<u16>().is_ok_and(|port| port != 0)
}

/// Bound a conversation history to the request budgets, newest-first. Leading
/// assistant turns are removed while at least one later turn can remain; a
/// singleton assistant turn is retained for compatibility with existing wire
/// requests.
/// Returns the retained history and how many older turns were omitted.
pub fn bound_history(history: &[Message]) -> (Vec<Message>, usize) {
    let prepared = bound_history_cow_with_report(history, Cow::Borrowed);
    (prepared.messages, prepared.report.omitted_history_turns)
}

/// Machine-readable account of the transformations applied while preparing a
/// provider history window.
///
/// `changed_history_turns` counts retained turns changed by the caller's
/// preparation hook (for example secret redaction). `elided_history_turns`
/// counts retained turns whose prepared text exceeded the per-turn budget.
/// Together with `omitted_history_turns`, these fields make every lossy
/// history transformation visible without including any sensitive content in
/// the report itself.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HistoryReport {
    /// Number of turns supplied before preparation and bounding.
    pub input_history_turns: usize,
    /// Number of turns retained in the prepared request window.
    pub sent_history_turns: usize,
    /// Number of older turns omitted entirely.
    pub omitted_history_turns: usize,
    /// Number of retained turns changed by the preparation hook.
    pub changed_history_turns: usize,
    /// Number of retained turns shortened by middle elision.
    pub elided_history_turns: usize,
    /// UTF-8 text bytes retained across the sent turns, excluding JSON and
    /// provider framing overhead.
    pub sent_history_text_bytes: usize,
    /// Encoded JSON bytes occupied by the retained message objects and their
    /// separating commas, excluding the surrounding array brackets.
    pub sent_history_json_bytes: usize,
}

/// A bounded history together with a complete loss/preparation report.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "inspect report before sending a request built from this history"]
pub struct PreparedHistory {
    /// Bounded turns, kept in their original chronological order.
    pub messages: Vec<Message>,
    /// Non-sensitive diagnostics for the preparation and bounding pass.
    pub report: HistoryReport,
}

/// [`bound_history`] with machine-readable diagnostics for elision as well as
/// whole-turn omission.
pub fn bound_history_with_report(history: &[Message]) -> PreparedHistory {
    bound_history_cow_with_report(history, Cow::Borrowed)
}

/// [`bound_history`] with a per-turn preparation hook (redaction, normalization)
/// applied to each turn's text *before* the byte budget elides it, so the
/// budget is measured against what will actually be sent.
pub fn bound_history_with(
    history: &[Message],
    prepare: impl Fn(&str) -> String,
) -> (Vec<Message>, usize) {
    let prepared = bound_history_prepared_with_report(history, prepare);
    (prepared.messages, prepared.report.omitted_history_turns)
}

/// [`bound_history_with`] plus diagnostics describing every retained turn
/// changed by `prepare` and every retained turn elided by the byte budget.
pub fn bound_history_prepared_with_report(
    history: &[Message],
    prepare: impl Fn(&str) -> String,
) -> PreparedHistory {
    bound_history_cow_with_report(history, |text| Cow::Owned(prepare(text)))
}

/// Prepare and bound history without requiring the preparation hook to clone
/// clean or oversized input. A hook such as
/// [`crate::redact::redact_secrets_cow`] can borrow an unchanged turn; the
/// final retained [`Message`] is allocated only after the per-turn ceiling is
/// known.
pub fn bound_history_cow_with_report<'a>(
    history: &'a [Message],
    prepare: impl Fn(&'a str) -> Cow<'a, str>,
) -> PreparedHistory {
    struct PreparedTurn {
        message: Message,
        changed: bool,
        elided: bool,
    }

    let mut retained_reversed: Vec<PreparedTurn> = Vec::new();
    let mut retained_bytes = 0_usize;
    for turn in history.iter().rev() {
        if retained_reversed.len() >= MAX_REQUEST_HISTORY_TURNS {
            break;
        }
        let prepared = prepare(&turn.text);
        let changed = prepared.as_ref() != turn.text;
        let (text, elided, message_bytes) = bound_history_turn(turn.role, &prepared);
        let cost = message_bytes.saturating_add(usize::from(!retained_reversed.is_empty()));
        if !retained_reversed.is_empty()
            && retained_bytes.saturating_add(cost) > MAX_REQUEST_HISTORY_BYTES
        {
            break;
        }
        retained_bytes = retained_bytes.saturating_add(cost);
        retained_reversed.push(PreparedTurn {
            message: Message {
                role: turn.role,
                text,
            },
            changed,
            elided,
        });
    }
    retained_reversed.reverse();
    let mut omitted_history_turns = history.len().saturating_sub(retained_reversed.len());
    while retained_reversed.len() > 1
        && retained_reversed
            .first()
            .is_some_and(|turn| turn.message.role == Role::Assistant)
    {
        retained_reversed.remove(0);
        omitted_history_turns = omitted_history_turns.saturating_add(1);
    }
    let report = HistoryReport {
        input_history_turns: history.len(),
        sent_history_turns: retained_reversed.len(),
        omitted_history_turns,
        changed_history_turns: retained_reversed.iter().filter(|turn| turn.changed).count(),
        elided_history_turns: retained_reversed.iter().filter(|turn| turn.elided).count(),
        sent_history_text_bytes: retained_reversed
            .iter()
            .map(|turn| turn.message.text.len())
            .fold(0_usize, usize::saturating_add),
        sent_history_json_bytes: retained_reversed
            .iter()
            .enumerate()
            .map(|(index, turn)| {
                history_turn_wire_bytes(turn.message.role, &turn.message.text)
                    .saturating_add(usize::from(index > 0))
            })
            .fold(0_usize, usize::saturating_add),
    };
    PreparedHistory {
        messages: retained_reversed
            .into_iter()
            .map(|turn| turn.message)
            .collect(),
        report,
    }
}

fn bound_history_turn(role: Role, text: &str) -> (String, bool, usize) {
    let initial_limit = text.len().min(MAX_REQUEST_TURN_BYTES);
    if elided_history_turn_wire_bytes(role, text, initial_limit) <= MAX_REQUEST_HISTORY_BYTES {
        let bounded = elide_middle(text, initial_limit);
        let bytes = history_turn_wire_bytes(role, &bounded);
        return (bounded, initial_limit < text.len(), bytes);
    }

    // Find the largest raw-text budget whose JSON representation still fits.
    // The oracle computes slices without allocating, so hostile escape-heavy
    // text causes exactly one final sample allocation rather than one per
    // search step.
    let mut low = 0_usize;
    let mut high = initial_limit;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if elided_history_turn_wire_bytes(role, text, middle) <= MAX_REQUEST_HISTORY_BYTES {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let bounded = elide_middle(text, low);
    let bytes = history_turn_wire_bytes(role, &bounded);
    (bounded, low < text.len(), bytes)
}

fn elided_history_turn_wire_bytes(role: Role, text: &str, max_bytes: usize) -> usize {
    if text.len() <= max_bytes {
        return history_turn_wire_bytes(role, text);
    }
    const MARKER: &str = "\n\n… [bytes elided] …\n\n";
    let retained_budget = max_bytes.saturating_sub(MARKER.len());
    if retained_budget == 0 {
        return history_turn_wire_bytes(role, &text[..floor_char_boundary(text, max_bytes)]);
    }
    let head_budget = retained_budget / 2;
    let tail_budget = retained_budget.saturating_sub(head_budget);
    let head = &text[..floor_char_boundary(text, head_budget)];
    let tail = &text[ceil_char_boundary(text, text.len().saturating_sub(tail_budget))..];
    history_turn_wire_overhead(role)
        .saturating_add(json_string_contents_len(head))
        .saturating_add(json_string_contents_len(MARKER))
        .saturating_add(json_string_contents_len(tail))
        .saturating_add(2)
}

fn history_turn_wire_bytes(role: Role, text: &str) -> usize {
    history_turn_wire_overhead(role)
        .saturating_add(json_string_contents_len(text))
        .saturating_add(2)
}

fn history_turn_wire_overhead(role: Role) -> usize {
    // Serialized message object excluding the content string itself and its
    // two quote bytes. Field order does not affect the total.
    b"{\"role\":\"\",\"content\":}".len() + role.as_str().len()
}

fn json_string_contents_len(text: &str) -> usize {
    text.chars().fold(0_usize, |bytes, character| {
        let encoded = match character {
            '"' | '\\' | '\u{0008}' | '\t' | '\n' | '\u{000c}' | '\r' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            _ => character.len_utf8(),
        };
        bytes.saturating_add(encoded)
    })
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
/// Ollama receives the same OpenAI-shaped function definitions but no
/// `tool_choice` field because `/api/chat` does not expose one. Response
/// ingestion still enforces jagent's exact-one-call rule.
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
    let url = config.endpoint()?;
    let _extension_json_bytes = validate_extra_body_fields(extra_body_fields)?;
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
    let api_key = config.api_key.as_deref();
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
    if !body.is_object() {
        return Err(ProviderError::InvalidConfiguration(
            "provider request body is not a JSON object".into(),
        ));
    }
    for (field, value) in extra_body_fields {
        body[*field] = value.clone();
    }
    let body = serde_json::to_string(&body).map_err(|_| {
        ProviderError::InvalidConfiguration("provider request JSON could not be encoded".into())
    })?;
    if body.len() > MAX_REQUEST_JSON_BYTES {
        return Err(ProviderError::InvalidConfiguration(format!(
            "encoded request body exceeds the {MAX_REQUEST_JSON_BYTES}-byte limit"
        )));
    }
    let request = HttpRequest { url, headers, body };
    request.validate_transport()?;
    Ok(BuiltRequest {
        request,
        omitted_history_turns,
    })
}

fn validate_extra_body_fields(
    extra_body_fields: &[(&'static str, Value)],
) -> Result<usize, ProviderError> {
    const MAX_EXTRA_BODY_FIELDS: usize = 16;
    const MAX_EXTRA_BODY_FIELD_BYTES: usize = 64;
    const RESERVED: [&str; 8] = [
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "stream",
        "stream_options",
        "options",
    ];
    if extra_body_fields.len() > MAX_EXTRA_BODY_FIELDS {
        return Err(ProviderError::InvalidConfiguration(
            "too many provider extension fields".into(),
        ));
    }
    let mut encoded_bytes = 2_usize; // enclosing object braces
    for (index, (field, value)) in extra_body_fields.iter().enumerate() {
        if field.is_empty()
            || field.len() > MAX_EXTRA_BODY_FIELD_BYTES
            || !field.bytes().all(|byte| matches!(byte, 0x21..=0x7e))
        {
            return Err(ProviderError::InvalidConfiguration(
                "provider extension field name is invalid".into(),
            ));
        }
        if RESERVED.contains(field) {
            return Err(ProviderError::InvalidConfiguration(
                "provider extension may not replace a reserved field".into(),
            ));
        }
        if extra_body_fields[..index]
            .iter()
            .any(|(previous, _)| previous == field)
        {
            return Err(ProviderError::InvalidConfiguration(
                "duplicate provider extension field".into(),
            ));
        }
        let value_bytes = encoded_json_len(value, MAX_REQUEST_EXTENSION_JSON_BYTES)?;
        let field_bytes = json_string_contents_len(field)
            .checked_add(2) // field-name quotes
            .and_then(|bytes| bytes.checked_add(1)) // colon
            .and_then(|bytes| bytes.checked_add(value_bytes))
            .and_then(|bytes| bytes.checked_add(usize::from(index > 0)))
            .ok_or_else(|| {
                ProviderError::InvalidConfiguration(
                    "provider extension byte accounting overflowed".into(),
                )
            })?;
        encoded_bytes = encoded_bytes.checked_add(field_bytes).ok_or_else(|| {
            ProviderError::InvalidConfiguration(
                "provider extension byte accounting overflowed".into(),
            )
        })?;
        if encoded_bytes > MAX_REQUEST_EXTENSIONS_JSON_BYTES {
            return Err(ProviderError::InvalidConfiguration(format!(
                "provider extensions exceed the {MAX_REQUEST_EXTENSIONS_JSON_BYTES}-byte encoded limit"
            )));
        }
    }
    Ok(encoded_bytes)
}

fn encoded_json_len(value: &Value, limit: usize) -> Result<usize, ProviderError> {
    struct CountingWriter {
        bytes: usize,
        limit: usize,
    }

    impl Write for CountingWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            if buffer.len() > self.limit.saturating_sub(self.bytes) {
                return Err(io::Error::other("encoded JSON byte limit exceeded"));
            }
            self.bytes += buffer.len();
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut writer = CountingWriter { bytes: 0, limit };
    serde_json::to_writer(&mut writer, value).map_err(|_| {
        ProviderError::InvalidConfiguration(format!(
            "provider extension would make the encoded request body exceed its {limit}-byte extension limit"
        ))
    })?;
    Ok(writer.bytes)
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
    /// The provider stopped at a generation bound (its output-token limit or,
    /// where explicitly reported, its model context-window limit); the
    /// integration should present the text as partial rather than complete.
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
/// token-limit and usage metadata. The size and duplicate-object-member checks
/// happen before `serde_json` constructs the retained [`Value`].
pub fn parse_chat_response_full_bytes(
    provider: Provider,
    body: &[u8],
) -> Result<ChatResponse, ProviderError> {
    let response = decode_response_value(body)?;
    parse_chat_response_full(provider, &response)
}

/// Extract the assistant text from one non-streaming chat response.
/// A provider-reported generation bound is surfaced as a visible trailing
/// note, never as an error, so partial answers stay reviewable.
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
            "\n\n[Response reached a provider generation limit and may be incomplete. \
             Ask to continue or adjust the model/token limits.]",
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
    crate::json::from_slice(body)
        .map_err(|error| ProviderError::MalformedResponse(error.to_string()))
}

/// Validate the completion envelope used by the high-level Agent path.
///
/// The lower-level chat extractors intentionally retain their historical
/// tolerance for sparse OpenAI-compatible and older Ollama fixtures. Turning
/// model output into an Agent action needs a stronger contract: exactly one
/// completed choice/message, an unambiguous completion reason, and no
/// provider-declared filtering or pause. Otherwise partial or mismatched text
/// could happen to look like a complete `run` action.
pub(crate) fn validate_agent_response_envelope(
    provider: Provider,
    response: &Value,
) -> Result<(), ProviderError> {
    let object = response
        .as_object()
        .ok_or_else(|| malformed_response("top-level response is not a JSON object"))?;
    reject_provider_error_field(object)?;

    match provider {
        Provider::Anthropic => {
            let content = object
                .get("content")
                .and_then(Value::as_array)
                .ok_or_else(|| malformed_response("Anthropic response has no content array"))?;
            let mut has_tool_use = false;
            for block in content {
                let block = block.as_object().ok_or_else(|| {
                    malformed_response("Anthropic content block is not an object")
                })?;
                let kind = block.get("type").and_then(Value::as_str).ok_or_else(|| {
                    malformed_response("Anthropic content block has no string type")
                })?;
                has_tool_use |= kind == "tool_use";
                if kind == "text" && !block.get("text").is_some_and(Value::is_string) {
                    return Err(malformed_response(
                        "Anthropic text block has no string text",
                    ));
                }
            }

            match object.get("stop_reason").and_then(Value::as_str) {
                Some("end_turn" | "stop_sequence") if !has_tool_use => Ok(()),
                Some("end_turn" | "stop_sequence") => Err(malformed_response(
                    "Anthropic response contains tool_use with a non-tool stop_reason",
                )),
                Some("max_tokens" | "model_context_window_exceeded") => Ok(()),
                Some("tool_use") if has_tool_use => Ok(()),
                Some("tool_use") => Err(malformed_response(
                    "Anthropic tool_use stop has no tool_use content block",
                )),
                Some("pause_turn") => Err(malformed_response(
                    "Anthropic response paused before the turn completed",
                )),
                Some("refusal") => Err(malformed_response(
                    "Anthropic response ended with a model refusal",
                )),
                Some(_) => Err(malformed_response(
                    "Anthropic response has an unknown stop_reason",
                )),
                None => Err(malformed_response(
                    "Anthropic response has no string stop_reason",
                )),
            }
        }
        Provider::OpenAiCompatible => {
            let choices = object
                .get("choices")
                .and_then(Value::as_array)
                .ok_or_else(|| malformed_response("OpenAI response has no choices array"))?;
            if choices.len() != 1 {
                return Err(malformed_response(
                    "OpenAI response must contain exactly one choice",
                ));
            }
            let choice = choices[0]
                .as_object()
                .ok_or_else(|| malformed_response("OpenAI choice is not an object"))?;
            if let Some(index) = choice.get("index") {
                if index.as_u64() != Some(0) {
                    return Err(malformed_response(
                        "OpenAI response choice index is not zero",
                    ));
                }
            }
            let message = choice
                .get("message")
                .and_then(Value::as_object)
                .ok_or_else(|| malformed_response("OpenAI choice has no message object"))?;
            let has_tool_calls = match message.get("tool_calls") {
                None | Some(Value::Null) => false,
                Some(Value::Array(calls)) => !calls.is_empty(),
                Some(_) => {
                    return Err(malformed_response(
                        "OpenAI message tool_calls is not an array",
                    ))
                }
            };
            if message.get("refusal").is_some_and(|refusal| match refusal {
                Value::String(text) => !text.is_empty(),
                Value::Null => false,
                _ => true,
            }) {
                return Err(malformed_response(
                    "OpenAI response contains a model refusal",
                ));
            }
            match choice.get("finish_reason").and_then(Value::as_str) {
                Some("stop") if !has_tool_calls => Ok(()),
                Some("stop") => Err(malformed_response(
                    "OpenAI response contains tool calls with stop finish_reason",
                )),
                Some("length") => Ok(()),
                Some("tool_calls") if has_tool_calls => Ok(()),
                Some("tool_calls") => Err(malformed_response(
                    "OpenAI tool_calls finish has no non-empty tool_calls array",
                )),
                Some("function_call") => Err(malformed_response(
                    "OpenAI legacy function_call responses are unsupported by the Agent protocol",
                )),
                Some("content_filter") => Err(malformed_response(
                    "OpenAI response was stopped by content filtering",
                )),
                Some(_) => Err(malformed_response(
                    "OpenAI response has an unknown finish_reason",
                )),
                None => Err(malformed_response(
                    "OpenAI response has no string finish_reason",
                )),
            }
        }
        Provider::Ollama => {
            object
                .get("message")
                .and_then(Value::as_object)
                .ok_or_else(|| malformed_response("Ollama response has no message object"))?;
            match object.get("done") {
                Some(Value::Bool(true)) => {}
                Some(Value::Bool(false)) => {
                    return Err(malformed_response("Ollama response is not marked complete"))
                }
                _ => {
                    return Err(malformed_response(
                        "Ollama response has no boolean done field",
                    ))
                }
            }
            match object.get("done_reason") {
                None => Ok(()),
                Some(Value::String(reason)) if matches!(reason.as_str(), "stop" | "length") => {
                    Ok(())
                }
                Some(Value::String(_)) => Err(malformed_response(
                    "Ollama response has an unknown done_reason",
                )),
                Some(_) => Err(malformed_response(
                    "Ollama response has a non-string done_reason",
                )),
            }
        }
    }
}

/// Reject an explicitly incomplete, filtered, refused, or unknown completion
/// status without requiring the complete high-level Agent envelope.
///
/// Native-tool parsing predates [`crate::response::AgentResponse`] and its
/// low-level `Value` API intentionally accepts sparse compatibility fixtures.
/// Missing completion metadata therefore stays compatible here, but metadata
/// that is present must never authorize a potentially partial tool action.
pub(crate) fn validate_tool_response_completion(
    provider: Provider,
    response: &Value,
) -> Result<(), ProviderError> {
    if let Some(object) = response.as_object() {
        reject_provider_error_field(object)?;
    }
    match provider {
        Provider::Anthropic => {
            let has_tool_use = response
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|blocks| {
                    blocks
                        .iter()
                        .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
                });
            match response.get("stop_reason") {
                None | Some(Value::Null) => Ok(()),
                Some(Value::String(reason))
                    if matches!(reason.as_str(), "end_turn" | "stop_sequence") && !has_tool_use =>
                {
                    Ok(())
                }
                Some(Value::String(reason))
                    if matches!(reason.as_str(), "end_turn" | "stop_sequence") =>
                {
                    Err(malformed_response(
                        "Anthropic response contains tool_use with a non-tool stop_reason",
                    ))
                }
                Some(Value::String(reason))
                    if matches!(
                        reason.as_str(),
                        "max_tokens" | "model_context_window_exceeded"
                    ) =>
                {
                    Ok(())
                }
                // A sparse low-level fixture may carry `tool_use` without a
                // call. That still resolves to `NoToolCall`, never an action;
                // retain compatibility while rejecting the dangerous inverse
                // (a real call paired with a non-tool completion reason).
                Some(Value::String(reason)) if reason == "tool_use" => Ok(()),
                Some(Value::String(reason))
                    if matches!(reason.as_str(), "pause_turn" | "refusal") =>
                {
                    Err(malformed_response(
                        "Anthropic response did not complete normally",
                    ))
                }
                Some(Value::String(_)) => Err(malformed_response(
                    "Anthropic response has an unknown stop_reason",
                )),
                Some(_) => Err(malformed_response(
                    "Anthropic response has a non-string stop_reason",
                )),
            }
        }
        Provider::OpenAiCompatible => {
            if let Some(Value::Array(choices)) = response.get("choices") {
                if choices.len() > 1 {
                    return Err(malformed_response(
                        "OpenAI tool response contains more than one choice",
                    ));
                }
                if let Some(choice) = choices.first().and_then(Value::as_object) {
                    if choice
                        .get("index")
                        .is_some_and(|index| !index.is_null() && index.as_u64() != Some(0))
                    {
                        return Err(malformed_response(
                            "OpenAI tool response choice index is not zero",
                        ));
                    }
                }
            }
            if response
                .pointer("/choices/0/message/refusal")
                .is_some_and(|refusal| match refusal {
                    Value::String(text) => !text.is_empty(),
                    Value::Null => false,
                    _ => true,
                })
            {
                return Err(malformed_response(
                    "OpenAI response contains a model refusal",
                ));
            }
            let has_tool_calls = response
                .pointer("/choices/0/message/tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty());
            match response.pointer("/choices/0/finish_reason") {
                None | Some(Value::Null) => Ok(()),
                Some(Value::String(reason)) if reason == "stop" && !has_tool_calls => Ok(()),
                Some(Value::String(reason)) if reason == "stop" => Err(malformed_response(
                    "OpenAI response contains tool calls with stop finish_reason",
                )),
                Some(Value::String(reason)) if reason == "length" => Ok(()),
                // Missing calls remain a harmless `NoToolCall` on this sparse
                // compatibility API; the strict Agent envelope rejects them.
                Some(Value::String(reason)) if reason == "tool_calls" => Ok(()),
                Some(Value::String(reason)) if reason == "function_call" && has_tool_calls => {
                    Err(malformed_response(
                        "OpenAI tool calls conflict with legacy function_call completion",
                    ))
                }
                Some(Value::String(reason)) if reason == "function_call" => Ok(()),
                Some(Value::String(reason)) if reason == "content_filter" => Err(
                    malformed_response("OpenAI response was stopped by content filtering"),
                ),
                Some(Value::String(_)) => Err(malformed_response(
                    "OpenAI response has an unknown finish_reason",
                )),
                Some(_) => Err(malformed_response(
                    "OpenAI response has a non-string finish_reason",
                )),
            }
        }
        Provider::Ollama => {
            match response.get("done") {
                None | Some(Value::Null) | Some(Value::Bool(true)) => {}
                Some(Value::Bool(false)) => {
                    return Err(malformed_response("Ollama response is not marked complete"))
                }
                Some(_) => {
                    return Err(malformed_response(
                        "Ollama response has a non-boolean done field",
                    ))
                }
            }
            match response.get("done_reason") {
                None | Some(Value::Null) => Ok(()),
                Some(Value::String(reason)) if matches!(reason.as_str(), "stop" | "length") => {
                    Ok(())
                }
                Some(Value::String(_)) => Err(malformed_response(
                    "Ollama response has an unknown done_reason",
                )),
                Some(_) => Err(malformed_response(
                    "Ollama response has a non-string done_reason",
                )),
            }
        }
    }
}

fn reject_provider_error_field(
    object: &serde_json::Map<String, Value>,
) -> Result<(), ProviderError> {
    if object.get("error").is_some_and(|error| !error.is_null()) {
        return Err(malformed_response(
            "provider returned an error instead of a completion",
        ));
    }
    Ok(())
}

fn malformed_response(detail: &'static str) -> ProviderError {
    ProviderError::MalformedResponse(detail.to_string())
}

/// Did the provider stop at a generation bound that can truncate the reply?
/// Shared by the text and native-tool ingestion paths so both report the same
/// condition.
pub(crate) fn reached_token_limit(provider: Provider, response: &Value) -> bool {
    match provider {
        Provider::Anthropic => {
            matches!(
                response.get("stop_reason").and_then(Value::as_str),
                Some("max_tokens" | "model_context_window_exceeded")
            )
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
        chat.model = format!("model-{secret}");
        chat.base_url = format!("https://user:{secret}@example.test");
        let config_debug = format!("{chat:?}");
        assert!(config_debug.contains("[REDACTED]"));
        assert!(!config_debug.contains(secret));

        // The same value in the body must stay out of Debug too. Request
        // context is sensitive even when it does not match a known token
        // shape, so Debug reports only the encoded length.
        let mut valid = config(Provider::Anthropic);
        valid.api_key = Some(secret.into());
        let request = build_chat_request(&valid, None, &[user(secret)]).unwrap();
        let request_debug = format!("{request:?}");
        assert!(request_debug.contains("header_count"));
        assert!(request_debug.contains("body_bytes"));
        assert!(!request_debug.contains(secret));

        let direct = HttpRequest {
            url: format!("https://user:{secret}@example.test/?key={secret}"),
            headers: vec![
                ("cookie".into(), format!("session={secret}")),
                ("proxy-authorization".into(), secret.into()),
                (format!("x-private-{secret}"), secret.into()),
            ],
            body: String::new(),
        };
        let direct_debug = format!("{direct:?}");
        assert!(direct_debug.contains("header_count: 3"));
        assert!(!direct_debug.contains(secret));
        assert!(!direct_debug.contains("cookie"));
    }

    #[test]
    fn request_metrics_and_transport_validation_are_content_free() {
        let secret = "private-header-and-body-value";
        let request = HttpRequest {
            url: "https://example.test/v1/chat/completions".into(),
            headers: vec![
                ("content-type".into(), "application/json".into()),
                ("x-private".into(), secret.into()),
            ],
            body: format!(r#"{{"prompt":"{secret}"}}"#),
        };
        let metrics = request.validate_transport().unwrap();
        assert_eq!(metrics.url_bytes, request.url.len());
        assert_eq!(metrics.header_count, 2);
        assert_eq!(
            metrics.header_bytes,
            request
                .headers
                .iter()
                .map(|(name, value)| name.len() + value.len())
                .sum::<usize>()
        );
        assert_eq!(metrics.body_bytes, request.body.len());
        assert!(!format!("{metrics:?}").contains(secret));
        assert!(!format!("{request:?}").contains(secret));
    }

    #[test]
    fn transport_validation_rejects_noncanonical_headers_and_bodies() {
        let valid = || HttpRequest {
            url: "https://example.test/v1".into(),
            headers: vec![("content-type".into(), "application/json".into())],
            body: "{}".into(),
        };

        let mut cases = Vec::new();
        let mut request = valid();
        request.headers[0].0 = "Content-Type".into();
        cases.push(request);
        let mut request = valid();
        request.headers.push(("bad:name".into(), "value".into()));
        cases.push(request);
        let mut request = valid();
        request
            .headers
            .push(("x-value".into(), "line\nbreak".into()));
        cases.push(request);
        let mut request = valid();
        request
            .headers
            .push(("content-type".into(), "application/json".into()));
        cases.push(request);
        let mut request = valid();
        request.headers[0].1 = "text/plain".into();
        cases.push(request);
        let mut request = valid();
        request.headers.clear();
        cases.push(request);
        let mut request = valid();
        request.body = "[]".into();
        cases.push(request);
        let mut request = valid();
        request.body = "{not-json}".into();
        cases.push(request);
        let mut request = valid();
        request.body = r#"{"model":"first","model":"second"}"#.into();
        cases.push(request);
        let mut request = valid();
        request.url = "x".repeat(MAX_REQUEST_URL_BYTES + 1);
        cases.push(request);
        for invalid_url in [
            "ftp://example.test/v1",
            "http://example.test/v1",
            "https://example.test/v1?secret=value",
            "https://example.test/v1#fragment",
            "https://example.test\\other",
            "https://example.test/line\nbreak",
            "https://user:secret@example.test/v1",
            "https://example.test:0/v1",
            "https://2130706433/v1",
        ] {
            let mut request = valid();
            request.url = invalid_url.into();
            cases.push(request);
        }
        let mut request = valid();
        request
            .headers
            .extend((0..MAX_REQUEST_HEADERS).map(|index| (format!("x-{index}"), String::new())));
        cases.push(request);
        let mut request = valid();
        request
            .headers
            .push(("x-large".into(), "x".repeat(MAX_REQUEST_HEADER_BYTES)));
        cases.push(request);

        for request in cases {
            assert!(request.validate_transport().is_err());
        }
    }

    #[test]
    fn config_caps_tokens_and_rejects_port_zero() {
        let mut chat = config(Provider::OpenAiCompatible);
        chat.max_tokens = MAX_REQUEST_MAX_TOKENS;
        assert!(chat.validate().is_ok());
        chat.max_tokens = MAX_REQUEST_MAX_TOKENS + 1;
        assert!(chat.validate().is_err());

        chat.max_tokens = 512;
        chat.base_url = "https://example.test:0/v1".into();
        assert!(chat.validate().is_err());
        chat.base_url = "http://127.0.0.2:0/v1".into();
        assert!(chat.validate().is_err());
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

        let config = config(Provider::OpenAiCompatible);
        assert_eq!(
            config.endpoint().unwrap(),
            "https://api.openai.com/v1/chat/completions"
        );
        let mut invalid = config;
        invalid.api_key = Some(" unsafe ".into());
        assert!(invalid.endpoint().is_err());
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        assert_eq!("CLAUDE".parse::<Provider>().unwrap(), Provider::Anthropic);
        assert!("x"
            .repeat(MAX_PROVIDER_NAME_BYTES + 1)
            .parse::<Provider>()
            .is_err());
        assert!("open\nai".parse::<Provider>().is_err());
        assert!("\u{202e}openai".parse::<Provider>().is_err());
        assert!(format!("{}openai", " ".repeat(MAX_PROVIDER_NAME_BYTES))
            .parse::<Provider>()
            .is_err());
        let unknown = "secret-provider-marker";
        let error = unknown.parse::<Provider>().unwrap_err().to_string();
        assert!(!error.contains(unknown));

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
    fn api_keys_are_exact_visible_ascii_header_values() {
        for hostile in [
            "",
            " secret-leading-123",
            "secret-trailing-123 ",
            "secret\tpart-123",
            "secret\npart-123",
            "secret part-123",
            "clé-api-123",
            "secret\u{00a0}part-123",
            "secret\u{200b}part-123",
        ] {
            let mut chat = config(Provider::OpenAiCompatible);
            chat.api_key = Some(hostile.into());
            let error = chat.validate().expect_err("unsafe header value accepted");
            let rendered = error.to_string();
            if !hostile.is_empty() {
                assert!(
                    !rendered.contains(hostile),
                    "credential leaked through error"
                );
            }
            assert!(
                build_chat_request(&chat, None, &[user("hi")]).is_err(),
                "request builder accepted {hostile:?}"
            );
        }

        let key = "sk-safe_ABC123.+/=";
        let mut chat = config(Provider::OpenAiCompatible);
        chat.api_key = Some(key.into());
        let request = build_chat_request(&chat, None, &[user("hi")]).unwrap();
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "authorization" && value == &format!("Bearer {key}")));
    }

    #[test]
    fn base_urls_must_be_https_unless_the_endpoint_is_loopback() {
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

        // Local OpenAI-compatible servers and provider proxies commonly use
        // clear-text loopback HTTP. The exception is based on the host, not
        // the provider label.
        for loopback in [
            "http://localhost:11434",
            "http://LOCALHOST",
            "http://127.0.0.1:11434",
            "http://127.6.6.6",
            "http://[::1]:11434",
        ] {
            for provider in [
                Provider::Anthropic,
                Provider::OpenAiCompatible,
                Provider::Ollama,
            ] {
                let mut local = config(provider);
                local.base_url = loopback.into();
                assert!(
                    local.validate().is_ok(),
                    "{loopback} is loopback for {provider:?}"
                );
            }
        }
        for remote in [
            "http://example.com",
            "http://localhost.example.com",
            "http://127.0.0.1.example.com",
            "http://[::2]:11434",
            "http://127.0.0.1:not-a-port",
            "http://127.0.0.1:65536",
        ] {
            for provider in [
                Provider::Anthropic,
                Provider::OpenAiCompatible,
                Provider::Ollama,
            ] {
                let mut endpoint = config(provider);
                endpoint.base_url = remote.into();
                assert!(
                    endpoint.validate().is_err(),
                    "{remote} must not pass as loopback for {provider:?}"
                );
            }
        }
    }

    #[test]
    fn base_urls_and_models_reject_smuggled_components() {
        for hostile in [
            "https:///missing-authority",
            "https://:443",
            "https://[::1",
            "https://example.com:99999",
            "https://user:secret@example.com/v1",
            "https://example.com/v1?api-key=secret",
            "https://example.com/v1#fragment",
            "https://example.com\\v1",
            "https://%65xample.com",
            "https://example..com",
            "https://-example.com",
            "https://example-.com",
            "https://exa_mple.com",
            "https://.",
            "https://2130706433",
            "https://0x7f000001",
            "https://127.1",
            "https://999.1.1.1",
            "https://例子.example",
            "https://exam\u{200b}ple.com",
            "https://example.com/\u{202e}v1",
            "https://example.com/v 1",
            " https://example.com/v1",
            "https://example.com/v1 ",
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

        for valid in [
            "https://api-1.example/v1",
            "https://xn--bcher-kva.example/v1",
            "https://127.0.0.1/v1",
            "https://[2001:db8::1]/v1",
        ] {
            let mut endpoint = config(Provider::OpenAiCompatible);
            endpoint.base_url = valid.into();
            assert!(endpoint.validate().is_ok(), "{valid:?} must be accepted");
        }

        for hostile in [
            "gpt\u{202e}4o",
            "gpt\u{200b}4o",
            "gpt\t4o",
            "gpt\n4o",
            " gpt-4o",
            "gpt-4o ",
        ] {
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
    fn detailed_history_report_surfaces_elision_preparation_and_omission() {
        let oversized = "界".repeat(MAX_REQUEST_TURN_BYTES / "界".len() + 1);
        let history = vec![
            assistant("orphaned assistant"),
            user("secret"),
            user(&oversized),
        ];
        let prepared = bound_history_prepared_with_report(&history, |text| {
            if text == "secret" {
                "[REDACTED]".to_string()
            } else {
                text.to_string()
            }
        });

        // Leading assistant context is omitted, while the two user turns are
        // retained. The report distinguishes preparation, middle-elision,
        // and whole-turn loss.
        assert_eq!(prepared.report.input_history_turns, 3);
        assert_eq!(prepared.report.sent_history_turns, 2);
        assert_eq!(prepared.report.omitted_history_turns, 1);
        assert!(prepared.report.sent_history_json_bytes <= MAX_REQUEST_HISTORY_BYTES);
        assert_eq!(prepared.report.changed_history_turns, 1);
        assert_eq!(prepared.report.elided_history_turns, 1);
        assert_eq!(
            prepared.report.sent_history_text_bytes,
            prepared
                .messages
                .iter()
                .map(|message| message.text.len())
                .sum::<usize>()
        );
        assert_eq!(prepared.messages[0].text, "[REDACTED]");
        assert!(prepared.messages[1].text.contains("bytes elided"));

        let prepared =
            bound_history_prepared_with_report(&history[1..2], |_| "[REDACTED]".to_string());
        assert_eq!(prepared.report.changed_history_turns, 1);
        assert_eq!(prepared.report.elided_history_turns, 0);
        assert_eq!(prepared.report.omitted_history_turns, 0);
        assert_eq!(prepared.messages[0].text, "[REDACTED]");
    }

    #[test]
    fn history_json_budget_uses_an_exact_serde_compatible_oracle() {
        for role in [Role::User, Role::Assistant] {
            for text in [
                "plain",
                "quote \" and slash \\",
                "line\nnull\0tab\t",
                "编译🙂",
            ] {
                let expected = serde_json::to_string(&json!({
                    "role": role.as_str(),
                    "content": text,
                }))
                .unwrap()
                .len();
                assert_eq!(history_turn_wire_bytes(role, text), expected, "{text:?}");
            }
        }

        let hostile = "\0\"\\\n".repeat(MAX_REQUEST_TURN_BYTES / 4);
        let prepared = bound_history_with_report(&[user(&hostile)]);
        assert_eq!(prepared.messages.len(), 1);
        assert_eq!(prepared.report.elided_history_turns, 1);
        assert!(prepared.report.sent_history_json_bytes <= MAX_REQUEST_HISTORY_BYTES);
        let actual = prepared
            .messages
            .iter()
            .map(|message| {
                serde_json::to_string(&json!({
                    "role": message.role.as_str(),
                    "content": message.text,
                }))
                .unwrap()
            })
            .collect::<Vec<_>>()
            .join(",")
            .len();
        assert_eq!(prepared.report.sent_history_json_bytes, actual);
    }

    #[test]
    fn provider_extensions_cannot_replace_core_or_stream_fields() {
        let config = config(Provider::OpenAiCompatible);
        let allowed = build_request_with(
            &config,
            None,
            &[user("hello")],
            false,
            &[("response_format", json!({"type": "json_object"}))],
        )
        .unwrap();
        let body: Value = serde_json::from_str(&allowed.request.body).unwrap();
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["model"], config.model);

        for reserved in [
            "model",
            "messages",
            "system",
            "max_tokens",
            "temperature",
            "stream",
            "stream_options",
            "options",
        ] {
            let secret = "secret-extension-value";
            let error = build_request_with(
                &config,
                None,
                &[user("hello")],
                false,
                &[(reserved, json!(secret))],
            )
            .unwrap_err()
            .to_string();
            assert!(!error.contains(secret));
        }

        assert!(build_request_with(
            &config,
            None,
            &[],
            false,
            &[("extension", json!(1)), ("extension", json!(2))],
        )
        .is_err());
        assert!(build_request_with(&config, None, &[], false, &[("", json!(1))]).is_err());
        let too_many = vec![("extension", json!(null)); 17];
        assert!(build_request_with(&config, None, &[], false, &too_many).is_err());
    }

    #[test]
    fn complete_encoded_request_body_has_a_final_ceiling() {
        let config = config(Provider::OpenAiCompatible);
        let oversized = "x".repeat(MAX_REQUEST_JSON_BYTES);
        let error = build_request_with(
            &config,
            None,
            &[],
            false,
            &[("future_extension", json!(oversized))],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("encoded request body"));

        let ordinary = build_request_with(
            &config,
            Some("system"),
            &[user("hello")],
            false,
            &[("future_extension", json!({"enabled": true}))],
        )
        .unwrap();
        assert!(ordinary.request.body.len() < MAX_REQUEST_JSON_BYTES);
    }

    #[test]
    fn extension_budgets_measure_encoded_values_before_request_assembly() {
        let escaped = json!("\0\"\\\n".repeat(1024));
        assert_eq!(
            encoded_json_len(&escaped, MAX_REQUEST_EXTENSION_JSON_BYTES).unwrap(),
            serde_json::to_string(&escaped).unwrap().len()
        );

        // Raw bytes remain below the limit while JSON escaping pushes the
        // encoded value over it.
        let escape_heavy = json!("\n".repeat(MAX_REQUEST_EXTENSION_JSON_BYTES / 2 + 1));
        let error = build_request_with(
            &config(Provider::OpenAiCompatible),
            None,
            &[],
            false,
            &[("future_extension", escape_heavy)],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("encoded request body"));

        let large = "x".repeat(700 * 1024);
        let extensions = [
            ("future_a", json!(large)),
            ("future_b", json!("x".repeat(700 * 1024))),
            ("future_c", json!("x".repeat(700 * 1024))),
        ];
        let error = build_request_with(
            &config(Provider::OpenAiCompatible),
            None,
            &[],
            false,
            &extensions,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("provider extensions exceed"));
    }

    #[test]
    fn extension_failures_never_echo_names_or_values() {
        let secret = "extension-secret-marker";
        for fields in [
            vec![("model", json!(secret))],
            vec![(secret, json!(1)), (secret, json!(2))],
            vec![(
                "future_extension",
                json!(format!(
                    "{secret}{}",
                    "x".repeat(MAX_REQUEST_EXTENSION_JSON_BYTES)
                )),
            )],
        ] {
            let error = build_request_with(
                &config(Provider::OpenAiCompatible),
                None,
                &[],
                false,
                &fields,
            )
            .unwrap_err()
            .to_string();
            assert!(!error.contains(secret));
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
        assert!(text.contains("generation limit"));

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
