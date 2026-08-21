//! Versioned Agent protocol and delivery capability discovery.
//!
//! Integrations often have two independent participants: a terminal owns the
//! UI and transport while a shell owns execution. Guessing whether the peer
//! understands provider-native tools or streaming can silently pair a request
//! with the wrong decoder. This module provides a small sans-IO contract that
//! can travel through an environment variable, IPC message, or diagnostic
//! command without transporting credentials or user context.

use crate::{AgentProtocol, Provider};
use std::fmt;

/// Current capability-token schema version.
pub const AGENT_CAPABILITIES_VERSION: u16 = 1;
/// Canonical version-1 capability token emitted by all built-in providers.
pub const AGENT_CAPABILITIES_V1_WIRE: &str =
    "jagent-agent/1;protocols=text,native-tools;delivery=complete,streaming";
/// Capability tokens are deliberately tiny; reject an effectively unbounded
/// environment or IPC value before splitting it.
pub const MAX_AGENT_CAPABILITIES_WIRE_BYTES: usize = 256;

const PROTOCOL_TEXT: u8 = 1 << 0;
const PROTOCOL_NATIVE_TOOLS: u8 = 1 << 1;
const DELIVERY_COMPLETE: u8 = 1 << 0;
const DELIVERY_STREAMING: u8 = 1 << 1;

/// How one provider response body is delivered to the integration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AgentDelivery {
    /// One complete, bounded response envelope.
    #[default]
    Complete,
    /// Incremental SSE or NDJSON frames ending in an explicit completion.
    Streaming,
}

impl AgentDelivery {
    /// Stable ASCII name used by the capability token.
    pub const fn as_wire_name(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::Streaming => "streaming",
        }
    }
}

/// Versioned set of protocols and delivery modes supported by one peer.
///
/// Fields stay private so later schema versions can grow without making
/// callers construct internally inconsistent bitsets. Use
/// [`agent_capabilities`] for a provider or [`Self::from_wire`] for a peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AgentCapabilities {
    version: u16,
    protocols: u8,
    delivery: u8,
}

impl AgentCapabilities {
    const V1_ALL: Self = Self {
        version: AGENT_CAPABILITIES_VERSION,
        protocols: PROTOCOL_TEXT | PROTOCOL_NATIVE_TOOLS,
        delivery: DELIVERY_COMPLETE | DELIVERY_STREAMING,
    };

    /// Capability schema version carried by this value.
    pub const fn version(self) -> u16 {
        self.version
    }

    /// Whether the peer supports this exact action protocol and delivery
    /// combination.
    pub const fn supports(self, protocol: AgentProtocol, delivery: AgentDelivery) -> bool {
        let protocol_bit = match protocol {
            AgentProtocol::Text => PROTOCOL_TEXT,
            AgentProtocol::NativeTools => PROTOCOL_NATIVE_TOOLS,
        };
        let delivery_bit = match delivery {
            AgentDelivery::Complete => DELIVERY_COMPLETE,
            AgentDelivery::Streaming => DELIVERY_STREAMING,
        };
        self.protocols & protocol_bit != 0 && self.delivery & delivery_bit != 0
    }

    /// Select the first supported protocol from the caller's preference
    /// order. An empty list or a peer with no matching protocol returns
    /// `None`; negotiation never silently invents a fallback.
    pub fn negotiate(
        self,
        preferred: &[AgentProtocol],
        delivery: AgentDelivery,
    ) -> Option<AgentProtocol> {
        preferred
            .iter()
            .copied()
            .find(|protocol| self.supports(*protocol, delivery))
    }

    /// Select the first protocol supported by both this capability set and a
    /// peer's. This is the normal split-process negotiation entry point;
    /// [`Self::negotiate`] remains useful after an integration has already
    /// computed or received one effective set.
    pub fn negotiate_with(
        self,
        peer: Self,
        preferred: &[AgentProtocol],
        delivery: AgentDelivery,
    ) -> Option<AgentProtocol> {
        preferred.iter().copied().find(|protocol| {
            self.supports(*protocol, delivery) && peer.supports(*protocol, delivery)
        })
    }

    /// Canonical bounded ASCII token suitable for an environment variable or
    /// IPC field. It contains capabilities only—never endpoint, credential,
    /// model, history, or terminal context.
    pub fn to_wire(self) -> String {
        let protocols = match self.protocols {
            PROTOCOL_TEXT => "text",
            PROTOCOL_NATIVE_TOOLS => "native-tools",
            _ => "text,native-tools",
        };
        let delivery = match self.delivery {
            DELIVERY_COMPLETE => "complete",
            DELIVERY_STREAMING => "streaming",
            _ => "complete,streaming",
        };
        format!(
            "jagent-agent/{};protocols={protocols};delivery={delivery}",
            self.version
        )
    }

    /// Parse a strict capability token.
    ///
    /// Unknown versions, fields, capability names, duplicates, empty sets,
    /// whitespace, and overlong input fail closed. Field order is fixed so
    /// each logical set has one canonical spelling.
    pub fn from_wire(value: &str) -> Result<Self, CapabilityError> {
        if value.len() > MAX_AGENT_CAPABILITIES_WIRE_BYTES {
            return Err(CapabilityError::TooLarge);
        }
        let mut fields = value.split(';');
        let version_text = fields
            .next()
            .and_then(|header| header.strip_prefix("jagent-agent/"))
            .ok_or(CapabilityError::Malformed)?;
        if version_text.is_empty()
            || !version_text.bytes().all(|byte| byte.is_ascii_digit())
            || (version_text.len() > 1 && version_text.starts_with('0'))
        {
            return Err(CapabilityError::Malformed);
        }
        let version = version_text
            .parse::<u16>()
            .map_err(|_| CapabilityError::Malformed)?;
        if version != AGENT_CAPABILITIES_VERSION {
            return Err(CapabilityError::UnsupportedVersion(version));
        }
        let protocol_field = fields
            .next()
            .and_then(|field| field.strip_prefix("protocols="))
            .ok_or(CapabilityError::Malformed)?;
        let delivery_field = fields
            .next()
            .and_then(|field| field.strip_prefix("delivery="))
            .ok_or(CapabilityError::Malformed)?;
        if fields.next().is_some() {
            return Err(CapabilityError::Malformed);
        }

        let protocols = parse_bits(
            protocol_field,
            &[
                ("text", PROTOCOL_TEXT),
                ("native-tools", PROTOCOL_NATIVE_TOOLS),
            ],
        )?;
        let delivery = parse_bits(
            delivery_field,
            &[
                ("complete", DELIVERY_COMPLETE),
                ("streaming", DELIVERY_STREAMING),
            ],
        )?;
        Ok(Self {
            version,
            protocols,
            delivery,
        })
    }
}

fn parse_bits(value: &str, known: &[(&str, u8)]) -> Result<u8, CapabilityError> {
    if value.is_empty() {
        return Err(CapabilityError::Malformed);
    }
    let mut bits = 0_u8;
    let mut last_index = None;
    for item in value.split(',') {
        let (index, bit) = known
            .iter()
            .enumerate()
            .find_map(|(index, (name, bit))| (*name == item).then_some((index, *bit)))
            .ok_or(CapabilityError::Malformed)?;
        if last_index.is_some_and(|last| index <= last) || bits & bit != 0 {
            return Err(CapabilityError::Malformed);
        }
        last_index = Some(index);
        bits |= bit;
    }
    Ok(bits)
}

/// Provider capabilities enforced by jagent's request and response codecs.
///
/// The explicit provider match is intentional. When a provider gains or loses
/// a protocol or delivery form, compiler exhaustiveness makes this table and
/// request preparation change together.
pub const fn agent_capabilities(provider: Provider) -> AgentCapabilities {
    match provider {
        Provider::Anthropic | Provider::OpenAiCompatible | Provider::Ollama => {
            AgentCapabilities::V1_ALL
        }
    }
}

/// Failure to decode a peer capability token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityError {
    TooLarge,
    Malformed,
    UnsupportedVersion(u16),
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => write!(
                formatter,
                "agent capability token exceeds the {MAX_AGENT_CAPABILITIES_WIRE_BYTES}-byte limit"
            ),
            Self::Malformed => write!(formatter, "malformed agent capability token"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported agent capability version {version}")
            }
        }
    }
}

impl std::error::Error for CapabilityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_provider_advertises_the_matrix_its_codecs_implement() {
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let capabilities = agent_capabilities(provider);
            assert_eq!(capabilities.version(), AGENT_CAPABILITIES_VERSION);
            for protocol in [AgentProtocol::Text, AgentProtocol::NativeTools] {
                for delivery in [AgentDelivery::Complete, AgentDelivery::Streaming] {
                    assert!(capabilities.supports(protocol, delivery));
                }
            }
        }
    }

    #[test]
    fn negotiation_respects_preference_and_never_guesses_an_empty_fallback() {
        let capabilities = agent_capabilities(Provider::Ollama);
        assert_eq!(
            capabilities.negotiate(
                &[AgentProtocol::NativeTools, AgentProtocol::Text],
                AgentDelivery::Streaming,
            ),
            Some(AgentProtocol::NativeTools)
        );
        assert_eq!(capabilities.negotiate(&[], AgentDelivery::Complete), None);

        let text_only =
            AgentCapabilities::from_wire("jagent-agent/1;protocols=text;delivery=complete")
                .unwrap();
        assert_eq!(
            capabilities.negotiate_with(
                text_only,
                &[AgentProtocol::NativeTools, AgentProtocol::Text],
                AgentDelivery::Complete,
            ),
            Some(AgentProtocol::Text)
        );
        assert_eq!(
            capabilities.negotiate_with(
                text_only,
                &[AgentProtocol::Text],
                AgentDelivery::Streaming,
            ),
            None
        );
    }

    #[test]
    fn canonical_wire_round_trips_and_malformed_or_future_tokens_fail_closed() {
        let capabilities = agent_capabilities(Provider::Anthropic);
        assert_eq!(
            AgentCapabilities::from_wire(&capabilities.to_wire()),
            Ok(capabilities)
        );
        assert_eq!(capabilities.to_wire(), AGENT_CAPABILITIES_V1_WIRE);
        let text_only =
            AgentCapabilities::from_wire("jagent-agent/1;protocols=text;delivery=complete")
                .unwrap();
        assert!(text_only.supports(AgentProtocol::Text, AgentDelivery::Complete));
        assert!(!text_only.supports(AgentProtocol::NativeTools, AgentDelivery::Complete));
        assert_eq!(
            text_only.negotiate(
                &[AgentProtocol::NativeTools, AgentProtocol::Text],
                AgentDelivery::Complete,
            ),
            Some(AgentProtocol::Text)
        );
        for malformed in [
            "",
            "jagent-agent/1",
            "jagent-agent/01;protocols=text;delivery=complete",
            "jagent-agent/+1;protocols=text;delivery=complete",
            "jagent-agent/1;delivery=complete,streaming;protocols=text,native-tools",
            " jagent-agent/1;protocols=text,native-tools;delivery=complete,streaming",
            "jagent-agent/1;protocols=text,text;delivery=complete",
            "jagent-agent/1;protocols=native-tools,text;delivery=complete",
            "jagent-agent/1;protocols=text;delivery=streaming,complete",
            "jagent-agent/1;protocols=future;delivery=complete",
            "jagent-agent/1;protocols=;delivery=complete",
        ] {
            assert_eq!(
                AgentCapabilities::from_wire(malformed),
                Err(CapabilityError::Malformed),
                "accepted {malformed:?}"
            );
        }
        assert_eq!(
            AgentCapabilities::from_wire(
                "jagent-agent/2;protocols=text,native-tools;delivery=complete,streaming"
            ),
            Err(CapabilityError::UnsupportedVersion(2))
        );
        assert_eq!(
            AgentCapabilities::from_wire(&"x".repeat(MAX_AGENT_CAPABILITIES_WIRE_BYTES + 1)),
            Err(CapabilityError::TooLarge)
        );
    }
}
