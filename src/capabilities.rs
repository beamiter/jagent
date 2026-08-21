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

/// Latest supported capability-token schema version.
pub const AGENT_CAPABILITIES_VERSION: u16 = 2;
/// Canonical legacy capability token accepted for version-1 peers.
pub const AGENT_CAPABILITIES_V1_WIRE: &str =
    "jagent-agent/1;protocols=text,native-tools;delivery=complete,streaming";
/// Canonical opt-in version-2 capability token for all built-in providers.
///
/// Version 2 advertises exact protocol/delivery pairs instead of implying the
/// Cartesian product of two independent sets.
pub const AGENT_CAPABILITIES_V2_WIRE: &str = "jagent-agent/2;modes=text+complete,text+streaming,native-tools+complete,native-tools+streaming";
/// Capability tokens are deliberately tiny; reject an effectively unbounded
/// environment or IPC value before splitting it.
pub const MAX_AGENT_CAPABILITIES_WIRE_BYTES: usize = 256;

const MODE_TEXT_COMPLETE: u8 = 1 << 0;
const MODE_TEXT_STREAMING: u8 = 1 << 1;
const MODE_NATIVE_TOOLS_COMPLETE: u8 = 1 << 2;
const MODE_NATIVE_TOOLS_STREAMING: u8 = 1 << 3;
const ALL_MODES: u8 = MODE_TEXT_COMPLETE
    | MODE_TEXT_STREAMING
    | MODE_NATIVE_TOOLS_COMPLETE
    | MODE_NATIVE_TOOLS_STREAMING;

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
/// [`agent_capabilities`] for compatibility-first provider discovery,
/// [`agent_capabilities_for_peer`] for a reply, or [`Self::from_wire`] for a
/// peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AgentCapabilities {
    version: u16,
    modes: u8,
}

impl AgentCapabilities {
    const V2_ALL: Self = Self {
        version: AGENT_CAPABILITIES_VERSION,
        modes: ALL_MODES,
    };

    /// Convert an exact matrix to the largest compatibility-oriented v1
    /// rectangle selected by a deterministic preference order. Every mode in
    /// the resulting Cartesian product was present in `self`; downgrade may
    /// omit a usable mode but can never advertise an unsupported crossing.
    const fn safe_v1_downgrade(self) -> Self {
        let modes = if self.modes & ALL_MODES == ALL_MODES {
            ALL_MODES
        } else if self.modes & (MODE_TEXT_COMPLETE | MODE_NATIVE_TOOLS_COMPLETE)
            == (MODE_TEXT_COMPLETE | MODE_NATIVE_TOOLS_COMPLETE)
        {
            MODE_TEXT_COMPLETE | MODE_NATIVE_TOOLS_COMPLETE
        } else if self.modes & (MODE_TEXT_COMPLETE | MODE_TEXT_STREAMING)
            == (MODE_TEXT_COMPLETE | MODE_TEXT_STREAMING)
        {
            MODE_TEXT_COMPLETE | MODE_TEXT_STREAMING
        } else if self.modes & (MODE_NATIVE_TOOLS_COMPLETE | MODE_NATIVE_TOOLS_STREAMING)
            == (MODE_NATIVE_TOOLS_COMPLETE | MODE_NATIVE_TOOLS_STREAMING)
        {
            MODE_NATIVE_TOOLS_COMPLETE | MODE_NATIVE_TOOLS_STREAMING
        } else if self.modes & (MODE_TEXT_STREAMING | MODE_NATIVE_TOOLS_STREAMING)
            == (MODE_TEXT_STREAMING | MODE_NATIVE_TOOLS_STREAMING)
        {
            MODE_TEXT_STREAMING | MODE_NATIVE_TOOLS_STREAMING
        } else if self.modes & MODE_TEXT_COMPLETE != 0 {
            MODE_TEXT_COMPLETE
        } else if self.modes & MODE_NATIVE_TOOLS_COMPLETE != 0 {
            MODE_NATIVE_TOOLS_COMPLETE
        } else if self.modes & MODE_TEXT_STREAMING != 0 {
            MODE_TEXT_STREAMING
        } else if self.modes & MODE_NATIVE_TOOLS_STREAMING != 0 {
            MODE_NATIVE_TOOLS_STREAMING
        } else {
            panic!("provider Agent capability matrix must not be empty")
        };
        Self { version: 1, modes }
    }

    /// Capability schema version carried by this value.
    pub const fn version(self) -> u16 {
        self.version
    }

    /// Whether the peer supports this exact action protocol and delivery
    /// combination.
    pub const fn supports(self, protocol: AgentProtocol, delivery: AgentDelivery) -> bool {
        let mode = match (protocol, delivery) {
            (AgentProtocol::Text, AgentDelivery::Complete) => MODE_TEXT_COMPLETE,
            (AgentProtocol::Text, AgentDelivery::Streaming) => MODE_TEXT_STREAMING,
            (AgentProtocol::NativeTools, AgentDelivery::Complete) => MODE_NATIVE_TOOLS_COMPLETE,
            (AgentProtocol::NativeTools, AgentDelivery::Streaming) => MODE_NATIVE_TOOLS_STREAMING,
        };
        self.modes & mode != 0
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
        if self.version == 1 {
            let protocols = match self.modes & (MODE_TEXT_COMPLETE | MODE_TEXT_STREAMING) != 0 {
                true if self.modes & (MODE_NATIVE_TOOLS_COMPLETE | MODE_NATIVE_TOOLS_STREAMING)
                    != 0 =>
                {
                    "text,native-tools"
                }
                true => "text",
                false => "native-tools",
            };
            let delivery = match self.modes & (MODE_TEXT_COMPLETE | MODE_NATIVE_TOOLS_COMPLETE) != 0
            {
                true if self.modes & (MODE_TEXT_STREAMING | MODE_NATIVE_TOOLS_STREAMING) != 0 => {
                    "complete,streaming"
                }
                true => "complete",
                false => "streaming",
            };
            return format!(
                "jagent-agent/{};protocols={protocols};delivery={delivery}",
                self.version
            );
        }

        let mut modes = String::new();
        for (name, bit) in [
            ("text+complete", MODE_TEXT_COMPLETE),
            ("text+streaming", MODE_TEXT_STREAMING),
            ("native-tools+complete", MODE_NATIVE_TOOLS_COMPLETE),
            ("native-tools+streaming", MODE_NATIVE_TOOLS_STREAMING),
        ] {
            if self.modes & bit == 0 {
                continue;
            }
            if !modes.is_empty() {
                modes.push(',');
            }
            modes.push_str(name);
        }
        format!("jagent-agent/{};modes={modes}", self.version)
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
        match version {
            1 => {
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
                let protocols = parse_bits(protocol_field, &[("text", 1), ("native-tools", 2)])?;
                let deliveries = parse_bits(delivery_field, &[("complete", 1), ("streaming", 2)])?;
                let mut modes = 0;
                if protocols & 1 != 0 && deliveries & 1 != 0 {
                    modes |= MODE_TEXT_COMPLETE;
                }
                if protocols & 1 != 0 && deliveries & 2 != 0 {
                    modes |= MODE_TEXT_STREAMING;
                }
                if protocols & 2 != 0 && deliveries & 1 != 0 {
                    modes |= MODE_NATIVE_TOOLS_COMPLETE;
                }
                if protocols & 2 != 0 && deliveries & 2 != 0 {
                    modes |= MODE_NATIVE_TOOLS_STREAMING;
                }
                Ok(Self { version, modes })
            }
            AGENT_CAPABILITIES_VERSION => {
                let mode_field = fields
                    .next()
                    .and_then(|field| field.strip_prefix("modes="))
                    .ok_or(CapabilityError::Malformed)?;
                if fields.next().is_some() {
                    return Err(CapabilityError::Malformed);
                }
                let modes = parse_bits(
                    mode_field,
                    &[
                        ("text+complete", MODE_TEXT_COMPLETE),
                        ("text+streaming", MODE_TEXT_STREAMING),
                        ("native-tools+complete", MODE_NATIVE_TOOLS_COMPLETE),
                        ("native-tools+streaming", MODE_NATIVE_TOOLS_STREAMING),
                    ],
                )?;
                Ok(Self { version, modes })
            }
            _ => Err(CapabilityError::UnsupportedVersion(version)),
        }
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

/// Provider capabilities emitted for compatibility-first discovery.
///
/// The explicit provider match is intentional. When a provider gains or loses
/// a protocol or delivery form, compiler exhaustiveness makes this table and
/// request preparation change together. The default remains version 1 so a
/// newly upgraded endpoint can still initiate discovery with a 0.7 peer that
/// predates version 2. Use [`agent_capabilities_v2`] only after an integration
/// has explicit evidence that its peer accepts version 2, or prefer
/// [`agent_capabilities_for_peer`] after decoding the peer's token.
pub const fn agent_capabilities(provider: Provider) -> AgentCapabilities {
    agent_capabilities_v2(provider).safe_v1_downgrade()
}

/// Provider capabilities encoded with the exact-pair version-2 schema.
///
/// This is deliberately opt-in. Sending its wire value to an unprobed
/// version-1 peer would make that peer reject discovery with
/// [`CapabilityError::UnsupportedVersion`].
pub const fn agent_capabilities_v2(provider: Provider) -> AgentCapabilities {
    match provider {
        Provider::Anthropic | Provider::OpenAiCompatible | Provider::Ollama => {
            AgentCapabilities::V2_ALL
        }
    }
}

/// Emit provider capabilities in a schema version the decoded peer accepts.
///
/// Since [`AgentCapabilities::from_wire`] only constructs supported versions,
/// this cannot select an unknown schema. A version-1 peer receives the legacy
/// Cartesian-product token; a version-2 peer receives exact mode pairs.
pub const fn agent_capabilities_for_peer(
    provider: Provider,
    peer: AgentCapabilities,
) -> AgentCapabilities {
    match peer.version() {
        1 => agent_capabilities(provider),
        AGENT_CAPABILITIES_VERSION => agent_capabilities_v2(provider),
        _ => agent_capabilities(provider),
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
    fn every_provider_advertises_a_v1_default_and_an_exact_v2_opt_in() {
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let compatible = agent_capabilities(provider);
            let exact = agent_capabilities_v2(provider);
            assert_eq!(compatible.version(), 1);
            assert_eq!(compatible.to_wire(), AGENT_CAPABILITIES_V1_WIRE);
            assert_eq!(exact.version(), AGENT_CAPABILITIES_VERSION);
            assert_eq!(exact.to_wire(), AGENT_CAPABILITIES_V2_WIRE);
            for protocol in [AgentProtocol::Text, AgentProtocol::NativeTools] {
                for delivery in [AgentDelivery::Complete, AgentDelivery::Streaming] {
                    assert!(exact.supports(protocol, delivery));
                    if compatible.supports(protocol, delivery) {
                        assert!(
                            exact.supports(protocol, delivery),
                            "v1 downgrade overclaimed {protocol:?}+{delivery:?} for {provider:?}"
                        );
                    }
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
        let capabilities = agent_capabilities_v2(Provider::Anthropic);
        assert_eq!(
            AgentCapabilities::from_wire(&capabilities.to_wire()),
            Ok(capabilities)
        );
        assert_eq!(capabilities.to_wire(), AGENT_CAPABILITIES_V2_WIRE);
        let legacy = AgentCapabilities::from_wire(AGENT_CAPABILITIES_V1_WIRE).unwrap();
        assert_eq!(legacy.version(), 1);
        assert_eq!(legacy.to_wire(), AGENT_CAPABILITIES_V1_WIRE);
        for protocol in [AgentProtocol::Text, AgentProtocol::NativeTools] {
            for delivery in [AgentDelivery::Complete, AgentDelivery::Streaming] {
                assert!(legacy.supports(protocol, delivery));
            }
        }
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
            "jagent-agent/2",
            "jagent-agent/2;modes=",
            "jagent-agent/2;protocols=text;delivery=complete",
            "jagent-agent/2;modes=text+complete;text=native-tools+complete",
            "jagent-agent/2;modes=text+complete,text+complete",
            "jagent-agent/2;modes=native-tools+complete,text+complete",
            "jagent-agent/2;modes=text+complete,future+complete",
            "jagent-agent/2;modes=text+complete ",
        ] {
            assert_eq!(
                AgentCapabilities::from_wire(malformed),
                Err(CapabilityError::Malformed),
                "accepted {malformed:?}"
            );
        }
        assert_eq!(
            AgentCapabilities::from_wire("jagent-agent/3;modes=text+complete"),
            Err(CapabilityError::UnsupportedVersion(3))
        );
        assert_eq!(
            AgentCapabilities::from_wire(&"x".repeat(MAX_AGENT_CAPABILITIES_WIRE_BYTES + 1)),
            Err(CapabilityError::TooLarge)
        );
    }

    #[test]
    fn version_two_preserves_exact_protocol_delivery_pairs() {
        let split = AgentCapabilities::from_wire(
            "jagent-agent/2;modes=text+complete,native-tools+streaming",
        )
        .unwrap();
        assert_eq!(
            split.to_wire(),
            "jagent-agent/2;modes=text+complete,native-tools+streaming"
        );
        assert!(split.supports(AgentProtocol::Text, AgentDelivery::Complete));
        assert!(!split.supports(AgentProtocol::Text, AgentDelivery::Streaming));
        assert!(!split.supports(AgentProtocol::NativeTools, AgentDelivery::Complete));
        assert!(split.supports(AgentProtocol::NativeTools, AgentDelivery::Streaming));

        let local = agent_capabilities_v2(Provider::Ollama);
        assert_eq!(
            local.negotiate_with(
                split,
                &[AgentProtocol::NativeTools, AgentProtocol::Text],
                AgentDelivery::Complete,
            ),
            Some(AgentProtocol::Text)
        );
        assert_eq!(
            local.negotiate_with(
                split,
                &[AgentProtocol::Text, AgentProtocol::NativeTools],
                AgentDelivery::Streaming,
            ),
            Some(AgentProtocol::NativeTools)
        );
    }

    #[test]
    fn v1_downgrade_never_invents_a_crossed_mode() {
        let asymmetric = AgentCapabilities {
            version: AGENT_CAPABILITIES_VERSION,
            modes: MODE_TEXT_COMPLETE | MODE_NATIVE_TOOLS_STREAMING,
        };
        let downgraded = asymmetric.safe_v1_downgrade();
        assert_eq!(
            downgraded.to_wire(),
            "jagent-agent/1;protocols=text;delivery=complete"
        );
        for protocol in [AgentProtocol::Text, AgentProtocol::NativeTools] {
            for delivery in [AgentDelivery::Complete, AgentDelivery::Streaming] {
                assert!(
                    !downgraded.supports(protocol, delivery)
                        || asymmetric.supports(protocol, delivery),
                    "downgrade invented {protocol:?}+{delivery:?}"
                );
            }
        }
    }

    #[test]
    fn version_one_compatibility_keeps_its_legacy_product_and_cannot_smuggle_v2() {
        let legacy = AgentCapabilities::from_wire(
            "jagent-agent/1;protocols=text,native-tools;delivery=complete",
        )
        .unwrap();
        assert_eq!(legacy.version(), 1);
        assert!(legacy.supports(AgentProtocol::Text, AgentDelivery::Complete));
        assert!(legacy.supports(AgentProtocol::NativeTools, AgentDelivery::Complete));
        assert!(!legacy.supports(AgentProtocol::Text, AgentDelivery::Streaming));
        assert!(!legacy.supports(AgentProtocol::NativeTools, AgentDelivery::Streaming));
        assert_eq!(
            legacy.to_wire(),
            "jagent-agent/1;protocols=text,native-tools;delivery=complete"
        );

        assert_eq!(
            AgentCapabilities::from_wire(
                "jagent-agent/1;modes=text+complete,native-tools+streaming"
            ),
            Err(CapabilityError::Malformed)
        );
        assert_eq!(
            AgentCapabilities::from_wire(
                "jagent-agent/2;protocols=text,native-tools;delivery=complete"
            ),
            Err(CapabilityError::Malformed)
        );
    }

    fn decode_like_a_version_one_peer(value: &str) -> Result<AgentCapabilities, CapabilityError> {
        if !value.starts_with("jagent-agent/1;") {
            let version = value
                .strip_prefix("jagent-agent/")
                .and_then(|tail| tail.split(';').next())
                .and_then(|version| version.parse().ok())
                .unwrap_or(0);
            return Err(CapabilityError::UnsupportedVersion(version));
        }
        AgentCapabilities::from_wire(value)
    }

    #[test]
    fn rolling_upgrade_emission_is_old_peer_safe_and_peer_version_aware() {
        for provider in [
            Provider::Anthropic,
            Provider::OpenAiCompatible,
            Provider::Ollama,
        ] {
            let default = agent_capabilities(provider);
            assert_eq!(
                decode_like_a_version_one_peer(&default.to_wire()),
                Ok(default),
                "a new endpoint's first advertisement broke an old peer"
            );

            let exact = agent_capabilities_v2(provider);
            assert_eq!(
                decode_like_a_version_one_peer(&exact.to_wire()),
                Err(CapabilityError::UnsupportedVersion(2))
            );
            assert_eq!(AgentCapabilities::from_wire(&exact.to_wire()), Ok(exact));

            let old_peer =
                AgentCapabilities::from_wire("jagent-agent/1;protocols=text;delivery=complete")
                    .unwrap();
            let new_peer = AgentCapabilities::from_wire(
                "jagent-agent/2;modes=text+complete,native-tools+streaming",
            )
            .unwrap();
            assert_eq!(
                agent_capabilities_for_peer(provider, old_peer).to_wire(),
                AGENT_CAPABILITIES_V1_WIRE
            );
            assert_eq!(
                agent_capabilities_for_peer(provider, new_peer).to_wire(),
                AGENT_CAPABILITIES_V2_WIRE
            );
        }
    }
}
