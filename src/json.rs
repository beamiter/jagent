//! JSON decoding shared by untrusted model-response and action boundaries.
//!
//! `serde_json::Value` normally keeps the last value for a repeated object
//! member. That is convenient for many application payloads, but it makes an
//! agent protocol ambiguous: another JSON implementation may keep the first
//! value, so the same bytes can name two different commands or completion
//! states. An allocation-light preflight retains only the names in each open
//! object while validating the complete structure. The ordinary
//! `serde_json::Value` decoder then constructs the single retained response
//! tree, preserving whichever compatible `serde_json` features the embedding
//! dependency graph selected.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;

const DUPLICATE_MEMBER: &str = "duplicate JSON object member";

pub(crate) fn from_slice(input: &[u8]) -> Result<Value, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(input);
    NoDuplicateMembers::deserialize(&mut deserializer)?;
    deserializer.end()?;
    serde_json::from_slice(input)
}

pub(crate) fn from_str(input: &str) -> Result<Value, serde_json::Error> {
    from_slice(input.as_bytes())
}

struct NoDuplicateMembers;

impl<'de> Deserialize<'de> for NoDuplicateMembers {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer
            .deserialize_any(NoDuplicateMembersVisitor)
            .map(|()| Self)
    }
}

struct NoDuplicateMembersVisitor;

impl<'de> Visitor<'de> for NoDuplicateMembersVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object members")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element::<NoDuplicateMembers>()?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut members = HashSet::new();
        while let Some(member) = object.next_key::<String>()? {
            if !members.insert(member) {
                // Do not reflect a provider/model-controlled member name in
                // logs. Stop before decoding the duplicate's value too.
                return Err(de::Error::custom(DUPLICATE_MEMBER));
            }
            object.next_value::<NoDuplicateMembers>()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unique_decoder_matches_serde_json_for_unambiguous_values() {
        let inputs = [
            "null",
            "true",
            "-9223372036854775808",
            "18446744073709551615",
            "1.25e10",
            r#""text""#,
            r#"[null,true,7,"text",{"nested":[1,2,3]}]"#,
            r#"{"text":"hello","nested":{"enabled":true},"empty":[]}"#,
        ];

        for input in inputs {
            assert_eq!(
                from_str(input).unwrap(),
                serde_json::from_str::<Value>(input).unwrap(),
                "{input}"
            );
        }
    }

    #[test]
    fn duplicate_members_fail_at_every_depth_without_echoing_the_name() {
        let cases = [
            r#"{"command":"safe","command":"dangerous"}"#,
            r#"{"outer":{"finish_reason":"stop","finish_reason":"length"}}"#,
            r#"[{"tool_calls":[],"tool_calls":[{"function":{}}]}]"#,
            r#"{"command":"safe","\u0063ommand":"dangerous"}"#,
        ];

        for input in cases {
            let error = from_str(input).unwrap_err().to_string();
            assert!(error.contains(DUPLICATE_MEMBER), "{input}: {error}");
            assert!(!error.contains("command"), "{error}");
            assert!(!error.contains("finish_reason"), "{error}");
            assert!(!error.contains("tool_calls"), "{error}");
        }

        assert_eq!(
            from_str(&json!({"command": "safe"}).to_string()).unwrap(),
            json!({"command": "safe"})
        );
    }

    #[test]
    fn trailing_or_malformed_input_stays_invalid() {
        for input in [r#"{"ok":true} trailing"#, r#"{"ok":true"#] {
            assert!(from_str(input).is_err(), "{input}");
        }
    }
}
