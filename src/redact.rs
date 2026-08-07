//! redact — high-confidence secret scrubbing for AI-bound text.
//!
//! Scoped narrowly: integrations run this over explicitly AI-bound context,
//! history, and prompts before they serialize a provider request.
//! Goal is to stop "I pasted my .env" / "I ran `aws sts get-session-token`"
//! accidents — not to be a general DLP. Patterns are conservative: we only
//! match shapes whose false-positive rate is essentially zero (AWS access
//! key ids, GitHub PATs, Slack tokens, JWTs, PEM block headers, credentialed
//! URLs, and explicit bearer headers). Generic
//! "looks like a hex string" detection would gut routine command output
//! (git SHAs, hashes) so we deliberately avoid it.
//!
//! Replacement format: `[REDACTED:<kind>]` — short enough to keep the
//! surrounding token context legible for the model, distinctive enough to
//! survive copy/paste through the AI panel back to the user.

use regex::Regex;
use std::sync::OnceLock;

struct SecretPattern {
    replacement: &'static str,
    regex: Regex,
}

/// Each pattern carries a regex replacement. Most replace the entire match;
/// the URL and bearer forms retain the non-secret framing through named
/// captures so logs remain useful after scrubbing. Order is significant: the
/// specific token formats run before the generic bearer-header rule, and PEM
/// blocks come first so their complete multi-line body is removed.
fn patterns() -> &'static [SecretPattern] {
    static CELL: OnceLock<Vec<SecretPattern>> = OnceLock::new();
    CELL.get_or_init(|| {
        let pats: &[(&str, &str)] = &[
            // PEM private key block (any flavor): RSA, EC, OPENSSH, plain
            // PRIVATE KEY. Span includes body so the whole secret is gone.
            (
                "[REDACTED:private-key]",
                r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            ),
            // AWS access key ids (long-lived + STS). Format is fixed.
            (
                "[REDACTED:aws-access-key]",
                r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
            ),
            // GitHub fine-grained PAT (long form).
            ("[REDACTED:github-pat]", r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
            // GitHub classic tokens: ghp_, gho_, ghu_, ghs_, ghr_.
            ("[REDACTED:github-token]", r"\bgh[opusr]_[A-Za-z0-9]{36,}\b"),
            // Slack bot / user / app / refresh tokens.
            (
                "[REDACTED:slack-token]",
                r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
            ),
            // JWT (header.payload.signature). Loose but the three-segment
            // base64url structure with `eyJ` header prefix is distinctive.
            (
                "[REDACTED:jwt]",
                r"\beyJ[A-Za-z0-9_=-]{8,}\.eyJ[A-Za-z0-9_=-]{8,}\.[A-Za-z0-9_=.+/-]{8,}\b",
            ),
            // Anthropic API keys — protect the user's own key if it shows
            // up in `env | grep` output etc. Format: sk-ant-<base64ish>.
            (
                "[REDACTED:anthropic-key]",
                r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b",
            ),
            // OpenAI keys (sk-, sk-proj-). The 20+ tail catches both.
            (
                "[REDACTED:openai-key]",
                r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b",
            ),
            // A password embedded in URI userinfo. Preserve the scheme and
            // username as useful context, but remove everything between the
            // first credential separator and the host delimiter. The paired
            // `://` and `@` keep this from matching ordinary `name:value`
            // output or a bare host:port.
            (
                "${prefix}:[REDACTED:url-password]@",
                r"(?i)(?P<prefix>\b[a-z][a-z0-9+.\-]*://[^\s:/@]+):[^\s/@]+@",
            ),
            // RFC 6750 bearer credentials that are opaque rather than JWTs.
            // The explicit authentication scheme keeps long hashes and ids in
            // ordinary command output untouched.
            (
                "${scheme} [REDACTED:bearer-token]",
                r"(?i)(?P<scheme>\bbearer)[ \t]+[A-Za-z0-9._~+/=-]{16,}",
            ),
        ];
        pats.iter()
            .map(|(replacement, pattern)| SecretPattern {
                replacement,
                regex: Regex::new(pattern).expect("redact pattern compiles"),
            })
            .collect()
    })
}

/// Walk the input through every pattern, replacing matches with
/// `[REDACTED:<kind>]`. Allocates only when something matches — short
/// circuits on a clean string so the common case (most block output) is
/// just a couple of regex `is_match` probes.
pub fn redact_secrets(input: &str) -> String {
    let mut current = std::borrow::Cow::Borrowed(input);
    for pattern in patterns() {
        if pattern.regex.is_match(&current) {
            current = std::borrow::Cow::Owned(
                pattern
                    .regex
                    .replace_all(&current, pattern.replacement)
                    .into_owned(),
            );
        }
    }
    current.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_nothing_to_redact() {
        let s = "the quick brown fox 1234567890 deadbeefcafef00d";
        assert_eq!(redact_secrets(s), s);
    }

    #[test]
    fn redacts_aws_access_key() {
        let s = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:aws-access-key]"), "got {out}");
        assert!(!out.contains("AKIA"));
    }

    #[test]
    fn redacts_aws_sts_access_key() {
        let s = "ASIAY34FZKBOKMUTVV7A is current STS";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:aws-access-key]"), "got {out}");
    }

    #[test]
    fn redacts_github_classic_token() {
        // Classic GitHub PATs are exactly 36 chars after the prefix.
        let s = "git remote set-url origin https://x:ghp_1234567890abcdefghijABCDEFGHIJ123456@github.com/";
        let out = redact_secrets(s);
        // Once the token is scrubbed, the surrounding credentialed-URL rule
        // deliberately collapses the whole password slot as well. This is the
        // same layered result shell callers produced before that shared rule
        // moved into jagent.
        assert!(out.contains("[REDACTED:url-password]"), "got {out}");
        assert!(!out.contains("ghp_"));
    }

    #[test]
    fn redacts_github_fine_grained_pat() {
        let body = "X".repeat(82);
        let s = format!("token: github_pat_{body}");
        let out = redact_secrets(&s);
        assert!(out.contains("[REDACTED:github-pat]"), "got {out}");
    }

    #[test]
    fn redacts_slack_token() {
        let s = "SLACK_TOKEN=xoxb-12345-67890-abcdefghijklmnop";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:slack-token]"), "got {out}");
    }

    #[test]
    fn redacts_jwt() {
        // Realistic-shape JWT (header.payload.signature, base64url).
        let s = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:jwt]"), "got {out}");
        assert!(!out.contains("eyJzdWIi"));
    }

    #[test]
    fn redacts_pem_private_key_block_inclusive() {
        let s = "before\n-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\nLOTSOFGARBAGE==\n-----END OPENSSH PRIVATE KEY-----\nafter";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:private-key]"));
        assert!(!out.contains("b3BlbnNzaC1rZXktdjE"));
        assert!(out.contains("before"));
        assert!(out.contains("after"));
    }

    #[test]
    fn redacts_anthropic_key() {
        let s = "ANTHROPIC_API_KEY=sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:anthropic-key]"), "got {out}");
    }

    #[test]
    fn redacts_url_password_but_keeps_the_non_secret_origin_context() {
        let s = "psql postgres://alice:p%40ss:word@db.internal:5432/app";
        let out = redact_secrets(s);
        assert_eq!(
            out,
            "psql postgres://alice:[REDACTED:url-password]@db.internal:5432/app"
        );
        assert!(!out.contains("p%40ss"));

        // A URL without a password and ordinary host:port output are not
        // credentials and remain useful to the model.
        let safe = "https://example.test/path localhost:5432";
        assert_eq!(redact_secrets(safe), safe);
    }

    #[test]
    fn redacts_opaque_bearer_credentials_without_hiding_the_scheme() {
        let s = "Authorization: bEaReR mF_9.B5f-4.1JqM0X+Yz==";
        let out = redact_secrets(s);
        assert_eq!(out, "Authorization: bEaReR [REDACTED:bearer-token]");
        assert!(!out.contains("mF_9"));

        // Requiring a realistically long credential avoids eating prose and
        // CLI arguments that merely happen to follow the word "bearer".
        let safe = "document bearer abc123";
        assert_eq!(redact_secrets(safe), safe);
    }

    #[test]
    fn does_not_redact_short_git_sha_or_plain_uuid() {
        // Common content we MUST leave alone.
        let s = "commit deadbeefcafef00d1234567890abcdef01234567 (HEAD -> main)\nuuid: 550e8400-e29b-41d4-a716-446655440000";
        assert_eq!(redact_secrets(s), s);
    }

    #[test]
    fn multiple_secrets_in_same_input_all_redacted() {
        let s = "AKIAIOSFODNN7EXAMPLE then ghp_1234567890abcdefghijABCDEFGHIJ123456 done";
        let out = redact_secrets(s);
        assert!(out.contains("[REDACTED:aws-access-key]"));
        assert!(out.contains("[REDACTED:github-token]"));
        assert!(!out.contains("AKIA"));
        assert!(!out.contains("ghp_"));
    }
}
