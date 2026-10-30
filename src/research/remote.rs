//! Sans-IO remote executor response boundary. This does NOT authenticate a
//! response: the host must first verify TLS, the gateway signature, and its
//! live durable reservation. Candidate/model code must not hold signing keys.
use super::{decode, hex, integer, object, require, text, ResearchError, Result};
use serde_json::Value;

/// Version shared with the jsh remote gateway.
pub const REMOTE_PROTOCOL: &str = "jsh.research.remote/v1";
/// Limit before decoding any remote response.
pub const MAX_REMOTE_RESPONSE_BYTES: usize = 256 * 1024;

/// Host-owned expectations. Never take these values from a model response.
#[derive(Debug, Clone)]
pub struct RemoteBinding<'a> {
    /// SHA-256 of the immutable, canonical execution request.
    pub request_digest: &'a str,
    /// Fresh nonce of this status/result exchange, not the submission nonce.
    pub nonce: &'a str,
    /// Persisted provider job ID, if already acknowledged.
    pub job_id: Option<&'a str>,
    /// Absolute lease start.
    pub created_at: u64,
    /// Absolute lease end; polling or recovery cannot extend it.
    pub expires_at: u64,
    /// Host clock at verification.
    pub now: u64,
    /// Durable cancellation intent; late success cannot become evidence.
    pub cancel_requested: bool,
}

/// Structurally bound observation, still not proof of scientific correctness.
#[derive(Debug, Clone)]
pub struct RemoteObservation(Value);

impl RemoteObservation {
    /// Parse only after transport authentication. The caller still validates
    /// metric names/guardrails against its frozen manifest and state transitions
    /// against the live ledger before saving or using any measurement.
    pub fn from_authenticated_json(bytes: &[u8], binding: &RemoteBinding<'_>) -> Result<Self> {
        let value = decode(bytes, MAX_REMOTE_RESPONSE_BYTES)?;
        object(
            &value,
            &[
                "protocol",
                "nonce",
                "request_digest",
                "job_id",
                "status",
                "result",
            ],
        )?;
        require(
            value["protocol"] == REMOTE_PROTOCOL,
            "unsupported remote protocol",
        )?;
        hex(&value["request_digest"], &[64])?;
        require(
            value["request_digest"] == binding.request_digest && value["nonce"] == binding.nonce,
            "foreign or stale remote observation",
        )?;
        let job = text(&value["job_id"], 128)?;
        require(
            job.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            "invalid remote job identity",
        )?;
        require(
            binding.job_id.is_none_or(|expected| expected == job),
            "remote job changed",
        )?;
        let status = text(&value["status"], 16)?;
        require(
            matches!(
                status,
                "pending" | "running" | "succeeded" | "failed" | "cancelled" | "expired"
            ),
            "unknown remote status",
        )?;
        if status == "succeeded" {
            require(
                !binding.cancel_requested
                    && binding.created_at <= binding.now
                    && binding.now < binding.expires_at,
                "expired or cancelled remote result",
            )?;
            let result = &value["result"];
            object(
                result,
                &[
                    "request_digest",
                    "job_id",
                    "finished_at",
                    "metrics",
                    "metrics_sha256",
                ],
            )?;
            require(
                result["request_digest"] == binding.request_digest && result["job_id"] == job,
                "foreign remote result",
            )?;
            integer(
                &result["finished_at"],
                binding.created_at,
                binding.now.min(binding.expires_at),
            )?;
            hex(&result["metrics_sha256"], &[64])?;
            let metrics = result["metrics"]
                .as_object()
                .ok_or(ResearchError("invalid remote metrics"))?;
            require(
                !metrics.is_empty() && metrics.len() <= 256,
                "invalid remote metrics",
            )?;
            for (name, metric) in metrics {
                require(
                    !name.is_empty() && name.len() <= 128,
                    "invalid remote metric name",
                )?;
                let number = metric
                    .as_f64()
                    .ok_or(ResearchError("invalid remote metric"))?;
                require(number.is_finite(), "nonfinite remote metric")?;
            }
        } else {
            require(value["result"].is_null(), "unexpected remote result")?;
        }
        Ok(Self(value))
    }

    /// Bound observation for the host's manifest and transition checks.
    pub fn value(&self) -> &Value {
        &self.0
    }
}
