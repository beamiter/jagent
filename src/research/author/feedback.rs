//! Typed host observations, never model-supplied measurements or instructions.
use crate::research::{
    finite, integer, object, require, trial_id, ResearchContext, ResearchError, Result,
};
use serde_json::Value;
use std::collections::HashSet;

fn status(value: &Value) -> Result<&str> {
    let value = value.as_str().ok_or(ResearchError("expected status"))?;
    require(
        matches!(
            value,
            "kept"
                | "archived"
                | "screened_out"
                | "invalid"
                | "failed"
                | "interrupted"
                | "author_failed"
        ),
        "invalid feedback category",
    )?;
    Ok(value)
}

fn validate_binding(record: &Value, context: &ResearchContext) -> Result<()> {
    let Some(id) = record["trial_id"].as_str() else {
        return Ok(());
    };
    let snapshot = context.as_json();
    let visible = snapshot["recent_trials"]
        .as_array()
        .expect("validated context trials")
        .iter()
        .find(|trial| trial["id"] == id);
    if let Some(trial) = visible {
        require(
            record["status"] == trial["status"],
            "feedback status contradicts visible trial",
        )?;
        for stage in ["screen", "confirm"] {
            // Both sides have already been validated as finite numbers or
            // absent evidence. Compare numeric meaning, not JSON spelling.
            require(
                record["scores"][stage].as_f64() == trial["scores"][stage].as_f64(),
                "feedback scores contradict visible trial",
            )?;
        }
    } else {
        require(
            snapshot["omitted_trials"]
                .as_u64()
                .expect("validated omitted count")
                > 0,
            "feedback trial is absent from complete history",
        )?;
    }
    Ok(())
}

pub(super) fn validate(value: &Value, version: u64, context: &ResearchContext) -> Result<()> {
    let records = value
        .as_array()
        .ok_or(ResearchError("expected feedback array"))?;
    require(records.len() <= 4, "too many feedback records")?;
    let mut previous = 0;
    let mut trials = HashSet::new();
    for record in records {
        if version == 1 {
            status(record)?;
            continue;
        }
        object(
            record,
            &[
                "attempt",
                "status",
                "stage",
                "failure_code",
                "trial_id",
                "scores",
            ],
        )?;
        let attempt = integer(&record["attempt"], 1, 1000)?;
        require(attempt > previous, "feedback is not in attempt order")?;
        previous = attempt;
        let outcome = status(&record["status"])?;
        require(
            matches!(
                record["stage"].as_str(),
                Some(
                    "prepare"
                        | "model_request"
                        | "validate_plan"
                        | "commit_candidate"
                        | "submit"
                        | "evaluate"
                        | "finished"
                        | "recovered_without_replay"
                )
            ),
            "invalid feedback stage",
        )?;
        let code = &record["failure_code"];
        let stage = record["stage"].as_str().expect("validated stage");
        let valid_code = match outcome {
            "kept" | "archived" | "screened_out" => code.is_null(),
            "failed" => code == "evaluation_failed",
            "invalid" => code == "evaluation_invalid",
            "interrupted" => code == "interrupted",
            _ => matches!(
                (stage, code.as_str()),
                // Older ledgers without a stage use the host's prepare placeholder.
                ("prepare", Some("preparation_failed" | "legacy_unknown"))
                    | ("model_request", Some("transport_failed"))
                    | ("validate_plan", Some("plan_rejected"))
                    | ("commit_candidate", Some("candidate_rejected"))
                    | ("submit", Some("admission_rejected"))
                    | ("evaluate", Some("evaluation_unresolved"))
                    | ("recovered_without_replay", Some("legacy_unknown"))
            ),
        };
        require(valid_code, "feedback failure code contradicts status")?;
        let bound = !record["trial_id"].is_null();
        if bound {
            let id = trial_id(&record["trial_id"])?;
            require(trials.insert(id), "duplicate feedback trial")?;
            require(
                matches!(
                    stage,
                    "submit" | "evaluate" | "finished" | "recovered_without_replay"
                ),
                "feedback trial precedes admission",
            )?;
        }
        object(&record["scores"], &["screen", "confirm"])?;
        for stage in ["screen", "confirm"] {
            let score = &record["scores"][stage];
            if !score.is_null() {
                require(bound, "unbound feedback cannot carry scores")?;
                finite(score, -1e100)?;
            }
        }
        require(
            record["scores"]["confirm"].is_null() || !record["scores"]["screen"].is_null(),
            "confirmation feedback lacks screen evidence",
        )?;
        if matches!(outcome, "kept" | "archived" | "screened_out") {
            require(
                bound
                    && !record["scores"]["screen"].is_null()
                    && matches!(
                        record["stage"].as_str(),
                        Some("finished" | "recovered_without_replay")
                    ),
                "successful evaluation lacks completed evidence",
            )?;
            require(
                if outcome == "screened_out" {
                    record["scores"]["confirm"].is_null()
                } else {
                    !record["scores"]["confirm"].is_null()
                },
                "feedback confirmation evidence contradicts status",
            )?;
        }
        if matches!(outcome, "failed" | "invalid") {
            require(bound, "evaluation failure lacks a trial identity")?;
            require(
                matches!(stage, "finished" | "recovered_without_replay"),
                "evaluation failure lacks a completed stage",
            )?;
        }
        validate_binding(record, context)?;
    }
    Ok(())
}
