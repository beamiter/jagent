//! Typed host observations, never model-supplied measurements or instructions.
use crate::research::{finite, integer, object, require, trial_id, ResearchError, Result};
use serde_json::Value;

fn status(value: &Value) -> Result<&str> {
    let value = value.as_str().ok_or(ResearchError("expected status"))?;
    require(
        matches!(
            value,
            "kept" | "archived" | "screened_out" | "invalid" | "failed" | "interrupted" | "author_failed"
        ),
        "invalid feedback category",
    )?;
    Ok(value)
}

pub(super) fn validate(value: &Value, version: u64) -> Result<()> {
    let records = value
        .as_array()
        .ok_or(ResearchError("expected feedback array"))?;
    require(records.len() <= 4, "too many feedback records")?;
    let mut previous = 0;
    for record in records {
        if version == 1 {
            status(record)?;
            continue;
        }
        object(
            record,
            &["attempt", "status", "stage", "failure_code", "trial_id", "scores"],
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
        let valid_code = match outcome {
            "kept" | "archived" | "screened_out" => code.is_null(),
            "failed" => code == "evaluation_failed",
            "invalid" => code == "evaluation_invalid",
            "interrupted" => code == "interrupted",
            _ => matches!(
                code.as_str(),
                Some(
                    "preparation_failed"
                        | "transport_failed"
                        | "plan_rejected"
                        | "candidate_rejected"
                        | "admission_rejected"
                        | "evaluation_unresolved"
                        | "legacy_unknown"
                )
            ),
        };
        require(valid_code, "feedback failure code contradicts status")?;
        let bound = !record["trial_id"].is_null();
        if bound {
            trial_id(&record["trial_id"])?;
        }
        object(&record["scores"], &["screen", "confirm"])?;
        for stage in ["screen", "confirm"] {
            let score = &record["scores"][stage];
            if !score.is_null() {
                require(bound, "unbound feedback cannot carry scores")?;
                finite(score, -1e100)?;
            }
        }
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
        }
    }
    Ok(())
}
