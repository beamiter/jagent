use jagent::research::remote::{RemoteBinding, RemoteObservation, REMOTE_PROTOCOL};
use serde_json::{json, Value};

fn response() -> Value {
    json!({"protocol": REMOTE_PROTOCOL, "nonce": "fresh", "request_digest": "a".repeat(64),
        "job_id": "dlc-mock-1", "status": "succeeded", "result": {
        "request_digest": "a".repeat(64), "job_id": "dlc-mock-1", "finished_at": 110,
        "metrics": {"loss": 1.0}, "metrics_sha256": "b".repeat(64)}})
}

#[test]
fn remote_binding_rejects_replay_and_identity_changes() {
    let digest = "a".repeat(64);
    let binding = RemoteBinding {
        request_digest: &digest,
        nonce: "fresh",
        job_id: Some("dlc-mock-1"),
        created_at: 100,
        expires_at: 200,
        now: 120,
        cancel_requested: false,
    };
    assert!(RemoteObservation::from_authenticated_json(
        &serde_json::to_vec(&response()).unwrap(),
        &binding
    )
    .is_ok());
    for field in ["nonce", "job_id", "request_digest", "protocol", "status"] {
        let mut value = response();
        value[field] = json!("foreign");
        assert!(RemoteObservation::from_authenticated_json(
            &serde_json::to_vec(&value).unwrap(),
            &binding
        )
        .is_err());
    }
    for field in ["job_id", "request_digest", "metrics_sha256"] {
        let mut value = response();
        value["result"][field] = json!("foreign");
        assert!(RemoteObservation::from_authenticated_json(
            &serde_json::to_vec(&value).unwrap(),
            &binding
        )
        .is_err());
    }
    for now in [99, 200, 201] {
        let other = RemoteBinding {
            now,
            ..binding.clone()
        };
        assert!(RemoteObservation::from_authenticated_json(
            &serde_json::to_vec(&response()).unwrap(),
            &other
        )
        .is_err());
    }
    let cancelled = RemoteBinding {
        cancel_requested: true,
        ..binding.clone()
    };
    assert!(RemoteObservation::from_authenticated_json(
        &serde_json::to_vec(&response()).unwrap(),
        &cancelled
    )
    .is_err());
    let mut value = response();
    value["result"]["finished_at"] = json!(121);
    assert!(RemoteObservation::from_authenticated_json(
        &serde_json::to_vec(&value).unwrap(),
        &binding
    )
    .is_err());
    value["result"]["finished_at"] = json!(110);
    value["result"]["metrics"]["loss"] = json!(true);
    assert!(RemoteObservation::from_authenticated_json(
        &serde_json::to_vec(&value).unwrap(),
        &binding
    )
    .is_err());
}
