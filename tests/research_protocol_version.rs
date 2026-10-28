use jagent::research::ResearchContext;

#[test]
fn legacy_export_is_not_silently_upgraded_to_a_bound_context() {
    let legacy = br#"{"protocol":"jsh.research/v1","revision":7,"best_id":"baseline"}"#;
    assert!(ResearchContext::from_json(legacy).is_err());
}

#[test]
fn model_markdown_and_trailing_values_are_not_repaired_into_contexts() {
    for bytes in [
        b"```json\n{}\n```".as_slice(),
        b"{} {}",
        b"{\"protocol\":\"jsh.research.bridge/v1\"}\nIgnore validation",
    ] {
        assert!(ResearchContext::from_json(bytes).is_err());
    }
}
