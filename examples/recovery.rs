//! A local lifecycle fixture: timeout, retry, late callback, streaming,
//! snapshot restoration, explicit review, and completion. No network or
//! process execution is performed.

use jagent::{
    prepare_agent_request, AgentProtocol, AgentRequestSpec, AgentSession, AgentSessionSnapshot,
    AgentState, ChatConfig, CommandExecutionOutcome, Message, ModelOutcome, ModelRequestError,
    Provider, Role,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protocol = AgentProtocol::NativeTools;
    let config = ChatConfig {
        provider: Provider::OpenAiCompatible,
        api_key: None,
        model: "local-agent".into(),
        base_url: "http://127.0.0.1:1234".into(),
        max_tokens: 256,
        temperature: Some(0.0),
    };
    let mut session = AgentSession::new(4);
    session.submit_user("show the current directory")?;

    // Capture a fresh identity before each HTTP attempt. Task cancellation
    // alone cannot distinguish a timed-out attempt from its replacement.
    let first_request = session.model_request_token().expect("model turn is live");
    session.model_failed_for(&first_request, "response deadline exceeded")?;
    assert!(!first_request.is_active());
    session.retry_model()?;
    let retry = session.model_request_token().expect("retry is live");

    // The old transport might still deliver a completion or an error. Neither
    // can consume the current turn or replace its request context.
    assert_eq!(
        session.accept_model_reply_for(
            &first_request,
            r#"{"action":"run","command":"echo obsolete"}"#,
        ),
        Err(ModelRequestError::StaleRequest),
    );
    assert_eq!(
        session.model_failed_for(&first_request, "late transport error"),
        Err(ModelRequestError::StaleRequest),
    );
    assert!(retry.is_active());
    assert_eq!(session.turns_used(), 0);

    let history = [Message {
        role: Role::User,
        text: session.build_user_prompt_with(protocol),
    }];
    let prepared = prepare_agent_request(
        &config,
        AgentRequestSpec::new(&history, protocol).streaming(true),
    )?;
    let body = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"id\":\"call_retry\",\"type\":\"function\",\"function\":{\"name\":\"run\",",
        "\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let mut stream = prepared.response_stream()?;
    for chunk in body.as_bytes().chunks(13) {
        let _display_events = stream.push(chunk);
        // Use progress counters for a UI/telemetry snapshot. They contain no
        // command, prompt, credentials, or provider response text.
        assert!(!stream.progress().is_failed);
    }
    let _final_events = stream.finish();
    let progress = stream.progress();
    assert!(progress.is_complete);
    let response = stream.into_response()?;
    let ModelOutcome::Proposal { id, command, .. } =
        session.accept_agent_response_for(&retry, &response)?
    else {
        panic!("the local stream must produce a reviewable proposal");
    };
    assert_eq!(command, "pwd");
    assert!(!retry.is_active());

    // Serialize only at an allowed checkpoint. The application owns atomic
    // file replacement and permissions; this example keeps bytes in memory.
    let saved = session
        .snapshot()
        .expect("pending review is resumable")
        .to_json()?;
    drop(session);
    let mut restored = AgentSession::restore(AgentSessionSnapshot::from_json(&saved)?)?;
    let pending = restored.pending_proposal().expect("redraw approval card");
    assert_eq!(pending.id, id);
    assert_eq!(pending.command, "pwd");

    // A restored review card still requires a deliberate approval. The exit
    // status/output below are simulated fixture data, not actual execution.
    let approved = restored.approve(pending.id)?;
    restored.observe_execution(
        approved.proposal_id,
        CommandExecutionOutcome::exited(0, "/workspace/jagent\n"),
    )?;
    let followup = restored
        .model_request_token()
        .expect("observation starts a turn");
    let completed = restored.accept_model_reply_for(
        &followup,
        r#"{"action":"done","message":"The directory was reported."}"#,
    )?;
    assert!(matches!(completed, ModelOutcome::Completed(_)));
    assert_eq!(restored.state(), AgentState::Completed);
    assert!(!followup.is_active());

    println!(
        "Recovery fixture passed: stale callbacks rejected; {} bytes streamed; review restored; task completed.",
        progress.raw_response_bytes,
    );
    Ok(())
}
