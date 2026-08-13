use jagent::{
    prepare_agent_request, AgentProtocol, AgentRequestSpec, AgentSession, ChatConfig, Message,
    ModelOutcome, Provider, Role, StreamEvent,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protocol = AgentProtocol::NativeTools;
    let mut session = AgentSession::new(4);
    session.submit_user("show the current directory")?;

    let history = [Message {
        role: Role::User,
        text: session.build_user_prompt_with(protocol),
    }];
    let config = ChatConfig {
        provider: Provider::OpenAiCompatible,
        api_key: None,
        model: "local-agent".into(),
        base_url: "http://127.0.0.1:1234".into(),
        max_tokens: 256,
        temperature: Some(0.0),
    };
    let prepared = prepare_agent_request(
        &config,
        AgentRequestSpec::new(&history, protocol).streaming(true),
    )?;

    assert!(prepared.is_streaming());
    assert_eq!(prepared.protocol(), AgentProtocol::NativeTools);
    assert!(prepared.request.body.contains("\"stream\":true"));
    assert!(prepared.request.body.contains("\"name\":\"run\""));

    // A real transport feeds response-body chunks as it receives them. This
    // fixture is a complete OpenAI-compatible SSE response, split at awkward
    // byte boundaries to demonstrate that AgentStream owns reassembly.
    let response_bytes = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"run\",",
        "\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],",
        "\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
    )
    .as_bytes();

    let mut stream = prepared.response_stream()?;
    let mut events = Vec::new();
    for chunk in response_bytes.chunks(17) {
        events.extend(stream.push(chunk));
    }
    events.extend(stream.finish());

    assert!(events
        .iter()
        .any(|event| matches!(event, StreamEvent::ToolCall(_))));
    assert!(events.contains(&StreamEvent::Done));

    // A low-level ToolCall event is display data, not authorization. Only the
    // completed response enters the session and becomes a reviewable proposal.
    let response = stream.into_response()?;
    let ModelOutcome::Proposal { id, command, .. } = session.accept_agent_response(&response)?
    else {
        panic!("the local stream fixture must produce a command proposal");
    };
    assert_eq!(command, "pwd");

    // Approval still returns a value; this example intentionally executes
    // nothing and therefore does not fabricate an observation.
    let approved = session.approve(id)?;
    assert_eq!(approved.command, "pwd");

    Ok(())
}
