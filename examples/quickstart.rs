use jagent::{
    agent_user_prompt, prepare_agent_request, AgentProtocol, AgentRequestSpec, AgentSession,
    ChatConfig, EnvironmentMeta, Message, ModelOutcome, Provider, Role,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protocol = AgentProtocol::Text;
    let mut session = AgentSession::new(8);

    // This deliberately resembles a leaked credential: the prepared request
    // must redact it even though the session retains the user's original text.
    const SECRET: &str = "ghp_1234567890abcdefghijABCDEFGHIJ123456";
    session.submit_user(format!(
        "show the current directory; an accidental token was pasted: {SECRET}"
    ))?;

    let environment = EnvironmentMeta {
        cwd: "/workspace/jagent".into(),
        shell: "bash".into(),
        os: "linux".into(),
        git: None,
    };
    let history = [Message {
        role: Role::User,
        text: agent_user_prompt(
            &session.build_user_prompt_with(protocol),
            &environment,
            None,
        ),
    }];

    // This loopback configuration needs no API key. jagent remains sans-IO,
    // so constructing the request cannot contact the endpoint.
    let config = ChatConfig {
        provider: Provider::OpenAiCompatible,
        api_key: None,
        model: "local-agent".into(),
        base_url: "http://127.0.0.1:1234".into(),
        max_tokens: 512,
        temperature: Some(0.0),
    };
    let prepared = prepare_agent_request(&config, AgentRequestSpec::new(&history, protocol))?;

    assert_eq!(
        prepared.request.url,
        "http://127.0.0.1:1234/chat/completions"
    );
    assert_eq!(
        prepared.report.request_body_bytes,
        prepared.request.body.len()
    );
    assert!(prepared.report.redaction_enabled);
    assert_eq!(prepared.report.history.input_history_turns, 1);
    assert_eq!(prepared.report.history.sent_history_turns, 1);
    assert_eq!(prepared.report.history.changed_history_turns, 1);
    assert_eq!(prepared.report.history.elided_history_turns, 0);
    assert_eq!(prepared.report.history.omitted_history_turns, 0);
    assert!(!prepared.request.body.contains(SECRET));
    assert!(prepared.request.body.contains("[REDACTED:github-token]"));

    // A real integration performs `prepared.request` with its own transport.
    // The local fixture below demonstrates bounded response decoding and the
    // explicit review flow without network or process I/O.
    let response_bytes = br#"{
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "{\"action\":\"run\",\"command\":\"pwd\"}"
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8}
    }"#;
    let response = prepared.parse_response(response_bytes)?;
    assert_eq!(
        response.usage().and_then(|usage| usage.input_tokens),
        Some(20)
    );

    let outcome = session.accept_agent_response(&response)?;
    let ModelOutcome::Proposal {
        id,
        command,
        danger,
    } = outcome
    else {
        panic!("the local response fixture must produce a command proposal");
    };
    assert_eq!(command, "pwd");
    assert!(danger.is_none());

    // Approval returns a value but never executes it.
    let approved = session.approve(id)?;
    assert_eq!(approved.command, "pwd");

    // Simulate the integration executing that exact approved value and
    // recording the actual exit status/output. No command is run here.
    session.observe(approved.proposal_id, 0, "/workspace/jagent\n")?;

    Ok(())
}
