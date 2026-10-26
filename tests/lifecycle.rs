//! Public-API integration coverage. Provider payloads and execution outcomes
//! are local fixtures; these tests perform no network or process I/O.

use jagent::{
    prepare_agent_request, AgentProtocol, AgentRequestSpec, AgentSession, AgentSessionSnapshot,
    AgentState, ChatConfig, CommandExecutionOutcome, Message, ModelOutcome, ModelRequestError,
    Provider, Role,
};
use serde_json::{json, Value};

const COMMAND: &str = "printf '%s' '你好🦀'";

fn sse(frames: &[Value]) -> String {
    frames
        .iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect()
}

fn fixtures(provider: Provider, protocol: AgentProtocol) -> (Vec<u8>, Vec<u8>) {
    let native = protocol == AgentProtocol::NativeTools;
    let arguments = json!({"command": COMMAND});
    let action = json!({"action": "run", "command": COMMAND}).to_string();
    let (complete, streaming) = match provider {
        Provider::Anthropic => {
            let block = if native {
                json!({"type": "tool_use", "id": "call_1", "name": "run", "input": arguments})
            } else {
                json!({"type": "text", "text": action})
            };
            let initial = if native {
                json!({"type": "tool_use", "id": "call_1", "name": "run", "input": {}})
            } else {
                json!({"type": "text", "text": ""})
            };
            let delta = if native {
                json!({"type": "input_json_delta", "partial_json": arguments.to_string()})
            } else {
                json!({"type": "text_delta", "text": action})
            };
            let reason = if native { "tool_use" } else { "end_turn" };
            (
                json!({"content": [block], "stop_reason": reason}),
                sse(&[
                    json!({"type": "content_block_start", "index": 0, "content_block": initial}),
                    json!({"type": "content_block_delta", "index": 0, "delta": delta}),
                    json!({"type": "content_block_stop", "index": 0}),
                    json!({"type": "message_delta", "delta": {"stop_reason": reason}}),
                    json!({"type": "message_stop"}),
                ]),
            )
        }
        Provider::OpenAiCompatible => {
            let message = if native {
                json!({"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "run", "arguments": arguments.to_string()}
                }]})
            } else {
                json!({"role": "assistant", "content": action})
            };
            let delta = if native {
                json!({"tool_calls": [{
                    "index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "run", "arguments": arguments.to_string()}
                }]})
            } else {
                json!({"content": action})
            };
            let reason = if native { "tool_calls" } else { "stop" };
            (
                json!({"choices": [{"message": message, "finish_reason": reason}]}),
                sse(&[
                    json!({"choices": [{"delta": delta, "finish_reason": null}]}),
                    json!({"choices": [{"delta": {}, "finish_reason": reason}]}),
                ]) + "data: [DONE]\n\n",
            )
        }
        Provider::Ollama => {
            let message = if native {
                json!({"role": "assistant", "content": "", "tool_calls": [{
                    "function": {"name": "run", "arguments": arguments}
                }]})
            } else {
                json!({"role": "assistant", "content": action})
            };
            let frame = json!({"message": message, "done": true, "done_reason": "stop"});
            let stream = format!("{frame}\n");
            (frame, stream)
        }
    };
    (complete.to_string().into_bytes(), streaming.into_bytes())
}

#[test]
fn all_provider_modes_preserve_review_recovery_and_request_identity() {
    for provider in [
        Provider::Anthropic,
        Provider::OpenAiCompatible,
        Provider::Ollama,
    ] {
        for protocol in [AgentProtocol::Text, AgentProtocol::NativeTools] {
            let (complete, streaming) = fixtures(provider, protocol);
            // None is complete delivery; every other case is independently
            // streamed, including splits within UTF-8 and JSON escapes.
            for chunk_size in [None, Some(1), Some(13), Some(1024)] {
                let mut session = AgentSession::new(3);
                session.submit_user("print the greeting").unwrap();
                let request = session.model_request_token().unwrap();
                let history = [Message {
                    role: Role::User,
                    text: session.build_user_prompt_with(protocol),
                }];
                let config = ChatConfig {
                    provider,
                    api_key: Some("fixture-key".into()),
                    model: "fixture-model".into(),
                    base_url: "https://example.invalid".into(),
                    max_tokens: 256,
                    temperature: None,
                };
                let prepared = prepare_agent_request(
                    &config,
                    AgentRequestSpec::new(&history, protocol).streaming(chunk_size.is_some()),
                )
                .unwrap();
                let response = if let Some(chunk_size) = chunk_size {
                    let mut stream = prepared.response_stream().unwrap();
                    for chunk in streaming.chunks(chunk_size) {
                        let _ = stream.push(chunk);
                        assert!(!stream.progress().is_failed, "{provider:?}/{protocol:?}");
                    }
                    let _ = stream.finish();
                    let progress = stream.progress();
                    assert!(progress.is_complete);
                    assert!(!progress.is_failed);
                    assert_eq!(progress.raw_response_bytes, streaming.len());
                    assert_eq!(
                        progress.tool_calls,
                        usize::from(protocol == AgentProtocol::NativeTools)
                    );
                    stream.into_response().unwrap()
                } else {
                    prepared.parse_response(&complete).unwrap()
                };
                let ModelOutcome::Proposal { id, command, .. } = session
                    .accept_agent_response_for(&request, &response)
                    .unwrap()
                else {
                    panic!("{provider:?}/{protocol:?} must produce a proposal");
                };
                assert_eq!(command, COMMAND);
                assert!(!request.is_active());
                assert_eq!(
                    session.state(),
                    AgentState::AwaitingApproval { proposal_id: id }
                );
                let saved = session.snapshot().unwrap().to_json().unwrap();
                assert_eq!(
                    session.accept_agent_response_for(&request, &response),
                    Err(ModelRequestError::StaleRequest),
                );
                assert_eq!(session.snapshot().unwrap().to_json().unwrap(), saved);

                // Restoring a review never executes the command. Its exact
                // text/id must still be displayed and explicitly approved.
                let mut restored =
                    AgentSession::restore(AgentSessionSnapshot::from_json(&saved).unwrap())
                        .unwrap();
                let pending = restored.pending_proposal().unwrap();
                assert_eq!((pending.id, pending.command), (id, COMMAND));
                let approved = restored.approve(id).unwrap();
                restored
                    .observe_execution(
                        approved.proposal_id,
                        CommandExecutionOutcome::exited(0, "你好🦀\n"),
                    )
                    .unwrap();
                let followup = restored.model_request_token().unwrap();
                assert!(followup.is_active());
                assert_eq!(
                    restored.accept_agent_response_for(&request, &response),
                    Err(ModelRequestError::StaleRequest),
                );
                restored
                    .accept_model_reply_for(
                        &followup,
                        r#"{"action":"done","message":"Greeting displayed."}"#,
                    )
                    .unwrap();
                assert_eq!(restored.state(), AgentState::Completed);
                assert!(!followup.is_active());
            }
        }
    }
}
