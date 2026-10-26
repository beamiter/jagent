//! Local request-preparation microbenchmark. Run with `cargo run --release --example bench_requests`.
//! Measures preparation only; performs no HTTP or process execution.

use jagent::{
    prepare_agent_request, AgentProtocol, AgentRequestSpec, ChatConfig, Message, Provider, Role,
};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let mut config = ChatConfig::new(Provider::OpenAiCompatible);
    config.base_url = "http://127.0.0.1:1234".into();
    for (name, text, redact) in [
        (
            "40x5k plain",
            "compile log and diagnostics\n".repeat(190),
            false,
        ),
        (
            "40x5k default redaction",
            "compile log and diagnostics\n".repeat(190),
            true,
        ),
        ("single escape-heavy", "\0\"\\\n".repeat(49152), false),
    ] {
        let count = if name.starts_with("single") { 1 } else { 40 };
        let history: Vec<Message> = (0..count)
            .map(|i| Message {
                role: if i % 2 == 0 {
                    Role::User
                } else {
                    Role::Assistant
                },
                text: text.clone(),
            })
            .collect();
        for _ in 0..8 {
            let _ = black_box(
                prepare_agent_request(
                    &config,
                    AgentRequestSpec::new(&history, AgentProtocol::Text).redact_secrets(redact),
                )
                .unwrap(),
            );
        }
        let mut times = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..120 {
                let _ = black_box(
                    prepare_agent_request(
                        black_box(&config),
                        AgentRequestSpec::new(black_box(&history), AgentProtocol::Text)
                            .redact_secrets(redact),
                    )
                    .unwrap(),
                );
            }
            times.push(start.elapsed().as_nanos() / 120);
        }
        times.sort();
        println!("{name}: median {} ns/request, rounds {:?}", times[3], times);
    }
}
