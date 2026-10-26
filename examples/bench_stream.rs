//! Local SSE parsing microbenchmark. Run with `cargo run --release --example bench_stream`.
//! Measures parsing only; performs no HTTP or process execution.

use jagent::{Provider, StreamEvent, StreamParser};
use std::hint::black_box;
use std::time::Instant;
fn main() {
    let frame = format!("data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n", "x".repeat(64));
    let body = frame.repeat(2000) + "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    for chunk_size in [1, 64, 4096, body.len()] {
        let start = Instant::now();
        let mut total = 0;
        for _ in 0..50 {
            let mut parser = StreamParser::new(Provider::OpenAiCompatible);
            for chunk in body.as_bytes().chunks(chunk_size) {
                for event in parser.push(black_box(chunk)) {
                    if let StreamEvent::TextDelta(text) = event {
                        total += text.len();
                    }
                }
            }
            black_box(parser.finish());
        }
        assert_eq!(total, 50 * 2000 * 64);
        println!(
            "chunk={chunk_size} bytes={} iterations=50 elapsed_ms={:.3}",
            body.len(),
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
