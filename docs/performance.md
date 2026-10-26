# Request and streaming performance

These measurements cover local work inside the sans-IO core. They do not
measure provider generation speed, HTTP latency, terminal rendering, or total
task completion time. Both implementations preserve the same byte budgets,
protocol validation, and explicit command-review boundary.

## 2026-10-26 comparison

Baseline: `d5492320686879c8b92106d2e31e4293ff125a5a`.
Updated: the request-history and streaming changes in this revision.
Both variants used Rust 1.99.1 release builds on the same shared Linux host.
Compilation time is excluded. Host contention was visible, so treat the
numbers as a reproducible workload comparison, not a performance SLA.

### Request preparation

The updated path borrows the already prepared history instead of cloning and
bounding it a second time, and reuses encoded-size accounting. It independently
validates the prepared window before provider encoding and retains the final
transport-envelope validation. Invalid configuration is rejected before the
redaction scan.

| Input | Baseline per request | Updated per request | Time reduction |
|---|---:|---:|---:|
| 40 turns × 5,320 bytes, redaction disabled | 2.390 ms | 0.848 ms | 64.5% |
| Same history, default redaction enabled | 3.783 ms | 2.112 ms | 44.2% |
| One 192 KiB escape-heavy turn, redaction disabled | 4.048 ms | 3.096 ms | 23.5% |

The comparison ran baseline/updated/updated/baseline processes. Each process
used eight warmups and seven batches of 120 preparations per workload. The
table uses the median of the fourteen per-batch average timings for each
variant. The default-redaction row retains the actual production default;
the other rows isolate preparation costs with redaction disabled explicitly.

The public builders and optimized path are also checked for identical request
bytes across 120 combinations of provider, protocol, delivery, system-prompt
choice, and history shape. Speed alone is not a correctness gate.

- Harness: [`examples/bench_requests.rs`](../examples/bench_requests.rs)
- Raw data: [`2026-10-26-request-preparation.txt`](benchmarks/2026-10-26-request-preparation.txt)

### Streaming parsing

The workload is a 282,081-byte OpenAI-compatible SSE response with 2,000 text
deltas. Each row measures 50 complete parses. Complete lines can now be
borrowed from incoming chunks, fragmented-line/SSE buffers retain capacity,
and tool fragments avoid redundant JSON cloning. Small chunks retain a
lightweight byte path so scanning overhead does not erase the improvement.

| Transport chunk | Baseline, 50 parses | Updated, 50 parses | Time reduction |
|---|---:|---:|---:|
| 1 byte | 410.210 ms | 343.293 ms | 16.3% |
| 64 bytes | 235.691 ms | 203.611 ms | 13.6% |
| 4,096 bytes | 244.112 ms | 187.818 ms | 23.1% |
| Whole response | 232.844 ms | 187.042 ms | 19.7% |

These are individual measured runs per chunk size, not confidence intervals.
The recorded streaming comparison used a separate harness lockfile with
`serde_json 1.0.154` and `regex 1.13.1`, held constant for both variants. The
repository lockfile uses `serde_json 1.0.151`; the checked-in example reproduces
the workload, while exact timing can differ with that dependency graph. The
[recorded harness lockfile](benchmarks/2026-10-26-stream-Cargo.lock) preserves
the dependency versions used for the table.
Tests separately check all two-chunk split points for representative text,
tool, UTF-8, BOM, CRLF, EOF, and failure fixtures, together with exact budget
boundaries. Progress counters are diagnostics and do not change acceptance.

- Harness: [`examples/bench_stream.rs`](../examples/bench_stream.rs)
- Raw data: [baseline](benchmarks/2026-10-26-stream-before.txt) and
  [updated](benchmarks/2026-10-26-stream-after.txt)

## Reproduce locally

Run from the checkout you want to measure:

```text
cargo run --locked --release --example bench_requests
cargo run --locked --release --example bench_stream
```

The fixtures use an invalid/local endpoint only to build request values; no
request is sent and no command is executed. For a baseline comparison, create
a separate checkout at the baseline SHA, copy these two example files into
its `examples/` directory, and run the same commands. Keep compiler, build
profile, machine, and workload fixed; alternate baseline/updated runs and
compare repeated samples. Do not benchmark concurrent builds.
