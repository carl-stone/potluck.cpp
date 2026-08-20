# Benchmarks

All numbers below were measured on 2026-08-20.

## Hardware and build

- Host: Apple Mac16,1, Apple M4, 10 CPU cores
- Memory: 16 GiB unified memory
- OS: Darwin 25.5.0, arm64
- Build: Release, local HiGHS enabled
- Fixture: `models/Qwen3.5-0.8B-Q4_0.gguf` (526.50 MiB)
- Workers: two local `potluck-worker` processes, CPU-only (`ngl=0`)

Build command:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DPOTLUCK_HIGHS=ON
cmake --build build -j8 --target potluck-head potluck-server potluck-worker llama-cli
```

## `potluck-server --bench`

Command:

```sh
build/bin/potluck-server \
  -m models/Qwen3.5-0.8B-Q4_0.gguf \
  --workers 2 --port 18081 --n-predict 8 --bench
```

Measured output:

```text
bench worker host window       weight-bytes gpu-layers decode-tok/s peak-rss-mb
bench      0 127.0.0.1       [0,12)    281518032          0       286.31       847.0
bench      1 127.0.0.1       [12,24)    281518032          0       152.52      1109.5
bench cluster prefill-tok/s 129.01 decode-tok/s 81.11 aggregate-tok/s 58.23 ms/token 17.17 wire-bytes/token 75.0 coordinator-peak-rss-mb 126.6 worker-peak-rss-mb-max 1109.5
```

The benchmark request uses the fixed prompt `The capital of France is`, an eight-token decode, and greedy sampling. `decode-tok/s` in the worker table is measured inside each stage. `peak-rss-mb` is `getrusage(RUSAGE_SELF).ru_maxrss`; macOS values are converted from bytes to MiB.

## `potluck-head --bench`

Workers were started with:

```sh
build/bin/potluck-worker models/Qwen3.5-0.8B-Q4_0.gguf 127.0.0.1 18101
build/bin/potluck-worker models/Qwen3.5-0.8B-Q4_0.gguf 127.0.0.1 18102
printf '127.0.0.1:18101\n127.0.0.1:18102\n' >/tmp/potluck-head-workers.txt
```

Coordinator command:

```sh
build/bin/potluck-head models/Qwen3.5-0.8B-Q4_0.gguf \
  /tmp/potluck-head-workers.txt 8 127.0.0.1 --bench
```

Measured output:

```text
bench worker host window       weight-bytes gpu-layers decode-tok/s peak-rss-mb
bench      0 127.0.0.1       [0,12)    281518032          0       305.17       849.9
bench      1 127.0.0.1       [12,24)    281518032          0       167.79      1112.5
bench cluster prefill-tok/s 170.90 decode-tok/s 80.03 aggregate-tok/s 59.23 ms/token 16.88 wire-bytes/token 30.0 coordinator-peak-rss-mb 126.4 worker-peak-rss-mb-max 1112.5
```

The head benchmark proves the coordinator can request per-stage metrics without tearing down the live chain. A later HTTP request after `--bench` metrics returned successfully in the same smoke run.

## Interpretation and limits

The worker weight column is a resident-weight estimate for an unsplit GGUF:
model file bytes multiplied by the stage's layer fraction. Sharded deployments
should use the actual shard file size in deployment records. RSS includes
runtime, KV, graph buffers, and touched mapped pages; it is not a pure weight
measurement.

These are single-machine fixture numbers. Local workers are a test harness:
they do not create memory, and they are not evidence of cluster scaling. On one
machine, all worker shards still consume the same physical RAM and memory
bandwidth. Production capacity scaling requires separate machines, each holding
and computing its own layer shard. Multiple local workers can be useful for
protocol and correctness tests, or with multiple physical accelerators; this
M4 run has one unified-memory accelerator.

No pre-change batched-prefill measurement was captured in this run, so this
file does not claim a speedup ratio. The post-change path sends one batched
prefill message per prompt.

## Verification scope

These benchmarks and the acceptance suite intentionally use small fixture
models that fit the test host. The current correctness fixture is Qwen3.5
0.8B; other small models may be used when they exercise a supported path.
Verifying 27B correctness, performance, or end-to-end execution is an explicit
non-goal. 27B is a deployment target, not an acceptance target.
