# Benchmarks

> Product measurements must run the integrated resource-aware direct-peer ZeroMQ piped-ring server with automatic placement, per-window prefetch, per-device offload, continuous batching, and slots.

## Integrated 27B product acceptance

The following measurements came from a supervised 2026-08-23 run of the integrated direct-peer ZeroMQ PRP server. This is an operating point, not a speedup claim.

- Model: Gemma 3 27B Q4_K_M, 62 layers
- Devices: Apple M4 Mac, Apple M1 Mac, and Linux PC with NVIDIA GTX 1650 SUPER
- Build: Release
- Discovery and placement: automatic DNS-SD discovery, live probing, head participation, six repeated windows, and automatic window-shard distribution
- Request: `The capital of France is`, greedy sampling, 32 streamed completion tokens

The route used M4 windows `[0,12)` and `[31,43)` with 12 Metal layers each, M1 windows `[12,22)` and `[43,53)` with 10 Metal layers each, and PC windows `[22,31)` and `[53,62)` with 6 CUDA layers followed by a CPU tail.

Measured client-side results:

```text
TTFT                 38.943 s
32-token total       57.159 s
decode throughput     1.702 token/s
```

Two simultaneous 16-token conversations, one streaming and one non-streaming, both returned HTTP 200 in 46.874 s and 47.366 s. Killing the M1 worker during another request returned HTTP 503, the controller restored a ready three-device ring, and the next 16-token request returned HTTP 200 in 47.100 s.

These values include the current PRP route, network transfers of intermediate activations, per-window synchronization, and client-visible HTTP streaming. They do not isolate prefill, compare against another topology, or establish a regression threshold.

## Historical component measurements

The numbers below measure static, manually configured, coordinator-routed raw-TCP component paths that ADR 0006 and ADR 0007 required Potluck to remove. They remain only as dated evidence and must not guide product configuration or performance claims.

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

Exact token-parity checks use the Qwen3.5 0.8B fixture so a full-model reference fits one test host. The supervised Gemma 3 27B run establishes integrated three-device operation and a measured performance point; it does not claim reference-token parity or a speedup ratio.
