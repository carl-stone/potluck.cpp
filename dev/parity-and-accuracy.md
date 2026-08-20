# Feature parity and inference accuracy

This document separates two claims:

- **Inference accuracy**: generated tokens match the full-model reference on the
  0.8B fixture.
- **Feature parity**: a runtime capability has a named acceptance check.

`CHAIN PASSED` proves inference accuracy only. It does not prove feature parity.
The reference capability set is documented by [prima.cpp](https://github.com/OpenCPIL/prima.cpp);
this project keeps a modern llama.cpp base and uses a versioned TCP transport.

## Re-verified on 2026-08-20

The checks below ran from this checkout on an Apple M4 with the Qwen3.5 0.8B
fixture. The full command is `bash tests/potluck/run_all.sh`.

| Capability | Definition of done | Check |
|---|---|---|
| Hidden-state static pipeline and tail LM head | Multiple worker outputs match the full-model reference | `test_chain.sh` |
| Contiguous N-worker tessellation | 1, 2, 4, and 6 worker splits pass | `test_chain.sh` entries in `run_all.sh` |
| Piped-ring routing | Every worker owns at least two windows and all four scenarios pass | `test_ring.sh` (including 2x[4,4] three-cycle case) |
| Throughput-aware scheduling | Profiled weights and scheduler produce a passing chain | `test_sched.sh` |
| HiGHS LP placement | LP path honors the configured placement constraints | `test_lp.sh` |
| Worker profiling and removal | Profile results drive the drop-slowest path | `test_remove.sh` |
| Prefetch | Page-cache warm path completes and keeps output identical | `test_prefetch.sh` |
| GPU layer planning | CPU/Metal offload budgets preserve the checked output; CUDA still checks the plan and permits only an evidenced near-tie mismatch | `test_gpu.sh` |
| Sampling | Temperature, top-p, and seed behavior meet the script contract | `test_sampler.sh` |
| Speculative decoding | Speculative output matches the target greedy stream | `test_spec.sh` |
| Dynamic batch sequences | Multiple sequence IDs match isolated requests | `test_batch.sh` |
| Prompt, chat, and streaming output | Chat template, text output, and stream checks pass | `test_chat.sh` |
| HTTP server behavior | Prompt changes output, missing prompt is 400, health/models work, SSE parses, and chat output matches `llama-cli` | `test_server.sh` and `test_vs_llama_cli.sh` |
| Per-window GGUF shards | Generated shards load, a sharded chain matches the unsplit chain, and a wrong assignment names both windows | `potluck-shard` plus `test_shard.sh` |
| Versioned transport | Protocol mismatch reports local and peer versions | protocol tests and worker handshake path |
| Metadata-only coordinator | Head runs without `--parity-check` and loads no model weights | `test_chain.sh` default path and server startup |

The GPU check asserts exact greedy parity on CPU and Metal. CUDA backend
numerics can vary at a near tie; the script accepts that case only when the
log identifies CUDA and the mismatch is present, while plan and nonzero
offload checks remain mandatory.

The ring test explicitly covers these scenarios:

- 2 workers, windows `[6,6]`;
- 3 workers, windows `[4,4,4]`;
- 2 workers, uneven windows `[3,9]`;
- 2 workers, windows `[4,4]` over three cycles.

## Inherited or unverified

These items are not claimed as newly verified by the current M4 run:

- Quantization breadth outside the checked Q4_0 fixture. Modern llama.cpp owns
  the quantization implementations, but this checkout has no Q8_0 stage fixture
  in the ignored `models/` directory. Re-run the chain with a Q8_0 or other
  non-Q4_0 fixture before making a window-path claim for that format.
- Cross-machine throughput and the SSH launch path. The code and deployment
  commands are present; they need two reachable hosts built from the same
  commit.
- 27B end-to-end inference. The 18.97 GB model is not safe to load on this
  16 GiB unified-memory host. No 27B completion or throughput number is claimed.
- PC GPU model and operating system. Earlier notes conflict, so this document
  makes no hardware claim for that machine until it is inspected directly.

## Audit corrections

- The ring acceptance script includes the previously missing 2-worker
  `[4,4]` three-cycle scenario. It is a checked case, not an inherited claim.
- This repository makes no claim about the PC GPU model or operating system.
  Earlier records conflict and were removed from the parity matrix.
- Quantization breadth has no passing Q8_0 window test in this checkout. It is
  explicitly inherited and unverified, not feature parity.

## Architecture scope

The distributed window loader currently targets the `qwen35` dense path. The
static route assigns one contiguous window to each worker. Ring mode is a
separate route and loads several disjoint windows per worker. Raw TCP is
intentional; ZeroMQ transport compatibility is not a project requirement.

The full-model reference remains an explicit correctness harness. Production
runs should omit `--parity-check`; the coordinator then loads GGUF metadata and
vocabulary only. One chain serves one request at a time, and the HTTP server
returns 429 while it is busy.

CI configures `POTLUCK_HIGHS=OFF` to avoid the network-only HiGHS
`FetchContent` dependency. In that build, `test_lp.sh` reports an explicit
skip; the LP feature check is re-verified locally with `POTLUCK_HIGHS=ON`.
