# Component verification and inference accuracy

This document records evidence from the current unfinished implementation. It
does not define the product architecture or a release gate. The binding
decision is [ADR 0006](decisions/0006-piped-ring-server-product.md), amended by
[ADR 0007](decisions/0007-prima-direct-ring-zeromq.md): Potluck is a
resource-aware, direct-peer ZeroMQ piped-ring-only OpenAI-compatible server with
automatic profiling and selection, heterogeneous window placement, per-window
prefetch, per-device accelerator placement, shard-only loading, continuous
batching, and conversation slots.

Static execution and other alternative architecture paths must be removed from
the source and tests. Passing a static, standalone ring, batch-protocol, shard,
or HTTP component check does not make Potluck a finished product.

This document separates two component claims:

- **Inference accuracy**: generated tokens match the full-model reference on
  small test fixtures.
- **Component feature evidence**: a current mechanism has a named check.

`CHAIN PASSED` proves fixture inference accuracy only. It does not prove the
product contract. The reference capability set is documented by
[prima.cpp](https://github.com/OpenCPIL/prima.cpp). Potluck keeps a modern
llama.cpp base. The present versioned raw-TCP transport is conflicting legacy
implementation; ADR 0007 requires prima.cpp's direct-ring ZeroMQ model.

## Verification scope

Fixture accuracy is exercised with small model files that fit the test host.
The current fixture is Qwen3.5 0.8B; other small models may exercise a
component primitive. Verifying 27B correctness, performance, or end-to-end
execution remains outside the fixture test scope. It does not change the
ADR 0006 product architecture or release gate.

## Re-verified on 2026-08-21

The component checks below ran from this checkout on an Apple M4 with the
Qwen3.5 0.8B fixture. The command is `bash tests/potluck/run_all.sh`. The same
source and fixture also passed the 18-check component suite on the Arch PC
(`x86_64`) with `POTLUCK_HIGHS=OFF`; the PC run skips CLI comparison because
CUDA/x86 numerical variance is covered by the fixture policy.


| Component | Checked behavior | Check and product boundary |
|---|---|---|
| Legacy static hidden-state path | Multiple worker outputs match the full-model reference | `test_chain.sh`; removal target, not product execution |
| Legacy static worker tessellation | 1, 2, 4, and 6 worker splits pass | `test_chain.sh`; must be rewritten or deleted during cutover |
| Mixed recurrent/attention window primitive | A worker window with no attention layers preserves fixture accuracy | `test_no_attn.sh`; component evidence only |
| Standalone piped-ring routing | Every worker owns at least two windows and all four scenarios pass | `test_ring.sh`; must move into the server |
| Throughput measurement primitive | Profiled weights can drive a passing legacy schedule | `test_sched.sh`; manual application is not product behavior |
| HiGHS placement primitive | LP path honors configured constraints | `test_lp.sh`; automatic integration remains missing |
| Schedule-time worker removal primitive | Profile results drive one drop-slowest operation | `test_remove.sh`; live selection remains missing |
| Whole-file prefetch primitive | Page-cache warm path completes and keeps output identical | `test_prefetch.sh`; conflicts with required per-window prefetch |
| Static GPU planning primitive | CPU/Metal budgets preserve checked output; CUDA permits only an evidenced near-tie mismatch | `test_gpu.sh`; per-device/window placement remains missing |
| Sampling primitive | Temperature, top-p, and seed meet the script contract | `test_sampler.sh` |
| Standalone speculative decoding | Output matches the target greedy stream | `test_spec.sh`; not integrated with the product server |
| Batch sequence protocol | Multiple sequence IDs match isolated requests | `test_batch.sh`; not continuous HTTP batching or slots |
| Prompt, chat, and streaming component | Template, text, and stream checks pass | `test_chat.sh`; not the complete OpenAI contract |
| Legacy static HTTP server | Prompt/error/health/models/SSE checks and CLI comparison pass | `test_server.sh`, `test_vs_llama_cli.sh`; server architecture must be replaced |
| Per-window GGUF shard primitive | Generated shards load and reject wrong assignments | `potluck-shard`, `test_shard.sh`; automatic ring integration remains missing |
| Legacy versioned raw-TCP transport | Protocol mismatch reports local and peer versions | protocol tests and worker handshake path; removal target, not product communication |
| Metadata-only coordinator primitive | Head can omit the full fixture reference | `test_chain.sh`; production must also use shard-only worker loading |

The GPU check asserts exact greedy parity on CPU and Metal. CUDA backend
numerics can vary at a near tie; the script accepts that case only when the
log identifies CUDA and the mismatch is present, while plan and nonzero
offload checks remain mandatory.

## CI platform limits

The macOS job runs the complete acceptance matrix on its Metal-capable host.
The Ubuntu job still runs the CPU pipeline and all portable checks, but sets
`POTLUCK_SKIP_GPU_TESTS=1` because its hosted runner has no accelerator.
It also sets `POTLUCK_SKIP_CLI_PARITY=1`: x86 CPU kernels can choose a different
near-tie token for a distributed stage split than the monolithic `llama-cli`
reference. The M4 run remains the named check for exact server/CLI text parity;
`test_chain.sh` remains the cross-platform full-model correctness check.

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
- PC GPU model and operating system. Earlier notes conflict, so this document
  makes no hardware claim for that machine until it is inspected directly.

## Audit corrections

- The ring acceptance script includes the previously missing 2-worker
  `[4,4]` three-cycle scenario. It is a checked case, not an inherited claim.
- This repository makes no claim about the PC GPU model or operating system.
  Earlier records conflict and were removed from the parity matrix.
- Quantization breadth has no passing Q8_0 window test in this checkout. It is
  explicitly inherited and unverified, not feature parity.

## Architecture status

The current distributed window loader targets the `qwen35` dense path. The
source still contains a static route, a separate opt-in ring route, manually
applied scheduling, whole-file prefetch, static accelerator budgets, a
standalone batch path, and a one-chain HTTP server that returns 429 while busy.

Every item in that list is an unfinished gap against ADR 0006. The clean
cutover must remove static execution and connect automatic live placement,
per-window prefetch, per-device accelerator placement, shard-only loading,
continuous batching, and conversation slots to the OpenAI-compatible ring
server. Component checks in this document must be rewritten around that
integrated product path.

The full-model reference remains test-only and explicit. Product binaries must
not load the full model as a correctness reference.

CI configures `POTLUCK_HIGHS=OFF` to avoid the network-only HiGHS
`FetchContent` dependency. In that build, `test_lp.sh` reports an explicit
skip; the LP feature check is re-verified locally with `POTLUCK_HIGHS=ON`.
