# Component verification and inference accuracy

This document records component and integrated-run evidence. It does not define
the product architecture or release gate. The binding decision is
[ADR 0006](decisions/0006-piped-ring-server-product.md), amended by
[ADR 0007](decisions/0007-prima-direct-ring-zeromq.md): Potluck is a
resource-aware, direct-peer ZeroMQ piped-ring-only OpenAI-compatible server
with automatic profiling and selection, HiGHS-solved heterogeneous window
placement (HALDA, [ADR 0010](decisions/0010-prima-feature-parity-baseline.md)),
per-window prefetch, per-device accelerator placement, complete-model
window-bounded loading, continuous batching, conversation slots, speculative
decoding, quantized-model support, and a potluck completion CLI beside the
server.

Any remaining static or alternate architecture check is removal work, not a
product path. Passing a component check does not prove every large
platform-specific release scenario.

This document separates two component claims:

- **Inference accuracy**: generated tokens match the full-model reference on
  small test fixtures.
- **Component feature evidence**: a current mechanism has a named check.

The direct adjacent-peer ZeroMQ data path is now implemented inside
`potluck-server`. It is functional evidence for the current server path, not
proof of the complete product contract.

## Verification scope

Fixture accuracy is exercised with small model files that fit the test host.
The Qwen3.5 0.8B fixture provides exact token and component checks. A
supervised Gemma 3 27B run provides integrated multi-device and performance
evidence; it does not claim full reference-token parity. These results do not
change the ADR 0006 product architecture or release gate.

## Direct-ring smoke evidence on 2026-08-21

The Qwen3.5 0.8B fixture passed a local two-worker CPU smoke through
`potluck-server`. Each worker owned repeated disjoint windows and sent hidden
state directly to its cyclic next peer over ZeroMQ.

An M4 head also bootstrapped one Linux CPU worker over SSH. The server logged
windows `[0,12)` and `[12,24)`, and a non-stream completion with `n_predict: 2`
returned ` located in`. The M4 was controller-only in this smoke; it did not
execute a window and the result does not demonstrate heterogeneous placement.

These observations establish the direct server transport, repeated-window
route, local worker launch, and SSH bootstrap boundaries only. They do not
establish automatic discovery, live profiling or selection, resource-aware
placement, full-model distribution, continuous batching, conversation slots,
resilience, security, or full API parity.

| Component | Checked behavior | Check and product boundary |
|---|---|---|
| Direct server ring | Adjacent workers exchange hidden state directly over ZeroMQ; ingress enters rank 0 and final results return to the head | Local `potluck-server` smoke and integrated suite |
| Repeated disjoint windows | HALDA assigns repeated windows from live profile data and the ring traverses them in cycle order | `test-potluck-halda` plus integrated server route |
| SSH and DNS-SD bootstrap | The head discovers candidates, profiles them, and forms the cyclic ring through scoped SSH bootstrap | Discovery smoke and integrated server route |
| Window-bounded full-model loading | Each worker loads assigned global layer bounds from one complete GGUF | Integrated route and `test-potluck-qwen35-stages` |
| Sampling and speculative decoding | Temperature-zero output stays stable with draft state; accepted counts and trim state cross the ring | Speculative server smoke and integrated suite |
| Prompt, chat, and streaming component | Template, text, error, health, model, and stream behavior pass the component checks | `test_server.sh`; the OpenAI-compatible surface is still a subset |
| Protocol and transport components | Direct ZeroMQ messages and worker protocol behavior pass their named checks | `run_all.sh`; component evidence only |
| Full-model reference | Small-fixture output can be compared with a monolithic reference | Test-only accuracy evidence; product workers must not load a full model as a reference |

The available backend checks remain subject to numerical differences. Exact
greedy parity on CPU and Metal is useful fixture evidence; CUDA may choose a
near-tie token. No such component result establishes automatic placement or
the complete server contract.

## Platform and fixture limits

The Qwen3.5 0.8B fixture fits the test hosts and is the named small-fixture
target. The supervised Gemma 3 27B runs used the model file
`/Users/carlstone/models/gemma-3-27b-it-Q4_K_M.gguf` on an M4 head, an M1,
and a Linux CUDA device. The assignment log and measured streaming results
are recorded in `docs/BENCHMARKS.md`.

The fixture establishes exact small-model behavior and nonzero speculative
acceptance. The 27B runs establish integrated three-device operation,
full-model window loading, per-window prefetch, and measured performance.
Neither result claims the full llama-server API surface or a general model
compatibility matrix.

## Inherited or unverified

These items remain outside the current named evidence:

- Quantization breadth outside the checked fixtures. Quantized GGUF support is
  implemented, but no complete distributed quantization matrix is claimed.
- Cross-machine token-state migration and safe retry after a worker change.
  The current recovery path is bounded and retryable, but it does not migrate
  an active sequence.
- Broader llama-server HTTP parity, model management, embeddings, tools, and
  multimodal behavior.
- Longer platform-specific runs beyond the supervised 27B operating point.

## Architecture status

The current verification covers the `qwen35` dense fixture and the Gemma 3
27B operating point. The server owns a direct adjacent-peer ZeroMQ ring with
HALDA-solved repeated windows, local and SSH worker launch, DNS-SD discovery,
per-window prefetch, per-device CPU, Metal, and CUDA placement, speculative
decoding, continuous batching, and conversation slots.

The remaining scope is broader API and model coverage, longer platform runs,
and token-state migration after a worker failure. The full-model reference
remains test-only and explicit. Product binaries must not load the full model
as a correctness reference.

A build without the HiGHS-backed HALDA placement required by
[ADR 0010](decisions/0010-prima-feature-parity-baseline.md) does not establish
automatic placement.
