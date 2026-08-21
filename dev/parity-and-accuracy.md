# Component verification and inference accuracy

This document records evidence from the current unfinished implementation. It
does not define the product architecture or a release gate. The binding
decision is [ADR 0006](decisions/0006-piped-ring-server-product.md), amended by
[ADR 0007](decisions/0007-prima-direct-ring-zeromq.md): Potluck is a
resource-aware, direct-peer ZeroMQ piped-ring-only OpenAI-compatible server with
automatic profiling and selection, heterogeneous window placement, per-window
prefetch, per-device accelerator placement, shard-only loading, continuous
batching, and conversation slots.

Any remaining static or alternate architecture check is removal work, not a
product path. Passing a batch-protocol, shard, or HTTP component check does not
make Potluck a finished product.

This document separates two component claims:

- **Inference accuracy**: generated tokens match the full-model reference on
  small test fixtures.
- **Component feature evidence**: a current mechanism has a named check.

The direct adjacent-peer ZeroMQ data path is now implemented inside
`potluck-server`. It is functional evidence for the current server path, not
proof of the complete product contract.

## Verification scope

Fixture accuracy is exercised with small model files that fit the test host.
The current fixture is Qwen3.5 0.8B; other small models may exercise a
component primitive. Verifying 27B correctness, performance, or end-to-end
execution remains outside the fixture test scope. It does not change the
ADR 0006 product architecture or release gate.

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
placement, shard automation, continuous batching, conversation slots,
resilience, security, or full API parity.

| Component | Checked behavior | Check and product boundary |
|---|---|---|
| Direct server ring | Adjacent workers exchange hidden state directly over ZeroMQ; ingress enters rank 0 and final results return to the head | Local `potluck-server` smoke; direct product path, but not the complete release gate |
| Repeated disjoint windows | The current route assigns two windows per worker where model layers permit | Server startup and completion smoke; fixed route, not live heterogeneous scheduling |
| SSH bootstrap | The head launches an explicitly named Linux worker and forms the cyclic ring | M4-head to Linux-worker 0.8B smoke; functional launch evidence, not automatic discovery |
| Mixed recurrent/attention window primitive | A worker window with no attention layers preserves fixture accuracy | Retained component check; not product completion |
| Per-window GGUF shard primitive | Generated shards load and reject wrong assignments | `potluck-shard` component check; automatic creation and deployment remain missing |
| Sampling primitive | Temperature, top-p, and seed meet the component contract | Retained component check; full request sampling controls remain missing |
| Prompt, chat, and streaming component | Template, text, error, health, model, and stream behavior pass the component checks | `test_server.sh`; the OpenAI-compatible surface is still only a subset |
| Protocol and transport components | Direct ZeroMQ messages and worker protocol behavior pass their named checks | `run_all.sh`; component evidence only |
| Full-model reference | Small-fixture output can be compared with a monolithic reference | Test-only accuracy evidence; product workers must not load a full model as a reference |

The available backend checks remain subject to numerical differences. Exact
greedy parity on CPU and Metal is useful fixture evidence; CUDA may choose a
near-tie token. No such component result establishes automatic placement or
the complete server contract.

## Platform and fixture limits

The Qwen3.5 0.8B fixture fits the test hosts and is the current named smoke
target. The M4-to-Linux check proves functional SSH bootstrap and direct
ring communication only. It does not measure cross-machine throughput,
automatic discovery, live selection, heterogeneous placement, or head work.

The fixture does not establish quantization breadth outside Q4_0, correctness
or performance for 27B, or end-to-end product completion. No claim is made
about PC GPU performance or other uninspected hardware.

## Inherited or unverified

These items remain unverified by the current smoke evidence:

- Quantization breadth outside the checked Q4_0 fixture. Modern llama.cpp owns
  the quantization implementations, but no other distributed stage fixture is
  named here.
- Cross-machine throughput and resilience. SSH bootstrap and one functional
  completion passed, but reconnect, topology rebuild, retry, and recovery are
  not implemented.
- Automatic discovery, live profiling, device selection, head resource
  reservation, and per-device accelerator placement.
- Shard creation, transfer, validation, selection, and caching automation.
- Continuous HTTP batching, conversation slots, and full API parity.

## Architecture status

The current distributed window loader targets the `qwen35` dense path. The
server now owns a direct adjacent-peer ZeroMQ ring with two repeated disjoint
windows per worker where layers permit. Local worker launch and explicit SSH
bootstrap are implemented.

The remaining gaps are automatic discovery, live profiling and selection,
resource-aware placement, per-window prefetch, independent per-device
accelerator placement, shard automation, continuous batching, conversation
slots, resilience, security, and full OpenAI-compatible API parity. The direct
ring smoke is evidence for one server path, not product completion.

The full-model reference remains test-only and explicit. Product binaries must
not load the full model as a correctness reference.

A build without optional placement-solver support does not establish automatic
placement; the release gate still requires the complete server lifecycle.
