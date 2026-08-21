# ADR 0003: The parity reference is opt-in

- Date: 2026-08-20
- Status: Superseded by ADR 0006

## Context

The coordinator compared chain output against an in-process full-model
reference by default. That required loading the whole model in the head
process, which is fatal for a 27B model on a 16 GiB head. The comparison also
forced `n_ubatch = 1` so that batched numerics matched the isolated
single-token path, sacrificing prefill throughput.

## Decision

ADR 0006 strengthens this boundary. A full-model reference is a test-only
fixture mechanism and must not be a Potluck product capability or product
binary option. Product heads and workers load only metadata and assigned window
shards.

## Consequences

- The head needs only the GGUF metadata (vocab + hparams + chat template), so
  head memory is independent of model size.
- The batched prefill path is free to use large ubatches without violating a
  reference comparison.
- Accuracy gates on small fixtures may use a separate full-model test process;
  deployed Potluck binaries never load that reference.
