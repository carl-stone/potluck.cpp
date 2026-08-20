# ADR 0003: The parity reference is opt-in

- Date: 2026-08-20
- Status: Accepted

## Context

The coordinator compared chain output against an in-process full-model
reference by default. That required loading the whole model in the head
process, which is fatal for a 27B model on a 16 GiB head. The comparison also
forced `n_ubatch = 1` so that batched numerics matched the isolated
single-token path, sacrificing prefill throughput.

## Decision

Invert the flag: the in-process reference runs only under `--parity-check`
(default off). Tests that verify token equality against the full model pass
the flag explicitly. The throughput path never loads the whole model.

## Consequences

- The head needs only the GGUF metadata (vocab + hparams + chat template), so
  head memory is independent of model size.
- The batched prefill path is free to use large ubatches without violating a
  reference comparison.
- Accuracy gates on the 0.8B fixture use `--parity-check`; a stranger's
  deployment never pays for it.
