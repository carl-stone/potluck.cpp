# ADR 0004: Context and sequence sizing come from need

- Date: 2026-08-20
- Status: Accepted

## Context

`potluck_runtime.h` hardcoded `n_seq_max = 64` and `n_outputs_max = 64`. On
Qwen3.8-27B the DeltaNet recurrent state is ~150 MiB per linear layer per
sequence: 48 linear layers x 64 sequences is ~9.6 GiB of state nothing uses,
plus 63.5 MB of reserved logits.

## Decision

The head decides `n_ctx`, `n_seq_max`, and `n_ubatch` once for the cluster and
carries them in the protocol `node_config`. Defaults: `n_ctx` 4096,
`n_seq_max` 1, `n_ubatch` 512. `--ctx N` sets the context on head and server;
`--batch N` sets `n_seq_max` (the multi-sequence path).

## Consequences

- Default 27B recurrent-state reservation drops from ~9.6 GiB to ~150 MiB per
  worker.
- Batched prefill sends the whole prompt in one message with one graph launch
  per token per stage, not one round trip per token.
- The protocol version bump to 5 absorbs the config change.
