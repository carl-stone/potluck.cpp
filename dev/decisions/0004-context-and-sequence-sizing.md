# ADR 0004: Context and sequence sizing come from need

- Date: 2026-08-20
- Status: Superseded by ADR 0006
- Context ordering amended by: ADR 0012

## Context

`potluck_runtime.h` hardcoded `n_seq_max = 64` and `n_outputs_max = 64`. On
Qwen3.8-27B the DeltaNet recurrent state is ~150 MiB per linear layer per
sequence: 48 linear layers x 64 sequences is ~9.6 GiB of state nothing uses,
plus 63.5 MB of reserved logits.

## Decision

ADR 0006 requires bounded conversation slots and continuous batching in the
integrated ring server. Context, slot count, and batch sizes must be selected
from the model's state cost and the live per-device resource plan. A
single-sequence default is not a product configuration, and users must not
manually assemble a separate batch execution mode.

## Consequences

- Sequence-state allocation remains bounded by the selected slot count.
- Product placement accounts for context and per-slot state on every selected
  worker before admitting the topology.
- The protocol carries the selected context and batching limits as part of the
  integrated server lifecycle.
- The protocol version bump to 5 absorbs the config change.
