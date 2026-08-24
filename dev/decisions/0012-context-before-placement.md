# ADR 0012: Select context before placement

- Date: 2026-08-23
- Status: Accepted
- Decision owner: Carl Stone
- Amends: ADR 0004 context and sequence sizing

## Context

HALDA needs the per-layer KV cost before it can build its capacity rows. The
placement also determines which devices can support the selected context, so
deriving context from the live plan creates a circular dependency.

## Decision

Potluck selects `n_ctx` before placement from model metadata and the head
reserve. The selected context, slot count, and batch limits are scheduler
inputs. HALDA receives the already known `kv_per_layer` value and solves the
window and accelerator allocation from it.

The protocol carries the selected context and batching limits through the
integrated ring configuration. Placement does not resize context after the
solve.

## Consequences

- Context sizing is deterministic before the first HALDA iteration.
- Capacity estimates include the KV cost of the selected context on every
  device.
- A placement failure reports the selected context and resource shortfall
  instead of silently changing user-visible context behavior.
- ADR 0004 remains historical and is amended for this ordering.
