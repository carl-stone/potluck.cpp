# ADR 0011: HALDA solver owns device exclusion

- Date: 2026-08-23
- Status: Accepted
- Decision owner: Carl Stone
- Supersedes: ADR 0008 pre-launch capacity admission

## Context

ADR 0008 lets the probe admit only candidates whose measured usable capacity
can contribute to the model. HALDA cannot price a device that the controller
removes before solving. The solver must see every reachable device so memory,
compute, storage, and accelerator tradeoffs remain part of one optimization.

## Decision

The pre-launch probe keeps only hard rejections: an unreachable candidate, a
candidate with the wrong Potluck build id, or a Windows candidate while Windows
is not supported. Every other reachable candidate enters the HALDA solve.

HALDA owns capacity-based device exclusion. After solving, a device whose
window count is one is removed unless it is rank 0. The controller re-solves
with the reduced device set until the device set is stable. The head is not a
HALDA device when resource-aware participation excludes it.

The probe still measures and reports live memory and device capabilities. It
does not make the placement decision. No heuristic admission path remains.

## Consequences

- A reachable device can be priced as a slow, memory-poor, or accelerator-rich
  option before the solver decides whether to use it.
- The controller can remove weak devices only through the solved allocation.
- Probe failures remain visible startup errors, while capacity decisions are
  deterministic solver output.
- ADR 0008 remains historical and is superseded for admission behavior.
