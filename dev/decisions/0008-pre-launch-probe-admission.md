# ADR 0008: Pre-launch probe and capacity admission

- Date: 2026-08-21
- Status: Accepted
- Decision owner: Carl Stone
- Supersedes: nothing

## Context

A worker could previously report its accelerator only after launch. The head
therefore started every discovered device before it knew whether the device
could hold its assigned windows. This made admission and heterogeneous window
sizing reactive, and a slow or unsuitable device could wedge startup.

## Decision

Potluck probes every candidate before launching its ring worker. The probe uses
the same worker backend discovery and memory queries as the in-ring
`profile_result` message. SSH candidates run `potluck-worker --probe`; the
head runs the same worker executable locally. A failed or timed-out probe
excludes that candidate and is reported to the user.

The controller admits candidates by measured usable capacity. Accelerator
capacity reserves at least 512 MiB or one eighth of reported accelerator total;
host capacity reserves at least 2 GiB or one eighth of reported host total.
Usable capacity is the larger budget, not their sum, because unified-memory
hosts report one physical pool. Candidates are ordered by usable capacity and
the smallest prefix covering the model and KV-memory estimate is admitted.

The in-ring profile remains authoritative after launch. It is used to confirm
live placement inputs and to choose per-device CPU, CUDA, or Metal execution.
The route then assigns repeated disjoint windows proportionally to admitted
capacity, capped by each device's feasible layer count.

## Consequences

- Devices that cannot be probed or cannot contribute required capacity are
  excluded before model distribution and worker launch.
- Startup can fail with a measured capacity shortfall instead of launching an
  infeasible ring.
- Probe and in-ring profile implementations must stay behaviorally aligned.
- Capacity measurements are live and can change between probe and launch; the
  post-launch profile remains the placement authority.
