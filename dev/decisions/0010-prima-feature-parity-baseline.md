# ADR 0010: Prima.cpp feature baseline

- Date: 2026-08-23
- Status: Accepted
- Decision owner: Carl Stone
- Supersedes: nothing
- Amends: ADR 0006 manual workload controls and client-surface boundary

## Context

On 2026-08-23, Carl Stone reviewed prima.cpp's published feature set and
directed that Potluck must carry every item in it. prima.cpp is the behavioral
progenitor of Potluck's distributed runtime, so its documented capabilities
form the product baseline.

Three documented positions conflicted with that direction:

- ADR 0006 banned manual placement inputs outright and treated any additional
  client surface as legacy.
- The gap matrix called the solver-free placement route intentional.
- The gap matrix tracked speculative decoding only as an upstream capability
  with no product claim.

This decision records the user's direction and overrides those positions.

## Decision

Potluck requires all twelve capabilities below inside the one integrated
piped-ring product path from ADR 0006. Items already fixed by ADR 0006 and
ADR 0007 stay binding; the rest join the product contract.

1. mmap lazy weight loading. Workers map their assigned window shards; the OS
   faults pages in on demand. The scheduler balances memory pressure per
   device, and per-window prefetch coordinates which ranges become resident.
2. Piped-ring parallelism with prefetching, as already decided.
3. Heterogeneity-aware layer-to-device allocation (HALDA). Window and
   accelerator allocation is solved from live profiles as an optimization
   problem, using HiGHS as prima.cpp does. HiGHS is MIT licensed, builds on
   macOS, Linux, and Windows, and keeps placement behavior aligned with the
   progenitor; there is no reason to hand-roll a competing solver. A heuristic
   route may seed solver inputs but never replaces the solved allocation.
4. Automatic device selection, as already decided.
5. Quantized models. Quantized GGUF weights are a normal supported input.
   llama.cpp owns the quantization formats, and named checks establish
   distributed breadth.
6. Speculative decoding. The ring serves draft-based speculation end to end,
   for server requests and CLI sessions, reusing llama.cpp's speculative
   machinery. This is release-gate work, not an optional experiment.
7. Dynamic batching of concurrent requests, as already decided (continuous
   batching).
8. Platforms. macOS and Linux are supported now. Windows is on the roadmap
   and is not required yet.
9. Two client modes. The potluck command family provides a completion CLI and
   an OpenAI-compatible server endpoint over the same ring runtime and
   scheduler. Neither mode is a second execution architecture.
10. Optional manual workload setting. A user may override the computed
    workload per device, like prima.cpp's -lw and -ngl, as an explicit expert
    control. Automatic scheduling remains the default and the advertised path.
    This amends ADR 0006: manual ranks, worker files, host lists, ports, and
    launch steps stay removed, and an override is never the scheduler.
11. Accelerators. CUDA and Metal are the required accelerator backends; CPU is
    always available. Other backends are not required yet.
12. Command family. User-facing binaries carry the potluck name. Their
    inference-facing flags mirror prima.cpp's llama-cli and llama-server
    usage so knowledge transfers directly. Distributed launch flags such as
    --world, --rank, --master, --next, and the data and signal ports stay
    internal to the controller, per ADR 0007.

## Consequences

- The controller gains a HiGHS build dependency. Scripts already reference a
  POTLUCK_HIGHS option, but no CMake code consumes it; wiring it up is
  unfinished work.
- The release gate grows named checks for solved placement, speculative
  decoding, quantized models, the completion CLI, and platform claims.
- Documents that called manual workload inputs non-product or the solver-free
  route intentional must be corrected to match this decision.
