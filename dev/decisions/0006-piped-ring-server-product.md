# ADR 0006: Piped-ring server product

- Date: 2026-08-21
- Status: Accepted
- Decision owner: Carl Stone
- Supersedes: the download and storage restriction in ADR 0002
- Transport and ring topology amended by:
  [ADR 0007](0007-prima-direct-ring-zeromq.md)
- Manual workload controls and the client-surface boundary are amended by:
  [ADR 0010](0010-prima-feature-parity-baseline.md)
- Model storage and loading are amended by:
  [ADR 0013](0013-full-model-per-device.md)

## Context

Potluck exists to run models that are too large or too slow for one household
computer by combining several heterogeneous, comparatively weak devices. The
user-facing result must behave like a normal local inference server. A client
or agent harness connects to one endpoint on the head machine; it must not need
to understand workers, layer bounds, shards, ranks, ports, or scheduling.

Previous work treated a static contiguous pipeline, manual worker files, a
single-chain HTTP wrapper, or staged subsets of the intended runtime as
acceptable configurations. They are not the product, provisional product
modes, fallback modes, or architectures to preserve. Keeping those execution
paths in Potluck would invite the project to drift away from its purpose.

## Decision authority

[prima.cpp](https://github.com/OpenCPIL/prima.cpp) is Potluck's behavioral
progenitor for distributed inference. When an unresolved technical behavioral
choice concerns ring execution, scheduling, placement, profiling, prefetch,
accelerator use, device selection or removal, batching, sequence handling, or
another distributed-runtime semantic, Potluck defers to prima.cpp's documented
and coded behavior by default. Match its observable behavior and purpose.
Exact source compatibility, old ggml internals, wire payloads, and command-line
flags are not required. ADR 0007 makes prima.cpp's direct-ring topology and
ZeroMQ communication model explicit product requirements.

An agent must not replace prima.cpp behavior with a preferred alternative. A
technical departure requires an explicit user decision recorded in an accepted
Potluck ADR. Existing accepted Potluck decisions include modern llama.cpp,
shard-only loading, the ring-only server contract, and the direct-ring ZeroMQ
communication decision in ADR 0007.

Prima.cpp is not the user-experience reference. Potluck's usability north star
is that a person selects a model and uses one normal local server without
understanding or configuring distributed inference. Automatic discovery,
profiling, selection, placement, sharding, startup, recovery, batching, and
slots must hide the cluster machinery. Prima.cpp's manual ranks, topology,
files, flags, and launch flow must not be copied into the Potluck product.

## Decision

The only Potluck execution architecture is a piped ring. Every production
request runs through repeated, disjoint layer windows assigned around the ring.
Static one-window-per-device execution and other distributed execution
architectures must be removed from the Potluck implementation, command line,
server, tests, and product documentation. Historical records may describe
removed work only when they are marked as superseded and non-canonical.

A finished Potluck product includes all of the following as one integrated
server path:

1. **Piped-ring execution.** Each selected device can own several disjoint
   windows. Requests traverse those windows through direct ZeroMQ connections
   between adjacent ring peers; the head does not relay windows it does not
   execute.
2. **Heterogeneity-aware window assignment.** Window sizes and placement are
   selected from measured compute, accelerator, memory, storage, and network
   capabilities. Equal splits and user-supplied weights or bounds are not a
   product scheduler.
3. **Automatic device profiling and selection.** The controller discovers,
   profiles, admits, and excludes devices automatically. A device is selected
   only when it is required for capacity or improves the feasible schedule.
4. **Per-window prefetch.** Each worker prefetches its next assigned window in
   coordination with ring execution. Warming an entire full-model file is not
   the required behavior.
5. **Per-device accelerator placement.** The scheduler independently chooses
   CPU, CUDA, or Metal placement for each device and window from current usable
   capacity. One global GPU-layer boundary is not sufficient.
6. **Per-device GGUF shard loading.** A worker loads only the tensors required
   by its assigned windows. A complete model file may be downloaded or stored
   on any device, but a worker must not load the complete model into memory.
   Distribution and caching policy must preserve this loading invariant.
7. **Continuous batching.** The server schedules active requests together
   through the ring instead of serializing the cluster behind one request.
8. **Conversation slots.** Independent conversations have bounded state,
   sequence identity, cache affinity, isolation, and lifecycle management.
9. **OpenAI-compatible head server.** The head exposes the documented OpenAI
   request, response, error, and streaming contract needed by ordinary client
   and agent harnesses. Unsupported fields must not be silently ignored.
10. **Resource-aware head participation.** The head may also own ring windows,
    but user activity has priority. Placement must account for current CPU
    load, memory pressure, accelerator use, and an explicit reserve for the
    user and operating system. The scheduler must reduce or remove head work
    when those resources are not safely available.

These capabilities are one product contract. They are not optional follow-on
features and must not be split into provisional releases that ship another
execution architecture or an incomplete server model.

## Finished-product boundary

Potluck is not a finished product until one server process can automatically
form and operate this resource-aware piped ring and serve concurrent,
state-isolated OpenAI-compatible conversations through the head endpoint.
Passing isolated routing, profiling, prefetch, batching, shard, or HTTP checks
does not satisfy this boundary.

The following are explicitly unfinished and must be removed rather than kept as
Potluck capabilities:

- static contiguous execution;
- manual topology or placement controls, including bounds, weights, ranks,
  workers files, and host lists;
- standalone profiling whose output must be applied by hand;
- whole-file prefetch in place of per-window prefetch;
- one global GPU-layer budget in place of per-device placement;
- one-request server execution with HTTP 429 admission;
- stateless chat without conversation slots;
- batching available only through a separate CLI;
- a ring CLI disconnected from the OpenAI-compatible server.

Diagnostics and acceptance checks must exercise the decided piped-ring product
architecture. They must not preserve a second execution architecture inside
Potluck. No alternative distributed architecture is being evaluated.
Engineering work must make a clean cutover to this decision.

## Consequences

- `potluck-server` must own the piped-ring runtime. Ring execution cannot remain
  exclusive to `potluck-head`.
- Static code, flags, tests, and documentation must be deleted or rewritten;
  they are not compatibility surfaces.
- Device profiling, selection, placement, prefetch, batching, slots, and the
  HTTP API must share one runtime and one lifecycle.
- The head is a possible worker, not a free or permanently idle worker. Its
  contribution is decided from live available resources after preserving the
  user's reserve.
- Shards remain the unit of worker loading. ADR 0002's requirement that every
  device avoid downloading or storing a full GGUF is removed; only the
  no-full-model-load requirement remains.
- Component tests do not establish product completion. The release gate must
  cover the complete server path described here.
