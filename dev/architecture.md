# potluck.cpp architecture

## Binding product decision

Potluck has one execution architecture: a resource-aware piped ring. The
complete product decision is [ADR 0006](decisions/0006-piped-ring-server-product.md);
[ADR 0007](decisions/0007-prima-direct-ring-zeromq.md) fixes its direct-ring
topology and ZeroMQ communication model.
Static contiguous execution, manual placement, and a single-request server are
unfinished legacy implementation and must be removed, not retained as product
modes or fallbacks.

The direct ring is now implemented inside `potluck-server`. The controller
launches local workers and can bootstrap explicitly named workers over SSH.
This is a working transport and launch path, not the finished automatic
cluster lifecycle.

A finished Potluck deployment presents one OpenAI-compatible endpoint on the
head machine. The controller automatically discovers and profiles the
available devices, selects the useful devices, assigns repeated disjoint layer
windows around the ring, starts the workers, and continuously serves isolated
conversation slots. Users and client harnesses do not configure model files,
workers, ranks, ports, weights, bounds, or execution modes.

## Behavioral reference and usability north star

Prima.cpp is the behavioral progenitor of Potluck's distributed runtime.
Unresolved technical behavior defaults to prima.cpp: preserve how its ring,
scheduler, profiler, prefetch, accelerator planning, device selection,
batching, and sequence machinery behave and what they accomplish. Existing
accepted Potluck ADRs define deliberate differences. Any new technical
departure requires explicit user approval and another accepted ADR.

This deference does not extend to prima.cpp's manual user flow. Potluck must be
easy to use as one normal local server. The controller hides discovery,
profiling, topology, windows, model distribution, placement, and startup. The
user selects a model and connects a standard client or agent harness to the
head; distributed configuration is not part of the normal interface.

## Product processes and data flow

1. The head discovers reachable workers and observes current CPU, memory,
   accelerator, storage, and network capacity.
2. The scheduler reserves resources for the user and operating system on the
   head. The head may own ring windows only when its currently available
   resources make that safe and useful.
3. The scheduler selects the devices and assigns several disjoint windows to
   each device. Assignment is heterogeneity-aware and minimizes the limiting
   stage subject to live resource constraints.
4. The controller ensures the complete GGUF is present on each selected
   device, transferring it automatically and reusing a copy whose checksum
   matches.
5. Each worker opens that model file but loads only its assigned layer
   windows. The full file may remain on disk; no worker or production head
   loads the complete model into memory.
6. Per-window prefetch is not implemented yet. Admission sizes assignments so
   each device can hold all of its windows, so full-file presence is sufficient
   for the current schedule.
7. CPU, CUDA, and Metal placement is selected independently for each device and
   window from current usable capacity.
8. Active conversation slots are continuously batched through the ring. Each
   slot keeps isolated sequence state and cache affinity.
9. The tail samples or selects the next token and the head returns the
   documented OpenAI-compatible JSON or streaming event.

The implemented data plane in `potluck-server` uses ZeroMQ message-oriented
communication between adjacent ring peers. Each worker binds its receive
socket, connects directly to its cyclic next peer, and sends final results back
to the head. Ingress goes to rank 0; the head does not relay intermediate
windows it does not execute.

DNS-SD candidate discovery, bounded pre-launch probing and admission,
resource-weighted selection and placement, automatic SSH launch, and
full-model distribution with checksum validation are implemented. Adaptive
load changes, per-window prefetch, recovery migration, security, and full API
parity remain unfinished.

## Piped-ring execution

Every selected worker can own several disjoint layer windows. A request visits
those windows in ring order and can visit the same device more than once during
one model pass. Window sizes and ownership come from the automatic scheduler in
the finished product; equal splits and user-supplied static bounds are not
product behavior.

The current server route assigns repeated disjoint windows from pre-launch
usable-capacity measurements. DNS-SD supplies candidate nodes and the server
launches them through SSH after probing and admission. Each worker reports its
accelerator kind and free memory before and after the schedule; the head uses
those profiles for per-window CPU, Metal, or CUDA placement.

Each selected device receives from its previous ring peer and sends directly to
its next ring peer. Rank 0 is both a ring peer and the client-facing controller.
Its control responsibility does not make it a relay for other peers' window
transitions. Automatic setup configures these connections without exposing
ranks, addresses, ports, or launch order to the user.

The ring supports prompt prefill, continuous decode, multiple active sequences,
slot lifecycle, cancellation, and per-window accelerator placement through the
same server runtime. Per-window prefetch is currently unimplemented because
admission keeps every assigned window resident on its device; it remains a
release-gate requirement. A transport smoke does not satisfy the architecture.

## Model distribution and window loading

`potluck-shard` remains an optional disk-saving and format-validation tool for
offline workflows. Normal server startup does not require manually generated
shards or a deployment script.

The controller distributes the complete source GGUF to each admitted remote
device, verifies its checksum, and reuses a valid copy on later starts. A
worker opens that file with the layer bounds for its assigned windows and
loads only those windows. A complete GGUF may exist on disk, but production
execution must not load the complete model into memory.

## Scheduling and head resource protection

Profiling and placement are automatic parts of startup and service operation,
not separate commands whose output a user applies by hand. The scheduler must
consider:

- measured CPU and accelerator throughput;
- current CPU load and accelerator use;
- available RAM, VRAM, swap policy, and memory pressure;
- an explicit resource reserve on the head for the user and operating system;
- storage capacity and per-window read behavior;
- network reachability, bandwidth, and latency;
- model, context, and per-slot state requirements.

The scheduler selects only a feasible set of devices. It can exclude a weak
device when the remaining devices have enough capacity and become faster
without it. It must retain a slower device when that device is required to fit
the model. It must reduce or remove the head's work when current user activity
consumes the resources that were available during placement.

## OpenAI-compatible server

The head is the sole client-facing endpoint. The product server must provide
the documented OpenAI-compatible chat/completion, streaming, model, error,
usage, sampling, stop, and cancellation behavior required by ordinary
harnesses. Unsupported request fields must be rejected explicitly rather than
ignored.

Conversation slots and continuous batching are implemented in the integrated
server path. Slots own bounded sequence state, cache affinity, isolation,
cancellation, and lifecycle. The API surface and live failure migration remain
smaller than the full llama.cpp server contract.

## Current implementation gap

The direct adjacent-peer ZeroMQ path now runs in `potluck-server`:

- Local two-worker CPU smoke passed with repeated disjoint windows.
- An M4 head bootstrapped one Linux CPU worker over SSH for the Qwen3.5 0.8B
  fixture. The two-token result was ` located in`, and the server logged
  windows `[0,12)` and `[12,24)`.
- The same M4 head automatically discovered the Linux node through DNS-SD,
  accepted its first SSH host key in a Potluck-specific trust file, launched
  the worker, and returned the same two-token result.
- The local two-worker smoke now reports each worker's accelerator through the
  ring protocol and places window layers automatically: fully on M4 Metal and
  on a GTX 1650 SUPER CUDA device, CPU-only when no device exists.

The remaining product gaps are explicit:

- DNS-SD discovery, bounded pre-launch probing, capacity admission,
  heterogeneous window sizing, automatic model distribution, and accelerator
  placement are implemented in the startup lifecycle.
- Per-window prefetch remains unimplemented. Admission keeps every assigned
  window resident on its device, so whole-file presence covers the current
  schedule.
- Head resource reservation and automatic head participation are implemented
  at startup. They do not yet adapt to changing user load.
- The server provides continuous HTTP batching, isolated conversation slots,
  per-request sampling, and a single rebuild attempt with 30-second backoff.
- Worker failure migration and safe request retry are unfinished. A failed
  request receives a rebuild error; completed token state is not migrated.
- The OpenAI-compatible HTTP surface is a subset; full request, response,
  error, usage, streaming, and cancellation parity is unfinished.
- Authentication, encryption, credential handling, and tenant or prompt
  privacy controls are unfinished. The deployment boundary remains a trusted
  LAN.

These gaps must be removed through a clean cutover. They are not supported
product configurations, provisional releases, or alternate architectures.
Product tests must exercise the integrated piped-ring server.
