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
conversation slots. Users and client harnesses do not configure shards,
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
profiling, topology, windows, shards, placement, startup, and recovery. The
person selects a model and connects a standard client or agent harness to the
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
4. The controller creates or locates the required GGUF shards and transfers or
   reuses them on each selected device.
5. Each worker loads only its assigned window shards. A complete GGUF may exist
   on disk, but no worker or production head loads the complete model.
6. Workers prefetch their next assigned window as part of ring execution.
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

DNS-SD candidate discovery and automatic SSH launch are implemented. Live
profiling and admission, resource-aware selection and placement, shard
automation, recovery, security, and full API parity remain unfinished.

## Piped-ring execution

Every selected worker can own several disjoint layer windows. A request visits
those windows in ring order and can visit the same device more than once during
one model pass. Window sizes and ownership come from the automatic scheduler in
the finished product; equal splits and user-supplied static bounds are not
product behavior.

The current server route implements two repeated disjoint windows per worker
where the model layers permit. DNS-SD supplies candidate nodes and the server
launches them through SSH. Each worker reports its accelerator kind and free
memory before the schedule is sent, and the head budgets that memory across the
worker's windows to fill per-window layer offload on Metal or CUDA. Device
admission, heterogeneous window sizing, and selection are still not live.

Each selected device receives from its previous ring peer and sends directly to
its next ring peer. Rank 0 is both a ring peer and the client-facing controller.
Its control responsibility does not make it a relay for other peers' window
transitions. Automatic setup must eventually configure these connections
without exposing ranks, addresses, ports, or launch order to the user.

The ring must support prompt prefill, continuous decode, multiple active
sequences, slot lifecycle, per-window prefetch, and per-window accelerator
placement through the same server runtime. A transport smoke does not satisfy
the architecture.

## Shards and loading

`potluck-shard` currently proves that independently loadable GGUF window files
can preserve source metadata and global block indices. The product controller
must integrate shard creation, selection, transfer, validation, and caching.

The current server accepts explicit shard inputs and assigns workers only the
windows described by that configuration. Shard creation, transfer, validation,
selection, and caching are not yet automated.

Shards are the unit of loading:

- every shard carries the metadata required to load its windows;
- block tensor names keep their global `blk.<index>` names;
- boundary shards carry the required embedding, normalization, and output
  tensors;
- shard metadata identifies the source model and covered windows;
- a worker rejects assignments outside its shard windows.

A device may download or retain the complete source GGUF. This is a deployment
choice, not a loading choice. Production execution must load only the assigned
window tensors.

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

Conversation slots and continuous batching are required product behavior.
Slots own bounded sequence state, conversation identity, cache affinity,
isolation, cancellation, and lifecycle. The server must admit and schedule
concurrent work; global one-chain serialization and HTTP 429-on-busy behavior
are unfinished implementation.

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

- DNS-SD candidate discovery and SSH launch are automatic, as is accelerator
  profiling with per-window layer placement. Device admission, heterogeneous
  selection, and heterogeneous window sizing are not.
- Head resource reservation and adaptive head participation are not
  implemented.
- Shard creation, transfer, validation, selection, and caching remain manual.
- The server does not yet provide continuous HTTP batching or isolated
  conversation slots.
- Worker failure handling lacks reconnect, ring rebuild, slot migration, and
  safe retry behavior.
- The OpenAI-compatible HTTP surface is a subset; full request, response,
  error, usage, streaming, and cancellation parity is unfinished.
- Authentication, encryption, credential handling, and tenant or prompt
  privacy controls are unfinished. The deployment boundary remains a trusted
  LAN.

These gaps must be removed through a clean cutover. They are not supported
product configurations, provisional releases, or alternate architectures.
Product tests must exercise the integrated piped-ring server.
