# potluck.cpp architecture

## Binding product decision

Potluck has one execution architecture: a resource-aware piped ring. The
complete product decision is [ADR 0006](decisions/0006-piped-ring-server-product.md);
[ADR 0007](decisions/0007-prima-direct-ring-zeromq.md) fixes its direct-ring
topology and ZeroMQ communication model.
Static contiguous execution, manual placement, and a single-request server are
unfinished legacy implementation and must be removed, not retained as product
modes or fallbacks.

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

The data plane uses ZeroMQ message-oriented communication between adjacent ring
peers. Hidden states do not return to the head between windows unless the head
owns the next window. The custom PTLK/raw-TCP data plane and
coordinator-relayed route are legacy removal targets.

## Piped-ring execution

Every selected worker can own several disjoint layer windows. A request visits
those windows in ring order and can visit the same device more than once during
one model pass. Window sizes and ownership come from the automatic scheduler;
equal splits and user-supplied static bounds are not product behavior.

Each selected device receives from its previous ring peer and sends directly to
its next ring peer. Rank 0 is both a ring peer and the client-facing controller.
Its control responsibility does not make it a relay for other peers' window
transitions. Automatic setup must configure these connections without exposing
ranks, addresses, ports, or launch order to the user.

The ring must support prompt prefill, continuous decode, multiple active
sequences, slot lifecycle, per-window prefetch, and per-window accelerator
placement through the same server runtime. A standalone ring demonstration
does not satisfy the architecture.

## Shards and loading

`potluck-shard` currently proves that independently loadable GGUF window files
can preserve source metadata and global block indices. The product controller
must integrate shard creation, selection, transfer, validation, and caching.

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

The present source does not implement this architecture end to end:

- `potluck-server` uses a static equal or manually bounded chain;
- `potluck-head --ring` is a separate opt-in path;
- profiling produces manually consumed weights instead of live placement;
- prefetch warms a model file rather than coordinating the next ring window;
- accelerator planning uses static/global budgets;
- shard generation and deployment are manual;
- batch sequence primitives are not a continuous HTTP scheduler;
- the server has no conversation slots and serializes requests with a mutex.

These are gaps to remove through a clean cutover. They are not supported
product configurations, provisional releases, or alternate architectures.
Product tests must exercise the integrated piped-ring server. Static execution
tests must be deleted or rewritten rather than used as product evidence.
