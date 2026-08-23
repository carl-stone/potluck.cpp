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
4. The controller derives route-keyed GGUF window shards from the selected model, transfers only each device's assigned shards, verifies their checksums, and reuses matching cached shards.
5. Each worker loads only its assigned window shards. A complete GGUF may remain on controller storage, but no production worker or head loads the complete model into memory.
6. Each worker synchronizes its current window output, sends it directly over
   ZeroMQ, and only then advises the mmap range for its next owned window.
   Advice is bounded to that window's mapped tensors; whole-file mmap prefetch
   is not the execution path.
7. CPU, CUDA, and Metal placement is selected independently for each device and
   window from current usable capacity.
8. Active conversation slots are continuously batched at each token step: the
   scheduler merges active sequence rows into one scheduled batch, preserving
   per-slot sequence state and cache affinity.
9. For autoregressive decode, token t+1 waits for the final logits returned
   from the completed ring traversal for token t. The tail samples or selects
   the next token and the head returns the documented OpenAI-compatible JSON or
   streaming event.

The implemented data plane in `potluck-server` uses ZeroMQ message-oriented
communication between adjacent ring peers. Each worker binds its receive
socket, connects directly to its cyclic next peer, and sends final results back
to the head. Ingress goes to rank 0; the head does not relay intermediate
windows it does not execute.

DNS-SD candidate discovery, bounded pre-launch probing and admission,
resource-weighted selection and placement, automatic SSH launch, and
full-model distribution with checksum validation are implemented. Adaptive
load changes, recovery migration, security implementation, and full API parity
remain unfinished. The accepted trusted-LAN security baseline is defined by
[ADR 0009](decisions/0009-trusted-lan-curve-http-controls.md).

## Piped-ring execution

Every selected worker can own several disjoint layer windows. Prima's
Pipelined-Ring Parallelism (PRP) behavior in Potluck's piped ring traverses
every disjoint global window in cyclic ring order for each scheduled batch. A
batch can therefore return to the same device multiple times in one model
pass; it is not a single synchronous pass through one window per device.
Window sizes and ownership come from the automatic scheduler in the finished
product; equal splits and user-supplied static bounds are not product behavior.

At each window, computation completes and the worker synchronizes its output
before sending it directly over ZeroMQ to the next ring peer. The send occurs
before the worker advises the mmap range for its next owned window. This permits
bounded overlap between downstream window computation and OS page-cache work
for the advised range; it does not provide unrestricted compute/send overlap.

Continuous batching merges active sequence rows into the scheduled batch at one
token step. Token t+1 waits for final logits from the full repeated-window
traversal of token t, so ring work is not pipelined across token steps. There is
no separate cycle scheduler: cyclic window traversal is the route of each
scheduled batch.

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
same server runtime. A transport smoke does not satisfy the architecture.

## Model distribution and window loading

`potluck-shard` provides the GGUF window-shard format used by automatic startup. Users do not generate shards or run deployment commands.

After placement selects the repeated window route, the controller creates a checksum-keyed shard set, sends each selected device only the shards for its assigned windows, and reuses valid cached files on later starts. A complete source GGUF may remain on controller storage, but production workers load only assigned window shards.

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

## Current implementation status

The integrated direct-peer ZeroMQ server now implements repeated-window PRP traversal, automatic DNS-SD discovery, SSH bootstrap, live admission, resource-weighted placement, automatic window-shard distribution, per-window CPU/Metal/CUDA placement, bounded mmap advice, two conversation slots, and worker-loss recovery.

A supervised 2026-08-23 acceptance run served Gemma 3 27B Q4_K_M across an M4 Mac, an M1 Mac, and a Linux PC with a GTX 1650 SUPER. The automatic route assigned six repeated windows across all three devices. Two concurrent conversations, including one streaming response, completed with HTTP 200. Killing the M1 worker during generation returned the required retryable HTTP 503; the controller relaunched the worker, restored a ready three-device ring, and completed the next request.

Implemented lifecycle and reliability behavior:

- DNS-SD discovery, bounded pre-launch probing, capacity admission,
  heterogeneous window sizing, automatic model distribution, and accelerator
  placement are implemented in the startup lifecycle.
- Partial Potluck models suppress whole-file mmap prefetch. Each worker advises
  only the mapped tensors for its first owned window, then advises its next owned
  window after each ring pass.
- Head resource reservation and automatic head participation are re-profiled
  during idle topology checks; CPU load and host memory can remove or restore
  head windows.
- Ring workers answer sequence-checked control heartbeats during idle and active
  execution. A lost worker ends a non-streaming request with a retryable 503.
  A streaming request that already sent output ends with an SSE error; the client
  must discard the partial output before retrying. The server then reconnects
  and rebuilds from live admitted devices.
- Recovery uses bounded exponential backoff with deterministic jitter and a
  terminal health reason after repeated failures. Topology checks never rebuild
  an active request.

The remaining product gaps are:

- The OpenAI-compatible HTTP surface is a subset; full request, response,
  error, usage, streaming, and cancellation parity is unfinished.
- Authentication, encryption, credential handling, and tenant or prompt privacy
  controls remain unfinished in code. The accepted trusted-LAN baseline is
  documented in [ADR 0009](decisions/0009-trusted-lan-curve-http-controls.md);
  the deployment boundary remains a trusted LAN.

These gaps must be removed through a clean cutover. They are not supported
product configurations, provisional releases, or alternate architectures.
Product tests must exercise the integrated piped-ring server.
