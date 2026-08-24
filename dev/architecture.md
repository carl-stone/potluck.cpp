# potluck.cpp architecture

## Binding product decision

Potluck has one execution architecture: a resource-aware piped ring. The
complete product decision is [ADR 0006](decisions/0006-piped-ring-server-product.md);
[ADR 0007](decisions/0007-prima-direct-ring-zeromq.md) fixes its direct-ring
topology and ZeroMQ communication model.
Static contiguous execution, manual placement that displaces the automatic
scheduler, and a single-request server are unfinished legacy implementation
and must be removed, not retained as product modes or fallbacks.

The direct ring is implemented inside `potluck-server`. The controller
launches local workers, discovers advertised nodes, profiles them, solves the
HALDA placement, and bootstraps the ring through the same lifecycle used by
the completion CLI.

A finished Potluck deployment presents one OpenAI-compatible endpoint on the
head machine. The controller automatically discovers and profiles the
available devices, selects the useful devices, assigns repeated disjoint layer
windows around the ring, starts the workers, and continuously serves isolated
conversation slots. Users and client harnesses do not configure model files,
workers, ranks, ports, weights, bounds, or execution modes. The optional
expert workload override in
[ADR 0010](decisions/0010-prima-feature-parity-baseline.md) is the single
exception; automatic operation stays the default.

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

## Prima feature baseline

[ADR 0010](decisions/0010-prima-feature-parity-baseline.md) adopts prima.cpp's
published feature set as Potluck's product baseline. Alongside the piped-ring
contract above, the product requires:

- mmap lazy weight loading with scheduler-balanced per-device memory pressure;
- heterogeneity-aware layer-to-device allocation (HALDA) solved with HiGHS;
- support for quantized GGUF models;
- speculative decoding served end to end;
- macOS and Linux support now, with Windows on the roadmap;
- a completion CLI beside the server endpoint, both over one ring runtime;
- CUDA and Metal accelerators, CPU always available; other backends not yet;
- an optional manual workload override, with automatic scheduling the default.

## Product processes and data flow

1. The head discovers reachable workers and observes current CPU, memory,
   accelerator, storage, and network capacity.
2. The scheduler reserves resources for the user and operating system on the
   head. The head may own ring windows only when its currently available
   resources make that safe and useful.
3. The scheduler selects the devices and assigns several disjoint windows to
   each device. Assignment is heterogeneity-aware and minimizes the limiting
   stage subject to live resource constraints.
4. The controller ensures one complete GGUF per selected device, transfers it
   once when its digest is absent or stale, verifies its checksum, and reuses it
   across topology rebuilds.
5. Each worker passes the complete model path and its assigned global layer
   bounds to the window loader. It maps only those tensors; no production
   worker or head loads the complete model into memory.
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

DNS-SD candidate discovery, bounded pre-launch profiling and admission,
HiGHS-backed HALDA placement, automatic SSH launch, full-model distribution
with checksum validation, per-window prefetch, speculative decoding, and
the integrated completion CLI are implemented. Adaptive load changes,
token-state migration, and full llama-server API parity remain unfinished.
The accepted trusted-LAN security baseline is defined by
[ADR 0009](decisions/0009-trusted-lan-curve-http-controls.md).

## Piped-ring execution

Every selected worker can own several disjoint layer windows. Prima's
Pipelined-Ring Parallelism (PRP) behavior in Potluck's piped ring traverses
every disjoint global window in cyclic ring order for each scheduled batch. A
batch can therefore return to the same device multiple times in one model
pass; it is not a single synchronous pass through one window per device.
Window sizes and ownership come from the automatic scheduler in the finished
product; equal splits are never product behavior. An explicit expert override
may pin per-device layers and accelerator layers
([ADR 0010](decisions/0010-prima-feature-parity-baseline.md)); automatic
sizing stays the default.

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

The controller stores one complete GGUF per selected device under a stable model
path. It verifies the model digest before transfer and reuses the file across
topology rebuilds. Workers load only their assigned global layer windows from
that complete file.

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

The scheduler balances memory pressure across devices. Workers mmap their
assigned windows from the complete model and pages load lazily; admission and
per-window prefetch decide which ranges become resident.
allocation is solved with HiGHS as prima.cpp's HALDA does
([ADR 0010](decisions/0010-prima-feature-parity-baseline.md)); a
capacity-weighted heuristic alone does not satisfy the contract.

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

A completion CLI completes the potluck command family. `potluck-cli` offers
prima-style interactive completion and chat over the same integrated ring
runtime as the server ([ADR 0010](decisions/0010-prima-feature-parity-baseline.md));
it is a second client surface, not a second execution path. Its
inference-facing flags mirror prima.cpp's llama-cli usage, while distributed
launch flags stay internal.
## Current implementation status

The integrated direct-peer ZeroMQ server implements repeated-window PRP
traversal, automatic DNS-SD discovery, SSH bootstrap, live profiling and
admission, HiGHS-backed HALDA placement, full-model distribution, per-window
CPU/Metal/CUDA placement, prefetch modes, speculative decoding, continuous
batching, conversation slots, and bounded worker-loss recovery.

A supervised 2026-08-23 acceptance run served Gemma 3 27B Q4_K_M across an M4
Mac, an M1 Mac, and a Linux PC with a GTX 1650 SUPER. The automatic route
assigned six repeated windows across all three devices. Two concurrent
conversations, including one streaming response, completed with HTTP 200.
The measured operating point is recorded in `docs/BENCHMARKS.md`.

Implemented lifecycle and reliability behavior:

- DNS-SD discovery, bounded pre-launch profiling, solver admission,
  heterogeneous window sizing, automatic model distribution, and accelerator
  placement run in the startup lifecycle.
- Workers suppress whole-file mmap prefetch and advise only mapped tensors for
  the first owned window and each next owned window.
- The head reserve and automatic head participation are re-profiled during
  idle topology checks; current CPU load and host memory can remove or restore
  head windows.
- Ring workers use sequence-checked control heartbeats. A lost worker fails
  active work with a retryable result and triggers a bounded rebuild.

The remaining product gaps are broader llama-server API parity, wider
distributed model and modality coverage, and token-state migration after a
worker change. These are not supported alternate architectures.

Product tests must exercise the integrated piped-ring server.
