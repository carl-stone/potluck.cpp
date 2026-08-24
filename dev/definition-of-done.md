# Potluck package Definition of Done (draft)

Status: Draft for discussion. This document is not an accepted ADR.

This document proposes the completion contract for the Potluck package. It gives
agents one clear target: what must be true, what must be tested, and what proof
is needed before Potluck can be called done.

It does not choose a new architecture. Accepted decisions in `dev/decisions/`
remain binding. A decision that is still open belongs in
[`open-questions.md`](open-questions.md), not here as an unapproved choice.
Behavior already defined by prima.cpp or llama.cpp is inherited rather than
re-specified here, unless an accepted Potluck ADR changes it.

## Document roles

- The accepted ADRs define the product architecture and its boundaries.
- This document defines the proposed finished-product behavior and proof.
- [`open-questions.md`](open-questions.md) lists decisions that still need an
  answer before this document can become final.
- [`parity-and-accuracy.md`](parity-and-accuracy.md) records test evidence and
  the limits of that evidence.
- [`architecture.md`](architecture.md) describes the implementation design.

## What done means

Potluck is done when a person can select a model and use one normal local
inference server. The server must automatically discover and select useful
household devices, form the resource-aware direct ring, and serve concurrent
state-isolated conversations through the head endpoint. The person must not
need to manage ranks, workers, layer bounds, shards, ports, weights, or launch
steps.

All required behavior must run through one integrated `potluck-server` path.
The `potluck-cli` uses the same discovery, worker lifecycle, ring, scheduler,
and shutdown behavior. A component test, a static route, a manual cluster, or a
separate ring tool does not establish completion.

## Required product behavior

The following behavior is the proposed package target. It is drawn from ADRs
0006, 0007, 0009, 0010, 0011, 0012, and 0013 and from the product checks that
were previously recorded in `open-questions.md`.

### Normal user flow

- The user selects a model and starts the normal Potluck server or CLI.
- The controller discovers reachable candidates and starts the useful workers.
- The controller hides ranks, topology, ports, window bounds, model transfer,
  and worker launch details.
- Automatic scheduling is the normal path.
- The optional expert workload override is a constraint inside the automatic
  scheduler. It is not a separate execution path.
- The head may execute ring windows only after reserving safe CPU, memory, and
  accelerator capacity for the current user and operating system.

### Ring and placement

- The only production distributed architecture is a repeated-window piped
  ring.
- Each selected device may own several disjoint layer windows.
- Adjacent ring peers communicate directly over ZeroMQ.
- The head handles client ingress and final results but does not relay windows
  that it does not execute.
- The controller profiles live CPU, memory, storage, network, and accelerator
  capacity before and during service.
- HiGHS solves heterogeneity-aware HALDA placement from those live profiles.
- Capacity-based device exclusion belongs to the solver. A reachable device is
  not removed by a separate heuristic admission path.
- The route is rebuilt from live profiles when the topology or resource state
  changes.

### Model files and window loading

- The controller ensures one complete digest-checked GGUF file per selected
  device and reuses it across topology rebuilds.
- A worker maps and loads only the tensors in its assigned global layer windows.
- A worker must not load the complete model into memory.
- Route-specific `.potluck-shards` directories and the old shard-generation
  tool are not product behavior.
- mmap loading is lazy. Memory pressure is balanced per device by the
  scheduler.
- A worker advises the next assigned window after sending the current window's
  result. Whole-file prefetch is not the execution path.

### Inference and scheduling

- CPU is always available. CUDA and Metal are required accelerator backends.
- Accelerator placement is selected independently for each device and window.
- Quantized GGUF models are supported through the integrated route.
- Speculative decoding works for server requests and CLI sessions through the
  same ring lifecycle.
- Active prompt and decode rows are continuously batched through the ring.
- Independent conversations have bounded slots, sequence identity, isolation,
  cache affinity, cancellation, and lifecycle management.
- Context size, slot count, and batch limits are selected before placement and
  are carried through the ring configuration.

### Client surfaces

- The head exposes the OpenAI-compatible interfaces required by the current
  OMP Pi agent for every supported Potluck model:
  - native llama.cpp discovery through `GET /models` and `GET /props`;
  - model listing through `GET /v1/models`;
  - Chat Completions through `POST /v1/chat/completions`;
  - Responses through `POST /v1/responses`.
- Chat Completions and Responses support streamed and non-streamed requests,
  usage, sampling, stop behavior, errors, and cancellation.
- Chat Completions supports the message, tool, reasoning, and sampling fields
  that Pi sends, including partial streamed tool-call arguments and tool
  results.
- Responses supports `input`, `instructions`, reasoning, tools, tool choices,
  output events, function-call argument events, completion events, failure
  events, and the matching non-streamed response.
- Tool calls preserve call IDs, function names, arguments, parallel calls, and
  follow-up tool results across turns.
- Reasoning content is streamed and can be replayed in later turns without
  changing the model's rendered history.
- Closing or aborting a streamed request stops the work and frees its slot.
  An explicit cancellation route is not required for Pi compatibility.
- The existing `POST /v1/completions` route remains documented for ordinary
  clients. It is not a substitute for either Pi API.
- Unsupported request fields are rejected instead of silently ignored.
- The completion CLI and the HTTP server use one ring runtime and one
  scheduler.
- The package acceptance check uses the current OMP Pi agent without a custom
  Potluck provider patch.

### Recovery and security

- A worker loss or topology change does not leave the server hung.
- The server can rebuild a valid ring and serve a later request.
- The worker refresh behavior in the goals below protects the old worker until
  the new worker has been copied, checked, verified, and started successfully.
- The trusted-LAN security baseline uses ephemeral CURVE credentials for direct
  ring peers.
- When configured, one bearer API key protects every HTTP route.
- CORS accepts one exact configured origin. CORS is not authentication.
- Potluck does not claim public-Internet safety or provide built-in TLS in this
  baseline. Public-network HTTP needs an external TLS boundary.

## Supported scope and boundaries

This draft uses the following accepted scope:

- macOS and Linux are supported now. Windows remains outside the current target.
- The distributed graph targets the Qwen3.5 dense text path and named quantized
  fixtures. Other model architectures need their own graph and acceptance
  evidence before they can be called supported.
- CPU, CUDA, and Metal are in scope. Other accelerator backends are not required
  by the current baseline.
- A full-model reference may be used by small test fixtures only. Production
  Potluck processes must not load a full model as a correctness reference.
- Full current OMP Pi agent protocol support is required for supported models.
  The current named model scope remains the Qwen3.5 dense text path and named
  quantized fixtures; this does not claim support for every model architecture.
- The server has an Anthropic Messages compatibility route at
  `POST /v1/messages`. Anthropic support remains an optional compatibility
  surface until the scope question in `open-questions.md` is decided; it is
  not a required pass condition in this draft.
- Vision, audio, and video model support require separate model and acceptance
  evidence. A text-only model must be advertised as text-only.
- Public-Internet deployment, tenant isolation, embeddings, reranking, model
  routing, and other broader llama.cpp features are outside this draft. A
  future scope change needs an explicit decision and, where required, an ADR.

## Three current goals

These are the three approved product goals. The integrated server gate now has
direct evidence for the exercised local route. The limits of that evidence stay
explicit below.

### Goal 1: Separate conversations

Two chats must be able to run at the same time. Each later message must see
only the earlier messages from its own chat.

Done means:

- Each chat has a conversation ID.
- Each ID stays linked to its own slot and history.
- Two chats remain independent during concurrent use.
- The needed history survives a ring rebuild.
- The behavior when all slots are busy or a chat is idle is documented.

Proof:

- The integrated server gate runs two conversation IDs concurrently, sends
  follow-up prompts with their own prior markers, and checks isolated output.
- `/health` and the server log show separate slots and stable slot and sequence
  values across two turns.
- The worker-loss gate repeats the follow-ups after a ring rebuild and checks
  the same slot and sequence.

Current evidence: PROVEN for the exercised local two-worker route. Clients
send the full prior history on each turn; prompt KV reuse and token-state
migration are separate scope.


### Goal 2: API cancellation

An ordinary client must be able to stop a request through the normal HTTP
interface. This includes a user who stops a streamed answer by closing the
connection.

- A client can stop a streamed request by closing the connection or aborting
  the request.
- The server stops the work and frees the slot.
- A new request can use that slot without a hang or mixed state.
- The response and error behavior is documented.

Proof:

- The integrated server gate closes a streamed request after the first data
  event and checks the disconnect log, slot cleanup, and a successful follow-up.
- It aborts a non-streamed long request and checks the same cleanup and
  follow-up behavior.
- Same-ID preemption returns HTTP 503 with `request cancelled` and does not
  cancel another active slot.

Current evidence: PROVEN for the exercised local two-worker route and the
streaming and non-streaming HTTP paths.

### Goal 3: Worker refresh recovery

When a remote machine has an old or missing Potluck worker, the server must
update it and restore service without a hang.

Done means:

- The new worker is copied to a temporary location.
- The copy is verified before the current worker is stopped.
- The old worker is kept or relaunched if the new worker cannot start.
- An unavailable machine is left out of the next valid ring when possible.
- A recovered machine can join later.
- Active chat state survives the ring change.

Proof:

- `test-potluck-refresh` uses fake SSH and rsync commands to check a fresh
  install, checksum failure, staged probe failure, installed start failure with
  rollback, equal build, and foreign platform.
- The integrated worker-loss gate stops a worker, checks the retryable 503,
  waits for a successful rebuild, checks the expected worker count and old
  worker exit, and serves follow-ups for both conversations.

Current evidence: PROVEN for the staged refresh transaction and the exercised
local worker-loss rebuild. Token state is not migrated across a worker change.

## End-to-end package checks

All ten checks must pass through one integrated server process before Potluck
can be called a regular local inference server. The current evidence labels are
included only to show the starting point for this draft:

- `UNPROVEN` means the pass condition has not been observed.
- `PROVEN for the exercised route` means only that one named operating point
  passed. It is not a general platform or compatibility claim.

### RG-01 - Automatic discovery and admission

Scenario: Start one server with no worker, rank, bound, or weight configuration
and at least two discoverable candidates, including one unreachable or
under-capacity candidate. Submit a client request through the head.

Pass condition: The server profiles candidates, excludes the unsuitable
candidate, forms an admitted topology, and completes the request without manual
topology input.

Current evidence: UNPROVEN.

### RG-02 - Heterogeneous window placement

Scenario: Run the same server with admitted peers that have deliberately
different compute, memory, accelerator, and network capacity. Submit a request
that crosses more than one ring window.

Pass condition: The route assigns repeated disjoint windows from live
measurements and current pressure, with no equal split or manual weights, and
the request completes through that route.

Current evidence: PROVEN for the exercised 27B operating point.

### RG-03 - Adaptive head reserve

Scenario: While the head owns ring work and serves a request, apply and remove
controlled CPU, memory, or accelerator load on the head.

Pass condition: Placement evidence shows head-owned work shrinking or being
removed under load and returning only when the user reserve is safe, without an
unsafe or hung request.

Current evidence: UNPROVEN.

### RG-04 - Direct ZeroMQ ring

Scenario: Start a three-peer server and submit a request that crosses windows
while observing the topology and peer connection records.

Pass condition: Adjacent peers exchange request state directly over ZeroMQ, the
head handles ingress and the final result but does not relay unowned windows,
and no static or alternate transport path is used.

Current evidence: PROVEN for the exercised route.

### RG-05 - Full-model window loading

Scenario: Make a complete GGUF available for controller distribution, start the
multi-peer server, and submit a request through the head.

Pass condition: Each worker receives one stable complete model path, loads only
tensors for its assigned layer windows, and completes the request. No worker
loads the complete model into memory.

Current evidence: PROVEN for the exercised route.

### RG-06 - Window prefetch and accelerator placement

Scenario: Run a multi-window request across peers with CPU, CUDA, and Metal
capabilities for at least two ring cycles.

Pass condition: The route records an independent backend choice for each device
and window, and each worker records prefetch of its next assigned window before
that window's compute. Whole-file prefetch is not used.

Current evidence: UNPROVEN.

### RG-07 - Conversation slot affinity

Scenario: Start two concurrent conversations through one head endpoint, then
send isolated follow-ups to both while their slots remain active.

Pass condition: Each follow-up retains only its client-supplied prior context,
keeps affinity to its assigned slot and sequence, and completes without state
leakage or cross-conversation output.

Current evidence: PROVEN for the exercised local route. `test_server.sh`
checks two concurrent IDs, isolated markers, stable slot and sequence values
across two turns, cancellation, and the same bindings after worker rebuild.
The client sends the full prior history; prompt KV reuse and token-state
migration remain outside this evidence.

### RG-08 - Concurrent streaming batch

Scenario: Open two concurrent streaming requests with different prompt lengths
and overlapping decode work through one head endpoint.

Pass condition: Both streams produce interleaved progress through the same ring
batch and finish normally. No global 429 serialization or one-stream blocking
occurs.

Current evidence: PROVEN for the exercised local route.

### RG-09 - OpenAI and Pi agent contract

Scenario: Use an ordinary OpenAI-compatible client and the current OMP Pi agent
against the head. Exercise native model discovery, Chat Completions and
Responses requests, streamed and non-streamed output, sampling and stop
fields, reasoning, a tool call with streamed partial arguments, a tool result
follow-up, an invalid request, and a cancelled request.

Pass condition: Both APIs return the documented response, usage, streaming
event, tool-call, reasoning, cancellation, and error behavior. Pi discovers a
supported model and completes a normal agent turn without a custom Potluck
provider patch. Unsupported fields are rejected rather than silently ignored.

Current evidence: UNPROVEN. The component suite covers several HTTP behaviors,
but it does not prove the full Pi agent contract.

### RG-10 - Restart and topology recovery

Scenario: Complete a request, restart the server with valid model files and
placement data still present, then remove or add a peer or change its available
resources and submit another request.

Pass condition: Restart reuses valid files and placement data, automatically
re-profiles the changed topology, records the new result, and serves the next
request without manual configuration or a hang.

Current evidence: PROVEN for the worker-loss rebuild and staged refresh
transaction exercised by `test_server.sh` and `test-potluck-refresh`. Full
server restart plus peer add, removal, or changed-resource re-profiling remains
UNPROVEN.

## Evidence rules

- The check must exercise the integrated `potluck-server` path.
- A standalone ring, static route, manual topology, component-only check, or
  separate batching CLI cannot satisfy a package check.
- Each check records its model, platform, topology, client action, expected
  behavior, observed behavior, and log or test reference.
- A small test may prove a local invariant. It cannot replace a medium or large
  check when the real boundary includes devices, files, threads, or a network.
- The evidence must show behavior, not only source structure or startup logs.
- A claimed platform or model feature needs a named check on that platform or
  model class.
- When a worker or topology changes, the evidence must cover both failure and
  successful recovery behavior.

## Final package checklist

Potluck can move from this draft to an accepted completion contract only after
these questions are answered and recorded:

- [ ] The Pi discovery, Chat Completions, Responses, tool, reasoning, and
      cancellation contract is implemented and documented.
- [ ] Inherited llama.cpp slot, cache, context, and local HTTP behavior is
      integrated through Potluck and covered by evidence.
- [ ] Inherited prima.cpp distributed execution and recovery behavior is
      integrated through the Potluck ring and covered by evidence.
- [ ] All required product behavior above exists in the integrated runtime.
- [ ] RG-01 through RG-10 pass with recorded evidence.
- [ ] Conflicting static, manual, and alternate execution paths are removed.
- [ ] Documentation and user-facing options match the accepted contract.
