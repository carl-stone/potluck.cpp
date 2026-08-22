# potluck-server feature gap matrix

Status: implementation gap inventory under the binding
[ADR 0006](decisions/0006-piped-ring-server-product.md) and
[ADR 0007](decisions/0007-prima-direct-ring-zeromq.md). This document compares
the current source with the decided product. It is not an architecture menu and
does not authorize provisional product modes.

## Product contract

Potluck is a trusted-home-network server that automatically forms a
resource-aware piped ring across heterogeneous household computers. The head
exposes an OpenAI-compatible endpoint and may also compute when live CPU,
memory, and accelerator use leaves the user's reserved capacity available.

The product requires direct peer-to-peer piped-ring execution over ZeroMQ,
automatic profiling and device selection, heterogeneity-aware window
placement, per-window prefetch, assigned-window model loading,
per-device CPU/CUDA/Metal placement, continuous batching, and isolated
conversation slots. These requirements are one release gate. The current
server implements profiling, admission, placement, distribution, batching,
and slots; per-window prefetch remains intentionally unimplemented.
Admission keeps every assigned window resident on its device, so a full-file
copy is sufficient for the current schedule. Coordinated per-window prefetch
remains a release-gate requirement.

Static execution, manual placement, one-request serving, and separate CLI
implementations of ring or batching are conflicting legacy code. They must be
removed through a clean cutover, not preserved as diagnostics, fallbacks,
compatibility modes, experiments, or provisional releases.

Automatic setup and deployment remain required so a user does not manage model
files, ports, bounds, ranks, workers, or SSH. The initial security boundary is
a trusted LAN; broader model/API families and public-network hardening remain
separate scope decisions, but they do not weaken the architecture contract.

Prima.cpp is the behavioral progenitor and default authority for unresolved
technical distributed-runtime behavior. A departure requires explicit user
approval and an accepted ADR. For user flow, Potluck's ease-of-use directive
controls instead: prima.cpp's manual cluster interface is not a product model.


## How to read this matrix

The comparison baseline is the `llama-server` documentation in this checkout
(`tools/server/README.md`) plus the prima.cpp server and distributed-runtime
documentation:

- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [prima.cpp server README](https://github.com/OpenCPIL/prima.cpp/blob/main/examples/server/README.md)
- [prima.cpp argument definitions](https://github.com/OpenCPIL/prima.cpp/blob/main/common/arg.cpp)
- [prima.cpp server implementation](https://github.com/OpenCPIL/prima.cpp/blob/main/examples/server/server.cpp)

The baselines are capability families. A model or backend can still limit a
specific feature. The matrix does not claim that every upstream option works
for every model.

Status labels:

- **PRESENT**: an implemented component with a named check. This label never
  means that the product release gate passes.
- **PARTIAL**: a required component exists, but not in the integrated product
  path.
- **MISSING**: required behavior has no implementation in the product path.
- **CONFLICTING**: current behavior implements an architecture or server model
  prohibited by ADR 0006 and must be removed.
- **TEST ONLY**: a correctness mechanism excluded from product binaries and
  execution.
- **INTENTIONAL**: deliberately outside the selected compatibility or security
  boundary; never an exception to ADR 0006.
- **INHERITED**: available in ordinary llama.cpp but not in Potluck and not
  advertised as a Potluck feature.

## Executive boundary

The current source is unfinished and cannot be shipped as Potluck. The
implemented `potluck-server` path owns a direct adjacent-peer ZeroMQ ring,
automatic pre-launch admission, resource-weighted windows, model distribution,
continuous batching, slots, and a bounded rebuild path.

| Area | Required product behavior | Current source | Status |
|---|---|---|---:|
| Distributed execution | Piped ring only; several disjoint windows per selected device | `potluck-server` direct adjacent-peer ZeroMQ ring; repeated windows per worker where layers permit | PRESENT |
| Window placement | Automatic heterogeneous placement from live capability and resource pressure | Pre-launch probes size windows from usable accelerator/host capacity; in-ring profiles choose offload | PRESENT |
| Device profiling and selection | Automatic discovery, profiling, admission, and exclusion in the server lifecycle | DNS-SD candidates are probed before launch; failed and insufficient candidates are excluded | PRESENT |
| Head participation | Optional worker after reserving for current user CPU, memory, and accelerator use | Head probes itself, reserves memory, and joins when its budget is useful | PARTIAL |
| Prefetch | Next assigned window coordinated with ring execution | Per-window prefetch is not implemented | MISSING |
| Accelerator placement | Independent CPU/CUDA/Metal decision per device and window | In-ring profiles drive per-window CPU/CUDA/Metal offload | PRESENT |
| Model loading | Load only assigned layer windows from a controller-distributed full GGUF | Controller transfers and checksums the full model; workers load assigned windows only | PRESENT |
| Worker bootstrap | Automatic local and remote startup with readiness and topology lifecycle | Local/SSH launch, readiness, checksum reuse, and startup rebuild callback are implemented | PARTIAL |
| Continuous batching | Active HTTP requests scheduled together through the ring | Scheduler merges prompt and decode rows into one ring pass | PRESENT |
| Conversation slots | Isolated state, identity, cache affinity, cancellation, and lifecycle | Bounded slots with per-sequence KV, cancellation, and reuse | PRESENT |
| Client server | OpenAI-compatible contract on the ring head | Completion/chat, streaming, models, health, sampling, and explicit field rejection | PARTIAL |
| Resilience | Reconnect, ring rebuild, migration, and safe retry | One reset/reprobe/relaunch rebuild with 30-second backoff; migration/retry are missing | PARTIAL |
| Security | Authenticated and encrypted deployment with privacy controls | SSH trust is scoped; ring authentication, encryption, and privacy controls are unfinished | MISSING |

The local two-worker CPU smoke passed. An M4 head also automatically discovered
one Linux CPU worker through DNS-SD, accepted its key in the Potluck-specific
SSH trust file, launched it, and returned the two-token completion ` located in`.
The server logged windows `[0,12)` and `[12,24)`. The remote smoke did not
demonstrate heterogeneous placement or head computation.

The three-device fixture smoke now discovers and admits the M1 Metal worker and
the Linux CUDA worker automatically. It formed four windows `[0,6)`, `[6,12)`,
`[12,18)`, and `[18,24)`, loaded two windows on each worker, and returned
` Paris.\n` for a three-token completion. Worker logs report Metal on the M1
and CUDA on the PC, with no tensor-buffer failure. The head reserved its
resources and did not participate in this run. These checks prove discovery,
heterogeneous placement, model distribution, and the direct server path; they
do not prove prefetch, adaptive head participation, migration, or security.

## 1. Current potluck server contract

### Startup options currently implemented

`potluck-server` currently accepts this small set of bootstrap and inference
options:

| Option family | Current behavior | Status |
|---|---|---:|
| `-m`, `--model` | Model metadata source for the direct server route | PRESENT component |
| `--workers N` | Explicit local engineering input when DNS-SD is not used | PRESENT component |
| `--hosts`, `--launch ssh` | Explicit SSH bootstrap; normal startup also uses DNS-SD | PRESENT component |
| `--host`, `--port` | Bind the client-facing HTTP server | PRESENT component |
| `--ctx` | Set the cluster context size | PRESENT component |
| `--batch`, `--ubatch`, `--slots` | Set sequence, microbatch, and slot capacities | PRESENT component |
| `--head-share` | Enable automatic or disabled head participation | PRESENT component |
| `--temp`, `--top-p`, `--seed` | Set process-wide sampler defaults | PRESENT component |
| `--n-predict` | Set a process-wide generation limit | PRESENT component |
| `--bench` | Run a startup benchmark | PRESENT component; not a release gate |

Local ports and cyclic peer endpoints are allocated by the controller. Normal
startup hides topology, placement, profiling, and execution details.
Engineering bootstrap flags remain for deterministic local and explicit-host
checks.

### Request fields currently implemented

| Request | Accepted fields | Behavior that is not present |
|---|---|---|
| `POST /completion` | `prompt`, `n_predict`, `stream`, `temperature`, `top_p`, `top_k`, `seed` | Token arrays, mixed prompt parts, multiple prompts, cache reuse, stop strings, grammar, probabilities, timings, token returns, multimodal data |
| `POST /v1/chat/completions` | `messages`, `max_tokens`, `stream`, `reasoning_effort`, `temperature`, `top_p`, `top_k`, `seed` | Full OpenAI chat fields, model validation, `n`, stop, response formats, tools, multimodal content, reasoning parsing/control, cancellation |

Unknown JSON fields are rejected explicitly with HTTP 400.

### Response fields currently implemented

| Mode | Current response | Missing baseline behavior |
|---|---|---|
| Non-stream `/completion` | `content`, `n_predict`, `finish_reason: "stop"` | Prompt, token IDs, stop type/word, generation settings, timings, cache counts, truncation, probabilities, detailed usage |
| Stream `/completion` | Raw SSE `data: {"content": "..."}` plus `[DONE]` | Token metadata, stop field, progress, pings, response options, standard completion chunk shape |
| Non-stream chat | `id`, `object`, `created`, `model`, one assistant message, `finish_reason`, basic `usage` | `n`, detailed usage, timings, reasoning content, tool calls, model metadata, system fingerprint, response-format metadata |
| Stream chat | Role delta, content deltas, final stop delta, `[DONE]` | Tool/reasoning deltas, usage chunk, token/probability fields, progress and ping events, cancellation |

The current ID and timestamp are generated from wall-clock seconds. There is
no request registry or durable completion identity for later control or
cancellation.

## 2. HTTP endpoint matrix

The following is the endpoint-level compatibility inventory. "Missing" means
the route is not registered by `potluck-server`, not that the upstream route is
always useful for every model.

| Endpoint family | llama.cpp / prima capability | Potluck status | Required work or boundary |
|---|---|---:|---|
| `GET /health`, `GET /v1/health` | Public readiness endpoint; 503 while loading and 200 when ready | PARTIAL | `/health` is present only after startup; add `/v1/health` and loading/degraded states |
| `POST /completion` | Rich llama.cpp completion API | PARTIAL | Keep route, add request/response contract or return explicit unsupported errors |
| `POST /v1/completions` | OpenAI-compatible text completion | MISSING | Add route and OpenAI completion response/SSE shape |
| `POST /v1/chat/completions` | OpenAI chat, streaming and broad request fields | PARTIAL | Add field validation, sampling, structured output, tools, reasoning, multimodal, cancellation |
| `GET /v1/models` | Model metadata, aliases, capabilities | PARTIAL | Add metadata/capability fields and aliases; add multi-model routing only if selected |
| `POST /tokenize` | Token IDs and optional pieces | MISSING | Add text/token input options and piece-safe output |
| `POST /detokenize` | Convert token IDs to text | MISSING | Add route using the coordinator vocabulary |
| `POST /apply-template` | Render chat messages without inference | MISSING | Expose the existing template renderer |
| `POST /embedding` | Native embedding endpoint | MISSING | Add an embedding graph and distributed embedding transport |
| `POST /embeddings` | Native all-token/pooled embeddings | MISSING | Add pooling, normalization, and response format |
| `POST /v1/embeddings` | OpenAI embeddings | MISSING | Add input arrays, encoding format, usage, and embedding model validation |
| `POST /rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | Reranking model endpoint | MISSING | Add rank pooling and distributed reranker support |
| `POST /infill` | FIM/code infill | MISSING | Add FIM token construction, extra files, and completion options |
| `GET /props` | Inspect model, template, modalities, defaults, build info | MISSING | Add read-only property snapshot |
| `POST /props` | Change selected global properties when enabled | MISSING | Define safe mutable properties and concurrency rules |
| `GET /slots` | Per-slot state, parameters, and timing | MISSING | Add slot model before exposing this route |
| `POST /slots/{id}?action=save` | Save slot KV/prompt cache | MISSING | Define portable distributed slot serialization |
| `POST /slots/{id}?action=restore` | Restore slot KV/prompt cache | MISSING | Validate model file, topology, and context compatibility |
| `POST /slots/{id}?action=erase` | Erase slot state | MISSING | Add slot lifecycle |
| `GET /metrics` | Prometheus request/token/speculative metrics | MISSING | Add counters, gauges, and labels without exposing private prompt data |
| `GET /lora-adapters`, `POST /lora-adapters` | Inspect/apply global LoRA adapters | MISSING | Add adapter distribution and synchronized worker state |
| `POST /v1/cancel` | Cancel a task | MISSING | Add request registry, cancellation propagation, and chain reset |
| `POST /v1/chat/completions/control` | End reasoning on an active stream | MISSING | Add active-stream control and a stable request ID |
| `POST /v1/responses` | OpenAI Responses API | MISSING | Add request translation and response-item semantics |
| `POST /v1/responses/input_tokens` | Responses token counting | MISSING | Add count-only path |
| `POST /v1/chat/completions/input_tokens` | Chat token counting | MISSING | Add template-aware count-only path |
| `POST /v1/messages` | Anthropic Messages API | MISSING | Add request/stream/content-block translation |
| `POST /v1/messages/count_tokens` | Anthropic token counting | MISSING | Add count-only path |
| `/tools/*` | llama-server built-in tool API | MISSING | Add only with an explicit security model; never expose by default on a LAN |
| `/` and static assets | llama-server web UI | MISSING | Add a UI only if it is a project requirement |
| Router/model-management routes | Multiple models, load/unload, model presets | MISSING | Add a router process or explicitly keep one-model scope |

## 3. Request, sampling, and output feature families

### Prompt and context behavior

| Feature family | llama.cpp / prima behavior | Potluck status |
|---|---|---:|
| String prompt | Supported | PRESENT |
| Token-ID prompt | Supported | MISSING |
| Mixed string/token prompt | Supported | MISSING |
| Multiple prompts / `n_cmpl` | Supported | MISSING |
| BOS/EOS and special-token controls | Request/model metadata controls | PARTIAL; tokenizer defaults only |
| Chat-template rendering | Jinja/common template rendering | PARTIAL; internal only, no `/apply-template` or full kwargs |
| Prompt cache reuse | Common-prefix KV reuse | MISSING |
| Conversation ID / `id_slot` | Stable slot affinity | MISSING |
| Context shift and `n_keep` | Continue beyond context with controlled eviction | MISSING |
| Context checkpoints / cache RAM limits | Slot checkpoint and cache eviction controls | MISSING |
| KV cache type/offload/unified cache | Configurable KV storage | MISSING in distributed server |
| Warmup and cache idle behavior | Server startup/warmup and idle-cache policy | MISSING |
| Token count helpers | Count without generation | MISSING |

### Sampling and generation controls

The server currently wires only process-wide `temp`, `top_p`, and `seed` into
the worker configuration. It does not expose these controls per request.

| Feature family | Included baseline controls | Potluck status |
|---|---|---:|
| Temperature, top-p, top-k, min-p | Basic distribution controls | PARTIAL; temp/top-p startup defaults only |
| Seed and sampler order | Reproducibility and ordered sampler chain | PARTIAL; seed startup only; fixed worker sampler |
| Typical, top-n-sigma, adaptive-p, dynamic temperature | Advanced distribution controls | MISSING |
| Mirostat | Mirostat 1/2, tau, eta | MISSING |
| Repeat/presence/frequency penalties | Repetition control | MISSING |
| DRY and XTC | Repetition and tail-crossing samplers | MISSING |
| Ignore EOS and stop strings | Generation termination controls | MISSING |
| Logit bias | Token/string bias and bans | MISSING |
| Grammar / JSON schema | Constrained decoding | MISSING |
| Backend sampling | Backend-side sampler execution | MISSING |
| `n`, `n_predict`, `max_tokens` | Completion count and token limit | PARTIAL; one count, route-specific limit |
| Time limits and indentation | FIM/time-bounded generation controls | MISSING |
| Request LoRA scales | Per-request adapter selection | MISSING |
| Speculative decoding | Draft model, draft count, acceptance settings | MISSING; no product-server integration |
| Lookup decoding | Static/dynamic lookup caches | MISSING |

### Structured output, tools, and modalities

| Feature family | llama.cpp / prima behavior | Potluck status |
|---|---|---:|
| JSON object/schema response format | Grammar generated from request schema | MISSING |
| Function/tool calls | Native and generic tool-call parsing | MISSING |
| Parallel tool calls | Multiple tool calls in one response | MISSING |
| Reasoning output parsing | Separate reasoning/content fields and control | PARTIAL; only disables thinking when `reasoning_effort` is `"none"` |
| Image/audio/video input | MTMD projector and typed OpenAI content | MISSING |
| Media path and remote/base64 media | Server media loading policy | MISSING |
| FIM/infill | Prefix/suffix/middle and repository context | MISSING |
| Embedding pooling | None/mean/CLS/last/rank and normalization | MISSING |
| Reranking | Query/document rank model | MISSING |

## 4. State, scheduling, and operations

### Current behavior

- A request acquires one of the bounded conversation slots.
- The scheduler batches prompt rows and decode rows from active slots through
  one ring pass, with one sequence and sampler per slot.
- Streaming requests receive isolated token pieces and can cancel their slot.
- A transport or worker error fails active requests with
  `cluster is rebuilding; retry`, resets reachable workers, and attempts one
  reprobe/relaunch cycle. New requests receive 503 while rebuilding.
- `/health` reports worker, window, and slot state; `--bench` prints one
  startup result and is not a live metrics API.

### Missing normal-server behavior

| Capability | llama.cpp / prima behavior | Potluck status |
|---|---|---:|
| Multiple independent slots | Bounded parallel conversations | PRESENT |
| Queue and admission control | Fair queue, backpressure, deferred work | PARTIAL; bounded slot wait |
| Continuous/dynamic batching | Merge active slots into decode batches | PRESENT |
| Per-request cancellation | Stop one task without stopping other users | PRESENT for streams |
| Stream isolation | One stream must not block unrelated slots | PRESENT |
| Prompt-cache slot affinity | Reuse prefix on the same conversation | MISSING |
| Context eviction | Predictable limit, shift, and eviction policy | MISSING |
| Graceful shutdown | Stop accepting work, cancel tasks, close workers, reap children | PARTIAL |
| Worker health and reconnect | Heartbeats, failure detection, reconnect, topology rebuild | PARTIAL; bounded rebuild |
| Request retry semantics | Safe retry only before a token is committed | MISSING |
| Per-request usage/timings | Prompt/decode counters and timings | PARTIAL; basic usage only |
| Prometheus metrics | Queue, slot, token, and speculative counters | MISSING |
| Slot save/restore/erase | Persistent KV state | MISSING |
| Model loading states | Listen while loading and report 503 | MISSING; startup blocks before listen |
| Runtime properties | Inspect/change safe global defaults | MISSING |

Prompt-cache reuse, context eviction, request retry, and metrics are separate
server features. They must not reintroduce the old single-chain reset path.

## 5. llama.cpp runtime and deployment features not exposed by potluck-server

The current llama.cpp server documents these runtime families. Potluck's
distributed worker path does not expose them as server options or carry them in
the worker configuration:

| Runtime family | Examples from llama-server | Potluck status |
|---|---|---:|
| CPU execution | Threads, batch threads, CPU masks/ranges, priority, polling | MISSING; worker uses process defaults |
| Context execution | Batch/ubatch sizing, keep, SWA, flash attention, perf switches | PARTIAL; fixed/internal context choices |
| RoPE/context scaling | RoPE type/scale/base/frequency and YaRN controls | MISSING |
| KV storage | KV offload, K/V types, defrag, unified KV, cache RAM/checkpoints | MISSING |
| Model loading | mmap/mlock/direct I/O/load mode, NUMA, tensor checks/overrides | MISSING or fixed by worker |
| Device selection | Device list, tensor buffer overrides, split mode, tensor split, main GPU, fit | PARTIAL; one derived per-window GPU layer count |
| MoE controls | CPU MoE and per-layer CPU MoE | MISSING |
| Model acquisition | Local path only | MISSING URL, Hugging Face, Docker source, cache, offline policy |
| Adapters | LoRA and control vectors | MISSING |
| Logging | Log file, levels, colors, timestamps, prefixes | PARTIAL; basic process output only |
| HTTP runtime | HTTP thread count, timeout, SSE ping interval, body/API controls | MISSING or library defaults |
| CORS | Configurable origins, methods, headers, credentials | MISSING; wildcard headers are hard-coded |
| TLS and authentication | SSL certificate/key, API key/file | MISSING |
| Static web UI | Configurable static path, UI configuration | MISSING |
| Agent/MCP tools | Built-in tools, MCP proxy/config, tool runtime | MISSING and intentionally unsafe to enable by default |
| Model router | Models directory/presets, load/unload, multi-model dispatch | MISSING |

Some of these controls are low-value for a first distributed release. They
remain gaps whenever the product claim is "use potluck like llama-server".

## 6. Prima.cpp distributed-runtime comparison

Prima's distributed runtime is a separate baseline from its HTTP API. Potluck
now implements the core automatic controller path, while live load adaptation,
prefetch, migration, and full API parity remain incomplete.

| Prima capability or option family | Potluck implementation | Status | Notes |
|---|---|---:|---|
| Multi-device layer pipeline | Direct adjacent-peer ring with repeated disjoint windows sized from live capacity | PRESENT | Startup route is heterogeneous and resource weighted |
| Full-model greedy reference | Optional test-only fixture comparison | TEST ONLY | Correctness evidence; never product loading |
| Piped-ring execution | `potluck-server` workers connect directly to cyclic next peers over ZeroMQ | PRESENT | Head ingress goes to rank 0 and final results return to the head |
| Worker discovery/peer topology | DNS-SD candidates, scoped SSH trust-on-first-use, SSH launch, and cyclic endpoints | PARTIAL | Admission and bounded rebuild exist; live migration remains missing |
| `--world`, `--rank`, `--master`, `--next`, `--data-port` | Not user-facing server options | INTENTIONAL | Configure equivalent direct-ring topology internally |
| Layer-window scheduling | Pre-launch probes and proportional usable-capacity route | PRESENT | Recomputed on startup and rebuild |
| Prima `--lw` layer-window weights | No user-facing option | INTENTIONAL | Manual window weights are not a product control |
| GPU offload | In-ring profiles choose independent CPU/CUDA/Metal window placement | PRESENT | Placement is bounded by live usable memory |
| Device profiling | DNS-SD and local/SSH workers report pre-launch and in-ring profiles | PRESENT | Failed or insufficient candidates are excluded |
| LP placement | No solver; deterministic capacity-weighted route | INTENTIONAL | A solver is not required for the current product path |
| Prefetch | No coordinated per-window prefetch | MISSING | Implement prefetch in ring scheduling |
| Scheduler cycles | Route repeats disjoint windows for a second cycle | PRESENT | Integrated with slot batching |
| Force-prefetch policy | No user mode | INTENTIONAL | Prefetch must be automatic |
| Master priority | Startup head reserve and automatic participation | PARTIAL | Does not adapt to changing user load |
| Runtime device removal | Reset, reprobe, relaunch, and 30-second rebuild backoff | PARTIAL | No token-state migration or safe retry |
| Speculative decoding | No integrated product-server path | MISSING | Separate experiments do not satisfy the contract |
| Dynamic batching | HTTP slots merge prompt/decode rows through one ring pass | PRESENT | Bounded slot wait remains |
| Sampler breadth | Tail supports temperature, top-p, top-k, and seed per slot | PARTIAL | Not the full prima/llama sampler chain |
| Model copy behavior | Controller distributes and checksums the full GGUF on admitted devices | PARTIAL | Automatic transfer works; per-window prefetch remains missing |
| Transport and data topology | Direct adjacent-peer ZeroMQ ring with a separate final-result path | PRESENT | No head relay for windows the head does not execute |

Prima-specific flag aliases are not required. Their behavioral goals are
binding: automatic topology setup, heterogeneous ring-window placement,
per-window prefetch, per-device accelerator placement, device selection,
continuous batching, conversation slots, and recovery must operate through
`potluck-server`. Explicit worker, host, and placement inputs are not the
replacement product contract.

## 7. Security (deferred) and deployment gaps

Security remains unfinished. The initial product assumes a trusted home LAN,
but that assumption does not provide authentication, encryption, or tenant
isolation. Deployment reliability is a separate release requirement: discovery,
capability reporting, model transfer, process startup, health, and recovery
must work without manual network administration.

| Boundary | Current behavior | Status |
|---|---|---:|
| HTTP authentication | None | MISSING |
| HTTP encryption | No TLS options | MISSING |
| Worker authentication | ZeroMQ sockets have no identity or credential authentication configured | MISSING |
| Worker encryption | ZeroMQ payloads are unencrypted; no security configuration is enabled | MISSING |
| Model-file authorization | A worker validates its assigned model/window contract but not caller identity | PARTIAL |
| CORS | `Access-Control-Allow-Origin: *` and fixed methods/headers | PARTIAL; not safe for credentialed clients |
| Remote launch | Explicit batch-mode SSH bootstrap assumes trusted hosts and remote directories | PARTIAL |
| Firewall guidance | Manual tunnel/workaround is documented | PARTIAL |
| Secrets and API keys | No storage, rotation, or redaction policy | MISSING |
| Prompt/privacy controls | No request logging policy or tenant isolation | MISSING |

Keep the current "never expose the service to the public Internet" warning.
Do not call the trusted-LAN assumption a security implementation; API keys,
TLS, worker authentication, encryption, and privacy controls remain unfinished.


## 8. Model and backend scope

The distributed graph currently targets the Qwen3.5 dense `qwen35` path.
Ordinary llama.cpp binaries in the same checkout support many more model
architectures and modalities, but that support is not automatically inherited
by a layer-window graph.

| Scope item | Potluck status |
|---|---:|
| Qwen3.5 dense text generation | PRESENT; 0.8B fixture is the acceptance target |
| Other text architectures | INHERITED only; no distributed-window claim |
| Quantization breadth | PARTIAL; no complete distributed quantization matrix |
| Embedding-only models | MISSING in distributed server |
| Reranker models | MISSING in distributed server |
| Vision-language/audio/video models | MISSING in distributed server |
| LoRA/control-vector model variants | MISSING |
| Multi-model router | MISSING |
| 27B correctness/performance/end-to-end acceptance | INTENTIONAL non-goal |

"Works in llama.cpp" and "works in a potluck distributed window" are separate
claims. A new architecture requires a graph/window implementation and a
named acceptance check before it can be listed as supported.

## 9. Required implementation cutover

The direct ring core is implemented, but no intermediate state is a Potluck
product, supported configuration, or release. The remaining work is:

1. Keep the direct piped-ring runtime as the only distributed execution path
   and remove any remaining static routing, static bounds, and static tests.
2. Extend discovery with live profiling, device admission, and heterogeneous
   window assignment in the server lifecycle. Include current head CPU, memory,
   accelerator pressure, and a user resource reserve.
3. Ensure the controller distributes and checksums the full GGUF on each
   admitted device. Workers must load only their assigned layer windows from
   that file; normal startup must not require manually partitioned model files.
4. Implement coordinated per-window prefetch and independent per-device,
   per-window CPU/CUDA/Metal placement.
5. Connect sequence IDs to bounded conversation slots and a continuous batch
   scheduler that drives the same ring.
6. Make the head's OpenAI-compatible request, response, error, cancellation,
   usage, and streaming behavior operate through that scheduler.
7. Extend local and SSH bootstrap with automatic readiness, resource changes,
   topology rebuild, and clear recovery without exposing ranks, hosts, ports,
   bounds, or weights.
8. Ship controller and worker artifacts and a simple local interface that
   forms the complete server without a source build.

Manual topology, static execution, standalone ring/batch tools, and manually
applied profiling are removal targets. They must not remain in the product as
expert, diagnostic, compatibility, or fallback modes.

## 10. Product release gate

Potluck can ship only when named end-to-end checks prove all of the following
through one server process:

1. The head discovers and profiles the available computers without manual
   workers, ranks, bounds, or weights.
2. The scheduler selects devices and assigns repeated ring windows from
   measured heterogeneous capability and current resource pressure.
3. Head computation preserves an explicit user reserve and adapts when the
   user's CPU, memory, or accelerator use changes.
4. Every request uses direct peer-to-peer piped-ring execution over ZeroMQ; the
   head does not relay unowned windows, and no static or alternate distributed
   execution or transport path exists in the product.
5. Workers load only their assigned layer windows from a full GGUF. The
   controller may distribute the complete source file without loading it as
   the worker model.
6. Per-window prefetch and per-device CPU/CUDA/Metal placement operate in the
   ring scheduler.
7. Multiple conversation slots retain isolated state and cache affinity.
8. Continuous batching serves concurrent streaming requests without global
   429 serialization.
9. Ordinary OpenAI-compatible clients can use the head endpoint with the
   documented fields, errors, usage, streaming, and cancellation behavior.
10. A restart reuses valid model files and placement data, while device or
    resource changes cause a clear automatic re-profile and topology result.

Component checks, a standalone ring CLI, a static server, or a manual cluster
cannot satisfy this gate.

## 11. Definition of "regular local inference server"

Potluck is a regular local inference server only when the complete release gate
above passes. In particular, named checks must cover:

1. two isolated multi-turn conversations with cache affinity;
2. concurrent streaming through continuous ring batching;
3. OpenAI-compatible chat/completion request, response, usage, sampling, stop,
   error, and cancellation behavior;
4. automatic discovery, live profiling, selection, and resource-aware
   placement, including a busy head machine;
5. assigned-window loading, per-window prefetch, and per-device accelerator
   placement;
6. worker and topology change handling without a hang.

The existing distributed component checks remain useful evidence for the
individual mechanisms they exercise. They do not establish product completion,
and static checks must not survive the architecture cutover.
