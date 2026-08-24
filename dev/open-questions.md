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
per-device CPU/CUDA/Metal placement, continuous batching, isolated
conversation slots, speculative decoding, HiGHS-solved HALDA placement,
quantized-model support, and a potluck completion CLI beside the server
endpoint ([ADR 0010](decisions/0010-prima-feature-parity-baseline.md)).
These requirements are one release gate. The integrated server now implements
profiling, admission, HALDA placement, distribution, per-window prefetch,
continuous batching, slots, speculative decoding, and the completion CLI. The
remaining status below separates implemented behavior from release evidence
and broader llama-server API coverage.

Static execution, manual placement that displaces the automatic scheduler,
one-request serving, and separate CLI implementations of ring or batching are
conflicting legacy code. They must be removed through a clean cutover, not
preserved as diagnostics, fallbacks, compatibility modes, experiments, or
provisional releases. The optional expert workload override in
[ADR 0010](decisions/0010-prima-feature-parity-baseline.md) is not such a
displacement; automatic operation stays the default.

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

The ADR 0010 feature baseline is implemented in the integrated `potluck-server`
and `potluck-cli` runtime. This matrix tracks remaining compatibility scope and
release evidence; `PRESENT` does not mean that every large acceptance scenario
has been rerun on every platform.

| Area | Required product behavior | Current source | Status |
|---|---|---|---:|
| Distributed execution | Piped ring only; several disjoint windows per selected device | `potluck-server` direct adjacent-peer ZeroMQ ring with repeated PRP windows | PRESENT |
| Window placement | Automatic heterogeneous placement from live capability and resource pressure | HiGHS-backed HALDA placement with live profiles and solver-owned exclusion | PRESENT |
| Device profiling and selection | Automatic discovery, profiling, admission, and exclusion in the server lifecycle | DNS-SD candidates are profiled before launch; hard failures are rejected and HALDA removes weak devices | PRESENT |
| Head participation | Optional worker after reserving for current user CPU, memory, and accelerator use | Head reserve and participation are part of profile refresh and placement | PRESENT |
| Prefetch | Next assigned window coordinated with ring execution | `off`, `advise`, and `force` modes advise the next mapped window after send | PRESENT |
| Accelerator placement | Independent CPU/CUDA/Metal decision per device and window | HALDA and in-ring profiles select per-window offload | PRESENT |
| Model loading | Load only assigned layer windows from a controller-distributed full GGUF | One digest-checked full model per device; workers map assigned windows only | PRESENT |
| Worker bootstrap | Automatic local and remote startup with readiness and topology lifecycle | Local/SSH launch, readiness, discovery, checksum reuse, and bounded rebuild are implemented | PRESENT |
| Continuous batching | Active HTTP requests scheduled together through the ring | Scheduler merges prompt and decode rows into one ring pass | PRESENT |
| Conversation slots | Isolated state, identity, cache affinity, cancellation, and lifecycle | Bounded slots carry per-sequence KV and speculative state | PRESENT |
| Client server | OpenAI-compatible contract on the ring head | Completion/chat, streaming, models, health, sampling, errors, auth, and explicit field rejection | PARTIAL |
| Resilience | Reconnect, ring rebuild, migration, and safe retry | Bounded reset, reprobe, relaunch, and retryable failure; token migration remains missing | PARTIAL |
| Security | Authenticated and encrypted deployment with privacy controls | CURVE ring credentials, bearer HTTP auth, exact-origin CORS, and trusted-LAN boundary | PARTIAL |

The local integrated suite and the supervised 27B run provide named evidence
for the direct ring, HALDA placement, full-model loading, per-window
prefetch, batching, slots, speculative decoding, quantized inference, and
the HTTP server. The 27B operating point is recorded in
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

These checks do not establish every broader llama-server endpoint or token
migration behavior. The release-gate list below remains the authoritative
record of scenarios that still need platform-specific evidence.

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
| `POST /v1/completions` | `model`, `prompt`, `n`, `max_tokens`, `max_completion_tokens`, `stop`, `stream`, sampling controls, penalties, and logprobs | Full OpenAI completion fields, prompt arrays, streaming usage, and cancellation |
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
| `GET /health`, `GET /v1/health` | Public readiness endpoint; 503 while loading and 200 when ready | PARTIAL | `/health` reports loading, rebuilding, and failed states after startup; `/v1/health` and listen-while-loading remain missing |
| `POST /completion` | Rich llama.cpp completion API | PARTIAL | Keep route, add request/response contract or return explicit unsupported errors |
| `POST /v1/completions` | OpenAI-compatible text completion | PARTIAL | Route and non-stream response/logprobs shape exist; add the remaining OpenAI fields and streaming parity |
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
| `GET /props` | Inspect model, template, modalities, defaults, build info | PARTIAL | Read-only model, template, slot, and health properties exist; add the remaining metadata fields |
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
| Temperature, top-p, top-k, min-p | Basic distribution controls | PRESENT |
| Seed and sampler order | Reproducibility and ordered sampler chain | PARTIAL; seed is request-local, sampler order is fixed |
| Typical, top-n-sigma, adaptive-p, dynamic temperature | Advanced distribution controls | MISSING |
| Mirostat | Mirostat 1/2, tau, eta | MISSING |
| Repeat/presence/frequency penalties | Repetition control | PRESENT |
| DRY and XTC | Repetition and tail-crossing samplers | MISSING |
| Ignore EOS and stop strings | Generation termination controls | PARTIAL; stop strings are request-local, ignore-EOS is missing |
| Logit bias | Token/string bias and bans | MISSING |
| Grammar / JSON schema | Constrained decoding | MISSING |
| Backend sampling | Backend-side sampler execution | MISSING |
| `n`, `n_predict`, `max_tokens` | Completion count and token limit | PRESENT |
| Time limits and indentation | FIM/time-bounded generation controls | MISSING |
| Request LoRA scales | Per-request adapter selection | MISSING |
| Speculative decoding | Draft model, draft count, acceptance settings | PRESENT; draft state and CLI/server options are integrated |
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
| Runtime properties | Inspect/change safe global defaults | PARTIAL; read-only `/props` is present |

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
| Model acquisition | Local path, Hugging Face repo/file, cache, and offline mode | PARTIAL; Docker source and broader acquisition policy are missing |
| Adapters | LoRA and control vectors | MISSING |
| Logging | Log file, levels, colors, timestamps, prefixes | PARTIAL; basic process output only |
| HTTP runtime | HTTP thread count, timeout, SSE ping interval, body/API controls | MISSING or library defaults |
| CORS | Configurable origins, methods, headers, credentials | PRESENT; one exact configured origin is supported |
| TLS and authentication | SSL certificate/key, API key/file | PARTIAL; API-key auth is present, TLS remains external |
| Static web UI | Configurable static path, UI configuration | MISSING |
| Agent/MCP tools | Built-in tools, MCP proxy/config, tool runtime | MISSING and intentionally unsafe to enable by default |
| Model router | Models directory/presets, load/unload, multi-model dispatch | MISSING |

Some of these controls are low-value for a first distributed release. They
remain gaps whenever the product claim is "use potluck like llama-server".

## 6. Prima.cpp distributed-runtime comparison

Prima's distributed runtime is a separate baseline from its HTTP API. Potluck
implements the required automatic controller path in one integrated runtime.
The table records compatibility boundaries and remaining evidence, not
alternate execution modes.

| Prima capability or option family | Potluck implementation | Status | Notes |
|---|---|---|---|
| Multi-device layer pipeline | Direct adjacent-peer ring with repeated disjoint windows sized from live capacity | PRESENT | HALDA solves the startup route |
| Full-model greedy reference | Optional test-only fixture comparison | TEST ONLY | Correctness evidence; never product loading |
| Piped-ring execution | `potluck-server` workers connect directly to cyclic next peers over ZeroMQ | PRESENT | Head ingress goes to rank 0 and final results return to the head |
| Worker discovery/peer topology | DNS-SD candidates, scoped SSH trust-on-first-use, SSH launch, and cyclic endpoints | PARTIAL | Rebuild is bounded; token migration remains missing |
| `--world`, `--rank`, `--master`, `--next`, `--data-port` | Not user-facing server options | INTENTIONAL | Configure equivalent direct-ring topology internally |
| Layer-window scheduling | HiGHS-backed HALDA placement from live profiles | PRESENT | Recomputed on startup and rebuild |
| Prima `--lw` layer-window weights | Optional `-lw` / `--layer-window` expert override | PRESENT | Automatic scheduling remains the default |
| GPU offload | Independent CPU/CUDA/Metal placement per device and window | PRESENT | Placement is bounded by live usable memory |
| Device profiling | DNS-SD and local/SSH workers report pre-launch and in-ring profiles | PRESENT | Solver owns weak-device exclusion |
| HALDA placement | Deterministic HiGHS MILP with fixed options and fixed-point set classification | PRESENT | Exact fixture coverage is in `test-potluck-halda` |
| Prefetch | `off`, `advise`, and `force` per-window modes | PRESENT | The next mapped window is advised after send |
| Scheduler cycles | Route repeats disjoint windows for each scheduled batch | PRESENT | Integrated with slot batching |
| Force-prefetch policy | `--prefetch force` and `--force` | PRESENT | Forced mode is explicit expert control |
| Master priority | `--master-priority` with automatic head reserve | PRESENT | Live refresh can remove or restore head work |
| Runtime device removal | Reset, reprobe, relaunch, and bounded rebuild backoff | PARTIAL | Token-state migration and safe retry remain missing |
| Speculative decoding | Draft state, tail verification, accepted-count propagation, and CLI/server options | PRESENT | Temperature-zero parity is covered by the fixture smoke |
| Dynamic batching | HTTP slots merge prompt/decode rows through one ring pass | PRESENT | Bounded slot wait remains |
| Sampler breadth | Tail supports temperature, top-p, top-k, and seed per slot | PARTIAL | Full prima/llama sampler breadth is outside this cutover |
| Model copy behavior | Controller distributes and checksums one complete GGUF per admitted device | PRESENT | Workers load only assigned windows |
| Transport and data topology | Direct adjacent-peer ZeroMQ ring with a separate final-result path | PRESENT | No head relay for windows the head does not execute |
| Completion CLI mode | `potluck-cli` shares the server's ring lifecycle and runtime | PRESENT | Supports one-shot and interactive conversation |
| Platform support | macOS and Linux builds and discovery adapters | PRESENT | Windows remains roadmap |
| Quantized-model support | Quantized GGUF loading and fixture inference | PRESENT | Breadth beyond the named fixtures remains evidence work |

Prima-specific flag aliases are not required. Their behavioral goals are
binding: automatic topology setup, heterogeneous ring-window placement,
per-window prefetch, per-device accelerator placement, device selection,
continuous batching, conversation slots, and recovery must operate through
`potluck-server`. Explicit worker, host, and launch inputs stay removed; the
optional workload override accepted by
[ADR 0010](decisions/0010-prima-feature-parity-baseline.md) is not a competing
product contract.

## 7. Security and deployment gaps

The accepted trusted-LAN security baseline is implemented. It protects direct
ring peers with ephemeral CURVE credentials and protects HTTP routes with an
optional bearer key. TLS for public-network HTTP and tenant isolation remain
outside the baseline.

| Boundary | Current behavior | Status |
|---|---|---:|
| HTTP authentication | Optional bearer key on every route, including health and options | PRESENT |
| HTTP encryption | No built-in TLS; external TLS is required beyond a trusted LAN | INTENTIONAL |
| Worker authentication | Ephemeral ZeroMQ CURVE credentials per ring generation | PRESENT |
| Worker encryption | CURVE encrypts direct peer payloads | PRESENT |
| Model-file authorization | Workers validate the assigned model/window contract | PARTIAL |
| CORS | One exact configured origin; no wildcard or credentialed mode | PRESENT |
| Remote launch | Scoped SSH trust-on-first-use and one-shot credential bootstrap | PRESENT |
| Firewall guidance | Public-Internet exposure remains prohibited; external TLS is required | PRESENT |
| Secrets and API keys | API keys are validated and not written to logs or persistent model files | PARTIAL |
| Prompt/privacy controls | Trusted-LAN process boundary; tenant isolation and prompt policy remain missing | MISSING |

Keep the current "never expose the service to the public Internet" warning.


## 7.1 Accepted security architecture (ADR 0009)

This section preserves the historical proposal and rationale. Carl Stone
accepted the proposal and all four defaults on 2026-08-22. The binding decision
is [ADR 0009](decisions/0009-trusted-lan-curve-http-controls.md). This section
is not an unresolved question or a separate authorization surface.

### Threat boundary and default exposure

- The boundary is a trusted LAN plus the existing SSH host and user trust. A
  device admitted by discovery and SSH is trusted to run Potluck code. The
  accepted baseline does not protect a compromised head, worker, SSH account,
  or host OS.
- The HTTP service binds to all interfaces by default (`0.0.0.0`) as the
  accepted household-device usability default. `--host` can restrict it to
  loopback or another explicit address. Startup must state the bound address
  and that LAN clients can connect.
- `--api-key` is optional. When set, require `Authorization: Bearer <value>`
  for every HTTP route, compare without timing leaks, and never write the value
  to logs or persistent model or config files. Without it, trusted-LAN use
  stays unauthenticated.
- `--cors-origin` accepts one exact allowed origin. When omitted, send no
  CORS allow-origin header. Wildcard origins and credentialed CORS are not
  allowed. CORS is not authentication.
- Accepted endpoint policy: all routes, including `/health` and `/v1/models`,
  require the key whenever one is configured. No route is exempt.

### Direct-ring credentials

- Before launching workers, the head generates an ephemeral in-memory Z85
  ZeroMQ CURVE keypair for every selected ring peer. The head keeps the session
  credential set in memory only; it is not read from a model, config, or key
  file.
- The head passes each worker its own public and secret key plus the expected
  public key of its next peer through a one-shot bootstrap record on the
  existing authenticated SSH stdin before that worker joins the ring. The
  local peer uses an equivalent in-memory handoff. Secrets must not appear in
  shell arguments, environment variables, process listings, logs, readiness
  output, model files, persistent worker files, or build artifacts.
- Each direct adjacent-peer ZeroMQ connection uses CURVE peer authentication:
  the receiver is configured with its own server keypair and the sender is
  configured with its own client keypair plus the expected receiver public key.
  The head configures credentials but does not relay ring data.
- Missing or mismatched credentials fail closed. There is no unauthenticated
  ZeroMQ fallback, alternate transport, or bypass for local workers.

### Key lifecycle and failure behavior

- A credential generation covers one ring topology. Startup and every topology
  rebuild generate a fresh set, close sockets from the old generation, and
  reject stale keys. Recovery obtains fresh credentials through SSH before
  reconnecting a worker.
- The head never persists private keys. Workers keep only their assigned
  private key in process memory and erase it on clean shutdown; logs and error
  paths must redact all key material.
- An SSH bootstrap failure, peer authentication failure, or key-generation
  failure excludes the affected peer and leaves the ring in a rebuilding or
  security-bootstrap-failed health state. A non-streaming in-flight request
  fails with a retryable error. A streaming request emits a terminal SSE error,
  and the client must discard any partial output before retrying.
- CURVE protects ring traffic from unauthenticated LAN peers, but it does not
  make a compromised admitted host trustworthy. The HTTP API key is a bearer
  credential and does not encrypt HTTP traffic; public-Internet exposure
  remains prohibited.

### Resolved choices

The following four choices were previously listed as requiring explicit
acceptance. They are now resolved by Carl Stone's acceptance on 2026-08-22 and
are binding through ADR 0009. The prior alternatives remain recorded here to
preserve the historical context; the accepted defaults are:

1. **Endpoint coverage**: The prior alternative considered leaving `/health`
   available without a key for remote orchestration. The accepted policy is
   that every route requires a configured key, including `/health` and
   `/v1/models`.
2. **CORS shape**: The prior alternatives considered multiple origins and
   credentialed browser requests. The accepted shape is one exact allowed
   origin, with no wildcard and no credentialed CORS.
3. **SSH bootstrap carrier**: The prior alternative considered another
   SSH-confidential handoff. The accepted carrier is a one-shot bootstrap
   record on SSH stdin before daemonization; command-line and persistent-file
   handoff are prohibited. A local peer uses the equivalent in-memory handoff.
4. **HTTP encryption**: The prior alternative considered deferring TLS to a
   later decision. The accepted baseline has no built-in TLS for trusted-LAN
   HTTP; non-loopback deployment is not treated as Internet-safe without an
   external TLS boundary.

## 8. Model and backend scope

The distributed graph currently targets the Qwen3.5 dense `qwen35` path.
Ordinary llama.cpp binaries in the same checkout support many more model
architectures and modalities, but that support is not automatically inherited
by a layer-window graph.

| Scope item | Potluck status |
|---|---:|
| Qwen3.5 dense text generation | PRESENT; 0.8B fixture is the acceptance target |
| Other text architectures | INHERITED only; no distributed-window claim |
| Quantization breadth | PARTIAL; no complete distributed quantization matrix; required by ADR 0010 |
| Embedding-only models | MISSING in distributed server |
| Reranker models | MISSING in distributed server |
| Vision-language/audio/video models | MISSING in distributed server |
| LoRA/control-vector model variants | MISSING |
| Multi-model router | MISSING |
| 27B correctness/performance/end-to-end acceptance | PRESENT as a supervised operating point; not a general compatibility claim |

"Works in llama.cpp" and "works in a potluck distributed window" are separate
claims. A new architecture requires a graph/window implementation and a
named acceptance check before it can be listed as supported.

## 9. Required implementation cutover

The required implementation cutover is complete for the ADR 0010 baseline:

1. The direct piped-ring runtime is the only distributed execution path.
2. Discovery, live profiling, head reserve, HALDA admission, and heterogeneous
   window assignment run in the server lifecycle.
3. The controller distributes and checksums one complete GGUF per device, and
   workers load only assigned windows from it.
4. Per-window prefetch, per-device CPU/CUDA/Metal placement, HiGHS-backed
   HALDA allocation, continuous batching, slots, and speculative decoding
   share one runtime.
5. The server and completion CLI share discovery, profiling, worker launch,
   ring startup, scheduling, and shutdown.
6. Install scripts stage the worker artifacts and the local quick start does
   not require a compiler or manually partitioned model files.

Static routing, standalone ring/batch tools, the shard toolchain, and
manually-applied profiling are removed. Manual workload flags are expert
constraints inside the automatic solver; they are not a separate execution
path.

Remaining work is broader HTTP API parity, wider model and modality support,
token-state migration after worker changes, and platform-specific long-run
acceptance evidence.

## 10. Product release gate

Potluck can ship only when all ten named end-to-end checks below have their
pass conditions observed through one server process. The current release
evidence is partial: the local integrated suite and supervised 27B run cover
named product mechanisms, but they do not cover every release-gate scenario.
Resource costs are planning estimates for supervised acceptance.

ADR 0010 widens this gate further: named checks must also cover solved HALDA
placement, speculative decoding, quantized-model serving, the potluck
completion CLI, and the supported-platform claims.

### Evidence ledger for the 2026-08-24 completion run

- **RG-02:** `build/bin/potluck-server -m /Users/carlstone/models/gemma-3-27b-it-Q4_K_M.gguf --hosts carl@192.168.1.78,carlstone@192.168.1.72 --launch ssh --head-share auto --host 0.0.0.0 --port 18089 --n-predict 16 --slots 1 --ubatch 8 --ctx 2048 --prefetch advise --spec-type ngram-simple --spec-draft-n-max 4` on 2026-08-24 returned HTTP 200 through three automatic windows `[0,6)`, `[6,16)`, and `[16,62)`. The pre-fix chat request logged `drafted=0 accepted=0`; the post-fix targeted repeated-token request logged `drafted=12 accepted=12 accept-rate=1.000` through the same three-device route. **PROVEN for the exercised operating points.**
- **RG-03:** The same 27B command on 2026-08-24 logged changing head budgets and changing head-owned windows. It did not apply a controlled host load while a request was active. **UNPROVEN.**
- **RG-04:** `bash tests/potluck/run_all.sh` on 2026-08-24 passed the direct PRP trace checks; the 27B command above also logged three ring workers and adjacent-window PRP traffic before HTTP 200. **PROVEN for the exercised route.**
- **RG-05:** `bash tests/potluck/run_all.sh` on 2026-08-24 passed the no-shard and per-worker resident mapping checks; the 27B command above loaded one complete model path per worker and assigned windows. **PROVEN for the exercised route.**
- **RG-06:** `bash tests/potluck/run_all.sh` on 2026-08-24 passed per-window prefetch checks. The 27B logs show heterogeneous device assignment, but no single run proves every CPU, Metal, and CUDA combination in this gate. **UNPROVEN.**
- **RG-07:** `bash tests/potluck/run_all.sh` on 2026-08-24 passed the four-slot concurrency checks, but it did not exercise two isolated multi-turn conversations with follow-up affinity. **UNPROVEN.**
- **RG-08:** `bash tests/potluck/run_all.sh` on 2026-08-24 passed four concurrent streaming requests through one server and complete SSE streams. **PROVEN for the exercised local route.**
- **RG-09:** `bash tests/potluck/run_all.sh` on 2026-08-24 passed chat/completion, usage, streaming, stop, error, authentication, CORS, and shutdown-stream checks. It did not prove ordinary-client cancellation. **UNPROVEN.**
- **RG-10:** The 27B command on 2026-08-24 completed its initial request, but a later refresh hit `worker binary refresh failed` for the M1 SSH worker. Earlier benchmark mode recorded two successful refresh rebuilds, but this gate requires request-serving recovery. **UNPROVEN.**


1. **RG-01 - Automatic discovery and admission**
   - Scenario: Start one `potluck-server` with no worker, rank, bound, or
     weight configuration and at least two discoverable candidates, including
     one candidate that is unreachable or lacks required capacity. Submit a
     client request through the head.
   - Pass condition: The server probes and profiles candidates, excludes the
     unsuitable candidate, forms an admitted topology, and completes the
     request without manual topology input.
   - Resource cost: Medium - multiple discovered devices and one inference run.
   - Evidence status: UNPROVEN.

2. **RG-02 - Heterogeneous window placement**
   - Scenario: Run the same server with admitted peers that have deliberately
     different measured compute, memory, accelerator, and network capacity.
     Submit a request that traverses more than one ring window.
   - Pass condition: The observed route assigns repeated disjoint windows from
     those measurements and current pressure, with no equal split or manual
     weights, and the request completes through the route.
   - Resource cost: Large - heterogeneous peers and a multi-window model run.
   - Evidence status: PROVEN for this operating point.

3. **RG-03 - Adaptive head reserve**
   - Scenario: While the head owns ring work and serves a request, apply and
     remove controlled CPU, memory, or accelerator load on the head.
   - Pass condition: Live topology or placement evidence shows head-owned work
     shrinking or being removed under load and returning only when the explicit
     user reserve is safe, without an unsafe or hung request.
   - Resource cost: Large - sustained inference plus supervised host load.
   - Evidence status: UNPROVEN.

4. **RG-04 - Direct ZeroMQ ring**
   - Scenario: Start a three-peer server and submit a request that crosses
     windows while observing the server topology and peer connection records.
   - Pass condition: Adjacent peers exchange request state directly over
     ZeroMQ, the head handles ingress and the final result but does not relay
     unowned windows, and no static or alternate transport path is used.
   - Resource cost: Large - three peers and a captured multi-window request.
   - Evidence status: PROVEN for the exercised route.

5. **RG-05 - Full-model window loading**
   - Scenario: Make a complete GGUF available for controller distribution, start
     the multi-peer server, and submit a request through the head.
   - Pass condition: Each worker receives one stable complete model path, loads
     only tensors for its assigned layer windows, and completes the request; no
     worker loads the complete model into memory.
   - Resource cost: Large - model distribution, disk space, and memory checks.
   - Evidence status: PROVEN for the exercised route.

6. **RG-06 - Window prefetch and accelerator placement**
   - Scenario: Run a multi-window request across peers with CPU, CUDA, and
     Metal capabilities for at least two ring cycles.
   - Pass condition: The integrated route records an independent backend choice
     for each device and window, and each worker records prefetch of its next
     assigned window before that window's compute, without whole-file prefetch.
   - Resource cost: Large - heterogeneous accelerators and a repeated ring run.
   - Evidence status: UNPROVEN.

7. **RG-07 - Conversation slot affinity**
   - Scenario: Interleave multi-turn requests for two independent
     conversations through one head endpoint while both conversations remain
     active.
   - Pass condition: Each follow-up retains only its own prior context, keeps
     affinity to its assigned slot and cache, and completes without state
     leakage or cross-conversation output.
   - Resource cost: Medium - two active conversations and several turns.
   - Evidence status: UNPROVEN.

8. **RG-08 - Concurrent streaming batch**
   - Scenario: Open two concurrent streaming requests with different prompt
     lengths and overlapping decode work through one head endpoint.
   - Pass condition: Both streams produce interleaved progress through the same
     ring batch and finish normally, with no global 429 serialization or one
     stream blocking the other.
   - Resource cost: Large - two concurrent streams and a multi-window run.
   - Evidence status: PROVEN for the exercised local route.

9. **RG-09 - OpenAI client contract**
   - Scenario: Use an ordinary OpenAI-compatible client against the head for
     non-streaming and streaming chat/completion requests, documented sampling
     and stop fields, an invalid request, and a cancelled request.
   - Pass condition: Responses, usage, streaming termination, cancellation,
     and error status/body match the documented contract; unsupported fields
     are rejected rather than silently ignored.
   - Resource cost: Medium - stock client session and one server run.
   - Evidence status: UNPROVEN.

10. **RG-10 - Restart and topology recovery**
    - Scenario: Complete a request, restart the server with valid model files
      and placement data still present, then remove or add a peer or change
      its available resources and submit another request.
    - Pass condition: Restart reuses valid files and placement data, the server
      automatically re-profiles the changed topology, records the new result,
      and serves the next request without manual configuration or a hang.
    - Resource cost: Large - restart plus supervised device or resource change.
    - Evidence status: UNPROVEN.

Component checks, a standalone ring CLI, a static server, or a manual cluster
cannot satisfy this gate.

## 11. Definition of "regular local inference server"

Potluck is a regular local inference server only when all ten named checks
above have observed their pass conditions. The current evidence covers
individual mechanisms and operating points, but the full ten-check release
gate is not complete. In particular, named checks must cover:

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
