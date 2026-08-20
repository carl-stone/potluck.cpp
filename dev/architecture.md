# potluck.cpp architecture

## Goal

Potluck distributes a modern llama.cpp model across machines that each contribute a layer window. The key invariant is that a production worker receives a shard containing its window, not a full model copy.

The current window loader and hidden-state graph path target the `qwen35` dense architecture. Other llama.cpp architectures keep their normal behavior but are not a distributed-window claim.

## Processes and data flow

1. The head loads GGUF metadata with `no_alloc` and `LLAMA_LOAD_MODE_NONE`. It reads the vocabulary, layer count, model dimensions, and chat metadata without loading model weights.
2. The head computes contiguous bounds `[bounds[i], bounds[i+1])`, or accepts explicit `--bounds` that match the shard files.
3. The head sends one versioned `node_config` to worker 0. Worker 0 forwards the config to worker 1 and so on.
4. Each worker loads only its assigned window. It decodes every position to keep its local KV and recurrent state current.
5. Non-tail workers send an embedding row to the next worker. The tail owns the output tensors, samples or takes the argmax, and sends the token back to the head.
6. Prefill uses one `batch_decode` message for the prompt in the static pipeline. The tail asks for logits only on the trailing requested rows.

The default transport is raw POSIX TCP. Frames carry PTLK magic, protocol version 5, message type, shape, and bounded payload. Handshake sockets use a 60-second timeout; decode sockets use a 300-second timeout. The environment variables `POTLUCK_TIMEOUT_HANDSHAKE_S` and `POTLUCK_TIMEOUT_DECODE_S` override those defaults.

## Static pipeline and piped ring

The default static route assigns one contiguous window to each worker. This is the normal production path because it minimizes routing messages.

`potluck-head --ring W0,W1,...` selects the piped-ring route. The sizes describe each worker's window in one cycle. The coordinator routes every position through windows in ring order; a worker can own multiple disjoint windows and loads one stage context per window. Ring mode keeps the same hidden-state contract but adds network hops.

Static and ring modes are separate routes. A static multi-worker run is not a ring run.

## Shards

`potluck-shard` writes independently loadable GGUF files:

- every shard carries the source metadata;
- block tensor names keep their global `blk.<index>` names;
- `token_embd.*` is present in the first shard;
- `output_norm.*` and `output.*` are present in the last shard;
- `potluck.shard.index`, `.count`, `.start`, `.end`, and `.source.file` identify the window.

The worker validates the shard metadata before loading. A mismatched assignment fails with both the assigned window and the shard window instead of hanging.

The first and last shards are larger because they contain embedding and output tensors. This is expected.

## Server

`potluck-server` uses vendored `cpp-httplib` and `nlohmann::json`. It exposes:

- `POST /completion` with a required non-empty `prompt`;
- `POST /v1/chat/completions` with common llama.cpp chat-template rendering;
- `GET /health` with worker addresses and layer windows;
- `GET /v1/models`;
- SSE chunks with a role delta, content deltas, a final stop chunk, and `[DONE]`.

One chain serves one request at a time. A second request gets HTTP 429. A worker or transport failure returns HTTP 503 and does not intentionally terminate the HTTP process.

The server can spawn local workers for fixture tests. `--hosts a,b,c --launch ssh` starts workers on separate hosts; `scripts/deploy.sh` copies the binary and the matching shard. Separate hosts are required for memory scaling.

## Resource model

A shard split changes where model memory lives. It does not reduce the aggregate number of model bytes. Multiple workers on one machine share physical RAM, page cache, and memory bandwidth; they are a topology emulator, not a way to fit a model that does not fit the machine.

The intended deployment is one worker per physical machine, with each machine holding only its shard. The head stays metadata-only unless `--parity-check` is requested. `--parity-check` is a correctness harness for small fixtures, not a requirement for target model sizes.

## Scheduling and optional paths

- Default bounds use integer worker weights.
- `--profile` measures worker decode throughput and writes measured weights.
- `--lp` uses HiGHS to honor RAM and VRAM budgets when the local build has `POTLUCK_HIGHS=ON`.
- `--gpu-layers` and `--gpu-mem` derive per-stage offload counts.
- `--prefetch` warms the model file into the OS page cache during chain setup.
- `--draft` enables speculative verification; `--batch` enables multiple sequence IDs in one chain.

Each optional path has a named script under `tests/potluck/`. The parity document records which checks were run and which remain inherited or unverified.
