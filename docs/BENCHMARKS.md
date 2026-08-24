# Benchmarks

> Product measurements must run the integrated resource-aware direct-peer ZeroMQ piped-ring server with automatic placement, per-window prefetch, per-device offload, continuous batching, and slots.

## Integrated 27B product acceptance

The following measurements came from supervised 2026-08-23 and 2026-08-24 runs of the integrated direct-peer ZeroMQ PRP server. They are operating points, not speedup claims.

- Model file: `/Users/carlstone/models/gemma-3-27b-it-Q4_K_M.gguf`
- Model: Gemma 3 27B Q4_K_M, 62 layers
- Devices: Apple M4 Mac, Apple M1 Mac, and Linux PC with NVIDIA GTX 1650 SUPER
- Build: Release
- Server configuration: automatic DNS-SD discovery, live probing, head participation, `--launch ssh`, full-model distribution, per-window prefetch, and `--spec-type ngram-simple --spec-draft-n-max 4`
- Request: `The capital of France is`, greedy sampling, streamed chat, 12 completion tokens

The successful three-device route assigned windows `[0,6)`, `[6,16)`, and `[16,62)` to the M4 head, M1, and Linux PC. The server logged three ring workers, three windows, startup prefetch, runtime PRP events, and HTTP status 200.

Measured client-side results for two successful runs of this request:

```text
run 1 TTFT             8.472 s
run 1 total           11.918 s
run 1 decode rate       3.482 tok/s
run 2 TTFT             7.893 s
run 2 total           11.310 s
run 2 decode rate       3.511 tok/s
mean decode rate        3.497 tok/s (2 runs)
```
The decode rate is completion tokens divided by client time after TTFT.

The usage record for a later clean three-device run reported 14 prompt tokens, 12 completion tokens, and 26 total tokens. It returned HTTP 200 with 13.400 s TTFT and 39.166 s total client time. The pre-fix n-gram configuration logged `drafted=0 accepted=0 accept-rate=0.000` because the server kept the default 12-token lookup pattern while limiting the draft window to 4 tokens.

That short chat prompt had no repeated n-gram, so zero drafts was not a useful functional proof. The scheduler now keeps the n-gram lookup pattern no longer than the configured draft window. A post-fix targeted 27B completion with a repeated token sequence logged `drafted=12 accepted=12 accept-rate=1.000` through the same three-device ring.

The separate 27B benchmark run logged two successful topology refresh rebuilds. Its initial three-worker assignment was `[0,12)`, `[12,16)`, `[16,62)`; subsequent rebuilds recorded `[0,18)`, `[18,62)` and `[0,19)`, `[19,62)` after the M1 was removed by live placement. This confirms assignment refresh behavior, but those refreshes used benchmark mode rather than the streaming request.

These values include the current PRP route, network transfers of intermediate activations, per-window synchronization, and client-visible HTTP streaming. They do not isolate prefill, compare against another topology, or establish a regression threshold.

## Verification scope

The local integrated suite exercises the direct adjacent-peer ZeroMQ ring,
automatic placement, full-model window loading, per-window prefetch,
speculative decoding, quantized inference, continuous batching, slots, and
HTTP paths with the Qwen3.5 0.8B fixture.

The supervised Gemma 3 27B run establishes integrated three-device operation
and a measured performance point. It does not claim reference-token parity,
scaling, or a speedup ratio. Broader model architectures and full
llama-server API parity remain outside this release baseline.

Build and fixture commands are engineering checks. They are documented in the
repository quick start and test scripts, not as alternate Potluck runtimes.
