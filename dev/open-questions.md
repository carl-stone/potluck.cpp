# Potluck open questions

Status: Draft list of decisions that still need an answer.

This file is for unresolved product-scope and behavior decisions. It is not a
feature gap matrix, implementation status report, release checklist, or place
to record accepted architecture. The proposed package target and its proof are
in [`definition-of-done.md`](definition-of-done.md). Current test evidence is
in [`parity-and-accuracy.md`](parity-and-accuracy.md).

## How to use this document

An item belongs here only when agents cannot implement it correctly without a
new product choice. Once Carl decides an item, record the decision in an ADR
when it changes architecture or in the Definition of Done when it only fixes
completion scope. Remove the resolved question from this file.

The binding technical reference for unresolved distributed-runtime behavior is
prima.cpp, unless an accepted Potluck ADR says otherwise. The binding technical
reference for the engine, local slots, cache, context, and HTTP behavior is
llama.cpp. A behavior answered by either reference does not belong in this
document as a new Potluck question.

The binding architecture is in ADRs 0006 and 0007. This file does not authorize
a second execution path, a manual cluster, or a provisional product mode.

## Questions that affect the package Definition of Done

### 1. Client API scope

- Should the package include Anthropic Messages or other non-Pi client
  surfaces?

The route may remain available as an optional compatibility surface while this
scope question is open. Its implementation does not make it a required package
feature.

### 2. Model and backend scope

- Should the package support other text model architectures?
- Should the package support embedding and reranking models?
- Should the package support vision, audio, or video input?
- Should the package support LoRA or control-vector variants?
- Should the package support multiple models and model routing?
- Should the package support a broader distributed quantization matrix?

### 3. Optional server features

- Should the package require Prometheus metrics and detailed per-request
  timing?
- Should the package require runtime property inspection and safe property
  changes?
- Should the package require slot cache persistence?
- Should the package require LoRA adapter management?
- Should the package require advanced sampler controls, grammar, and
  JSON-schema response controls?
- Should the package require infill and repository-context requests?
- Should the package require a static web UI?
- Should the package require server-hosted agent or MCP tools?
- Should the package require model acquisition from local paths, Hugging Face,
  cache, or offline stores?
