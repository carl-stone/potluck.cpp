# ADR 0002: Per-device model shards

- Date: 2026-08-20
- Status: Accepted

## Context

prima.cpp requires a full local GGUF copy on every device. The headline
difference of potluck.cpp must be that no device ever downloads, stores, or
loads the whole model: a worker only needs the layers in its window.

The GGUF format supports exactly this: `gguf_init_empty()` plus
`gguf_add_tensor` writes any subset of tensors, and the existing window loader
keys off the original global `blk.<i>` indices.

## Decision

Add `potluck-shard`, which writes one GGUF per layer window. Every shard gets
the full metadata (`gguf_set_kv(out, in)`), so each shard is an independently
loadable model file, and extra `potluck.shard.*` KVs record the window. The
first shard also holds `token_embd`, the last also holds `output_norm` and
`output`. No `split.*` KVs: those belong to `gguf-split`'s scheme.

Workers validate that their assigned window lies inside their shard's window at
load time and fail loudly otherwise.

## Consequences

- A 4-device cluster moves ~18.97 GB / 4 bytes per device instead of 18.97 GB
  each.
- First and last shards are visibly larger (embedding/output tensors); the
  shard table says so.
- No protocol change: workers already take a model path on their command line.
