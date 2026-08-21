# ADR 0002: Per-device model shards

- Date: 2026-08-20
- Status: Superseded by ADR 0006

## Context

This decision introduced per-device shards. ADR 0006 keeps shards as the unit
of worker loading but removes the requirement that a device must not download
or store the complete model file. The binding invariant is that a worker never
loads the complete model into memory; it loads only its assigned ring-window
shards.

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

- A worker loads only the bytes needed for its assigned windows. A complete
  model file may also exist on the device's storage.
- First and last shards are visibly larger (embedding/output tensors); the
  shard table says so.
- No protocol change: workers already take a model path on their command line.
