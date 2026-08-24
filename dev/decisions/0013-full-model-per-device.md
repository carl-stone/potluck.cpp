# ADR 0013: Full GGUF per device with window loading

- Date: 2026-08-23
- Status: Accepted
- Decision owner: Carl Stone
- Supersedes: ADR 0002 per-device model shards

## Context

A route-specific shard file changes whenever HALDA re-solves placement. That
makes topology recovery expensive and prevents the scheduler from moving a
window without rebuilding storage artifacts. The llama.cpp Potluck window
loader already accepts global layer bounds and maps only the tensors in those
bounds.

## Decision

Each selected device keeps one complete GGUF file for the model. The head
ensures that file exists once per device, verifies it by digest, and reuses it
across topology rebuilds. A model selected from Hugging Face is fetched by the
device when possible; a local model is copied to the device model cache when
needed.

A worker receives the full model path and loads only its assigned windows by
passing the global start and end bounds to `stage_load`. It must not load the
complete model into memory. Route-specific `.potluck-shards` directories,
shard generation, shard metadata validation, and the `potluck-shard` binary
are removed.

## Consequences

- HALDA can re-solve placement without rebuilding or transferring shard files.
- Disk usage increases to one complete model per selected device, but transfer
  happens once per model and digest.
- mmap remains lazy and resident memory remains proportional to assigned
  windows.
- ADR 0002 remains historical and is superseded for storage and deployment.
