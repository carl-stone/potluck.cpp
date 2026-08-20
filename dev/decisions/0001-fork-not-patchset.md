# ADR 0001: Ship as a fork, not a patch set

- Date: 2026-08-20
- Status: Accepted

## Context

The distributed layer-window pipeline existed as four commits on top of
llama.cpp `058df671b`, plus a set of shell tests and prose docs in a separate
`local-ai-network` directory. The old layout had three mutually contradictory
accounts of "which patch is the real base", and a public visitor had no single
artifact to clone.

## Decision

Ship as a normal git repository: the full llama.cpp history preserved, the four
commits squashed into one `potluck: distributed layer-window inference over
TCP` commit on `main`, `origin` at `carl-stone/potluck.cpp` and `upstream` at
`ggml-org/llama.cpp`. The `patches/` directory and the old prose docs stay in
`local-ai-network` as history; they are not copied into the new repository.

## Consequences

- Attribution is preserved: `git log` shows the llama.cpp authors for all
  upstream lines.
- The fork delta is exactly one reviewable commit.
- Rebasability is explicit: `upstream` tracks llama.cpp, so the delta can be
  rebased when upstream moves.
