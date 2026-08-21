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
`ggml-org/llama.cpp`. Essential current product documentation and agent rules
live in this repository. The external `local-ai-network` patch archive and its
historical prose are not inputs to Potluck development and must not be required
by future sessions.

## Consequences

- Attribution is preserved: `git log` shows the llama.cpp authors for all
  upstream lines.
- The initial fork delta was one reviewable squashed commit; subsequent Potluck
  work is recorded as normal commits in this repository.
- Rebasability is explicit: `upstream` tracks llama.cpp, so the delta can be
  rebased when upstream moves.
- `potluck.cpp` is the sole canonical working repository. Historical patch
  files and superseded architecture documents remain external and are not
  copied back into the product repository.
