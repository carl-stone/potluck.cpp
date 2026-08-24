# Potluck development documentation

`potluck.cpp` is the sole canonical repository for Potluck development. Future
sessions must work from this repository and must not consult the former
`local-ai-network` patch archive for requirements, architecture, status, tests,
or operational guidance.

## Required reading

Read these files in order before changing Potluck:

1. [`decisions/0006-piped-ring-server-product.md`](decisions/0006-piped-ring-server-product.md)
   — the binding product architecture and finished-product boundary.
2. [`decisions/0007-prima-direct-ring-zeromq.md`](decisions/0007-prima-direct-ring-zeromq.md)
   — the binding direct-ring topology and ZeroMQ communication decision.
3. [`decisions/0010-prima-feature-parity-baseline.md`](decisions/0010-prima-feature-parity-baseline.md)
   — the prima.cpp feature baseline: HALDA placement with HiGHS, speculative
   decoding, quantized models, platforms, the CLI family, and manual workload.
4. [`architecture.md`](architecture.md) — the required runtime, data flow,
   scheduling, window-loading, head-resource, and server design.
5. [`definition-of-done.md`](definition-of-done.md) - the draft package
   completion contract, product goals, and end-to-end proof.
6. [`open-questions.md`](open-questions.md) - decisions that still need an
   answer before the completion contract can become final.
7. [`parity-and-accuracy.md`](parity-and-accuracy.md) - evidence from the
   current component checks and the limits of that evidence.
8. [`decisions/`](decisions/) - accepted architectural decisions and their
   supersession history.

`completion-goals.md` remains as a link for older notes; edit the goals only in
`definition-of-done.md`.

Repository-wide agent and test rules are in [`../AGENTS.md`](../AGENTS.md).
Dated component measurements are in
[`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md); they are not product
benchmarks or architecture guidance.

## Decision authority

Potluck descends behaviorally from
[prima.cpp](https://github.com/OpenCPIL/prima.cpp). When a technical
distributed-runtime behavior is not already fixed by a direct user decision or
accepted Potluck ADR, read prima.cpp's documentation and code and defer to its
behavior. Reproduce the outcome and semantics. Potluck uses modern llama.cpp,
but ADR 0007 specifically requires prima.cpp's direct peer-to-peer ring and
ZeroMQ communication model.

Any technical departure from prima.cpp requires explicit user approval and an
accepted ADR. Modern llama.cpp is the implementation base, not an authority for
replacing prima.cpp's distributed behavior with another architecture.

For user flow and usability, the governing directive is different: Potluck
must be easy to use as one normal local server. Automatic cluster operation
must hide devices, ranks, windows, model files, ports, and launch mechanics. Do
not copy prima.cpp's manual deployment interface into Potluck merely because
its technical behavior is the reference.

## Canonical boundary

The former patch series, prima-prefixed scripts, static-pipeline runbooks,
manual workers files, hardware snapshots, and cross-machine experiment notes
are intentionally not copied here. They describe superseded implementations,
contain conflicting machine information, or encode manual/static behavior that
ADR 0006 requires Potluck to remove.

The source of truth is the decided product contract, not the history of how the
current incomplete implementation was reached. Current hardware capacity is
also not a checked-in scheduling input: the product must discover and profile
live devices, including current user load on the head.

When a historical fact is needed to explain a durable decision, record only the
necessary context in an ADR. Do not restore an external historical document or
create a second product specification.

## Local push verification

Potluck does not spend GitHub Actions minutes on its build and component suite.
Each checkout must install the repository-managed hook once:

```sh
bash scripts/install-git-hooks.sh
```

The pre-push hook runs `scripts/pre-push-check.sh` locally. The check requires
the ignored Qwen3.5 0.8B fixture in `models/`, configures and incrementally
builds the required binaries, and runs the current component suite. Use
`POTLUCK_TEST_MODEL` to select another local path for the same fixture.
