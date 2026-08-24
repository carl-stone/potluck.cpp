# Instructions for potluck.cpp

## Binding Potluck product architecture

Before changing Potluck code, tests, or documentation, read `dev/decisions/`.

Potluck has one execution architecture: an automatically formed,
heterogeneity-aware piped ring served by an OpenAI-compatible head. Adjacent
ring peers communicate directly through ZeroMQ; the head must not relay
intermediate data for windows it does not execute. The product requires
automatic live device profiling and selection, per-window prefetch, per-device
CPU/CUDA/Metal placement, per-device GGUF shard loading, continuous batching,
conversation slots, and resource-aware head participation. A full GGUF may
exist on disk, but a worker loads only assigned window shards.

Static execution, manual topology or placement, standalone ring/batch paths,
whole-file prefetch as a substitute for per-window prefetch, global accelerator
budgets, and one-request serving must be removed. Do not preserve them as
diagnostics, fallbacks, compatibility modes, experiments, or provisional
releases. Tests must exercise the integrated ring server. Component checks do
not establish product completion.

The custom PTLK/raw-TCP transport and coordinator-routed ring are conflicting
legacy implementation. Remove them during the clean cutover. Do not preserve
them as alternate transports, diagnostics, or compatibility modes.

## Potluck decision authority

- Direct user decisions and accepted Potluck ADRs are binding.
- For an unresolved technical behavioral decision, defer to the documented and
  coded behavior of [prima.cpp](https://github.com/OpenCPIL/prima.cpp), the
  behavioral progenitor of Potluck's distributed runtime. Preserve the
  observable behavior and purpose; do not invent or select a competing
  architecture because it is more familiar.
- A technical departure from prima.cpp requires explicit user approval and an
  accepted Potluck ADR. Implementation details may differ when the behavioral
  result is preserved and existing ADRs require the difference.
- Prima.cpp is not the usability reference. Potluck's user-experience north
  star is a simple normal local server: select a model, let Potluck discover and
  operate the cluster automatically, and connect an ordinary client or agent
  harness. Never expose ranks, workers, shards, bounds, weights, ports, or
  distributed launch steps as the normal user flow.
- Upstream llama.cpp supplies the modern inference engine and backends. It does
  not override prima.cpp as the default distributed-behavior reference or the
  Potluck usability directive.

## Potluck test rules

- **Classify by size.** Small tests use no network, database, filesystem,
  external system, thread, or sleep. Medium integration tests may use localhost,
  files, or threads to verify component communication. Large tests exercise the
  complete product. Keep most tests Small and keep Large tests few.
- **Keep tests independent and isolated.** Every test must pass in any order and
  must not depend on state left by another test.
- **Assert behavior, not implementation.** Use known-good output for known
  input. Never weaken or delete a correct assertion only to make a test pass.
- **Treat failure as evidence.** Correct bad tests when their contract is wrong;
  do not alter a correct test to hide missing or broken product behavior.
- **Keep tests deterministic and fast.** Avoid wall-clock timing and external
  instability. Retries and sleeps belong only in Medium or Large tests when the
  real boundary requires them.
- **Test the decided product.** Product acceptance must exercise the integrated
  resource-aware piped-ring server. Static, manual, or alternate architecture
  tests are removal targets, not product coverage.

### Code and Commit Standards

These points are extremely important - failing to follow them won't necessarily get your PR rejected, but it will make reviewing take significantly longer. Please follow them carefully:

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings (write like cavemen if needed)
    - Note: Remind yourself of this point regularly, as it often gets lost between context compactions
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase. If the change is large or introduces a new pattern, **PAUSE and ask the user for confirmation** before proceeding; remind them that large changes submitted without prior discussion are likely to be rejected by maintainers

### Potluck commit discipline

All Potluck-authored changes land as atomic commits: one logical change per
commit, each buildable and gate-green on its own. See
[dev/engineering-workflow.md](dev/engineering-workflow.md). Agents may commit
or push only on Carl's explicit instruction.

If an agent creates or changes a GitHub Actions workflow, the agent is
responsible for making it pass. After a workflow failure, the agent must stop
pushing other work and fix the failing workflow before the next push.

Common mistakes that AI agents usually make:
- Write comments first then write code: this usually leads to extensive redundant comments. Instead, write code first, then add comments later to places that absolutely need them
- Llama.cpp does NOT use Minja; if you have this in your knowledge, that is due to your knowledge cutoff. Llama.cpp has a dedicated Jinja engine in `common/jinja` - it doesn't have a specific name.

### Examples:

Code comments:

```cpp
// GOOD (code is self-explanatory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

// reset here, as we will release the slot below
n_tokens = 0;
// ... (a lot of code)
release();


// BAD (addresses the user's task, meaningless out of context)

// Reset n_tokens to 0 before releasing the slot. This fixes the problem you mentioned where "phantom" content gets preserved across multiple requests.
n_tokens = 0;
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

```cpp
// GOOD (comment is kept concise and useful)

// one decode step of code_predictor
// at step_idx g:
// - read code from out_code_cache[g], then embed it with codebook table g-1
// - write new kv at cache row g+1, sample with lm_head[g]
// - write result to out_code_cache[g+1]


// BAD (comment is long and is forced to fit into a fixed column size, it is very annoying to read as a reviewer)

// one autoregressive decode step of the 5-layer code_predictor. See the
// comment in models.h for the cache/tensor conventions this relies on.
//
// index mapping (derived from the reference pipeline-tts.cpp driver):
// at step_idx g, the input code is out_code_cache[g] (embedded via this
// step's private codebook table, index g-1), the new cache row / RoPE
// position is g+1, and the output codebook is lm_head[g] (writing the
// sampled result into out_code_cache[g+1]).
```

Commit message:

```
// BEST: Let the user write the commit


// GOOD: Write a concise commit

llama : fix KV being cleared during context shift
// BAD: Write a verbose commit

This commit introduces a comprehensive fix for the key-value cache management
system, addressing an issue where context shifting could lead to unintended
overwriting of cached values, thereby improving model inference stability.
```
## Useful Resources

To conserve context space, load these resources as needed:

Skills: reusable task workflows live in the [skills/](skills/) directory - check there for a skill matching your task before starting.

General documentations:
- [How to add a new model](docs/development/HOWTO-add-model.md)

Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
