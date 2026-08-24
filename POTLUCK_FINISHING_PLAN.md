# Potluck finishing plan

Status: proposed work plan to complete the product defined by
ADR 0006 (piped-ring server product), ADR 0007 (direct ring over ZeroMQ), and
ADR 0010 (prima feature parity baseline). Destination when approved:
`dev/finishing-plan.md`.

Three further ADRs fall out of aligning with prima.cpp's HALDA and must be
written before the code lands: ADR 0011 supersedes ADR 0008's probe admission,
ADR 0012 amends ADR 0004's context sizing, and ADR 0013 supersedes ADR 0002's
per-device shard files. Section 2 states each one.

This plan closes every requirement in the prima.cpp feature baseline. It does
not add scope beyond that baseline and the integrated-server contract.

## 1. Requirement to work-package map

| # | Requirement (ADR 0010) | Current source state | Work package |
|---|---|---|---|
| 1 | mmap lazy weight load, scheduler balances memory pressure | mmap lazy load is already correct: `llama-model.cpp:1605` passes `prefetch=false` for a window, so no `MAP_POPULATE`. Placement is a capacity heuristic (`admission.cpp:66-286`) | WP1, WP2 |
| 2 | Piped ring with prefetch | Ring runs (`ring.cpp`, `scheduler.cpp`); prefetch exists but is env-gated (`potluck-worker/main.cpp:641,658-673`) | WP3 |
| 3 | HALDA placement solved with HiGHS | No solver, no HiGHS in CMake; `POTLUCK_HIGHS` appears only in scripts | WP1, WP2 |
| 4 | Automatic device selection | Implemented (DNS-SD discovery, probe, admission) | verify only |
| 5 | Quantized model support | Works (Q4_K_M shards) | verify only |
| 6 | Speculative decoding | Absent from Potluck; upstream `common/speculative.*` is available | WP4 |
| 7 | Dynamic batching of concurrent requests | Implemented (slots, continuous batching) | verify only |
| 8 | macOS and Linux now, Windows on the roadmap | Builds and runs on both; guard message exists | WP7 doc work |
| 9 | CLI completion mode and server API mode | Server only; no `potluck-cli` | WP5 |
| 10 | Optional manual workload override, automatic default | Absent | WP6 |
| 11 | CUDA and Metal placement | Implemented per window (`assign_gpu_layers`) | WP2 (solver owns it) |
| 12 | prima-shaped CLI under the `potluck` command family | Server flags only | WP5, WP6 |
| - | Easy install and setup | `scripts/install.sh` plus payload deploy | WP7 |

Removal targets found during the survey: empty `tools/potluck-head/`,
`-DPOTLUCK_HIGHS=OFF` in `scripts/install.sh:221`, `scripts/stage-mac-payload.sh:74`
and `docs/BENCHMARKS.md:47`, the env-only prefetch switches
`POTLUCK_PREFETCH_FORCE` and `POTLUCK_TRACE_PRP`, the heuristic span allocator
`allocate_round_spans`, and the whole shard toolchain listed in WP8.

**Critical anchors.** `tools/potluck-server/admission.cpp` holds the heuristic
WP2 deletes (`allocate_round_spans:66`, `assign_gpu_layers:991`) and the probe
parser WP1 extends (`:438-468`). `tools/potluck-worker/main.cpp` holds the ring
loop, the prefetch hook (`:658-672`), and the tail sampler state WP4 extends
(`:952-982`). `src/potluck-protocol.cpp` holds the config encoder (`:73,113`),
the profile magic (`:235`), and the unversioned batch header (`:515-521`) that
WP0 replaces. `common/potluck_runtime.h:106-113` defines the window-relative
`n_gpu_layers` contract the solver output feeds. `src/llama-model.cpp:1605` is
the lazy-mapping call that makes per-window prefetch meaningful.

## 2. Decisions this plan takes

Decisions 1 to 3 resolve conflicts between the accepted ADRs and prima.cpp's
HALDA behavior. Carl decided each of them on 2026-08-23, and each needs its own
ADR before the code lands. Decisions 4 to 8 are implementation choices inside
the ADRs as they stand.

1. **The solver owns device exclusion; the probe only measures. ADR 0011,
   supersedes ADR 0008 admission.** ADR 0008 has the pre-launch probe admit
   candidates by measured usable capacity. HALDA cannot price a device it never
   sees, so every reachable device now enters the MILP and the solver drops any
   device left with a one-layer window, rank 0 excepted, then re-solves until
   the device set is stable (`common/common.cpp:1620-1628`). The probe keeps
   only hard rejections: unreachable, wrong build id, or Windows.
2. **Context size is fixed before placement. ADR 0012, amends ADR 0004.**
   ADR 0004 derives context from the live per-device plan, but HALDA needs
   `n_ctx` before it can build `b_prime` and the capacity rows, so the two are
   circular. prima takes `cparams.n_ctx` as a scheduler input. Potluck selects
   `n_ctx` from model metadata and the head reserve first, then solves
   placement with `kv_per_layer` already known.
3. **Every device holds the full GGUF and loads only its assigned windows.
   ADR 0013, supersedes ADR 0002.** ADR 0002 gives each device a pre-carved
   shard file, which HALDA would invalidate on every re-solve. prima points
   each device at the same complete model file and loads only the assigned
   layers. Potluck does the same: `potluck_layer_start`/`potluck_layer_end`
   already load a window out of a full GGUF (`common/potluck_runtime.h:100-103`),
   so re-placement costs nothing and WP8 deletes the shard toolchain.
4. **HiGHS is a required build dependency.** CMake gets a real `POTLUCK_HIGHS`
   option, default ON, resolved by `find_package(HIGHS)` first and pinned
   `FetchContent` second. prima keeps a non-solver path
   (`common/common.cpp:1579-1590`); Potluck does not carry it, because ADR 0006
   bans fallback execution paths. A build without the solver is not the
   product.
5. **Device speed is profiled once per ring formation.** prima profiles at
   launch and Potluck does the same: no cache file, no expiry, no `--reprobe`.
   The head holds the measured speed fields in memory for the life of the ring.
   Only the live pressure fields, free host memory, free accelerator memory and
   CPU load, are re-read on each 30-second topology check, because those are
   what HALDA's capacity rows consume.
6. **Speculative verification happens on the tail worker.** prima has no
   speculative decoding, so there is no behavior to copy here. The tail owns
   per-sequence sampler state (`potluck-worker/main.cpp:952-982`). Drafts travel
   with the batch, the tail samples and compares position by position and
   accepts only the matching prefix into sampler state. The head never needs
   full logit vectors, and sampler penalties stay correct.
7. **The head runs the draft model.** The draft context lives in the head
   process next to the HTTP surface and is charged against the head reserve
   before the head is admitted as a compute device.
8. **One runtime library serves both front ends.** `tools/potluck-server/*.cpp`
   becomes `potluck-core`; `potluck-server` and the new `potluck-cli` link it.
   The CLI is a second client surface, not a second execution architecture.

## 3. Work packages

### WP0 - Wire format cutover (single owner, lands first)

Everything else depends on the wire format, so one change makes all of it at
once in `src/potluck-protocol.h` and `src/potluck-protocol.cpp`.

- `accel_profile` becomes `device_profile`: keep `rank`, `kind`, and the four
  memory fields, add `os_kind`, `cpu_gflops[]` and `accel_gflops[]` per
  measured GGML type, `mem_copy_delay_ms`, `accel_copy_delay_ms`,
  `disk_read_seq_gbps`, `disk_read_rnd_gbps`, and `n_cpu_threads`. WP1 states
  what each field means and how it is measured.
- `node_config` gains `prefetch_mode` (off, advise, force) and `n_cycles`.
- Batch payload gains a draft-token vector and an accepted-count field; both
  are propagated hop by hop and returned by the tail. The payload header today
  is unversioned: 20 bytes holding `n`, `clear_seq`, `trim_seq`, `trim_to`, and
  `n_logits` (`src/potluck-protocol.cpp:515-521`). WP0 puts a magic and version
  prefix in front of it, in the style of the `EAP2` profile magic
  (`src/potluck-protocol.cpp:235`).
- `config_version` moves from 1 to 2 (`src/potluck-protocol.cpp:73,113`), and
  the profile magic moves from `EAP2` to `EDP3`. Head and workers ship from one
  build and already check a shared build id, so this is a clean cutover with no
  compatibility shim.

Acceptance: `tests/test-potluck-protocol.cpp` round-trips every new field,
rejects truncated and mismatched payloads, and rejects both the old config
version and the old profile magic with a clear message.

### WP1 - Device profiling for HALDA

`accel_profile` carries only `rank`, `kind`, `free_bytes`, `total_bytes`,
`host_free_bytes`, `host_total_bytes` (`src/potluck-protocol.h:103-110`). The
WP0 `device_profile` adds the fields HALDA consumes, and
`potluck-worker --probe` measures them:

- `os_kind` (`macos`, `linux`). Required for set classification and for the
  disk-speed choice below; a Windows probe is rejected with a clear message.
- `cpu_gflops[t]` and `accel_gflops[t]` for each GGML type `t` the model uses,
  from a timed `ggml` matmul per type. The head sends the type list, which it
  reads from the GGUF.
- `mem_copy_delay_ms`: host memory-copy delay for one layer-sized buffer, and
  `accel_copy_delay_ms` when an accelerator is present.
- `disk_read_seq_gbps` and `disk_read_rnd_gbps`: measured over a 500 MiB
  temporary file, 100 MiB blocks sequential and 4 KiB blocks random, the file
  removed afterwards. prima shells out to `fio`; Potluck measures in process so
  the probe keeps no external dependency.
- `n_cpu_threads`.

The speed fields are measured once per ring formation and held by the head.
`host_free_bytes`, `free_bytes`, and CPU load are re-read on every topology
check (section 2.5).

The head derives the model constants from GGUF metadata, with no `llama.h`
change:

- `b` = summed bytes of one `blk.N.*` tensor set;
- `bi` = `token_embd` bytes; `bo` = `output_norm` plus `output` bytes;
- `b_prime` = `b + kv_per_layer`, reusing the existing
  `2 * n_head_kv * head_dim * 2 * n_ctx` expression (`admission.cpp:1013`);
- per-type FLOPs for one layer, from that block's tensor shapes.

`admission.cpp:438-468` parses the extended probe line; `probe_local_worker`
and `probe_remote_candidates` pass the type list.

Acceptance: `potluck-worker --probe --probe-types q4_K,q6_K,f32` prints a
complete profile on the M4, the M1, and the Linux PC. A second call on the same
host returns the cached speed fields in under 100 ms while its reported free
memory still tracks a multi-GiB allocation held between the two calls.

### WP2 - HALDA solver

New `tools/potluck-server/halda.h` and `halda.cpp`, a direct port of
prima.cpp's `assign_layers_to_device` (`common/common.cpp:860-1532`). Names
below match that source so the two can be diffed.

**Symbols.** `L` = layer count; `k` = cycle count; `W = L / k` = layers per
cycle. Devices `m = 0..M-1` in ring order, `m = 0` the head when the head
participates. `w_m` = layers in device `m`'s window, so the device holds
`w_m * k` layers across the whole model and `sum_m w_m = W`. `n_m` = how many
of those `w_m` layers run on the accelerator; `stage_load` already takes
exactly this window-relative count (`common/potluck_runtime.h:106-113`), and
the existing route builder already emits `k` rounds of contiguous per-device
spans (`admission.cpp:266-272`), so it consumes `k` and `w` unchanged.

**Compute-buffer estimate.** `c_cpu_m` and `c_gpu_m` are per-device byte
constants recomputed on every outer iteration, from `n_ubatch`, `n_ctx`,
`n_embd`, `n_ff`, `n_head`, `n_head_kv`, `n_vocab`:

```
act = (2*n_embd + n_ff*2 + n_embd*2)*n_ubatch          // norm, qcur, ffn gate/up/out/inp
    + n_ctx*n_ubatch*(1 + n_head)                      // kq_mask, kq
    + 3*n_ubatch                                       // inp_pos, inp_out_ids, inp_toks
c_cpu_m = act*4 + (is_head ? (n_embd + n_vocab)*n_ubatch*4 : 0)
c_gpu_m = has_accel_layers ? (act*4 + accel_ctx_reserve) : 0
```

`accel_ctx_reserve` is 700 MiB for CUDA and 300 MiB for Metal, the constants
prima uses (`src/llama.cpp:21964-21968`). prima's own comment calls this an
upper-bound prediction, not an exact value. Reproduce it as prima writes it,
including the assignment rather than accumulation at `src/llama.cpp:22075`, so
the two schedulers stay diffable.

**Set classification**, with `l_m = w_m * k`, `l_gpu_m = n_m * k`,
`b_cio_m = (bi / n_vocab + bo) * [m == 0] + c_cpu_m`, and `budget_m` = live
free host memory, or live free accelerator memory on macOS with Metal:

```
cond1 = l_m*b + (bi/n_vocab + bo)*[m==0] + kv_per_layer*l_m + c_cpu_m > budget_m
cond2 = cond1 numerator + c_gpu_m                                     > budget_m
cond3 = (l_m - l_gpu_m)*b_prime + (bi/n_vocab + bo)*[m==0] + c_cpu_m  > budget_m

M4 if forced_m or disk_gbps_m < 0.1        // paging from such a disk never pays
M1 if macos and not metal and cond1        // host-memory constrained
M2 if macos and metal     and cond2        // unified-memory constrained
M3 if linux               and cond3        // host-memory constrained, offload helps
M4 otherwise                               // sufficient memory
```

`disk_gbps_m` is `disk_read_seq_gbps` on Linux and `disk_read_rnd_gbps`
elsewhere, as prima selects it (`common/common.cpp:1027-1038`).

**Coefficients**, all milliseconds per layer, `EPS = 1e-9`, bytes converted by
`bytes / (gbps * 1e9) * 1000`:

```
t_cpu_m   = 1000 * sum_t flops_layer[t] / (cpu_gflops_m[t]*1e9 + EPS)
t_accel_m = 1000 * sum_t flops_layer[t] / (accel_gflops_m[t]*1e9 + EPS)
alpha_m   = t_cpu_m + mem_copy_delay_ms_m
beta_m    = t_accel_m - t_cpu_m + accel_copy_delay_ms_m - mem_copy_delay_ms_m

M1: a_m = alpha_m + disk_ms(b_prime, m)
M2: a_m = alpha_m + disk_ms(b, m)              b_m = beta_m
M3: a_m = alpha_m + disk_ms(b_prime, m)        b_m = beta_m - disk_ms(b_prime, m)   [accel only]
M4: a_m = alpha_m                              b_m = beta_m                         [accel only]

kappa  = t_cpu_0 + disk_ms(bi/n_vocab, 0) + (head in M4 ? 0 : disk_ms(bo, 0))
       + sum over m in M1 or M3 of disk_ms(c_cpu_m - host_free_m, m)
```

These disk terms are the I/O-pressure mechanism, and they must not be dropped:
`a_m` charges every extra layer on a memory-constrained device with the disk
read it will cause, so the solver moves layers toward devices with spare RAM or
faster disks; the M3 `b_m` term makes accelerator offload cancel that host
paging cost directly; `kappa` charges the fixed per-cycle reads, including the
amount by which a constrained device's compute buffer already overflows its
free memory.

**Capacity coefficients** in window units, denominator `L * b_prime`:

```
M1:  z_m =  (host_free_m  - b_cio_m) / (L*b_prime)
M2:  z_m =  (accel_free_m - b_cio_m - c_gpu_m) / (L*b_prime)
M3:  z_m =  (host_free_m  - b_cio_m) / (L*b_prime)
M4:  z_m = -(macos and metal ? accel_free_m - b_cio_m - c_gpu_m
                             : host_free_m  - b_cio_m) / (L*b_prime)
z_gpu_m  = max((accel_free_m - c_gpu_m) / (L*b_prime), 0)     [accel only]
```

Every term reads live free memory, which is what makes the placement track
memory pressure rather than nameplate capacity.

**MILP.** Columns `[w_0..w_{M-1}, n_0..n_{M-1}]`, all `kInteger`, bounds
`1 <= w_m <= L` and `0 <= n_m <= L`. Minimize, with offset `kappa`:

```
k * sum_m (a_m*w_m + b_m*n_m) + kappa,   the w_0 cost divided by master_priority
```

Rows in this exact order, `1 + 3M` of them:

```
0        : sum_m w_m = W                              (both bounds W)
1..M     : -w_m + n_m <= 0
M+1..2M  : RAM row, upper bound -W*z_m
           M1, M2: -w_m
           M3    : -w_m, plus +n_m when the device has an accelerator
           M4    : +w_m, plus -n_m when the device has a non-Metal accelerator
2M+1..3M : +n_m <= max(W*z_gpu_m, 0)
```

HiGHS options are fixed for determinism: `log_to_console=false`, `threads=1`,
`mip_rel_gap=0`, `random_seed=0`. Ties break toward the lower rank.

**Outer fixed point.** The coefficients depend on the sets, and the sets depend
on the allocation, so one solve is not enough:

1. Seed `w_m = round(budget_m / sum(budget) * L)`, raise any zero to 1 while
   decrementing the largest, then add `L - sum(w)` to the largest-budget device
   when positive or the smallest-budget device when negative. Seed `n_m = w_m`
   for accelerator devices, else 0.
2. Loop: `W = sum(w)`, `cur_k = L / W`; abort with a clear message if `W <= 1`
   or `L % W != 0`. Reclassify the sets; if no set changed, stop.
3. Recompute `c_cpu`, `c_gpu`, `kappa`, `a`, `b`, `z`, `z_gpu`, then solve for
   every `k` in the candidate list and keep the lowest objective. Candidates are
   the divisors of `L` up to `L / 2`, or the single `-k` override. A non-optimal
   status is skipped unless every device is in M4.
4. No feasible `k`: restore the previous feasible solution and stop.
5. Inspect the solution for `weak` (`w_m == 1 and n_m == 0`), `free_gpu`
   (`n_m < floor(W*z_gpu_m)`), `gpu_overload` (`w_m > n_m`), and `cpu_overload`
   (no accelerator and not M4). If not `weak` and `free_gpu` and either
   overload, force the slowest-disk non-M4 device into M4 and repeat from 2.
6. Otherwise adopt the solution; stop when it is unchanged from the previous
   iteration.

**Device removal.** This is now the only device-exclusion authority
(section 2.1): `admit_devices` keeps hard rejections only, and every reachable
device reaches the MILP. Outside the fixed-point loop, drop every device with
`w_m == 1` except rank 0, re-run the whole fixed point with the remaining
devices, and repeat until nothing is dropped. Map results back to original
ranks, 0 for removed devices.
If the existing head participation gate (`internal.h:77-84`) says the
head does not compute, the head is not a HALDA device at all and no master
priority applies; `master_priority` defaults to prima's 1.01.

The solver replaces `allocate_round_spans` (`admission.cpp:66`) and
`assign_gpu_layers` (`admission.cpp:991`); both are deleted.

Acceptance: Small tests with fixed device fixtures assert exact `k`, `w`, and
`n` vectors, and specifically that halving one fixture's free memory shrinks
that device's window, that dropping one fixture's disk bandwidth below
0.1 GB/s moves it to M4 rather than loading it with paged layers, that a slow
disk with sufficient memory still receives layers, that a memory-poor device is
removed, and that repeated runs on one fixture produce identical output.

### WP3 - Per-window prefetch as a product feature

The mechanism is already correct: `prefetch_next_owned`
(`potluck-worker/main.cpp:658-672`) advances to the next owned window and calls
`stages[next].model->prefetch(force)`, and both call sites run after the result
is sent to the next peer (`:933`, `:1048`), which is where prima puts it so the
read overlaps the rest of the ring's compute and communication
(`src/llama.cpp:18567-18574`). WP3 turns it into a product feature.

- Replace `POTLUCK_PREFETCH_FORCE` and `POTLUCK_TRACE_PRP` with the
  `prefetch_mode` field from WP0, set by `--prefetch off|advise|force`
  (prima's `--prefetch` plus `--force`).
- Keep the advisory call on the ring thread; run forced residency on a
  dedicated worker-side prefetch thread so the ring never blocks on page
  faults.
- `force` applies uniformly, as prima does. Forced residency runs on the
  worker-side prefetch thread, so a device that is paging slows itself down
  rather than stalling the ring.
- Confirm the mapped-range guard from the earlier `llama_mmap::get_range` fix
  still holds for offloaded tensors under forced prefetch.

Acceptance: a two-window local run shows prefetch of window `i+1` starting
after the send of window `i` in the worker trace; forced mode measurably lowers
time to first token on the 27B run; and a fixture device in M3 logs the
downgrade instead of forcing.

### WP4 - Speculative decoding

- Head: one `common_speculative` instance per slot, created from
  `common_speculative_init_from_params`, supporting `--spec-draft-model` with
  `draft-simple` and the ngram types that need no second model.
- Scheduler: for a slot in decode state, request a draft of up to `n_draft`
  tokens, append them to the ring batch with `n_logits = n_draft + 1`, and send
  the drafts in the WP0 field.
- Tail: sample each logits row, compare with the draft, stop at the first
  mismatch, accept only the matching prefix into sampler state, and return the
  accepted tokens with the accepted count.
- Head: emit accepted tokens, then trim rejected positions on the next cycle
  with the existing `trim_seq` and `trim_to` path, and feed accepted tokens
  back into the speculator.
- Charge the draft context against the head reserve before head admission.

Acceptance: with temperature 0 the token sequence is identical with and without
speculation on the same prompt; the accept rate and the speculative token
counters appear in the server log.

### WP5 - potluck-cli completion mode

- Extract `potluck-core` from the server sources; `potluck-server` links it
  with no behavior change.
- New `tools/potluck-cli` drives the same ring lifecycle: discovery, probe,
  HALDA, worker launch, then streaming completion to the terminal.
- prima-shaped flags: `-m`, `-hf`, `-p`, `-n`, `-c`, `-cnv`, `--temp`,
  `--top-p`, `--top-k`, `--seed`, `--prefetch`, `--force`.
- Delete the empty `tools/potluck-head/` directory.

Acceptance: `potluck-cli -m models/Qwen3.5-0.8B-Q4_0.gguf -p "..." -n 32`
prints a completion over a live two-worker ring, and `-cnv` holds a multi-turn
chat.

### WP6 - Optional manual workload override

- Expert flags on both front ends, literals matching prima
  (`common/arg.cpp:680-790`): `-lw` / `--layer-window` / `--n-layer-window`
  taking a comma-separated list of layers per window in ring order,
  `-gm` / `--gpu-mem N` in GiB, `-k` / `--n-cycles N`, `--master-priority F`
  with prima's 1.01 default, `--prefetch`, and `--force`.
- An override replaces only the quantity it names and the solver still decides
  the rest: `-lw` fixes `w` and leaves `n` and device selection solved, `-k`
  fixes the cycle count, `-gm` caps the accelerator budget fed into `z_gpu`.
  Every override is logged once at startup as an expert override.
- Validation copies prima's rules and fails at startup, never silently: values
  must be positive non-zero integers, at most 32 entries, one entry per
  admitted device, and `sum(-lw)` must divide `L` exactly. A `-lw` vector whose
  windows do not fit a device's live memory is rejected with the device name
  and the two byte figures.

Acceptance: `-lw` with a valid vector produces exactly those windows and still
solves accelerator layers; a vector that does not divide the layer count, one
with a zero entry, and one with too many entries each fail with a distinct
message; no flag means fully solved placement.

### WP7 - Install, setup, and documentation

- `scripts/install.sh` installs both front ends, resolves HiGHS, drops the
  stale `-DPOTLUCK_HIGHS=OFF`, and prints the quick start.
- README quick start: install on the head, run `potluck-node` on each worker
  device, then `potluck-server -hf <repo>` or `potluck-cli -p "..."`.
- Update `dev/open-questions.md` rows as each gap closes, record the 27B result
  in `docs/BENCHMARKS.md`, and write ADR 0011 (solver owns device exclusion,
  supersedes ADR 0008 admission), ADR 0012 (context sizing precedes placement,
  amends ADR 0004), and ADR 0013 (full GGUF per device, supersedes ADR 0002).

Acceptance: a clean checkout on macOS and on Linux reaches a running server
with one install command plus one run command.

### WP8 - Full-model window loading, shard toolchain removal

Decision 2.3. Each device keeps one complete GGUF and loads only the layers
HALDA assigns, so a re-solve costs nothing.

- `potluck-worker/main.cpp:624-625` stops building a shard filename and passes
  the plain model path to `stage_load`, which already takes `window.start` and
  `window.end` and maps only that range (`common/potluck_runtime.h:100-103`,
  `src/llama-model.cpp:1605`). Delete `validate_shard` (`:348-366`) and its two
  call sites; the window bounds are now checked against the model's own layer
  count.
- `potluck-server/ring.cpp` loses shard generation and transfer: the directory
  and prefix setup (`:59-113`), the remote `potluck-shard` command (`:155-160`),
  and the remote shard directory (`:1153-1154`).
- The head instead ensures the model itself is present once per device. It
  compares a digest with what the device reports and, when missing, copies the
  GGUF to `~/potluck/models/<name>.gguf` and caches it by digest. When the model
  came from `-hf`, the device fetches from Hugging Face directly instead of
  taking a head-to-worker copy. Either way it happens once per model per device,
  not once per topology.
- Delete `tools/potluck-shard/`, `tools/CMakeLists.txt:49`, and
  `tools/potluck-server/CMakeLists.txt:22`.
- Drop `potluck-shard` from `scripts/install.sh:179,196-197,222,237,240`,
  `scripts/stage-linux-payload.sh:95,99,102,163,165,167,198,205,229`,
  `scripts/stage-mac-payload.sh:75,79,82,136,138,140,177,216`,
  `scripts/pre-push-check.sh:41`, and `scripts/setup_and_test_pc.sh:58`.
- Drop it from `tests/potluck/test_packaging.sh:41-48,68,163-170` and
  `tests/potluck/test_server.sh:18-20`. `tests/gguf-model-data.cpp:720-728` is
  upstream multi-file GGUF handling and stays.

Acceptance: a two-worker local ring serves a completion with both workers
pointed at the same full `models/Qwen3.5-0.8B-Q4_0.gguf`, no `.potluck-shards`
directory is created anywhere, and worker resident memory stays proportional to
the assigned window rather than the whole file.

## 4. Execution order and parallelism

- Stage 1 (serial): WP0 wire format cutover, then WP8 full-model window
  loading. WP8 must land before the solver so re-placement is free.
- Stage 2 (parallel): WP1 profiling, WP2 solver against fixtures, WP5 core
  extraction and CLI skeleton, WP3 prefetch flags.
- Stage 3 (parallel): WP2 wiring into ring bring-up, WP4 speculative decoding,
  WP6 overrides.
- Stage 4 (serial): WP7 install and docs, full gate, 27B acceptance.

## 5. Test ladder

Development uses `models/Qwen3.5-0.8B-Q4_0.gguf` for every loop. The draft
model for speculative work is the same 0.8B file, which shares the qwen35
architecture and the 248320-token vocabulary with the 27B target.

1. Small: wire format round-trip, HALDA fixtures (exact `k`/`w`/`n`, memory
   pressure shrinking a window, slow disk moving a device to M4, weak-device
   removal, determinism), compute-buffer estimate, draft-acceptance function.
2. Medium: local two-worker ring on the 0.8B model for greedy speculative
   parity, prefetch ordering after send, window loading straight from the full
   GGUF, override handling, and four concurrent slots.
3. Large: three-device run on `~/models/Qwen3.8-27B-Q4_K_M.gguf` across the M4
   head, the M1, and the Linux CUDA PC, with automatic discovery, solved
   placement, forced prefetch, speculative decoding, and streaming chat. The
   run must log one set assignment per device and must not oscillate between
   placements across topology refreshes. Record tokens per second and time to
   first token against the current baseline of about 1.45 tokens per second and
   1.5 to 1.8 seconds warm.

New Small and Medium tests join the existing gate: the binary list in
`tests/potluck/run_all.sh:18,27` and the build target list in
`scripts/pre-push-check.sh:41` and `scripts/setup_and_test_pc.sh:58`.

## 6. Definition of done

- Every ADR 0010 requirement is implemented in the integrated ring server and
  the completion CLI, with no static, manual-only, or alternate execution path
  left in the tree.
- `bash tests/potluck/run_all.sh` passes on macOS and Linux, and the named
  end-to-end checks in `dev/open-questions.md` pass.
- The 27B three-device run completes a streaming chat with speculative
  decoding, and its numbers are recorded in `docs/BENCHMARKS.md`.
- Install is one command per device, and no user step mentions ranks, windows,
  ports, or shards.
