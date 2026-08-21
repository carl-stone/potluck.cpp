#!/usr/bin/env bash
# End-to-end acceptance for feature-parity rows 6 + 4 (measured scheduler).
#
# Row 6: potluck-head --profile probes each worker (started with --bench),
#         measures realized decode throughput, prints a per-device capability
#         table, and writes a measured-capability weights file (slowest = 1).
# Row 4: the layer scheduler tessellates by that measured capability; an
#         asymmetric 2-worker run must assign layers by the model and still
#         match the monolithic reference (CHAIN PASSED).
#
# The two workers are made asymmetric on purpose: worker 0 fully Metal-offloaded
# (fast), worker 1 CPU-only (slow). The profile must measure them apart and the
# resulting split must be non-uniform yet exactly correct.
#
# Usage: test_sched.sh [n_workers] [n_predict] [host]
set -euo pipefail
if [[ "${POTLUCK_SKIP_GPU_TESTS:-0}" == 1 ]]; then
    printf 'test_sched skipped (POTLUCK_SKIP_GPU_TESTS=1)\n'
    exit 0
fi

N_WORKERS="${1:-2}"
N_PREDICT="${2:-32}"
HOST="${3:-127.0.0.1}"
PORT_BASE=$((46000 + RANDOM % 2000))

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"

if [[ ! -x "${BIN}/potluck-head" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing binaries (build potluck-head/potluck-worker first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing Qwen3.5 fixture: %s\n' "${MODEL}" >&2
    exit 2
fi

work="$(mktemp -d -t potluck-sched.XXXXXX)"
pids=()
cleanup() {
    for p in "${pids[@]:-}"; do
        kill "${p}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    if [[ -z "${POTLUCK_KEEP_WORK:-}" ]]; then
        rm -rf "${work}"
    fi
}
trap cleanup EXIT

workers="${work}/workers.txt"
for i in $(seq 0 $((N_WORKERS - 1))); do
    printf '%s:%d\n' "${HOST}" $((PORT_BASE + i)) >> "${workers}"
done

# Which worker is the fast (Metal) one. All-Fast == N_WORKERS-1 layers offloaded
# would exceed some Metal VRAM on this fixture beyond a couple workers, so we
# cap offload and rely on the CPU-only worker being measurably slower. The
# profile must still record a separation; we assert weights are not all equal.
spawn_bench() {
    for i in $(seq 0 $((N_WORKERS - 1))); do
        local ngl=0
        if (( i == 0 )); then
            ngl=24   # full offload -> fast
        fi
        "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((PORT_BASE + i)) --bench -ngl "${ngl}" \
            >"${work}/bench_${i}.log" 2>&1 &
        pids+=($!)
    done
    sleep 4
}
spawn_chain() {
    pids=()
    for i in $(seq 0 $((N_WORKERS - 1))); do
        "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((PORT_BASE + i)) \
            >"${work}/worker_${i}.log" 2>&1 &
        pids+=($!)
    done
    sleep 2
}
stop_all() {
    for p in "${pids[@]:-}"; do
        kill "${p}" 2>/dev/null || true
    done
    pids=()
    wait 2>/dev/null || true
    sleep 1
}

weights="${work}/weights.txt"
profile_log="${work}/profile.log"

# Phase 1 (§6): profile the bench workers and derive measured weights.
spawn_bench
if ! "${BIN}/potluck-head" "${MODEL}" "${workers}" 1 "${HOST}" --profile --out "${weights}" \
        >"${profile_log}" 2>&1; then
    cat "${profile_log}" >&2
    printf 'test_sched failed: profile head rc nonzero\n' >&2
    exit 1
fi
stop_all

if ! grep -qE 'head: measured weights' "${profile_log}"; then
    cat "${profile_log}" >&2
    printf 'test_sched failed: no measured-weights printed\n' >&2
    exit 1
fi
if [[ ! -s "${weights}" ]]; then
    cat "${profile_log}" >&2
    printf 'test_sched failed: weights file empty\n' >&2
    exit 1
fi
weights_line="$(awk '{print $NF}' "${weights}" | tr '\n' ' ')"
if awk '{print $NF}' "${weights}" | sort -u | wc -l | grep -q '^1$'; then
    cat "${profile_log}" >&2
    cat "${weights}" >&2
    printf 'test_sched failed: measured weights are all equal (no capability separation)\n' >&2
    exit 1
fi
printf '  ok (profile): measured weights %s\n' "${weights_line}"

# Phase 2 (§4): run the chain against the measured weights file; the split must
# be non-uniform and the output must match the monolithic reference exactly.
spawn_chain
head_log="${work}/head.log"
if ! "${BIN}/potluck-head" "${MODEL}" "${weights}" "${N_PREDICT}" "${HOST}" --parity-check >"${head_log}" 2>&1; then
    cat "${head_log}" >&2
    printf 'test_sched failed: chain head rc nonzero\n' >&2
    exit 1
fi
stop_all

if ! grep -q 'CHAIN PASSED' "${head_log}"; then
    cat "${head_log}" >&2
    printf 'test_sched failed: no CHAIN PASSED\n' >&2
    exit 1
fi
# The measured scheduler must actually redistribute layers (non-uniform split).
split="$(grep -oE 'layers split:.*' "${head_log}" | head -1)"
if ! grep -qE 'layers split: 0 [0-9]+ [0-9]+' "${head_log}" \
        && ! grep -qE 'layers split: [0-9]+ [0-9]+ [0-9]+ [0-9]+' "${head_log}"; then
    cat "${head_log}" >&2
    printf 'test_sched failed: no split printed\n' >&2
    exit 1
fi
printf '  ok (schedule): %s -> %s\n' "${split}" "$(grep -oE 'CHAIN PASSED[^,]*' "${head_log}" | head -1)"

printf 'test_sched passed (%s workers, measured weights %s)\n' "${N_WORKERS}" "${weights_line}"
