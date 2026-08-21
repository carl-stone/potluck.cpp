#!/usr/bin/env bash
# End-to-end acceptance for feature-parity row 5 (HiGHS LP scheduler).
#
# The LP allocates layers per worker by minimizing the slowest stage's wall
# time per token subject to per-machine budgets from the workers file
# (ram_mb = layer weights + KV budget, vram_mb = GPU offload budget, 0 = CPU).
# The two workers are asymmetric (worker 0 Metal-offloaded/fast, worker 1
# CPU/slow) and worker 1 carries a hard RAM cap, so the LP must give it only a
# few layers instead of the ~1/4 share the measured weights would alone imply.
# The resulting split must still reproduce the monolithic reference exactly
# (CHAIN PASSED).
#
# Usage: test_lp.sh [n_predict] [host]
set -euo pipefail

N_PREDICT="${1:-32}"
HOST="${2:-127.0.0.1}"
PORT_BASE=$((49000 + RANDOM % 2000))

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"

if [[ -f "${REPO}/build/CMakeCache.txt" ]] &&
    grep -q '^POTLUCK_HIGHS:BOOL=OFF$' "${REPO}/build/CMakeCache.txt"; then
    printf 'test_lp skipped (POTLUCK_HIGHS=OFF in the build)\n'
    exit 0
fi

if [[ ! -x "${BIN}/potluck-head" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing binaries (build potluck-head/potluck-worker first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing Qwen3.5 fixture: %s\n' "${MODEL}" >&2
    exit 2
fi

work="$(mktemp -d -t potluck-lp.XXXXXX)"
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

# Worker 1 gets a 60 MiB RAM cap and 0 MiB VRAM (CPU-only). The fixture is
# ~0.55 GB over 24 layers (~23 MB/layer), so the cap admits at most a couple
# of layers there -- far fewer than the ~6 the measured weights would give it.
workers="${work}/workers.txt"
printf '%s:%d 1 4096 4096\n' "${HOST}" $((PORT_BASE + 0)) >> "${workers}"
printf '%s:%d 1 60 0\n' "${HOST}" $((PORT_BASE + 1)) >> "${workers}"

spawn_bench() {
    pids=()
    for i in 0 1; do
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
    for i in 0 1; do
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

# Phase 1: profile. The weights file must carry the input constraints through.
spawn_bench
if ! "${BIN}/potluck-head" "${MODEL}" "${workers}" 1 "${HOST}" --profile --out "${weights}" \
        >"${profile_log}" 2>&1; then
    cat "${profile_log}" >&2
    printf 'test_lp failed: profile head rc nonzero\n' >&2
    exit 1
fi
stop_all
if ! grep -q 'head: measured weights' "${profile_log}"; then
    cat "${profile_log}" >&2
    printf 'test_lp failed: no measured-weights printed\n' >&2
    exit 1
fi
# The budget fields must have survived into the measured file (worker 1: 60 0).
if ! awk '{ print $3, $4 }' "${weights}" | grep -q '60.0 0.0'; then
    cat "${profile_log}" >&2
    cat "${weights}" >&2
    printf 'test_lp failed: measured weights file lost the per-machine budgets\n' >&2
    exit 1
fi
printf '  ok (profile): %s\n' "$(tr '\n' ' ' < "${weights}")"

# Phase 2: the LP chain run against the constrained weights file.
spawn_chain
head_log="${work}/head.log"
if ! "${BIN}/potluck-head" "${MODEL}" "${weights}" "${N_PREDICT}" "${HOST}" --parity-check --lp >"${head_log}" 2>&1; then
    cat "${head_log}" >&2
    printf 'test_lp failed: chain head rc nonzero\n' >&2
    exit 1
fi
stop_all

if ! grep -q 'CHAIN PASSED' "${head_log}"; then
    cat "${head_log}" >&2
    printf 'test_lp failed: no CHAIN PASSED\n' >&2
    exit 1
fi
# The LP must print its plan and must have honored the RAM cap on worker 1.
lp_line="$(grep -oE 'LP plan: w=\[[0-9 ]+\] ngl=\[[0-9 ]+\] makespan=[0-9.]+' "${head_log}" | head -1)"
if [[ -z "${lp_line}" ]]; then
    cat "${head_log}" >&2
    printf 'test_lp failed: no LP plan printed\n' >&2
    exit 1
fi
w1="$(printf '%s\n' "${lp_line}" | sed -E 's/.*w=\[[0-9]+ ([0-9]+)\].*/\1/')"
w0="$(printf '%s\n' "${lp_line}" | sed -E 's/.*w=\[([0-9]+) [0-9]+\].*/\1/')"
ngl1="$(printf '%s\n' "${lp_line}" | sed -E 's/.*ngl=\[[0-9]+ ([0-9]+)\].*/\1/')"
if (( w0 + w1 != 24 )) || (( w1 < 1 )) || (( w1 > 3 )); then
    cat "${head_log}" >&2
    printf 'test_lp failed: LP violated the worker RAM cap (w1=%s, expected 1..3)\n' "${w1}" >&2
    exit 1
fi
if [[ "${ngl1}" != "0" ]]; then
    cat "${head_log}" >&2
    printf 'test_lp failed: CPU-only worker got ngl=%s, want 0\n' "${ngl1}" >&2
    exit 1
fi

printf '  ok (lp): %s\n' "${lp_line}"
printf 'test_lp passed (%s, LP split w=[%s,%s] ngl1=%s)\n' \
    "$(grep -oE 'CHAIN PASSED[^,]*' "${head_log}" | head -1)" "${w0}" "${w1}" "${ngl1}"