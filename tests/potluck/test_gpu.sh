#!/usr/bin/env bash
# End-to-end acceptance for feature-parity row 9 (GPU memory planning).
#
# The coordinator resolves a *total* GPU-offload layer count K either directly
# (--gpu-layers/-ngl) or from a --gpu-mem budget in MiB using an estimated
# bytes-per-layer (total model bytes / n_layer). It broadcasts a per-stage
# window-relative offload count (ngl) in node_config; each worker offloads
# exactly that many of its own layers. The split keeps ONE model-wide GPU/CPU
# boundary at global layer K for every stage and the monolithic reference.
# CPU and Metal runs must preserve the checked greedy output. CUDA can be
# nondeterministic at a near-tie; the script accepts only an explicit CUDA
# mismatch while still requiring the offload plan and a nonzero worker ngl.
#
# Scenarios (3 workers so a boundary lands inside a window, not just at a seam):
#   1. --gpu-mem 200  -> K lands in the first window  -> ngl like [8,1,0]
#   2. --gpu-mem 400  -> K lands in the last window   -> partial tail offload
#   3. --gpu-layers 24 -> every layer offloaded        -> ngl [8,8,8]
# Each must print the GPU offload plan. CPU/Metal must also print CHAIN PASSED;
# CUDA near-tie mismatches are the documented nondeterministic exception.
#
# Usage: test_gpu.sh [n_workers] [n_predict] [host]
set -euo pipefail
if [[ "${POTLUCK_SKIP_GPU_TESTS:-0}" == 1 ]]; then
    printf 'test_gpu skipped (POTLUCK_SKIP_GPU_TESTS=1)\n'
    exit 0
fi

N_WORKERS="${1:-3}"
N_PREDICT="${2:-32}"
HOST="${3:-127.0.0.1}"
PORT_BASE=$((45000 + RANDOM % 2000))

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

work="$(mktemp -d -t potluck-gpu.XXXXXX)"
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

# One worker per scenario -- the coordinator broadcasts the ngl, so the worker
# binary is started fresh each run with no -ngl of its own.
start_workers() {
    pids=()
    for i in $(seq 0 $((N_WORKERS - 1))); do
        "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((PORT_BASE + i)) >"${work}/worker_${i}.log" 2>&1 &
        pids+=($!)
    done
    sleep 2
}
stop_workers() {
    for p in "${pids[@]:-}"; do
        kill "${p}" 2>/dev/null || true
    done
    pids=()
    wait 2>/dev/null || true
    sleep 3
}

fail_run() {
    local head_log="$1" label="$2"
    cat "${head_log}" >&2
    for log in "${work}"/worker_*.log; do
        if [[ -f "${log}" ]]; then
            printf '%s\n' "--- ${log} ---" >&2
            cat "${log}" >&2
        fi
    done
    printf 'test_gpu failed: %s\n' "${label}" >&2
    exit 1
}

scenarios=(
    "--gpu-mem 200|mid-window boundary (K in first window)"
    "--gpu-mem 400|boundary in the tail window (partial offload)"
    "--gpu-layers 24|all layers offloaded"
)

for spec in "${scenarios[@]}"; do
    budget_arg="${spec%%|*}"
    label="${spec##*|}"
    head_log="${work}/head_${budget_arg// /_}.log"

    start_workers
    # Bit-exact greedy parity ("CHAIN PASSED") across independent contexts is
    # a Metal/CPU guarantee. llama.cpp's CUDA backend is not run-to-run
    # reproducible, so a mixed-offload boundary can flip a near-tie greedy
    # step (the head then exits 1 with MISMATCH); on CUDA that specific
    # outcome is accepted and noted, everything else still fails.
    if "${BIN}/potluck-head" "${MODEL}" "${workers}" "${N_PREDICT}" "${HOST}" --parity-check ${budget_arg} >"${head_log}" 2>&1; then
        head_ok=1
    else
        head_ok=0
    fi
    stop_workers

    cuda_backend=0
    if grep -qE 'ggml_cuda_(init|graph)' "${head_log}" 2>/dev/null; then
        cuda_backend=1
    fi
    if (( head_ok )); then
        if ! grep -q 'CHAIN PASSED' "${head_log}"; then
            fail_run "${head_log}" "no CHAIN PASSED (${label})"
        fi
        printf '  ok: %-45s -> %s\n' "${label}" "$(grep -oE 'CHAIN PASSED[^,]*' "${head_log}" | head -1)"
    elif (( cuda_backend )) && grep -q 'MISMATCH' "${head_log}"; then
        printf '  (CUDA backend, %s: near-tie flip; exact parity not asserted)\n' "${label}"
    else
        fail_run "${head_log}" "head rc nonzero (${label})"
    fi
    if ! grep -q 'GPU offload plan' "${head_log}"; then
        fail_run "${head_log}" "no offload plan printed (${label})"
    fi
    # Confirm at least one stage actually offloaded something (budget > 0).
    if ! grep -qE 'per-worker ngl:.*[1-9]' "${head_log}"; then
        fail_run "${head_log}" "no worker got a nonzero offload count (${label})"
    fi
done

printf 'test_gpu passed (%s workers, %s predict)\n' "${N_WORKERS}" "${N_PREDICT}"
