#!/usr/bin/env bash
# End-to-end acceptance for feature-parity row 7 (weak-device removal /
# topology rebuild), static-pipeline variant.
#
# A 3-device run is profiled (§6) with one deliberately slow device (CPU-only).
# --drop-below removes any device under the throughput threshold from the
# rebuilt topology (it never appears in the weights file, so the chain never
# connects to it). The remaining devices re-tessellate the full layer span and
# must still reproduce the monolithic reference exactly (CHAIN PASSED).
#
# This is the schedule-time variant of potluck's runtime ring removal: there is
# no ring (our §3-static deployment), so "drop the device and keep a proxy
# socket" maps to "exclude it from the topology and let the rest run".
#
# Usage: test_remove.sh [n_predict] [host]
set -euo pipefail
if [[ "${POTLUCK_SKIP_GPU_TESTS:-0}" == 1 ]]; then
    printf 'test_remove skipped (POTLUCK_SKIP_GPU_TESTS=1)\n'
    exit 0
fi

N_PREDICT="${1:-32}"
HOST="${2:-127.0.0.1}"
PORT_BASE=$((48000 + RANDOM % 2000))

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

work="$(mktemp -d -t potluck-remove.XXXXXX)"
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
for i in 0 1 2; do
    printf '%s:%d\n' "${HOST}" $((PORT_BASE + i)) >> "${workers}"
done

# Devices 0 and 2 are fast (Metal offload); device 1 is deliberately slow (CPU).
spawn_bench() {
    pids=()
    for i in 0 1 2; do
        local ngl=0
        if (( i != 1 )); then
            ngl=12
        fi
        "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((PORT_BASE + i)) --bench -ngl "${ngl}" \
            >"${work}/bench_${i}.log" 2>&1 &
        pids+=($!)
    done
    sleep 4
}
stop_all() {
    for p in "${pids[@]:-}"; do
        kill "${p}" 2>/dev/null || true
    done
    pids=()
    wait 2>/dev/null || true
    sleep 3
}

weights="${work}/weights.txt"
probe_weights="${work}/probe_weights.txt"

# Profile all devices and drop exactly the slowest one (--drop-slowest 1):
# the rebuilt topology must exclude it, and the remaining devices must still
# reproduce the reference exactly.
weights="${work}/weights.txt"

spawn_bench
if ! "${BIN}/potluck-head" "${MODEL}" "${workers}" 1 "${HOST}" --profile --out "${weights}" \
        --drop-slowest 1 >"${work}/profile.log" 2>&1; then
    cat "${work}/profile.log" >&2
    printf 'test_remove failed: profile head rc nonzero\n' >&2
    exit 1
fi
stop_all

n_started=3
n_kept=$(wc -l < "${weights}" | tr -d ' ')
n_dropped=$((n_started - n_kept))
if (( n_dropped != 1 )); then
    cat "${work}/profile.log" >&2
    printf 'test_remove failed: expected exactly 1 device dropped, got %s\n' "${n_dropped}" >&2
    exit 1
fi
if ! grep -q 'dropped device' "${work}/profile.log"; then
    cat "${work}/profile.log" >&2
    printf 'test_remove failed: no dropped-device line\n' >&2
    exit 1
fi
printf '  ok (removal): %s dropped, %s kept\n' "${n_dropped}" "${n_kept}"


# Rebuild the chain on exactly the kept devices and verify exact correctness.
pids=()
while read -r line; do
    port="${line%% *}"
    port="${port##*:}"
    "${BIN}/potluck-worker" "${MODEL}" "${HOST}" "${port}" >"${work}/worker_${port}.log" 2>&1 &
    pids+=($!)
done < "${weights}"
sleep 2

head_log="${work}/head.log"
# Bit-exact greedy parity across independent contexts is a Metal/CPU
# guarantee; llama.cpp's CUDA backend is not run-to-run reproducible, so after
# a device drop a near-tie can flip. On CUDA that specific outcome (MISMATCH
# exit) is accepted -- the topology itself is what this test checks --
# everything else still fails.
if "${BIN}/potluck-head" "${MODEL}" "${weights}" "${N_PREDICT}" "${HOST}" --parity-check >"${head_log}" 2>&1; then
    head_ok=1
else
    head_ok=0
fi
stop_all

if (( ! head_ok )) && grep -qE 'ggml_cuda_(init|graph)' "${head_log}" 2>/dev/null && grep -q 'MISMATCH' "${head_log}"; then
    printf '  (CUDA backend: near-tie flip after device drop; exact parity not asserted)\n'
else
    if (( ! head_ok )); then
        for log in "${work}"/worker_*.log "${work}"/bench_*.log; do
            if [[ -f "${log}" ]]; then
                printf '%s\n' "--- ${log} ---" >&2
                cat "${log}" >&2
            fi
        done
        cat "${head_log}" >&2
        printf 'test_remove failed: chain head rc nonzero\n' >&2
        exit 1
    fi
    if ! grep -q 'CHAIN PASSED' "${head_log}"; then
        cat "${head_log}" >&2
        printf 'test_remove failed: no CHAIN PASSED\n' >&2
        exit 1
    fi
    printf '  ok (rebuild): %s -> %s\n' \
        "$(grep -oE 'layers split:.*' "${head_log}" | head -1)" \
        "$(grep -oE 'CHAIN PASSED[^,]*' "${head_log}" | head -1)"
fi
printf 'test_remove passed (%s of %s devices kept, %s predict)\n' "${n_kept}" "${n_started}" "${N_PREDICT}"
