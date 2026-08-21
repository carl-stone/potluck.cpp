#!/usr/bin/env bash
# End-to-end correctness for the dynamic potluck-style chain: the coordinator
# tessellates the model across N worker stages and the generated tokens must
# exactly match a full-model reference run.
#
# Usage: test_chain.sh [n_workers] [n_predict] [host]
#
# Worker-count ceiling: 7 workers fails on a 16 GB M4 because the head's
# full-model reference plus 7 concurrent Metal contexts exceed device memory
# (decode aborts with a backend allocation assert). 2-6 workers is the
# verified range on this machine.
set -euo pipefail

N_WORKERS="${1:-3}"
N_PREDICT="${2:-48}"
HOST="${3:-127.0.0.1}"
# Random per-run port base so a slow-exiting worker from a previous run can
# never collide with a fresh run's listeners.
PORT_BASE=$((44000 + RANDOM % 3000))

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"
worker_models=()
if [[ -n "${POTLUCK_WORKER_MODELS:-}" ]]; then
    IFS=',' read -r -a worker_models <<<"${POTLUCK_WORKER_MODELS}"
    [[ "${#worker_models[@]}" -eq "${N_WORKERS}" ]] || {
        printf 'POTLUCK_WORKER_MODELS must contain %s paths\n' "${N_WORKERS}" >&2
        exit 2
    }
fi

if [[ ! -x "${BIN}/potluck-head" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing binaries (build potluck-head/potluck-worker first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing Qwen3.5 fixture: %s\n' "${MODEL}" >&2
    exit 2
fi

work="$(mktemp -d -t potluck-chain.XXXXXX)"
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

ngl_args=()
if [[ -n "${POTLUCK_NGL:-}" ]]; then
    ngl_args=(-ngl "${POTLUCK_NGL}")
fi

# Optional prompt override passed to potluck-head (greedy runs only here; the
# sampler gets its own self-check in test_sampler.sh).
extra_args=()
parity_args=()
if [[ "${POTLUCK_PARITY_CHECK:-1}" != 0 ]]; then
    parity_args+=(--parity-check)
fi
if [[ -n "${POTLUCK_PROMPT:-}" ]]; then
    extra_args+=(-p "${POTLUCK_PROMPT}")
fi
if [[ "${POTLUCK_BENCH:-0}" == 1 ]]; then
    extra_args+=(--bench)
fi
if [[ -n "${POTLUCK_BOUNDS:-}" ]]; then
    extra_args+=(--bounds "${POTLUCK_BOUNDS}")
fi
for i in $(seq 0 $((N_WORKERS - 1))); do
    worker_model="${MODEL}"
    if [[ "${#worker_models[@]}" -gt 0 ]]; then
        worker_model="${worker_models[i]}"
    fi
    [[ -f "${worker_model}" ]] || { printf 'missing worker model: %s\n' "${worker_model}" >&2; exit 2; }
    "${BIN}/potluck-worker" "${worker_model}" "${HOST}" $((PORT_BASE + i)) "${ngl_args[@]+"${ngl_args[@]}"}" >"${work}/worker_${i}.log" 2>&1 &
    pids+=($!)
done

# Give the workers time to bind their listeners before the coordinator connects.
sleep 2

head_log="${work}/head.log"
if ! "${BIN}/potluck-head" "${MODEL}" "${workers}" "${N_PREDICT}" "${HOST}" "${parity_args[@]+"${parity_args[@]}"}" "${ngl_args[@]+"${ngl_args[@]}"}" "${extra_args[@]+"${extra_args[@]}"}" >"${head_log}" 2>&1; then
    cat "${head_log}" >&2
    printf 'potluck chain test failed (head rc nonzero)\n' >&2
    exit 1
fi

if ! grep -q 'CHAIN PASSED' "${head_log}"; then
    cat "${head_log}" >&2
    printf 'potluck chain test failed (no CHAIN PASSED)\n' >&2
    exit 1
fi

grep -E 'head: [0-9]+ workers|CHAIN PASSED' "${head_log}"
if [[ -n "${POTLUCK_OUTPUT_FILE:-}" ]]; then
    python3 - "${head_log}" "${POTLUCK_OUTPUT_FILE}" <<'PY'
import re
import sys

log = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()
match = re.search(r"head: generated text \(\d+ tokens\):\n(.*?)\nCHAIN (?:PASSED|RUN:)", log, re.S)
if match is None:
    raise SystemExit("head log has no generated text block")
text = match.group(1)
# printf adds one newline after the detokenized text. Keep a real final LF
# emitted by the model, but discard that formatting newline.
if text.endswith("\n"):
    text = text[:-1]
open(sys.argv[2], "w", encoding="utf-8").write(text)
PY
fi
if [[ -n "${POTLUCK_PROMPT:-}" ]] && ! grep -q 'head: generated text' "${head_log}"; then
    cat "${head_log}" >&2
    printf 'potluck chain test failed (no generated text output)\n' >&2
    exit 1
fi
printf 'potluck chain test passed (%s workers, %s predict)\n' "${N_WORKERS}" "${N_PREDICT}"
