#!/usr/bin/env bash
# §11 speculative decoding: --draft <model> (self-draft) proposes draft_n
# greedy tokens per round and the chain verifies them one at a time against
# its own sequential argmax. The DoD: the accepted/committed tokens must be
# identical to a no-spec greedy run of the same chain.
#
# Usage: test_spec.sh [n_workers] [n_predict] [host]
set -euo pipefail

N_WORKERS="${1:-3}"
N_PREDICT="${2:-24}"
HOST="${3:-127.0.0.1}"
DRAFT_N="${DRAFT_N:-4}"
PORT_BASE=$((52000 + RANDOM % 3000))

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"
PROMPT="${PROMPT:-The capital of France is}"

if [[ ! -x "${BIN}/potluck-head" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing binaries (build potluck-head/potluck-worker first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing Qwen3.5 fixture: %s\n' "${MODEL}" >&2
    exit 2
fi

work="$(mktemp -d -t potluck-spec.XXXXXX)"
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
start_workers() {
    : > "${workers}"
    for i in $(seq 0 $((N_WORKERS - 1))); do
        printf '%s:%d\n' "${HOST}" $((PORT_BASE + i)) >> "${workers}"
    done
    for i in $(seq 0 $((N_WORKERS - 1))); do
        "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((PORT_BASE + i)) >"${work}/worker_${i}.log" 2>&1 &
        pids+=($!)
    done
    sleep 2
}

# A borderline greedy near-tie can flip run-to-run on the 0.8B Metal
# fixture, so the spec-vs-no-spec comparison is retried with fresh workers; a
# systematic verification bug fails every retry, a transient near-tie flip
# passes on the first clean comparison.
attempt=0
while (( attempt < 3 )); do
    attempt=$((attempt + 1))
    # Spec run. Workers exit once the head disconnects, so restart them for
    # the no-spec reference run.
    start_workers
    spec_log="${work}/spec_${attempt}.log"
    if ! "${BIN}/potluck-head" "${MODEL}" "${workers}" "${N_PREDICT}" "${HOST}" \
            --draft "${MODEL}" --draft-n "${DRAFT_N}" -p "${PROMPT}" >"${spec_log}" 2>&1; then
        tail -20 "${spec_log}" >&2
        printf 'spec test failed (head rc nonzero, attempt %d)\n' "${attempt}" >&2
        exit 1
    fi

    for p in "${pids[@]:-}"; do kill "${p}" 2>/dev/null || true; done
    wait 2>/dev/null || true
    pids=()
    PORT_BASE=$((PORT_BASE + 1000))
    start_workers
    plain_log="${work}/plain_${attempt}.log"
    if ! "${BIN}/potluck-head" "${MODEL}" "${workers}" "${N_PREDICT}" "${HOST}" \
            -p "${PROMPT}" >"${plain_log}" 2>&1; then
        tail -20 "${plain_log}" >&2
        printf 'spec test failed (no-spec head rc nonzero, attempt %d)\n' "${attempt}" >&2
        exit 1
    fi

    spec="$(grep -oE 'spec_tokens\([0-9]+\):[^$]*' "${spec_log}" | head -1 || true)"
    plain="$(grep -oE 'chain_tokens\([0-9]+\):[^$]*' "${plain_log}" | head -1 || true)"
    if [[ -n "${spec}" && -n "${plain}" ]]; then
        spec_toks="$(printf '%s' "${spec}" | cut -d: -f2-)"
        plain_toks="$(printf '%s' "${plain}" | cut -d: -f2-)"
        if [[ "${spec_toks}" == "${plain_toks}" ]]; then
            grep -E 'head: spec_tokens|head: spec accept' "${spec_log}"
            printf 'SPEC PASSED: %d workers, %d drafts/round, accepted output matches no-spec run\n' \
                "${N_WORKERS}" "${DRAFT_N}"
            exit 0
        fi
    fi
    for p in "${pids[@]:-}"; do kill "${p}" 2>/dev/null || true; done
    wait 2>/dev/null || true
    pids=()
    PORT_BASE=$((PORT_BASE + 1000))
done

printf 'SPEC MISMATCH after %d attempts\n  spec: %s\n nospc:%s\n' \
    "${attempt}" "${spec_toks:-missing}" "${plain_toks:-missing}" >&2
tail -20 "${spec_log}" >&2
exit 1