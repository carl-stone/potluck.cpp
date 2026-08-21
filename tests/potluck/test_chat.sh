#!/usr/bin/env bash
# §13 prompt/chat/text/streaming self-check for the potluck chain.
#
# The head now supports a chat template (--chat applies the model's built-in
# llama_chat_apply_template to the user message and renders the assistant
# marker) and a streaming output loop (--stream flushes each detokenized token
# to stdout as it is produced). Greedy parity with the full-model reference is
# held through the chat path (the reference is built from the same tokenized,
# template-formatted prompt), so CHAIN PASSED is still the exact gate.
#
# What we assert here:
#   1. --chat CHAIN PASSED: the template-formatted prompt, driven greedily,
#      exactly matches the full-model reference built from the same tokens.
#   2. --chat emits a streamed output ("head: stream:") as tokens are produced.
#   3. --chat materially changes the prompt vs raw -p (chat-formatted and raw
#      prompts of the same message produce different greedy streams), so the
#      template is genuinely applied.
#   4. plain --stream on a raw prompt also streams and passes chain parity.
#
# Usage: test_chat.sh [n_workers] [n_predict] [host]
set -euo pipefail

N_WORKERS="${1:-2}"
N_PREDICT="${2:-24}"
HOST="${3:-127.0.0.1}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"

if [[ ! -x "${BIN}/potluck-head" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing binaries (build potluck-head/potluck-worker first): %s\n' "${BIN}" >&2
    exit 2
fi

work="$(mktemp -d -t potluck-chat.XXXXXX)"
pids=()
cleanup() {
    for p in "${pids[@]:-}"; do kill "${p}" 2>/dev/null || true; done
    wait 2>/dev/null || true
    if [[ -z "${POTLUCK_KEEP_WORK:-}" ]]; then rm -rf "${work}"; fi
}
trap cleanup EXIT

# Start N_WORKERS workers on a fresh port base and run the head once with the
# given args. On success prints the head log; restores the chain state so the
# next scenario starts clean (workers exit once the chain closes anyway).
run_head() {
    local port_base=$((48000 + RANDOM % 3000))
    local workers_file="${work}/workers.txt"
    : > "${workers_file}"
    for i in $(seq 0 $((N_WORKERS - 1))); do
        printf '%s:%d\n' "${HOST}" $((port_base + i)) >> "${workers_file}"
    done
    for i in $(seq 0 $((N_WORKERS - 1))); do
        "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((port_base + i)) >"${work}/worker_${i}.log" 2>&1 &
        pids+=($!)
    done
    sleep 2
    local log="${work}/head.log"
    if ! "${BIN}/potluck-head" "${MODEL}" "${workers_file}" "${N_PREDICT}" "${HOST}" --parity-check "$@" >"${log}" 2>&1; then
        cat "${log}" >&2
        printf 'potluck chat test failed (head rc nonzero)\n' >&2
        exit 1
    fi
    for p in "${pids[@]:-}"; do kill "${p}" 2>/dev/null || true; done
    wait 2>/dev/null || true
    pids=()
    cat "${log}"
}

stream_of() {
    # The streamed pieces are printed raw and can contain newlines, so compare
    # the structured chain_tokens line instead: "head: chain_tokens(24): 303 ..".
    sed -n 's/^head: chain_tokens(.*: \([0-9 ]*\)$/\1/p'
}

MSG="What is the capital of France?"

# 1. Chat template path must hold exact greedy parity with the reference.
chat_out=$(run_head --temp 0 --chat "${MSG}")
if ! grep -q 'CHAIN PASSED' <<<"${chat_out}"; then
    printf 'potluck chat test failed: --chat did not reach CHAIN PASSED\n' >&2
    exit 1
fi

# 2. Chat path streams its output token by token.
if ! grep -q 'head: stream:' <<<"${chat_out}"; then
    printf 'potluck chat test failed: --chat did not stream (no `head: stream:`)\n' >&2
    exit 1
fi

# 3. Chat template materially changes the prompt: same raw message via -p (no
#    template) must run the model on different tokens and so produce a
#    different greedy stream.
raw_out=$(run_head --temp 0 -p "${MSG}" --stream)
chat_stream=$(stream_of <<<"${chat_out}")
raw_stream=$(stream_of <<<"${raw_out}")
if [[ -z "${chat_stream}" || -z "${raw_stream}" || "${chat_stream}" == "${raw_stream}" ]]; then
    printf 'potluck chat test failed: chat template did not change the prompt (identical streams)\n' >&2
    exit 1
fi

# 4. The raw -p run must also hold greedy parity with its own reference.
if ! grep -q 'CHAIN PASSED' <<<"${raw_out}"; then
    printf 'potluck chat test failed: raw -p --stream run did not hold parity\n' >&2
    exit 1
fi

# 5. Plain --stream on a raw prompt actually streams.
if ! grep -q 'head: stream:' <<<"${raw_out}"; then
    printf 'potluck chat test failed: --stream did not stream / hold parity\n' >&2
    exit 1
fi

printf 'potluck chat test passed (chat template + streaming, greedy parity held through chat)\n'
