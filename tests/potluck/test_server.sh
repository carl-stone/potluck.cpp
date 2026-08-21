#!/usr/bin/env bash
# Behavioral HTTP test for potluck-server.
# Usage: test_server.sh [n_workers] [n_predict] [host] [port]
set -euo pipefail

N_WORKERS="${1:-2}"
N_PREDICT="${2:-8}"
HOST="${3:-127.0.0.1}"
PORT="${4:-$((8100 + RANDOM % 400))}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
# shellcheck source=../../scripts/potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"
MODEL="${MODEL:-$(potluck_model_path)}"

source "${REPO}/tests/potluck/test_helpers.sh"
if [[ ! -x "${BIN}/potluck-server" || ! -x "${BIN}/potluck-worker" || ! -x "${BIN}/llama-cli" ]]; then
    printf 'missing binaries (build potluck-server, potluck-worker, llama-cli first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 2
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/potluck-server.XXXXXX")
SRV=""
stop_server() {
    if [[ -n "${SRV}" ]]; then
        kill "${SRV}" 2>/dev/null || true
        wait "${SRV}" 2>/dev/null || true
        SRV=""
    fi
    pkill -f potluck-wor[k]er 2>/dev/null || true
    sleep 1
}
cleanup() {
    local rc=$?
    if (( rc != 0 )) && [[ -f "${WORK}/server.log" ]]; then
        printf '%s\n' '--- server.log ---' >&2
        cat "${WORK}/server.log" >&2
    fi
    stop_server
    rm -rf "${WORK}" 2>/dev/null || true
    return "${rc}"
}
trap cleanup EXIT

"${BIN}/potluck-server" -m "${MODEL}" --workers "${N_WORKERS}" --port "${PORT}" \
    >"${WORK}/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
    grep -q 'listening on http' "${WORK}/server.log" 2>/dev/null && break
    if ! kill -0 "${SRV}" 2>/dev/null; then
        sed -n '$p' "${WORK}/server.log" >&2
        printf 'server exited during startup\n' >&2
        exit 1
    fi
    sleep 1
done
grep -q 'listening on http' "${WORK}/server.log" || {
    sed -n '$p' "${WORK}/server.log" >&2
    printf 'server never became ready\n' >&2
    exit 1
}

auth_header=(-H 'Content-Type: application/json')
FRANCE=$(curl -fsS "${auth_header[@]}" -d '{"prompt":"The capital of France is","n_predict":8}' \
    "http://${HOST}:${PORT}/completion")
EIFFEL=$(curl -fsS "${auth_header[@]}" -d '{"prompt":"The Eiffel Tower is in","n_predict":8}' \
    "http://${HOST}:${PORT}/completion")
python3 - "${FRANCE}" "${EIFFEL}" <<'PY'
import json, sys
first, second = map(json.loads, sys.argv[1:])
for result in (first, second):
    assert isinstance(result.get("content"), str) and result["content"]
    assert result["finish_reason"] == "stop"
    assert isinstance(result["n_predict"], int) and result["n_predict"] > 0
assert first["content"] != second["content"], (first, second)
PY

CHAT_REQ='{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8,"reasoning_effort":"none"}'
CHAT=$(curl -fsS "${auth_header[@]}" -d "${CHAT_REQ}" "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${CHAT}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result["object"] == "chat.completion"
assert result["choices"][0]["message"]["role"] == "assistant"
assert result["choices"][0]["message"]["content"]
assert result["choices"][0]["finish_reason"] == "stop"
usage = result["usage"]
assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
PY

MISSING_STATUS=$(curl -sS -o "${WORK}/missing.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{}' "http://${HOST}:${PORT}/completion")
[[ "${MISSING_STATUS}" == 400 ]]
python3 - "${WORK}/missing.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["error"] == "missing prompt"
PY

BAD_JSON_STATUS=$(curl -sS -o "${WORK}/bad-json.json" -w '%{http_code}' \
    "${auth_header[@]}" --data-binary '{' "http://${HOST}:${PORT}/completion")
[[ "${BAD_JSON_STATUS}" == 400 ]]
python3 - "${WORK}/bad-json.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["error"].startswith("invalid JSON:")
PY

INVALID_NUMBER_STATUS=$(curl -sS -o "${WORK}/invalid-number.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n_predict":"eight"}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_NUMBER_STATUS}" == 400 ]]
python3 - "${WORK}/invalid-number.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["error"].startswith("invalid n_predict:")
PY

INVALID_NEGATIVE_STATUS=$(curl -sS -o "${WORK}/negative-number.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n_predict":-1}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_NEGATIVE_STATUS}" == 400 ]]
python3 - "${WORK}/negative-number.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["error"].startswith("invalid n_predict:")
PY

INVALID_LARGE_STATUS=$(curl -sS -o "${WORK}/large-number.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n_predict":4294967296}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_LARGE_STATUS}" == 400 ]]
python3 - "${WORK}/large-number.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["error"].startswith("invalid n_predict:")
PY

INVALID_STREAM_STATUS=$(curl -sS -o "${WORK}/invalid-stream.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","stream":"yes"}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_STREAM_STATUS}" == 400 ]]
python3 - "${WORK}/invalid-stream.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["error"].startswith("invalid stream:")
PY

NOT_FOUND_STATUS=$(curl -sS -o "${WORK}/not-found.json" -w '%{http_code}' \
    "http://${HOST}:${PORT}/not-found")
[[ "${NOT_FOUND_STATUS}" == 404 ]]
python3 - "${WORK}/not-found.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["error"] == "not found"
PY

OPTIONS_STATUS=$(curl -sS -o /dev/null -w '%{http_code}' -X OPTIONS \
    "${auth_header[@]}" "http://${HOST}:${PORT}/completion")
[[ "${OPTIONS_STATUS}" == 204 ]]

HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
python3 - "${HEALTH}" "${N_WORKERS}" <<'PY'
import json, sys
health, expected = json.loads(sys.argv[1]), int(sys.argv[2])
assert health["status"] == "ok"
assert health["workers"] == expected
windows = sorted(health["windows"], key=lambda w: w["start"])
assert windows and windows[0]["start"] == 0
for previous, current in zip(windows, windows[1:]):
    assert current["start"] == previous["end"]
assert all(w["start"] < w["end"] for w in windows)
assert all(0 <= w["owner"] < expected for w in windows)
assert [w["index"] for w in windows] == list(range(len(windows)))
PY

MODELS=$(curl -fsS "http://${HOST}:${PORT}/v1/models")
python3 - "${MODELS}" "$(basename "${MODEL}")" <<'PY'
import json, sys
models, model = json.loads(sys.argv[1]), sys.argv[2]
assert models["object"] == "list"
assert models["data"] and models["data"][0]["id"] == model
PY

SSE=$(curl -fsS -N "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":3,"stream":true,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${SSE}" <<'PY'
import json, sys
lines = [line[6:] for line in sys.argv[1].splitlines() if line.startswith("data: ")]
assert lines and lines[-1] == "[DONE]"
chunks = [json.loads(line) for line in lines[:-1]]
assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) 
PY
RAW_SSE=$(curl -fsS -N "${auth_header[@]}" -d \
    '{"prompt":"Say hi","n_predict":3,"stream":true}' \
    "http://${HOST}:${PORT}/completion")
python3 - "${RAW_SSE}" <<'PY'
import json, sys
lines = [line[6:] for line in sys.argv[1].splitlines() if line.startswith("data: ")]
assert lines and lines[-1] == "[DONE]"
chunks = [json.loads(line) for line in lines[:-1]]
assert chunks and "".join(c.get("content", "") for c in chunks)
PY


stop_server
if [[ "${POTLUCK_SKIP_CLI_PARITY:-0}" == 1 ]]; then
    python3 - "${CHAT}" <<'PY'
import json, sys
chat_text = json.loads(sys.argv[1])["choices"][0]["message"]["content"]
assert chat_text
PY
    printf '  llama-cli text parity skipped (POTLUCK_SKIP_CLI_PARITY=1)\n'
else
    PROMPT='The capital of France is'
    CLI_OUT=$("${BIN}/llama-cli" -m "${MODEL}" -p "${PROMPT}" -n 8 --temp 0 --seed 1 -no-cnv -st \
        --no-display-prompt --chat-template-kwargs '{"enable_thinking": false}' --log-disable </dev/null 2>/dev/null)
    python3 - "${CHAT}" "${CLI_OUT}" "${PROMPT}" <<'PY'
import json, sys
chat, cli, prompt = sys.argv[1:]
chat_text = json.loads(chat)["choices"][0]["message"]["content"]
lines = cli.splitlines()
marker = next(i for i, line in enumerate(lines) if line.strip() == f"> {prompt}")
generated = []
for line in lines[marker + 1:]:
    if line.startswith("[ Prompt:"):
        break
    generated.append(line)
while generated and not generated[-1]:
    generated.pop()
expected = "\n".join(generated)
assert chat_text == expected, (chat_text, expected)
PY
fi

printf 'POTLUCK-SERVER TEST PASSED: %s workers, prompt/error/health/models/parity/SSE checks\n' "${N_WORKERS}"
