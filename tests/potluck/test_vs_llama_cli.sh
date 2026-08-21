#!/usr/bin/env bash
# Compare the OpenAI chat endpoint with llama-cli's greedy single-turn path.
# Usage: test_vs_llama_cli.sh [n_workers] [n_predict] [host] [port]
set -euo pipefail
if [[ "${POTLUCK_SKIP_CLI_PARITY:-0}" == 1 ]]; then
    printf 'test_vs_llama_cli skipped (POTLUCK_SKIP_CLI_PARITY=1)\n'
    exit 0
fi

N_WORKERS="${1:-2}"
N_PREDICT="${2:-16}"
HOST="${3:-127.0.0.1}"
PORT="${4:-$((8500 + RANDOM % 400))}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"

for executable in potluck-server potluck-worker llama-cli; do
    [[ -x "${BIN}/${executable}" ]] || { printf 'missing %s in %s\n' "${executable}" "${BIN}" >&2; exit 2; }
done
[[ -f "${MODEL}" ]] || { printf 'missing model: %s\n' "${MODEL}" >&2; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/potluck-vs-cli.XXXXXX")
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
    kill -0 "${SRV}" 2>/dev/null || { sed -n '$p' "${WORK}/server.log" >&2; exit 1; }
    sleep 1
done
grep -q 'listening on http' "${WORK}/server.log"

python3 - "${HOST}" "${PORT}" "${N_PREDICT}" "${WORK}" <<'PY'
import json, sys, urllib.request
host, port, n_predict, work = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
for index, prompt in enumerate(("The capital of France is", "The Eiffel Tower is in")):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "reasoning_effort": "none",
    }).encode()
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        result = json.load(response)
    with open(f"{work}/server_{index}.json", "w", encoding="utf-8") as output:
        json.dump({"prompt": prompt, "content": result["choices"][0]["message"]["content"]}, output)
PY
stop_server
python3 - "${BIN}" "${MODEL}" "${N_PREDICT}" "${WORK}" <<'PY'
import json, subprocess, sys
bin_dir, model, n_predict, work = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
for index in (0, 1):
    saved = json.load(open(f"{work}/server_{index}.json", encoding="utf-8"))
    prompt = saved["prompt"]
    command = [
        f"{bin_dir}/llama-cli", "-m", model, "-p", prompt,
        "-n", str(n_predict), "--temp", "0", "--seed", "1",
        "-no-cnv", "-st", "--no-display-prompt",
        "--chat-template-kwargs", '{"enable_thinking": false}',
        "--log-disable",
    ]
    output = subprocess.run(command, input="", stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True, text=True).stdout
    lines = output.splitlines()
    marker = next(i for i, line in enumerate(lines) if line.strip() == f"> {prompt}")
    generated = []
    for line in lines[marker + 1:]:
        if line.startswith("[ Prompt:"):
            break
        generated.append(line)
    while generated and not generated[-1]:
        generated.pop()
    expected = "\n".join(generated)
    assert saved["content"] == expected, (prompt, saved["content"], expected)
print("llama-cli parity passed for 2 prompts")
PY
