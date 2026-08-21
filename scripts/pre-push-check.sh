#!/usr/bin/env bash
# Local replacement for the removed potluck.cpp GitHub Actions build.
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
BUILD_DIR="${POTLUCK_BUILD_DIR:-${REPO}/build}"
MODEL="${POTLUCK_TEST_MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"
JOBS="${POTLUCK_BUILD_JOBS:-2}"

if [[ ! -f "${MODEL}" ]]; then
    printf 'pre-push: missing test fixture: %s\n' "${MODEL}" >&2
    printf 'pre-push: set POTLUCK_TEST_MODEL or place the Qwen3.5 0.8B fixture there\n' >&2
    exit 2
fi

case "$(uname -s)" in
    Darwin)
        : "${POTLUCK_SKIP_GPU_TESTS:=0}"
        : "${POTLUCK_SKIP_CLI_PARITY:=0}"
        ;;
    *)
        : "${POTLUCK_SKIP_GPU_TESTS:=1}"
        : "${POTLUCK_SKIP_CLI_PARITY:=1}"
        ;;
esac

export REPO MODEL
export BIN="${BUILD_DIR}/bin"
export POTLUCK_SKIP_GPU_TESTS POTLUCK_SKIP_CLI_PARITY
export POTLUCK_TIMEOUT_HANDSHAKE_S="${POTLUCK_TIMEOUT_HANDSHAKE_S:-60}"
export POTLUCK_TIMEOUT_DECODE_S="${POTLUCK_TIMEOUT_DECODE_S:-300}"

printf 'pre-push: configuring local release build\n'
cmake -S "${REPO}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DPOTLUCK_HIGHS=OFF \
    -DGGML_NATIVE=OFF

printf 'pre-push: building required binaries with %s jobs\n' "${JOBS}"
cmake --build "${BUILD_DIR}" --config Release --parallel "${JOBS}" \
    --target potluck-head potluck-worker potluck-server potluck-shard llama-cli

printf 'pre-push: running local component suite\n'
bash "${REPO}/tests/potluck/run_all.sh"
