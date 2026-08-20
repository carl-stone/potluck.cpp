#!/usr/bin/env bash
# One-shot build + full-suite run of the potluck tools on the Linux/CUDA PC.
#
# Run this ON the PC (Linux), from inside a potluck.cpp checkout. It builds
# potluck-head + potluck-worker + potluck-server + potluck-shard (CPU, plus
# CUDA when nvidia-smi is present), downloads the Qwen3.5 fixture, and runs
# the whole test suite.
#
# Env:
#   REPO       checkout to build        (default: this repo)
#   MODEL_DIR  fixture directory        (default: $REPO/models)
#   NPROC      parallel build jobs      (default: nproc)
#   SKIP_BUILD skip configure/build     (set 1 to only fetch model + run tests)
#
# Usage: bash setup_and_test_pc.sh [n_predict=32] [host=127.0.0.1]
set -euo pipefail

N_PREDICT="${1:-32}"
HOST="${2:-127.0.0.1}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-${REPO}/models}"
NPROC="${NPROC:-$(nproc 2>/dev/null || echo 4)}"
MODEL_URL="https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_0.gguf"

for tool in git cmake g++ pkg-config curl; do
    command -v "${tool}" >/dev/null 2>&1 || { echo "missing build tool: ${tool}" >&2; exit 2; }
done

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    if [[ ! -f "${REPO}/CMakeLists.txt" ]]; then
        echo "${REPO} is not a potluck.cpp checkout" >&2
        exit 2
    fi
    cd "${REPO}"

    echo "== configuring =="
    cmake_opts=(-DCMAKE_BUILD_TYPE=Release -DGGML_CPU=ON -DLLAMA_BUILD_TESTS=ON)
    NVCC="$(command -v nvcc 2>/dev/null || true)"
    if [[ -z "${NVCC}" && -x /opt/cuda/bin/nvcc ]]; then
        NVCC=/opt/cuda/bin/nvcc # Arch installs CUDA off-PATH
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${NVCC}" ]]; then
        cmake_opts+=(-DGGML_CUDA=ON)
        export PATH="$(dirname "${NVCC}"):${PATH}"
        echo "   (CUDA backend enabled via ${NVCC})"
    else
        echo "   (no nvidia-smi or no CUDA toolkit (nvcc); CPU-only build)"
    fi
    cmake -S . -B build "${cmake_opts[@]}" >/tmp/potluck_pc_cmake.log 2>&1 || {
        tail -20 /tmp/potluck_pc_cmake.log >&2; exit 2; }
    echo "== building (${NPROC} jobs) =="
    cmake --build build --target potluck-head potluck-worker potluck-server potluck-shard -j "${NPROC}" \
        >/tmp/potluck_pc_build.log 2>&1 || {
        tail -20 /tmp/potluck_pc_build.log >&2; exit 2; }
    echo "build ok"
else
    git -C "${REPO}" rev-parse --show-toplevel >/dev/null 2>&1 || {
        echo "SKIP_BUILD=1 but ${REPO} is not a checkout" >&2; exit 2; }
fi

echo "== fetching fixture =="
mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL_DIR}/Qwen3.5-0.8B-Q4_0.gguf" ]]; then
    curl -L --fail --progress-bar -o "${MODEL_DIR}/Qwen3.5-0.8B-Q4_0.gguf" "${MODEL_URL}"
fi
ls -l "${MODEL_DIR}/Qwen3.5-0.8B-Q4_0.gguf"

echo "== running the suite (n_predict=${N_PREDICT}, host=${HOST}) =="
export REPO="${REPO}"
export BIN="${REPO}/build/bin"
export MODEL="${MODEL_DIR}/Qwen3.5-0.8B-Q4_0.gguf"
bash "${REPO}/tests/potluck/run_all.sh" 2>&1 | tail -20
