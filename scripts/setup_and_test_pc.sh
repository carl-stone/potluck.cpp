#!/usr/bin/env bash
# One-shot build + integrated direct-ring suite on a Linux PC.
#
# Run this ON the PC (Linux), from inside a potluck.cpp checkout. It builds
# the retained potluck worker and server binaries, downloads the Qwen3.5
# fixture, and runs the whole test suite.
#
# Env:
#   REPO       checkout to build        (default: this repo)
#   MODEL_DIR  fixture directory        (default: $REPO/models)
#   NPROC      parallel build jobs      (default: potluck_build_jobs)
#   SKIP_BUILD skip configure/build     (set 1 to only fetch model + run tests)
#   POTLUCK_CUDA_ARCHITECTURES CUDA architectures (default: native)
#
# Usage: bash setup_and_test_pc.sh [n_predict=32] [host=127.0.0.1]
set -euo pipefail

N_PREDICT="${1:-32}"
HOST="${2:-127.0.0.1}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO}/scripts/potluck-safe.sh"
POTLUCK_MODEL_DIR="${POTLUCK_MODEL_DIR:-${MODEL_DIR:-}}"
export POTLUCK_MODEL_DIR
NPROC="${NPROC:-$(potluck_build_jobs)}"

for tool in git cmake g++ pkg-config curl; do
    command -v "${tool}" >/dev/null 2>&1 || { echo "missing build tool: ${tool}" >&2; exit 2; }
done

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    if [[ ! -f "${REPO}/CMakeLists.txt" ]]; then
        echo "${REPO} is not a potluck.cpp checkout" >&2
        exit 2
    fi
    cd "${REPO}"
    potluck_require_disk "${REPO}/build" "${POTLUCK_MIN_DISK_GIB:-10}"

    echo "== configuring =="
    cmake_opts=(-DCMAKE_BUILD_TYPE=Release -DGGML_CPU=ON -DLLAMA_BUILD_TESTS=ON -DPOTLUCK_HIGHS=ON)
    NVCC="$(command -v nvcc 2>/dev/null || true)"
    if [[ -z "${NVCC}" && -x /opt/cuda/bin/nvcc ]]; then
        NVCC=/opt/cuda/bin/nvcc # Arch installs CUDA off-PATH
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${NVCC}" ]]; then
        cmake_opts+=(-DGGML_CUDA=ON -DGGML_NATIVE=ON)
        if [[ -n "${POTLUCK_CUDA_ARCHITECTURES:-}" ]]; then
            cmake_opts+=("-DCMAKE_CUDA_ARCHITECTURES=${POTLUCK_CUDA_ARCHITECTURES}")
        fi
        export PATH="$(dirname "${NVCC}"):${PATH}"
        echo "   (CUDA backend enabled via ${NVCC}; architectures ${POTLUCK_CUDA_ARCHITECTURES:-native when supported})"
    else
        echo "   (no nvidia-smi or no CUDA toolkit (nvcc); CPU-only build)"
    fi
    cmake -S . -B build "${cmake_opts[@]}" >/tmp/potluck_pc_cmake.log 2>&1 || {
        tail -20 /tmp/potluck_pc_cmake.log >&2; exit 2; }
    echo "== building (${NPROC} jobs) =="
    POTLUCK_BUILD_JOBS="${NPROC}" potluck_build "${REPO}/build" \
        --target potluck-node potluck-worker potluck-server potluck-cli llama-cli test-potluck-discovery test-potluck-protocol test-potluck-probe test-potluck-transport test-potluck-halda test-potluck-qwen35-stages test-potluck-run \
        >/tmp/potluck_pc_build.log 2>&1 || {
        tail -20 /tmp/potluck_pc_build.log >&2; exit 2; }
    echo "build ok"
else
    git -C "${REPO}" rev-parse --show-toplevel >/dev/null 2>&1 || {
        echo "SKIP_BUILD=1 but ${REPO} is not a checkout" >&2; exit 2; }
fi

echo "== fetching fixture =="
MODEL_PATH="$(bash "${REPO}/scripts/fetch-model.sh")"
ls -l "${MODEL_PATH}"

echo "== running the suite (n_predict=${N_PREDICT}, host=${HOST}) =="
export REPO="${REPO}"
export BIN="${REPO}/build/bin"
export MODEL="${MODEL_PATH}"
export N_PREDICT HOST
bash "${REPO}/tests/potluck/run_all.sh"
