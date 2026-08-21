#!/usr/bin/env bash
# One-command device setup for Potluck engineering use.
#
# Builds the runtime binaries, fetches the pinned fixture model, and installs
# everything flat into one prefix directory (default: ~/potluck). The default
# prefix matches what potluck-server expects on remote worker devices, so
# worker machines should keep it.
#
# Usage: bash scripts/install.sh [--prefix DIR] [--jobs N] [--no-server]
#                                [--skip-build] [--start]
#
# After it finishes:
#   worker device:  ~/potluck/potluck-node
#   head device:    ~/potluck/potluck-server -m ~/potluck/<model file>
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"

prefix="${HOME}/potluck"
jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
build_dir="${POTLUCK_BUILD_DIR:-${REPO}/build}"
build_server=1
skip_build=0
start=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) [[ $# -ge 2 ]] || { echo "install: --prefix needs a value" >&2; exit 2; }; prefix="$2"; shift 2 ;;
        --jobs)   [[ $# -ge 2 ]] || { echo "install: --jobs needs a value" >&2; exit 2; }; jobs="$2"; shift 2 ;;
        --no-server) build_server=0; shift ;;
        --skip-build) skip_build=1; shift ;;
        --start) start=1; shift ;;
        -h|--help) grep '^#' "$0" | cut -c3-; exit 0 ;;
        *) echo "install: unknown option: $1" >&2; exit 2 ;;
    esac
done

missing=""
for tool in cmake curl pkg-config; do
    command -v "${tool}" >/dev/null 2>&1 || missing="${missing} ${tool}"
done
if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
    missing="${missing} c++"
fi
if ! pkg-config --exists libzmq 2>/dev/null; then
    missing="${missing} libzmq(zeromq)"
fi
if [[ -n "${missing}" ]]; then
    printf 'install: missing required tools:%s\n' "${missing}" >&2
    printf 'install: install them and run this script again\n' >&2
    exit 2
fi

if [[ "${skip_build}" != "1" ]]; then
    if [[ ! -f "${REPO}/CMakeLists.txt" ]]; then
        echo "install: ${REPO} is not a potluck.cpp checkout" >&2
        exit 2
    fi
    server_flag=ON
    if [[ "${build_server}" != "1" ]]; then
        server_flag=OFF
    fi
    printf 'install: configuring release build in %s\n' "${build_dir}"
    cmake -S "${REPO}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER="${server_flag}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_APP=OFF \
        -DPOTLUCK_HIGHS=OFF
    targets=(potluck-node potluck-worker)
    if [[ "${build_server}" == "1" ]]; then
        targets+=(potluck-server)
    fi
    printf 'install: building %s with %s jobs\n' "${targets[*]}" "${jobs}"
    cmake --build "${build_dir}" --config Release --parallel "${jobs}" \
        --target "${targets[@]}"
fi

printf 'install: fetching pinned model %s\n' "${POTLUCK_MODEL_FILE}"
model_path="$(bash "${REPO}/scripts/fetch-model.sh" --dest "${prefix}")"

mkdir -p "${prefix}"
bin_dir="${build_dir}/bin"
for name in potluck-node potluck-worker; do
    cp -f "${bin_dir}/${name}" "${prefix}/${name}"
done
if [[ "${build_server}" == "1" ]]; then
    cp -f "${bin_dir}/potluck-server" "${prefix}/potluck-server"
fi
model_file="$(basename "${model_path}")"
printf '\ninstalled to %s:\n' "${prefix}"
ls -lh "${prefix}" | sed -n '2,$p'
printf '\nnext steps:\n'
printf '  worker device: %s/potluck-node\n' "${prefix}"
printf '  head device:   %s/potluck-server -m %s/%s\n' "${prefix}" "${prefix}" "${model_file}"

if [[ "${start}" == "1" ]]; then
    exec "${prefix}/potluck-node"
fi
