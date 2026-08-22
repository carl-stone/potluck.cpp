#!/usr/bin/env bash
# One-command device setup for Potluck. A source checkout builds the runtime;
# a staged portable payload installs workers without a compiler or model copy.
#
# Usage: bash scripts/install.sh [--prefix DIR] [--jobs N] [--no-server]
#                                [--skip-build] [--payload DIR] [--no-model]
#                                [--start]
#
# After it finishes:
#   worker device:  ~/potluck/potluck-node
#   head device:    ~/potluck/potluck-server -m ~/potluck/<model file>
#
# A payload install copies only the staged binaries and libraries. The head
# downloads the pinned model on demand; worker payload installs use --no-model.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO}/scripts/potluck-safe.sh"
# shellcheck source=potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"
payload_for_platform() {
    case "$1" in
        'Darwin arm64'*|'Darwin aarch64'*) printf 'mac-arm64\n' ;;
        'Linux x86_64'*|'Linux amd64'*) printf 'linux-x86_64\n' ;;
        'Linux aarch64'*) printf 'linux-aarch64\n' ;;
        'Linux arm64'*) printf 'linux-arm64\n' ;;
        *) return 1 ;;
    esac
}

prefix="${HOME}/potluck"
jobs="$(potluck_build_jobs)"
build_dir="${POTLUCK_BUILD_DIR:-${REPO}/build}"
build_server=1
skip_build=0
payload_dir="${POTLUCK_PAYLOAD_DIR:-}"
no_model=0
start=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) [[ $# -ge 2 ]] || { echo "install: --prefix needs a value" >&2; exit 2; }; prefix="$2"; shift 2 ;;
        --jobs)   [[ $# -ge 2 ]] || { echo "install: --jobs needs a value" >&2; exit 2; }; jobs="$2"; shift 2 ;;
        --no-server) build_server=0; shift ;;
        --skip-build) skip_build=1; shift ;;
        --payload)
            [[ $# -ge 2 && -n "$2" ]] || { echo "install: --payload needs a directory" >&2; exit 2; }
            payload_dir="$2"
            skip_build=1
            build_server=0
            no_model=1
            shift 2
            ;;
        --no-model) no_model=1; shift ;;
        --start) start=1; shift ;;
        -h|--help) grep '^#' "$0" | cut -c3-; exit 0 ;;
        *) echo "install: unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${payload_dir}" && "${skip_build}" == "1" ]]; then
    case "$(uname -s) $(uname -m)" in
        'Darwin arm64'|'Darwin aarch64') candidate="${REPO}/dist/mac-arm64" ;;
        'Linux x86_64'|'Linux amd64') candidate="${REPO}/dist/linux-x86_64" ;;
        'Linux aarch64') candidate="${REPO}/dist/linux-aarch64" ;;
        'Linux arm64') candidate="${REPO}/dist/linux-arm64" ;;
        *) candidate="" ;;
    esac
    if [[ -n "${candidate}" && -f "${candidate}/potluck-build-id" ]]; then
        payload_dir="${candidate}"
        no_model=1
    fi
fi
if [[ -n "${payload_dir}" ]]; then
    no_model=1
fi

missing=""
if [[ -z "${payload_dir}" && "${skip_build}" != "1" ]]; then
    for tool in cmake pkg-config; do
        command -v "${tool}" >/dev/null 2>&1 || missing="${missing} ${tool}"
    done
    if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
        missing="${missing} c++"
    fi
    if ! pkg-config --exists libzmq 2>/dev/null; then
        missing="${missing} libzmq(zeromq)"
    fi
fi
if [[ "${no_model}" != "1" ]] && ! command -v curl >/dev/null 2>&1; then
    missing="${missing} curl"
fi
if [[ -n "${missing}" ]]; then
    printf 'install: missing required tools:%s\n' "${missing}" >&2
    printf 'install: install them and run this script again\n' >&2
    exit 2
fi
if [[ -n "${payload_dir}" ]]; then
    no_model=1
    [[ -d "${payload_dir}" && ! -L "${payload_dir}" ]] || {
        printf 'install: payload directory does not exist: %s\n' "${payload_dir}" >&2
        exit 2
    }
    [[ -f "${payload_dir}/potluck-build-id" &&
       ! -L "${payload_dir}/potluck-build-id" ]] || {
        printf 'install: payload has no potluck-build-id: %s\n' "${payload_dir}" >&2
        exit 2
    }
    expected_platform="$(payload_for_platform "$(uname -sm)")" || {
        printf 'install: unsupported device platform: %s\n' "$(uname -sm)" >&2
        exit 2
    }
    if command -v sha256sum >/dev/null 2>&1; then
        sha256_file() { sha256sum "$1" | cut -d' ' -f1; }
    elif command -v shasum >/dev/null 2>&1; then
        sha256_file() { shasum -a 256 "$1" | cut -d' ' -f1; }
    else
        printf 'install: payload mode needs sha256sum or shasum\n' >&2
        exit 2
    fi
    payload_files=()
    payload_platform=""
    commit_seen=0
    while read -r field value extra; do
        [[ -n "${field:-}" ]] || continue
        case "${field}" in
            platform)
                [[ -n "${value:-}" && -z "${extra:-}" &&
                   -z "${payload_platform}" ]] || {
                    printf 'install: invalid payload platform record\n' >&2
                    exit 2
                }
                payload_platform="${value}"
                ;;
            commit)
                [[ -n "${value:-}" && -z "${extra:-}" &&
                   "${commit_seen}" -eq 0 ]] || {
                    printf 'install: invalid payload commit record\n' >&2
                    exit 2
                }
                commit_seen=1
                ;;
            *)
                checksum="${field}"
                name="${value:-}"
                [[ -z "${extra:-}" &&
                   "${checksum}" =~ ^[[:xdigit:]]{64}$ &&
                   "${name}" =~ ^[A-Za-z0-9._+][A-Za-z0-9._+-]*$ &&
                   "${name}" != potluck-build-id &&
                   "${name}" != potluck-deploy.sha256 &&
                   -f "${payload_dir}/${name}" &&
                   ! -L "${payload_dir}/${name}" ]] || {
                    printf 'install: invalid payload checksum record\n' >&2
                    exit 2
                }
                actual="$(sha256_file "${payload_dir}/${name}")"
                [[ "${actual}" == "${checksum}" ]] || {
                    printf 'install: payload checksum mismatch: %s\n' "${name}" >&2
                    exit 2
                }
                found=0
                for known in "${payload_files[@]}"; do
                    [[ "${known}" == "${name}" ]] && found=1
                done
                [[ "${found}" == 0 ]] || {
                    printf 'install: duplicate payload checksum: %s\n' "${name}" >&2
                    exit 2
                }
                payload_files+=("${name}")
                ;;
        esac
    done < "${payload_dir}/potluck-build-id"
    [[ "${commit_seen}" -eq 1 && -n "${payload_platform}" &&
       "${payload_platform}" == "${expected_platform}" ]] || {
        printf 'install: payload platform does not match %s: %s\n' \
            "${expected_platform}" "${payload_platform:-missing}" >&2
        exit 2
    }
    for required in potluck-node potluck-worker; do
        found=0
        for known in "${payload_files[@]}"; do
            [[ "${known}" == "${required}" ]] && found=1
        done
        [[ "${found}" == 1 ]] || {
            printf 'install: payload manifest does not checksum %s\n' "${required}" >&2
            exit 2
        }
    done
    for name in "${payload_files[@]}"; do
        [[ -f "${payload_dir}/${name}" && ! -L "${payload_dir}/${name}" ]] || {
            printf 'install: payload missing %s\n' "${name}" >&2
            exit 2
        }
    done
    [[ -x "${payload_dir}/potluck-node" &&
       -x "${payload_dir}/potluck-worker" ]] || {
        printf 'install: payload worker binaries are not executable\n' >&2
        exit 2
    }
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
    potluck_require_disk "${build_dir}" "${POTLUCK_MIN_DISK_GIB:-10}"
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
    POTLUCK_BUILD_JOBS="${jobs}" potluck_build "${build_dir}" --config Release \
        --target "${targets[@]}"
fi

mkdir -p "${prefix}"
if [[ -n "${payload_dir}" ]]; then
    for name in "${payload_files[@]}"; do
        cp -f "${payload_dir}/${name}" "${prefix}/${name}"
    done
    cp -f "${payload_dir}/potluck-build-id" "${prefix}/potluck-build-id"
    chmod +x "${prefix}/potluck-node" "${prefix}/potluck-worker"
else
    bin_dir="${build_dir}/bin"
    for name in potluck-node potluck-worker; do
        [[ -x "${bin_dir}/${name}" ]] || {
            printf 'install: missing built binary: %s\n' "${bin_dir}/${name}" >&2
            exit 2
        }
        cp -f "${bin_dir}/${name}" "${prefix}/${name}"
    done
    if [[ "${build_server}" == "1" ]]; then
        [[ -x "${bin_dir}/potluck-server" ]] || {
            printf 'install: missing built binary: %s\n' "${bin_dir}/potluck-server" >&2
            exit 2
        }
        cp -f "${bin_dir}/potluck-server" "${prefix}/potluck-server"
    fi
fi

model_path=""
if [[ "${no_model}" != "1" ]]; then
    printf 'install: fetching pinned model %s\n' "${POTLUCK_MODEL_FILE}"
    model_path="$(bash "${REPO}/scripts/fetch-model.sh" --dest "${prefix}")"
fi

printf '\ninstalled to %s:\n' "${prefix}"
ls -lh "${prefix}" | sed -n '2,$p'
printf '\nnext steps:\n'
printf '  worker device: %s/potluck-node\n' "${prefix}"
if [[ -n "${model_path}" && -x "${prefix}/potluck-server" ]]; then
    model_file="$(basename "${model_path}")"
    printf '  head device:   %s/potluck-server -m %s/%s\n' \
        "${prefix}" "${prefix}" "${model_file}"
fi

if [[ "${start}" == "1" ]]; then
    exec "${prefix}/potluck-node"
fi
