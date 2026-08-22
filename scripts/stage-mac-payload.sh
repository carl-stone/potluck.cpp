#!/usr/bin/env bash
# Build and stage the portable macOS worker payload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
source "${REPO}/scripts/potluck-safe.sh"
build_dir="${REPO}/build-portable"
out_dir="${REPO}/dist/mac-arm64"

die() {
    printf 'stage-mac-payload: %s\n' "$*" >&2
    exit 1
}

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$(pwd -P)" "$1" ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            [[ $# -ge 2 && -n "$2" ]] || die '--build-dir needs a value'
            build_dir="$2"
            shift 2
            ;;
        --build-dir=*)
            build_dir="${1#*=}"
            [[ -n "${build_dir}" ]] || die '--build-dir needs a value'
            shift
            ;;
        --out)
            [[ $# -ge 2 && -n "$2" ]] || die '--out needs a value'
            out_dir="$2"
            shift 2
            ;;
        --out=*)
            out_dir="${1#*=}"
            [[ -n "${out_dir}" ]] || die '--out needs a value'
            shift
            ;;
        -h|--help)
            printf 'usage: %s [--build-dir DIR] [--out DIR]\n' "$0"
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

build_dir="$(absolute_path "${build_dir}")"
out_dir="$(absolute_path "${out_dir}")"
[[ "${out_dir}" != "/" && "${out_dir}" != "${REPO}" ]] ||
    die '--out must name a payload directory, not the repository root'
[[ ! -L "${out_dir}" ]] || die '--out must not be a symlink'

for command in cmake otool install_name_tool codesign lipo shasum awk git; do
    command -v "${command}" >/dev/null 2>&1 ||
        die "required command not found: ${command}"
done

potluck_require_disk "${build_dir}" "${POTLUCK_MIN_DISK_GIB:-10}"
cmake -S "${REPO}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DPOTLUCK_HIGHS=OFF
potluck_build "${build_dir}" --target potluck-node potluck-worker

worker="${build_dir}/bin/potluck-worker"
node="${build_dir}/bin/potluck-node"
[[ -x "${worker}" ]] || die "missing worker binary: ${worker}"
[[ -x "${node}" ]] || die "missing node binary: ${node}"

for binary in "${node}" "${worker}"; do
    archs="$(lipo -archs "${binary}")" ||
        die "cannot inspect architecture: ${binary}"
    [[ "${archs}" == *arm64* ]] ||
        die "binary is not arm64-capable: ${binary} (${archs})"
done

zmq_path="${POTLUCK_ZMQ:-}"
if [[ -n "${zmq_path}" && ! -f "${zmq_path}" ]]; then
    die "POTLUCK_ZMQ is not a file: ${zmq_path}"
fi
if [[ -z "${zmq_path}" ]] && command -v pkg-config >/dev/null 2>&1; then
    zmq_libdir="$(pkg-config --variable=libdir libzmq 2>/dev/null || true)"
    for candidate in "${zmq_libdir}/libzmq.5.dylib" "${zmq_libdir}/libzmq.dylib"; do
        if [[ -f "${candidate}" ]]; then
            zmq_path="${candidate}"
            break
        fi
    done
fi
if [[ -z "${zmq_path}" ]]; then
    for candidate in \
        /opt/homebrew/opt/zeromq/lib/libzmq.5.dylib \
        /usr/local/opt/zeromq/lib/libzmq.5.dylib; do
        if [[ -f "${candidate}" ]]; then
            zmq_path="${candidate}"
            break
        fi
    done
fi
if [[ -z "${zmq_path}" ]]; then
    zmq_path="$(otool -L "${worker}" |
        awk '$1 ~ /libzmq[^[:space:]]*\.dylib$/ { print $1; exit }')"
fi
[[ -f "${zmq_path}" ]] || die 'cannot resolve a libzmq dynamic library'

out_parent="$(dirname "${out_dir}")"
mkdir -p "${out_parent}"
staging_dir="$(mktemp -d "${out_parent}/.mac-arm64.payload.XXXXXX")"
cleanup() {
    if [[ -n "${staging_dir}" && -d "${staging_dir}" ]]; then
        rm -rf "${staging_dir}"
    fi
}
trap cleanup EXIT

cp "${node}" "${staging_dir}/potluck-node"
cp "${worker}" "${staging_dir}/potluck-worker"
cp "${zmq_path}" "${staging_dir}/libzmq.5.dylib"
chmod +x "${staging_dir}/potluck-node" "${staging_dir}/potluck-worker"

shipped_files=(potluck-node potluck-worker libzmq.5.dylib)
sodium_dependency="$(otool -L "${zmq_path}" |
    awk '$1 ~ /libsodium[^[:space:]]*\.dylib$/ { print $1; exit }')"
sodium_path="${POTLUCK_SODIUM:-}"
if [[ -n "${sodium_dependency}" || -n "${sodium_path}" ]]; then
    if [[ -z "${sodium_path}" && "${sodium_dependency}" == @rpath/* ]] &&
        command -v pkg-config >/dev/null 2>&1; then
        sodium_libdir="$(pkg-config --variable=libdir libsodium 2>/dev/null || true)"
        for candidate in "${sodium_libdir}/libsodium.26.dylib" "${sodium_libdir}/libsodium.dylib"; do
            if [[ -f "${candidate}" ]]; then
                sodium_path="${candidate}"
                break
            fi
        done
    elif [[ -z "${sodium_path}" && "${sodium_dependency}" == /* ]]; then
        sodium_path="${sodium_dependency}"
    fi
    [[ -f "${sodium_path}" ]] ||
        die "cannot resolve libzmq dependency: ${sodium_dependency:-libsodium}"
    cp "${sodium_path}" "${staging_dir}/libsodium.26.dylib"
    if [[ -n "${sodium_dependency}" ]]; then
        install_name_tool -change "${sodium_dependency}" \
            "@loader_path/libsodium.26.dylib" "${staging_dir}/libzmq.5.dylib"
    fi
    install_name_tool -id "@loader_path/libsodium.26.dylib" \
        "${staging_dir}/libsodium.26.dylib"
    codesign --force --sign - "${staging_dir}/libsodium.26.dylib"
    shipped_files+=(libsodium.26.dylib)
fi

install_name_tool -id "@rpath/libzmq.5.dylib" "${staging_dir}/libzmq.5.dylib"
codesign --force --sign - "${staging_dir}/libzmq.5.dylib"

for binary in "${staging_dir}/potluck-node" "${staging_dir}/potluck-worker"; do
    zmq_dependency="$(otool -L "${binary}" |
        awk '$1 ~ /libzmq[^[:space:]]*\.dylib$/ { print $1; exit }')"
    if [[ -n "${zmq_dependency}" ]]; then
        if [[ "${zmq_dependency}" != "@executable_path/libzmq.5.dylib" ]]; then
            install_name_tool -change "${zmq_dependency}" \
                "@executable_path/libzmq.5.dylib" "${binary}"
        fi
        if ! otool -l "${binary}" |
            awk '$1 == "path" && $2 == "@executable_path" { found = 1 }
                 END { exit !found }'; then
            install_name_tool -add_rpath "@executable_path" "${binary}"
        fi
    fi
    codesign --force --sign - "${binary}"
    if [[ -n "${zmq_dependency}" ]]; then
        dependency="$(otool -L "${binary}" |
            awk '$1 ~ /libzmq[^[:space:]]*\.dylib$/ { print $1; exit }')"
        [[ "${dependency}" == "@executable_path/libzmq.5.dylib" ]] ||
            die "binary does not use @executable_path/libzmq.5.dylib: ${binary}"
    fi
done

commit="$(git -C "${REPO}" rev-parse HEAD)" ||
    die 'cannot resolve the repository commit'
printf 'commit %s\n' "${commit}" > "${staging_dir}/potluck-build-id"
for file in "${shipped_files[@]}"; do
    checksum="$(shasum -a 256 "${staging_dir}/${file}" | awk '{ print $1 }')"
    [[ "${checksum}" =~ ^[[:xdigit:]]{64}$ ]] ||
        die "cannot checksum staged file: ${file}"
    printf '%s  %s\n' "${checksum}" "${file}" >> "${staging_dir}/potluck-build-id"
done

if [[ -e "${out_dir}" ]]; then
    [[ -d "${out_dir}" ]] || die "output path is not a directory: ${out_dir}"
    for entry in "${out_dir}"/* "${out_dir}"/.[!.]*; do
        [[ -e "${entry}" || -L "${entry}" ]] || continue
        case "$(basename "${entry}")" in
            potluck-node|potluck-worker|libzmq.5.dylib|libsodium.26.dylib|potluck-build-id)
                ;;
            *)
                die "refusing to replace non-payload output directory: ${out_dir}"
                ;;
        esac
    done
    rm -rf "${out_dir}"
fi
mv "${staging_dir}" "${out_dir}"
staging_dir=""
printf 'stage-mac-payload: staged %s\n' "${out_dir}"
