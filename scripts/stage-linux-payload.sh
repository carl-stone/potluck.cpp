#!/usr/bin/env bash
# Build and stage the portable Linux worker payload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
source "${REPO}/scripts/potluck-safe.sh"
build_dir="${REPO}/build-portable"
arch="$(uname -m)"
out_dir="${REPO}/dist/linux-${arch}"

die() {
    printf 'stage-linux-payload: %s\n' "$*" >&2
    exit 1
}

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$(pwd -P)" "$1" ;;
    esac
}

[[ "$(uname -s)" == "Linux" ]] || die 'this script stages a Linux payload and must run on Linux'

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

for command in cmake git sha256sum awk readelf ldd; do
    command -v "${command}" >/dev/null 2>&1 ||
        die "required command not found: ${command}"
done

# $ORIGIN runpath at build time lets staged binaries find shipped libraries.
cmake_opts=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_NATIVE=OFF
    -DGGML_CUDA=OFF
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_APP=OFF
    -DPOTLUCK_HIGHS=OFF
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
    '-DCMAKE_INSTALL_RPATH=$ORIGIN'
)
nvcc="$(command -v nvcc 2>/dev/null || true)"
if [[ -z "${nvcc}" && -x /opt/cuda/bin/nvcc ]]; then
    nvcc=/opt/cuda/bin/nvcc
fi
if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${nvcc}" ]]; then
    cmake_opts[3]=-DGGML_CUDA=ON
    if [[ -n "${POTLUCK_CUDA_ARCHITECTURES:-}" ]]; then
        cmake_opts+=("-DCMAKE_CUDA_ARCHITECTURES=${POTLUCK_CUDA_ARCHITECTURES}")
    fi
    export PATH="$(dirname "${nvcc}"):${PATH}"
fi
potluck_require_disk "${build_dir}" "${POTLUCK_MIN_DISK_GIB:-10}"
cmake -S "${REPO}" -B "${build_dir}" "${cmake_opts[@]}"
potluck_build "${build_dir}" --target potluck-node potluck-worker potluck-shard

worker="${build_dir}/bin/potluck-worker"
node="${build_dir}/bin/potluck-node"
shard="${build_dir}/bin/potluck-shard"
[[ -x "${worker}" ]] || die "missing worker binary: ${worker}"
[[ -x "${node}" ]] || die "missing node binary: ${node}"
[[ -x "${shard}" ]] || die "missing shard binary: ${shard}"

readelf_tag() {
    readelf -d "$1" | awk -v tag="$2" '
        $2 == "(" tag ")" {
            line = $0
            sub(/^[^[]*\[/, "", line)
            sub(/\][^]]*$/, "", line)
            print line
        }'
}
for binary in "${node}" "${worker}"; do
    if [[ -z "$(readelf_tag "${binary}" RUNPATH)" &&
          -z "$(readelf_tag "${binary}" RPATH)" ]]; then
        die 'binary lacks an $ORIGIN runpath'
    fi
done

zmq_path="${POTLUCK_ZMQ:-}"
if [[ -n "${zmq_path}" && ! -f "${zmq_path}" ]]; then
    die "POTLUCK_ZMQ is not a file: ${zmq_path}"
fi
zmq_soname() {
    readelf_tag "$1" SONAME | head -1
}
if [[ -z "${zmq_path}" ]] && command -v pkg-config >/dev/null 2>&1; then
    zmq_libdir="$(pkg-config --variable=libdir libzmq 2>/dev/null || true)"
    for candidate in "${zmq_libdir}/libzmq.so.5" "${zmq_libdir}/libzmq.so"; do
        if [[ -f "${candidate}" ]]; then
            zmq_path="${candidate}"
            break
        fi
    done
fi
if [[ -z "${zmq_path}" ]]; then
    for libdir in /usr/lib/"${arch}"/linux-gnu /usr/lib/x86_64-linux-gnu \
        /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib; do
        if [[ -f "${libdir}/libzmq.so.5" ]]; then
            zmq_path="${libdir}/libzmq.so.5"
            break
        fi
    done
fi
if [[ -z "${zmq_path}" ]] && command -v ldconfig >/dev/null 2>&1; then
    zmq_path="$(ldconfig -p |
        awk '/libzmq\.so\.5[[:space:]]/ { print $NF; exit }')"
fi
[[ -f "${zmq_path}" ]] || die 'cannot resolve a libzmq dynamic library'

out_parent="$(dirname "${out_dir}")"
mkdir -p "${out_parent}"
staging_dir="$(mktemp -d "${out_parent}/.linux-${arch}.payload.XXXXXX")"
cleanup() {
    if [[ -n "${staging_dir}" && -d "${staging_dir}" ]]; then
        rm -rf "${staging_dir}"
    fi
}
trap cleanup EXIT

cp "${node}" "${staging_dir}/potluck-node"
cp "${worker}" "${staging_dir}/potluck-worker"
cp "${shard}" "${staging_dir}/potluck-shard"
cp "${zmq_path}" "${staging_dir}/libzmq.so.5"
chmod +x "${staging_dir}/potluck-node" "${staging_dir}/potluck-worker" "${staging_dir}/potluck-shard"

shipped_files=(potluck-node potluck-worker potluck-shard libzmq.so.5)
zmq_needs() {
    readelf_tag "$1" NEEDED
}
sodium_soname="$(zmq_needs "${staging_dir}/libzmq.so.5" |
    awk '/^libsodium\.so/ { print; exit }')"
sodium_path="${POTLUCK_SODIUM:-}"
# Bundle libsodium whenever libzmq needs it. The deployment target may not
# provide the same system libraries as the build host.
if [[ -n "${sodium_path}" || -n "${sodium_soname}" ]]; then
    if [[ -z "${sodium_path}" ]] && command -v pkg-config >/dev/null 2>&1; then
        sodium_libdir="$(pkg-config --variable=libdir libsodium 2>/dev/null || true)"
        if [[ -f "${sodium_libdir}/${sodium_soname}" ]]; then
            sodium_path="${sodium_libdir}/${sodium_soname}"
        elif [[ -f "${sodium_libdir}/libsodium.so" ]]; then
            sodium_path="${sodium_libdir}/libsodium.so"
        fi
    fi
    if [[ -z "${sodium_path}" ]] && command -v ldconfig >/dev/null 2>&1; then
        sodium_path="$(ldconfig -p |
            awk -v soname="${sodium_soname:-libsodium.so}" \
                '$1 == soname { print $NF; exit }')"
    fi
    [[ -f "${sodium_path}" ]] ||
        die "cannot resolve libzmq dependency: ${sodium_soname:-libsodium}"
    sodium_name="$(zmq_soname "${sodium_path}")"
    [[ -n "${sodium_name}" ]] || sodium_name="${sodium_soname:-libsodium.so}"
    cp "${sodium_path}" "${staging_dir}/${sodium_name}"
    shipped_files+=("${sodium_name}")
fi

for binary in "${staging_dir}/potluck-node" "${staging_dir}/potluck-worker" "${staging_dir}/potluck-shard"; do
    zmq_needed="$(zmq_needs "${binary}" | awk '/^libzmq\.so/ { print; exit }')"
    if [[ -n "${zmq_needed}" && "${zmq_needed}" != "libzmq.so.5" ]]; then
        die "binary expects ${zmq_needed}, payload ships libzmq.so.5: ${binary}"
    fi
done
for staged in "${staging_dir}/potluck-node" "${staging_dir}/potluck-worker" \
    "${staging_dir}/potluck-shard" "${staging_dir}/libzmq.so.5"; do
    if (cd "${staging_dir}" &&
        LD_LIBRARY_PATH="${staging_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
            ldd "$(basename "${staged}")" | grep -q 'not found'); then
        die "staged file has unresolved libraries: ${staged}"
    fi
done

commit="$(git -C "${REPO}" rev-parse HEAD)" ||
    die 'cannot resolve the repository commit'
printf 'platform linux-%s\n' "${arch}" > "${staging_dir}/potluck-build-id"
printf 'commit %s\n' "${commit}" >> "${staging_dir}/potluck-build-id"
for file in "${shipped_files[@]}"; do
    checksum="$(sha256sum "${staging_dir}/${file}" | awk '{ print $1 }')"
    [[ "${checksum}" =~ ^[[:xdigit:]]{64}$ ]] ||
        die "cannot checksum staged file: ${file}"
    printf '%s  %s\n' "${checksum}" "${file}" >> "${staging_dir}/potluck-build-id"
done

if [[ -e "${out_dir}" ]]; then
    [[ -d "${out_dir}" ]] || die "output path is not a directory: ${out_dir}"
    for entry in "${out_dir}"/* "${out_dir}"/.[!.]*; do
        [[ -e "${entry}" || -L "${entry}" ]] || continue
        case "$(basename "${entry}")" in
            potluck-node|potluck-worker|potluck-shard|libzmq.so.5|libsodium.so.*|potluck-build-id)
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
printf 'stage-linux-payload: staged %s\n' "${out_dir}"
