#!/usr/bin/env bash
# Check portable payload installation without a compiler, model, or network.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/potluck-packaging.XXXXXX")"
cleanup() {
    rm -rf "${tmp}"
}
trap cleanup EXIT

payload="${tmp}/payload"
prefix="${tmp}/prefix"
mkdir -p "${payload}"
printf 'worker node\n' > "${payload}/potluck-node"
printf 'worker runtime\n' > "${payload}/potluck-worker"
printf 'portable zmq\n' > "${payload}/libzmq.5.dylib"
chmod +x "${payload}/potluck-node" "${payload}/potluck-worker"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}
{
    printf 'commit packaging-test\n'
    for name in potluck-node potluck-worker libzmq.5.dylib; do
        printf '%s  %s\n' "$(sha256_file "${payload}/${name}")" "${name}"
    done
} > "${payload}/potluck-build-id"

HOME="${tmp}/home" bash "${REPO}/scripts/install.sh" \
    --payload "${payload}" --prefix "${prefix}"

[[ -x "${prefix}/potluck-node" ]]
[[ -x "${prefix}/potluck-worker" ]]
[[ -f "${prefix}/libzmq.5.dylib" ]]
[[ ! -e "${prefix}/Qwen3.5-0.8B-Q4_0.gguf" ]]
printf 'PACKAGING PAYLOAD CHECK PASSED\n'
model="${tmp}/tiny.gguf"
route="${tmp}/route.txt"
bad_route="${tmp}/bad-route.txt"
fake_bin="${tmp}/fake-bin"
dist="${tmp}/dist"
remote_home="${tmp}/remote-home"
cache="${tmp}/shard-cache"
mkdir -p "${fake_bin}" "${dist}" "${remote_home}" "${cache}"
printf 'tiny model\n' > "${model}"

cat > "${fake_bin}/potluck-shard" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
model="$1"
shift
bounds_spec=""
outdir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bounds)
            bounds_spec="$2"
            shift 2
            ;;
        --outdir)
            outdir="$2"
            shift 2
            ;;
        *)
            exit 2
            ;;
    esac
done
printf '%s|%s|%s\n' "${model}" "${bounds_spec}" "${outdir}" >> "${FAKE_SHARD_LOG}"
IFS=, read -r -a bounds <<< "${bounds_spec}"
count=$((${#bounds[@]} - 1))
(( count > 0 ))
mkdir -p "${outdir}"
stem="$(basename "${model}")"
stem="${stem%.*}"
for ((i = 0; i < count; ++i)); do
    printf 'shard %d/%d layers %s-%s\n' "${i}" "${count}" \
        "${bounds[i]}" "${bounds[i + 1]}" \
        > "${outdir}/${stem}.potluck-${i}of${count}.gguf"
done
EOF

cat > "${fake_bin}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
command="${!#}"
case "${command}" in
    'uname -sm 2>/dev/null')
        printf '%s\n' "${FAKE_SSH_PLATFORM}"
        ;;
    'printf %s "$HOME"')
        printf '%s' "${FAKE_REMOTE_HOME}"
        ;;
    *)
        HOME="${FAKE_REMOTE_HOME}" bash -c "${command}"
        ;;
esac
EOF

cat > "${fake_bin}/scp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
last=$(( ${#args[@]} - 1 ))
destination="${args[last]}"
destination_path="${destination#*:}"
mkdir -p "${destination_path}"
sources=()
for ((i = 0; i < last; ++i)); do
    case "${args[i]}" in
        -q)
            ;;
        -o|-P)
            i=$((i + 1))
            ;;
        *)
            sources+=("${args[i]}")
            ;;
    esac
done
cp -f "${sources[@]}" "${destination_path}/"
EOF
chmod +x "${fake_bin}/potluck-shard" "${fake_bin}/ssh" "${fake_bin}/scp"

make_deploy_payload() {
    local dir="$1"
    local marker="$2"
    mkdir -p "${dir}"
    printf '%s node\n' "${marker}" > "${dir}/potluck-node"
    printf '%s worker\n' "${marker}" > "${dir}/potluck-worker"
    printf '%s library\n' "${marker}" > "${dir}/libzmq.5.dylib"
    chmod +x "${dir}/potluck-node" "${dir}/potluck-worker"
    {
        printf 'commit deploy-test\n'
        for name in potluck-node potluck-worker libzmq.5.dylib; do
            printf '%s  %s\n' "$(sha256_file "${dir}/${name}")" "${name}"
        done
    } > "${dir}/potluck-build-id"
}
make_deploy_payload "${dist}/mac-arm64" "mac-arm64"
make_deploy_payload "${dist}/linux-x86_64" "linux-x86_64"

cat > "${route}" <<'EOF'
# window owner start end
0 0 0 1
1 1 1 2
2 1 2 3
3 0 3 4
4 1 4 5
EOF
cat > "${bad_route}" <<'EOF'
0 0 0 1
1 1 2 3
EOF

PATH="${fake_bin}:${PATH}" \
POTLUCK_SHARD_BIN="${fake_bin}/potluck-shard" \
POTLUCK_DIST_DIR="${dist}" \
POTLUCK_SHARD_CACHE="${cache}" \
FAKE_REMOTE_HOME="${remote_home}" \
FAKE_SSH_PLATFORM='Darwin arm64' \
FAKE_SHARD_LOG="${tmp}/shard.log" \
bash "${REPO}/scripts/deploy.sh" \
    --model "${model}" --target fake-target --owner 1 --route-file "${route}"

remote_prefix="${remote_home}/potluck"
remote_files=("${remote_prefix}"/*)
[[ "${#remote_files[@]}" -eq 9 ]]
for name in \
    libzmq.5.dylib \
    potluck-build-id \
    potluck-deploy.sha256 \
    potluck-node \
    potluck-shards.manifest \
    potluck-worker \
    tiny.potluck-1of5.gguf \
    tiny.potluck-2of5.gguf \
    tiny.potluck-4of5.gguf; do
    [[ -f "${remote_prefix}/${name}" ]]
done
[[ ! -e "${remote_prefix}/tiny.gguf" ]]
[[ ! -e "${remote_prefix}/tiny.potluck-0of5.gguf" ]]
[[ ! -e "${remote_prefix}/tiny.potluck-3of5.gguf" ]]
[[ "$(<"${remote_prefix}/potluck-node")" == 'mac-arm64 node' ]]
[[ "$(<"${remote_prefix}/potluck-worker")" == 'mac-arm64 worker' ]]
[[ "$(<"${remote_prefix}/libzmq.5.dylib")" == 'mac-arm64 library' ]]
[[ "$(<"${remote_prefix}/tiny.potluck-1of5.gguf")" == 'shard 1/5 layers 1-2' ]]
[[ "$(<"${remote_prefix}/tiny.potluck-2of5.gguf")" == 'shard 2/5 layers 2-3' ]]
[[ "$(<"${remote_prefix}/tiny.potluck-4of5.gguf")" == 'shard 4/5 layers 4-5' ]]
[[ "$(<"${tmp}/shard.log")" == *'|0,1,2,3,4,5|'* ]]

model_sha="$(sha256_file "${model}")"
shard1_sha="$(sha256_file "${remote_prefix}/tiny.potluck-1of5.gguf")"
shard2_sha="$(sha256_file "${remote_prefix}/tiny.potluck-2of5.gguf")"
shard4_sha="$(sha256_file "${remote_prefix}/tiny.potluck-4of5.gguf")"
expected_manifest="$(printf 'version 1\nmodel tiny.gguf\nmodel_sha256 %s\nowner 1\nwindow_count 3\nwindow 1 1 1 2 tiny.potluck-1of5.gguf %s\nwindow 2 1 2 3 tiny.potluck-2of5.gguf %s\nwindow 4 1 4 5 tiny.potluck-4of5.gguf %s' \
    "${model_sha}" "${shard1_sha}" "${shard2_sha}" "${shard4_sha}")"
[[ "$(<"${remote_prefix}/potluck-shards.manifest")" == "${expected_manifest}" ]]

checksum_count=0
while read -r checksum name extra; do
    [[ -n "${checksum}" && -z "${extra:-}" ]]
    [[ -f "${remote_prefix}/${name}" ]]
    [[ "$(sha256_file "${remote_prefix}/${name}")" == "${checksum}" ]]
    checksum_count=$((checksum_count + 1))
done < "${remote_prefix}/potluck-deploy.sha256"
[[ "${checksum_count}" -eq 8 ]]

bad_output="${tmp}/bad-route.out"
if PATH="${fake_bin}:${PATH}" \
    POTLUCK_SHARD_BIN="${fake_bin}/potluck-shard" \
    POTLUCK_DIST_DIR="${dist}" \
    POTLUCK_SHARD_CACHE="${cache}" \
    FAKE_REMOTE_HOME="${remote_home}" \
    FAKE_SSH_PLATFORM='Darwin arm64' \
    FAKE_SHARD_LOG="${tmp}/shard.log" \
    bash "${REPO}/scripts/deploy.sh" \
        --model "${model}" --target fake-target --owner 1 --route-file "${bad_route}" \
        > "${bad_output}" 2>&1; then
    printf 'malformed route was accepted\n' >&2
    exit 1
fi
[[ "$(<"${bad_output}")" == *'route windows must be contiguous and non-empty'* ]]
printf 'PACKAGING DEPLOYMENT CHECK PASSED\n'
