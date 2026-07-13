#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
preflight="${repo_root}/scripts/gb200/nvlink_host_numa_preflight.sh"
build_script="${repo_root}/scripts/gb200/nvlink_host_numa_build.sh"
wrapper="${repo_root}/scripts/gb200/gb200.sh"
config_example="${repo_root}/scripts/gb200/gb200.conf.example"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

bash -n "$preflight"
bash -n "$build_script"
bash -n "$wrapper"
"$wrapper" --help | grep -Fq 'print-config'

config="${tmp_dir}/gb200.conf"
sed \
  -e 's/NODE_A_IP="CHANGE_ME"/NODE_A_IP="192.0.2.10"/' \
  -e 's/NODE_B_IP="CHANGE_ME"/NODE_B_IP="192.0.2.11"/' \
  -e 's/RUN_ID="gb200-CHANGE_ME"/RUN_ID="gb200-script-test"/' \
  -e "s|BUILD_DIR=\"/workspace/Mooncake/build-nvlink-host-numa\"|BUILD_DIR=\"${tmp_dir}/chosen-build\"|" \
  "$config_example" >"$config"

resolved="${tmp_dir}/resolved.log"
env NODE_A_IP=bad NODE_B_IP=bad RUN_ID=bad BUILD_DIR=/bad \
  PYTHONPATH=/bad LD_LIBRARY_PATH=/bad \
  "$wrapper" --config "$config" print-config >"$resolved"
grep -Fq 'NODE_A_IP=192.0.2.10' "$resolved"
grep -Fq 'NODE_B_IP=192.0.2.11' "$resolved"
grep -Fq 'RUN_ID=gb200-script-test' "$resolved"
grep -Fq "BUILD_DIR=${tmp_dir}/chosen-build" "$resolved"
grep -Fq 'METADATA_SERVER=http://192.0.2.10:8079/metadata' "$resolved"
grep -Fq 'MASTER_ADMIN_URL=http://192.0.2.10:9003' "$resolved"
grep -Fq 'CONSUMER_HOSTNAME_PREFIX=192.0.2.11:1240' "$resolved"
grep -Fq 'MC_FORCE_MNNVL=1' "$resolved"
if grep -Fq '/bad' "$resolved"; then
  printf 'wrapper inherited an ad-hoc shell environment variable\n' >&2
  exit 1
fi

diagnostic_config="${tmp_dir}/diagnostic.conf"
sed \
  -e 's/NODE_A_IP="192.0.2.10"/NODE_A_IP="127.0.0.1"/' \
  -e 's/MASTER_RPC_PORT=50051/MASTER_RPC_PORT=65430/' \
  -e 's/MASTER_ADMIN_PORT=9003/MASTER_ADMIN_PORT=65431/' \
  -e 's/METADATA_PORT=8079/METADATA_PORT=65432/' \
  "$config" >"$diagnostic_config"
diagnostic_log="${tmp_dir}/diagnostic.log"
env -u ADMIN_URL -u MASTER_ADMIN_URL -u NODE_A_IP \
  "$wrapper" --config "$diagnostic_config" diagnose >"$diagnostic_log" 2>&1
grep -Fq 'GET http://127.0.0.1:65431/get_all_segments' "$diagnostic_log"
grep -Fq 'GET http://127.0.0.1:65431/get_segments_detail' "$diagnostic_log"
grep -Fq 'GET http://127.0.0.1:65431/query_segment?segment=127.0.0.1:12345' \
  "$diagnostic_log"
grep -Fq 'GET http://127.0.0.1:65431/query_segment?segment=127.0.0.1%3A12345' \
  "$diagnostic_log"
if grep -Eq 'No host part|failed to set query|Could not parse the URL' "$diagnostic_log"; then
  printf 'diagnostics still depend on missing shell URL variables\n' >&2
  exit 1
fi

placeholder_log="${tmp_dir}/placeholder.log"
if "$wrapper" --config "$config_example" print-config >"$placeholder_log" 2>&1; then
  printf 'wrapper unexpectedly accepted an unedited example config\n' >&2
  exit 1
fi
grep -Eq 'NODE_[AB]_IP still contains CHANGE_ME' "$placeholder_log"

non_strict_missing_log="${tmp_dir}/non-strict-missing-probe.log"
if ! env -u MC_MNNVL_FABRIC_PROBE MC_REQUIRE_MNNVL_FABRIC=0 \
    "$preflight" >"$non_strict_missing_log" 2>&1; then
  printf 'non-strict preflight must report NOT_RUN with exit status 0\n' >&2
  exit 1
fi
grep -Fq 'FAIL  CUDA Fabric-handle capability was not probed because MC_MNNVL_FABRIC_PROBE is unset' \
  "$non_strict_missing_log"
grep -Fq 'RESULT NOT_RUN failures=' "$non_strict_missing_log"

missing_log="${tmp_dir}/missing-probe.log"
if env -u MC_MNNVL_FABRIC_PROBE MC_REQUIRE_MNNVL_FABRIC=1 \
    "$preflight" >"$missing_log" 2>&1; then
  printf 'strict preflight unexpectedly accepted an unset probe\n' >&2
  exit 1
fi
grep -Fq 'strict Fabric preflight requires MC_MNNVL_FABRIC_PROBE' \
  "$missing_log"
grep -Fq 'RESULT FAIL failures=' "$missing_log"

non_executable_probe="${tmp_dir}/fabric-probe"
: >"$non_executable_probe"
chmod 0644 "$non_executable_probe"
non_executable_log="${tmp_dir}/non-executable-probe.log"
if MC_REQUIRE_MNNVL_FABRIC=1 \
    MC_MNNVL_FABRIC_PROBE="$non_executable_probe" \
    "$preflight" >"$non_executable_log" 2>&1; then
  printf 'strict preflight unexpectedly accepted a non-executable probe\n' >&2
  exit 1
fi
grep -Fq 'MC_MNNVL_FABRIC_PROBE is not an executable file:' \
  "$non_executable_log"

python3 - "$build_script" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
markers = [
    'cmake --build "$build_dir" --parallel "$jobs"',
    '# Hardware validation must execute an actual Fabric allocation/export probe.',
    'export MC_REQUIRE_MNNVL_FABRIC=1',
    'export MC_MNNVL_FABRIC_PROBE=',
    '"${repo_root}/scripts/gb200/nvlink_host_numa_preflight.sh"',
]

positions = []
search_from = 0
for marker in markers:
    position = source.find(marker, search_from)
    if position < 0:
        raise SystemExit(f"missing build-flow marker: {marker}")
    positions.append(position)
    search_from = position + len(marker)

hardware_block = source[positions[1] : positions[-1] + len(markers[-1])]
if 'if [[ "$skip_preflight" != "1" ]]' not in hardware_block:
    raise SystemExit("hardware preflight no longer honors SKIP_PREFLIGHT")
configure_block = source[source.index('cmake -S "$repo_root"') : positions[0]]
extra_position = configure_block.index('"${extra_args[@]}"')
for fixed_option in ('-DWITH_EP=OFF', '-DUSE_CUDA=ON', '-DUSE_MNNVL=ON'):
    if extra_position > configure_block.index(fixed_option):
        raise SystemExit(f"CMAKE_EXTRA_ARGS can override fixed no-Torch option: {fixed_option}")
if 'cmake_cache_bool_is_true USE_EVENT_DRIVEN_COMPLETION' not in source:
    raise SystemExit("unit membership no longer follows event-completion CMake state")
if 'unit_tests+=(nvlink_event_driven_completion_test)' not in source:
    raise SystemExit("event-completion unit test is missing from dynamic membership")
PY

bool_helper="$(sed -n \
  '/^cmake_cache_bool_is_true() {$/,/^}$/p' "$build_script")"
if [[ -z "$bool_helper" ]]; then
  printf 'could not extract CMake cache boolean helper\n' >&2
  exit 1
fi
eval "$bool_helper"
for value in ON on true YES y custom-value 2; do
  printf 'USE_EVENT_DRIVEN_COMPLETION:BOOL=%s\n' "$value" >"${tmp_dir}/CMakeCache.txt"
  if ! cmake_cache_bool_is_true USE_EVENT_DRIVEN_COMPLETION \
      "${tmp_dir}/CMakeCache.txt"; then
    printf 'CMake true value was rejected: %s\n' "$value" >&2
    exit 1
  fi
done
for value in OFF off false NO n IGNORE NOTFOUND dependency-NOTFOUND 0; do
  printf 'USE_EVENT_DRIVEN_COMPLETION:BOOL=%s\n' "$value" >"${tmp_dir}/CMakeCache.txt"
  if cmake_cache_bool_is_true USE_EVENT_DRIVEN_COMPLETION \
      "${tmp_dir}/CMakeCache.txt"; then
    printf 'CMake false value was accepted: %s\n' "$value" >&2
    exit 1
  fi
done

printf 'GB200 script policy tests PASS\n'
