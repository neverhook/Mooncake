#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
preflight="${repo_root}/scripts/gb200/nvlink_host_numa_preflight.sh"
build_script="${repo_root}/scripts/gb200/nvlink_host_numa_build.sh"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

bash -n "$preflight"
bash -n "$build_script"

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
