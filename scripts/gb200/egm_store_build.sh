#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-egm-store-gb200}"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
build_unit_tests="${BUILD_UNIT_TESTS:-0}"
extra_args=()
if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"${CMAKE_EXTRA_ARGS}"
fi

case "$build_unit_tests" in
  0) cmake_build_unit_tests=OFF ;;
  1) cmake_build_unit_tests=ON ;;
  *)
    printf 'BUILD_UNIT_TESTS must be 0 or 1, got: %s\n' "$build_unit_tests" >&2
    exit 2
    ;;
esac

required_commands=(cmake ninja python3)
[[ "$build_unit_tests" == "0" ]] || required_commands+=(ctest)
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'required command is missing: %s\n' "$command_name" >&2
    exit 1
  }
done

printf 'source_sha=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
printf 'build_dir=%s\n' "$build_dir"

cmake -S "$repo_root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  "${extra_args[@]}" \
  -DBUILD_UNIT_TESTS="$cmake_build_unit_tests" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DWITH_STORE_RUST=OFF \
  -DWITH_EP=OFF \
  -DUSE_CUDA=ON \
  -DUSE_MNNVL=ON

build_targets=(mooncake_master store)
if [[ "$build_unit_tests" == "1" ]]; then
  build_targets+=(egm_store_pool_test nvlink_host_numa_allocation_test
    nvlink_transport_test)
fi
cmake --build "$build_dir" --parallel "$jobs" --target "${build_targets[@]}"

if [[ "$build_unit_tests" == "1" ]]; then
  ctest --test-dir "$build_dir" -L nvlink_vmm_unit --output-on-failure
  ctest --test-dir "$build_dir" -L egm_store_pool_unit --output-on-failure
fi

(
  cd "$repo_root"
  export PYTHONPYCACHEPREFIX="${build_dir}/python-cache"
  python3 -m py_compile scripts/gb200/egm_store_*.py
  python3 -m unittest scripts.gb200.test_egm_store_gb200 -v
)

if [[ "$build_unit_tests" == "1" ]]; then
  printf 'GB200 Store EGM build and focused tests PASS\n'
else
  printf 'GB200 Store EGM runtime build and script tests PASS (C++ unit tests disabled)\n'
fi
