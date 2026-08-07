#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-egm-store-gb200}"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
extra_args=()
if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"${CMAKE_EXTRA_ARGS}"
fi

for command_name in cmake ninja ctest python3; do
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
  -DBUILD_UNIT_TESTS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DWITH_STORE_RUST=OFF \
  -DWITH_EP=OFF \
  -DUSE_CUDA=ON \
  -DUSE_MNNVL=ON

cmake --build "$build_dir" --parallel "$jobs" --target \
  mooncake_master store egm_store_pool_test \
  nvlink_host_numa_allocation_test nvlink_transport_test

ctest --test-dir "$build_dir" -L nvlink_vmm_unit --output-on-failure
ctest --test-dir "$build_dir" -L egm_store_pool_unit --output-on-failure

(
  cd "$repo_root"
  export PYTHONPYCACHEPREFIX="${build_dir}/python-cache"
  python3 -m py_compile scripts/gb200/egm_store_*.py
  python3 -m unittest scripts.gb200.test_egm_store_gb200 -v
)

printf 'GB200 Store EGM build and focused tests PASS\n'
