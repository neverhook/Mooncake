#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build-nvlink-host-numa}"
jobs="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
extra_args=()
if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"${CMAKE_EXTRA_ARGS}"
fi

if ! command -v ctest >/dev/null 2>&1; then
  printf 'ctest is required\n' >&2
  exit 1
fi
ctest_help="$(ctest --help)"
if ! grep -Fq -- '--output-junit' <<<"$ctest_help"; then
  printf 'ctest with --output-junit support is required by the GB200 validation harness\n' >&2
  exit 1
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "${repo_root}/scripts/gb200/nvlink_host_numa_preflight.sh"
fi

cmake -S "$repo_root" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  -DBUILD_UNIT_TESTS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARK=OFF \
  -DWITH_STORE_RUST=OFF \
  -DWITH_EP=OFF \
  -DUSE_CUDA=ON \
  -DUSE_MNNVL=ON \
  "${extra_args[@]}"

cmake --build "$build_dir" --parallel "$jobs"

ctest_in_build() {
  (
    cd "$build_dir"
    ctest "$@"
  )
}

assert_label_exact() {
  local label="$1"
  shift
  local listing
  listing="$(ctest_in_build -N -L "$label")"
  printf '%s\n' "$listing"
  local actual_names expected_names
  actual_names="$(awk '$1 == "Test" {print $3}' <<<"$listing" | LC_ALL=C sort)"
  expected_names="$(printf '%s\n' "$@" | LC_ALL=C sort)"
  if [[ "$actual_names" != "$expected_names" ]]; then
    printf 'CTest membership mismatch for label %s\n' "$label" >&2
    diff -u <(printf '%s\n' "$expected_names") \
      <(printf '%s\n' "$actual_names") >&2 || true
    return 1
  fi
}

validate_junit() {
  local skipped_policy="$1"
  shift
  python3 - "$skipped_policy" "$@" <<'PY'
import pathlib
import sys
import xml.etree.ElementTree as ET

skipped_policy = sys.argv[1]
for raw_path in sys.argv[2:]:
    path = pathlib.Path(raw_path)
    if not path.is_file():
        raise SystemExit(f"CTest did not create JUnit output: {path}")
    root = ET.parse(path).getroot()
    testcases = list(root.iter("testcase"))
    if not testcases:
        raise SystemExit(f"JUnit result contains zero testcases: {path}")
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    declared_skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    testcase_skipped = sum(
        1 for testcase in testcases if testcase.find("skipped") is not None
    )
    skipped = max(declared_skipped, testcase_skipped)
    print(f"JUnit summary: tests={len(testcases)} skipped={skipped} path={path}")
    if skipped_policy == "forbid" and skipped:
        raise SystemExit(f"required result contains skipped={skipped}: {path}")
PY
}

assert_label_exact nvlink_host_numa_unit \
  client_integration_test \
  client_metrics_test \
  nvlink_host_numa_config_test \
  nvlink_host_numa_setup_test \
  nvlink_host_numa_store_test \
  nvlink_transport_fake_driver_test \
  nvlink_transport_metrics_test \
  nvlink_vmm_allocation_test \
  nvlink_host_numa_transfer_metadata_test \
  serializer_test

unit_junit="${build_dir}/nvlink-host-numa-unit.xml"
ctest_in_build -L nvlink_host_numa_unit \
  --output-on-failure --output-junit "$unit_junit"
validate_junit allow "$unit_junit"

if [[ "${RUN_HARDWARE_TESTS:-0}" == "1" ]]; then
  export MC_REQUIRE_MNNVL_FABRIC=1
  export MC_REQUIRE_NVLINK_HOST_NUMA_RDMA=1
  assert_label_exact nvlink_host_numa_hardware \
    nvlink_host_numa_fabric_test \
    nvlink_host_numa_store_hardware_test \
    nvlink_transport_fabric_hbm_test \
    nvlink_transport_ipc_cuda_malloc_test
  assert_label_exact nvlink_host_numa_rdma \
    nvlink_host_numa_rdma_smoke_test
  hardware_junit="${build_dir}/nvlink-host-numa-hardware.xml"
  rdma_junit="${build_dir}/nvlink-host-numa-rdma.xml"
  ctest_in_build -L nvlink_host_numa_hardware \
    --output-on-failure --output-junit "$hardware_junit"
  ctest_in_build -L nvlink_host_numa_rdma \
    --output-on-failure --output-junit "$rdma_junit"
  validate_junit forbid "$hardware_junit" "$rdma_junit"
fi
