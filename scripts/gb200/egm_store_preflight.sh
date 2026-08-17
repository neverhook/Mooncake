#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
failures=0

pass() { printf 'PASS  %s\n' "$*"; }
fail() {
  printf 'FAIL  %s\n' "$*" >&2
  failures=$((failures + 1))
}

check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    pass "command available: $1 ($(command -v "$1"))"
  else
    fail "required command is missing: $1"
  fi
}

printf 'GB200/NVL72 Store EGM preflight\n'
printf 'kernel=%s arch=%s user=%s\n' "$(uname -r)" "$(uname -m)" "$(id -un)"

check_command python3
check_command nvidia-smi
check_command nvcc

if [[ "$(uname -m)" == "aarch64" ]]; then
  pass 'GB200 Grace ARM64 architecture detected'
else
  fail "expected GB200 aarch64 host, got $(uname -m)"
fi

if [[ -n "${MC_USE_NVLINK_IPC+x}" ]]; then
  fail 'MC_USE_NVLINK_IPC is set; HOST_NUMA Fabric validation requires Fabric handles'
else
  pass 'MC_USE_NVLINK_IPC is unset'
fi
if [[ "${MC_MS_AUTO_DISC:-}" == "0" ]]; then
  pass 'Transfer Engine auto discovery is disabled'
else
  fail 'MC_MS_AUTO_DISC must be 0 for deterministic nvlink validation'
fi
if [[ "${MC_FORCE_MNNVL:-}" == "1" ]]; then
  pass 'MC_FORCE_MNNVL=1'
else
  fail 'MC_FORCE_MNNVL must be 1'
fi
if [[ -n "${MC_FORCE_TCP:-}" || -n "${MC_FORCE_HCA:-}" || -n "${MC_INTRANODE_NVLINK:-}" ]]; then
  fail 'MC_FORCE_TCP, MC_FORCE_HCA, and MC_INTRANODE_NVLINK must be unset'
else
  pass 'conflicting transport overrides are unset'
fi

gpu_rows=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_rows="$(nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id \
    --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$gpu_rows" ]]; then
    printf '%s\n' "$gpu_rows"
    pass 'NVIDIA driver and visible GPUs queried'
  else
    fail 'nvidia-smi returned no visible GPUs'
  fi
  if nvidia-smi topo -m; then
    pass 'GPU topology queried'
  else
    fail 'nvidia-smi topo -m failed'
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || fail 'nvcc cannot report its version'
fi

if command -v python3 >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  if PYTHONPATH="$script_dir" python3 - <<'PY'
import json

from egm_validation_common import collect_route_snapshot, route_snapshot_errors

snapshot = collect_route_snapshot()
errors = route_snapshot_errors(snapshot)
print(json.dumps({
    "backend": snapshot["backend"],
    "command_status": snapshot["command_status"],
    "nvlink_byte_counters": len(snapshot["nvlink_bytes"]),
    "c2c_capacity_counters": len(snapshot["c2c_capacity_gb_s"]),
    "c2c_error_counters": len(snapshot["c2c_errors"]),
    "fabric_gpus": len(snapshot["fabric"]),
}, sort_keys=True))
for error in errors:
    print(f"route observation error: {error}")
raise SystemExit(bool(errors))
PY
  then
    pass 'NVLink bytes, C2C capability/errors, and Fabric health are readable'
    pass 'C2C traffic qualification will use explicit C2C_ROUTE_INFERRED evidence'
  else
    fail 'nvidia-smi route observation is incomplete'
  fi
fi

active_rdma_ports=0
for state_file in /sys/class/infiniband/*/ports/*/state; do
  [[ -r "$state_file" ]] || continue
  if grep -qE '(^|[[:space:]])ACTIVE([[:space:]]|$)' "$state_file"; then
    pass "RDMA port active: ${state_file%/state}"
    active_rdma_ports=$((active_rdma_ports + 1))
  fi
done
(( active_rdma_ports > 0 )) || fail 'no active RDMA HCA port was found'

if [[ -r /proc/devices ]] &&
   grep -qE '(^|[[:space:]])nvidia-caps-imex-channels$' /proc/devices; then
  pass 'nvidia-caps-imex-channels is registered'
else
  fail 'nvidia-caps-imex-channels is not registered in /proc/devices'
fi

shopt -s nullglob
channels=(/dev/nvidia-caps-imex-channels/channel*)
if (( ${#channels[@]} == 0 )); then
  fail 'no IMEX channel device is present'
else
  accessible=0
  for channel in "${channels[@]}"; do
    if [[ -r "$channel" && -w "$channel" ]]; then
      pass "IMEX channel accessible: $channel"
      accessible=$((accessible + 1))
    fi
  done
  (( accessible > 0 )) || fail 'the launching user cannot access an IMEX channel'
fi

imex_active=0
if command -v systemctl >/dev/null 2>&1 &&
   systemctl is-active --quiet nvidia-imex.service 2>/dev/null; then
  imex_active=1
fi
if command -v pgrep >/dev/null 2>&1 && pgrep -x nvidia-imex >/dev/null 2>&1; then
  imex_active=1
fi
if (( imex_active == 1 )); then
  pass 'NVIDIA IMEX daemon is active'
elif [[ "${MC_IMEX_DAEMON_EXTERNAL:-0}" == "1" ]]; then
  pass 'IMEX daemon is declared external to this container'
else
  fail 'NVIDIA IMEX daemon not observed; set MC_IMEX_DAEMON_EXTERNAL=1 only when externally managed'
fi

gpu_bdfs=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_bdfs="$(nvidia-smi --query-gpu=index,pci.bus_id \
    --format=csv,noheader,nounits 2>/dev/null || true)"
fi
if [[ -z "$gpu_bdfs" ]]; then
  fail 'no visible GPU PCI BDF was reported'
else
  while IFS=',' read -r raw_index raw_bdf; do
    index="${raw_index//[[:space:]]/}"
    bdf="${raw_bdf//[[:space:]]/}"
    bdf="$(printf '%s' "$bdf" | tr '[:upper:]' '[:lower:]')"
    sysfs="/sys/bus/pci/devices/$bdf"
    if [[ ! -d "$sysfs" && ${#bdf} -gt 12 ]]; then
      sysfs="/sys/bus/pci/devices/${bdf:4}"
    fi
    if [[ ! -r "$sysfs/numa_node" ]]; then
      fail "GPU $index has no readable NUMA locality at $sysfs"
      continue
    fi
    numa_node="$(<"$sysfs/numa_node")"
    if [[ ! "$numa_node" =~ ^[0-9]+$ ]] ||
       [[ ! -d "/sys/devices/system/node/node${numa_node}" ]]; then
      fail "GPU $index maps to invalid NUMA node $numa_node"
      continue
    fi
    pass "GPU $index maps to online NUMA node $numa_node"
  done <<<"$gpu_bdfs"
fi

if (( failures == 0 )); then
  printf 'RESULT PASS\n'
  exit 0
fi
printf 'RESULT FAIL failures=%d\n' "$failures" >&2
exit 1
