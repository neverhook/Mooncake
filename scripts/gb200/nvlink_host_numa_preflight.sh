#!/usr/bin/env bash

set -uo pipefail

strict="${MC_REQUIRE_MNNVL_FABRIC:-0}"
failures=0

case "$strict" in
  0|1) ;;
  *)
    printf 'MC_REQUIRE_MNNVL_FABRIC must be 0 or 1, got: %s\n' "$strict" >&2
    exit 2
    ;;
esac

pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*" >&2; }
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

printf 'NVL72 HOST_NUMA Fabric preflight\n'
printf 'strict=%s kernel=%s arch=%s user=%s\n' \
  "$strict" "$(uname -r)" "$(uname -m)" "$(id -un)"

check_command nvidia-smi
check_command nvcc

if [[ -n "${MC_USE_NVLINK_IPC+x}" ]]; then
  fail "MC_USE_NVLINK_IPC is set; legacy NvlinkTransport will not use Fabric handles"
else
  pass "MC_USE_NVLINK_IPC is unset"
fi

case "${MC_MS_AUTO_DISC:-0}" in
  0)
    pass "Transfer Engine auto-discovery is disabled for deterministic NVLink installation"
    ;;
  1)
    if [[ -n "${MC_FORCE_MNNVL:-}" && -z "${MC_INTRANODE_NVLINK:-}" ]]; then
      pass "auto-discovery is forced to cross-node NVLink by MC_FORCE_MNNVL"
    else
      fail "MC_MS_AUTO_DISC=1 can select RDMA/intra-node transport; set MC_MS_AUTO_DISC=0 or MC_FORCE_MNNVL=1 without MC_INTRANODE_NVLINK"
    fi
    ;;
  *)
    fail "MC_MS_AUTO_DISC must be 0 or 1 for this harness"
    ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id \
      --format=csv,noheader; then
    pass "NVIDIA driver and visible GPUs queried"
  else
    fail "nvidia-smi cannot query visible GPUs"
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version || fail "nvcc exists but cannot report its version"
fi

if [[ -r /proc/devices ]] &&
   grep -qE '(^|[[:space:]])nvidia-caps-imex-channels$' /proc/devices; then
  pass "nvidia-caps-imex-channels is registered in /proc/devices"
else
  fail "nvidia-caps-imex-channels is not registered in /proc/devices"
fi

shopt -s nullglob
channels=(/dev/nvidia-caps-imex-channels/channel*)
if (( ${#channels[@]} == 0 )); then
  fail "no /dev/nvidia-caps-imex-channels/channel* device is present"
else
  accessible_channels=0
  for channel in "${channels[@]}"; do
    if [[ -r "$channel" && -w "$channel" ]]; then
      pass "IMEX channel is readable and writable: $channel"
      accessible_channels=$((accessible_channels + 1))
    else
      warn "IMEX channel is not readable and writable by $(id -un): $channel"
    fi
  done
  if (( accessible_channels == 0 )); then
    fail "the launching user cannot access any IMEX channel"
  fi
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
  pass "NVIDIA IMEX daemon is active"
elif [[ "${MC_IMEX_DAEMON_EXTERNAL:-0}" == "1" ]]; then
  pass "IMEX daemon is declared external to this execution environment"
else
  fail "NVIDIA IMEX daemon was not observed; set MC_IMEX_DAEMON_EXTERNAL=1 only when it is managed outside the container"
fi

gpu_rows=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_rows="$(nvidia-smi --query-gpu=index,pci.bus_id \
    --format=csv,noheader,nounits 2>/dev/null || true)"
fi
if [[ -z "$gpu_rows" ]]; then
  fail "no visible GPU PCI BDF was reported"
else
  while IFS=',' read -r gpu_index raw_bdf; do
    gpu_index="${gpu_index//[[:space:]]/}"
    bdf="${raw_bdf//[[:space:]]/}"
    bdf="$(printf '%s' "$bdf" | tr '[:upper:]' '[:lower:]')"
    sysfs="/sys/bus/pci/devices/$bdf"
    if [[ ! -d "$sysfs" && ${#bdf} -gt 12 ]]; then
      short_bdf="${bdf:4}"
      sysfs="/sys/bus/pci/devices/$short_bdf"
    fi
    if [[ ! -d "$sysfs" ]]; then
      fail "GPU $gpu_index PCI device is absent from sysfs: $bdf"
      continue
    fi
    if [[ ! -r "$sysfs/numa_node" ]]; then
      fail "GPU $gpu_index has no readable sysfs numa_node: $sysfs"
      continue
    fi
    numa_node="$(<"$sysfs/numa_node")"
    if [[ ! "$numa_node" =~ ^[0-9]+$ ]]; then
      fail "GPU $gpu_index has an unknown/negative NUMA node: $numa_node"
      continue
    fi
    if [[ ! -d "/sys/devices/system/node/node${numa_node}" ]]; then
      fail "GPU $gpu_index maps to non-online NUMA node $numa_node"
      continue
    fi
    pass "GPU $gpu_index PCI $bdf maps to online NUMA node $numa_node"
  done <<<"$gpu_rows"
fi

if [[ -n "${MC_MNNVL_FABRIC_PROBE:-}" ]]; then
  if [[ ! -x "${MC_MNNVL_FABRIC_PROBE}" ]]; then
    fail "MC_MNNVL_FABRIC_PROBE is not an executable file: ${MC_MNNVL_FABRIC_PROBE}"
  elif "${MC_MNNVL_FABRIC_PROBE}"; then
    pass "CUDA Fabric-handle capability probe succeeded"
  else
    fail "CUDA Fabric-handle capability probe failed: ${MC_MNNVL_FABRIC_PROBE}"
  fi
elif [[ "$strict" == "1" ]]; then
  fail "strict Fabric preflight requires MC_MNNVL_FABRIC_PROBE"
else
  warn "MC_MNNVL_FABRIC_PROBE is unset; the hardware CTest must perform the CUDA attribute/allocation probe"
  fail "CUDA Fabric-handle capability was not probed because MC_MNNVL_FABRIC_PROBE is unset"
fi

if (( failures == 0 )); then
  printf 'RESULT PASS\n'
  exit 0
fi

if [[ "$strict" == "1" ]]; then
  printf 'RESULT FAIL failures=%d strict=1\n' "$failures" >&2
  exit 1
fi

printf 'RESULT NOT_RUN failures=%d strict=0\n' "$failures" >&2
exit 0
