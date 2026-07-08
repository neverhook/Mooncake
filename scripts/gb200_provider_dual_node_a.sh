#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
REPO=${REPO:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}
SIZE_MB=${SIZE_MB:-256}

guess_primary_ip() {
    if command -v hostname >/dev/null 2>&1; then
        ips=$(hostname -I 2>/dev/null || true)
        set -- $ips
        if [ "$#" -gt 0 ]; then
            printf '%s\n' "$1"
            return
        fi
        hostname
        return
    fi
    printf '127.0.0.1\n'
}

NODE_A_IP=${NODE_A_IP:-$(guess_primary_ip)}
TRANSFER_PORT=${TRANSFER_PORT:-12355}
MASTER_PORT=${MASTER_PORT:-50051}
LOCAL_HOSTNAME=${LOCAL_HOSTNAME:-"$NODE_A_IP:$TRANSFER_PORT"}
MASTER_SERVER=${MASTER_SERVER:-"$NODE_A_IP:$MASTER_PORT"}
ETCD_IP=${ETCD_IP:-"$NODE_A_IP"}
GLOBAL_SEGMENT_SIZE_MB=${GLOBAL_SEGMENT_SIZE_MB:-$((SIZE_MB * 2))}
LOCAL_BUFFER_SIZE_MB=${LOCAL_BUFFER_SIZE_MB:-0}
PROTOCOL=${PROTOCOL:-nvlink,rdma}
MC_STORE_MEMCPY=${MC_STORE_MEMCPY:-1}
PYTHON_BIN=${PYTHON:-python3}

export REPO SIZE_MB NODE_A_IP TRANSFER_PORT LOCAL_HOSTNAME MASTER_SERVER ETCD_IP
export GLOBAL_SEGMENT_SIZE_MB LOCAL_BUFFER_SIZE_MB PROTOCOL MC_STORE_MEMCPY

exec sh "$REPO/scripts/gb200_env_exec.sh" "$PYTHON_BIN" - <<'PY'
import ctypes
import hashlib
import os
import signal
import sys
import time

try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
except ModuleNotFoundError:
    from store import MooncakeDistributedStore, ReplicateConfig

MB = 1024 * 1024


def getenv_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def make_payload(size):
    pattern = bytes(((i * 131 + 17) & 0xFF) for i in range(256))
    return bytearray(pattern * (size // len(pattern)) + pattern[: size % len(pattern)])


key = os.environ.get("KEY", "dual_nvlink_rdma_real_path")
size = getenv_int("SIZE_MB", 256) * MB
global_segment_size = getenv_int("GLOBAL_SEGMENT_SIZE_MB", max(512, getenv_int("SIZE_MB", 256) * 2)) * MB
local_buffer_size = getenv_int("LOCAL_BUFFER_SIZE_MB", 0) * MB
local_hostname = os.environ["LOCAL_HOSTNAME"]
metadata_server = os.environ["MC_METADATA_SERVER"]
master_server = os.environ["MASTER_SERVER"]
protocol = os.environ.get("PROTOCOL", "nvlink,rdma")
rdma_devices = os.environ.get("RDMA_DEVICES", "auto-discovery")

print("provider_config", {
    "local_hostname": local_hostname,
    "metadata_server": metadata_server,
    "master_server": master_server,
    "protocol": protocol,
    "rdma_devices": rdma_devices,
    "size": size,
    "global_segment_size": global_segment_size,
    "local_buffer_size": local_buffer_size,
    "scale_up_domain_id": os.environ.get("MC_NVLINK_SCALE_UP_DOMAIN_ID", ""),
    "host_numa_node": os.environ.get("MC_NVLINK_HOST_NUMA_NODE", ""),
    "MC_STORE_MEMCPY": os.environ.get("MC_STORE_MEMCPY", ""),
}, flush=True)

store = MooncakeDistributedStore()
rc = store.setup(
    local_hostname,
    metadata_server,
    global_segment_size,
    local_buffer_size,
    protocol,
    rdma_devices,
    master_server,
)
if rc != 0:
    raise SystemExit(f"provider setup failed rc={rc}")

payload = make_payload(size)
payload_view = (ctypes.c_char * len(payload)).from_buffer(payload)
payload_addr = ctypes.addressof(payload_view)
payload_sha256 = hashlib.sha256(payload).hexdigest()

cfg = ReplicateConfig()
cfg.replica_num = 1
cfg.preferred_segment = local_hostname

rc = store.put_from(key, payload_addr, len(payload), cfg)
if rc != 0:
    raise SystemExit(f"provider put_from failed rc={rc}")

print("provider_ready", {
    "key": key,
    "bytes": len(payload),
    "sha256": payload_sha256,
    "preferred_segment": local_hostname,
}, flush=True)

stopping = False


def handle_signal(signum, frame):
    global stopping
    stopping = True
    print(f"provider_stopping signal={signum}", flush=True)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

while not stopping:
    time.sleep(5)

print("provider_exit", flush=True)
sys.exit(0)
PY
