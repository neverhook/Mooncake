#!/usr/bin/env sh
set -eu

# Execute one command with the GB200 Mooncake validation environment.
# This script intentionally does not require "source"; all variables are scoped
# to the child process started by exec.

REPO=${REPO:-/workspace/Mooncake}
BUILD=${BUILD:-"$REPO/build-gb200"}
DOMAIN=${DOMAIN:-gb200-nvl}
RDMA_DEVICES=${RDMA_DEVICES:-mlx5_0,mlx5_1,mlx5_4,mlx5_5}
KEY=${KEY:-dual_nvlink_rdma_real_path}
SIZE_MB=${SIZE_MB:-256}

if [ -z "${MC_METADATA_SERVER:-}" ]; then
    if [ -n "${ETCD_IP:-}" ]; then
        MC_METADATA_SERVER="etcd://$ETCD_IP:2379"
    elif [ -n "${NODE_A_IP:-}" ]; then
        MC_METADATA_SERVER="etcd://$NODE_A_IP:2379"
    else
        MC_METADATA_SERVER="etcd://127.0.0.1:2379"
    fi
fi

if [ -z "${MASTER_SERVER:-}" ] && [ -n "${MASTER_IP:-}" ]; then
    MASTER_SERVER="$MASTER_IP:50051"
elif [ -z "${MASTER_SERVER:-}" ] && [ -n "${NODE_A_IP:-}" ]; then
    MASTER_SERVER="$NODE_A_IP:50051"
fi
MASTER_SERVER=${MASTER_SERVER:-127.0.0.1:50051}

export REPO BUILD DOMAIN RDMA_DEVICES KEY SIZE_MB
export MC_METADATA_SERVER MASTER_SERVER
export MC_ENABLE_NVLINK_HOST_NUMA=${MC_ENABLE_NVLINK_HOST_NUMA:-1}
export MC_NVLINK_HOST_NUMA_STRICT=${MC_NVLINK_HOST_NUMA_STRICT:-1}
export MC_NVLINK_HOST_NUMA_NODE=${MC_NVLINK_HOST_NUMA_NODE:-0}
export MC_NVLINK_SCALE_UP_DOMAIN_ID=${MC_NVLINK_SCALE_UP_DOMAIN_ID:-$DOMAIN}
export MC_LOG_LEVEL=${MC_LOG_LEVEL:-TRACE}
export MC_RPC_TIMEOUT_MS=${MC_RPC_TIMEOUT_MS:-3000}

# GB200 validation targets the modern DMA-BUF/GDR_C2C path. Keep this explicit
# because Mooncake's runtime default may still select the legacy nvidia-peermem
# ibv_reg_mr path when the variable is unset.
export WITH_NVIDIA_PEERMEM=${WITH_NVIDIA_PEERMEM:-0}

# Keep GB200 data-path validation on TCP control-plane RPC unless the caller
# explicitly starts processes outside this wrapper with a different setting.
unset MC_RPC_PROTOCOL

PYTHONPATH_PREPEND="$BUILD/mooncake-integration:$REPO/mooncake-wheel"
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PYTHONPATH_PREPEND:$PYTHONPATH"
else
    export PYTHONPATH="$PYTHONPATH_PREPEND"
fi

LD_LIBRARY_PATH_PREPEND="$BUILD/mooncake-common/etcd:$BUILD/mooncake-common:$BUILD/mooncake-common/src:$BUILD/mooncake-store/src:$BUILD/mooncake-transfer-engine/src:$BUILD/mooncake-integration:/usr/local/cuda/lib64"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_PREPEND:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_PREPEND"
fi

if [ "${1:-}" = "--print-env" ]; then
    printf 'REPO=%s\n' "$REPO"
    printf 'BUILD=%s\n' "$BUILD"
    printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
    printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
    printf 'MC_METADATA_SERVER=%s\n' "$MC_METADATA_SERVER"
    printf 'MASTER_SERVER=%s\n' "$MASTER_SERVER"
    printf 'MC_ENABLE_NVLINK_HOST_NUMA=%s\n' "$MC_ENABLE_NVLINK_HOST_NUMA"
    printf 'MC_NVLINK_HOST_NUMA_STRICT=%s\n' "$MC_NVLINK_HOST_NUMA_STRICT"
    printf 'MC_NVLINK_HOST_NUMA_NODE=%s\n' "$MC_NVLINK_HOST_NUMA_NODE"
    printf 'MC_NVLINK_SCALE_UP_DOMAIN_ID=%s\n' "$MC_NVLINK_SCALE_UP_DOMAIN_ID"
    printf 'MC_LOG_LEVEL=%s\n' "$MC_LOG_LEVEL"
    printf 'MC_RPC_TIMEOUT_MS=%s\n' "$MC_RPC_TIMEOUT_MS"
    printf 'WITH_NVIDIA_PEERMEM=%s\n' "$WITH_NVIDIA_PEERMEM"
    printf 'RDMA_DEVICES=%s\n' "$RDMA_DEVICES"
    printf 'KEY=%s\n' "$KEY"
    printf 'SIZE_MB=%s\n' "$SIZE_MB"
    exit 0
fi

if [ "$#" -eq 0 ]; then
    echo "usage: scripts/gb200_env_exec.sh [--print-env] <command> [args...]" >&2
    exit 2
fi

cd "$REPO"
exec "$@"
