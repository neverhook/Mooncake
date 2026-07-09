#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
REPO=${REPO:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd -P)}

if [ -z "${MASTER_SERVER:-}" ] && [ -z "${NODE_A_IP:-}" ]; then
    printf 'set NODE_A_IP=<node-a-ip> or MASTER_SERVER=<node-a-ip:50051>\n' >&2
    exit 2
fi

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

NODE_B_IP=${NODE_B_IP:-$(guess_primary_ip)}
TRANSFER_PORT=${TRANSFER_PORT:-12355}
LOCAL_HOSTNAME=${LOCAL_HOSTNAME:-"$NODE_B_IP:$TRANSFER_PORT"}
SIZE_MB=${SIZE_MB:-256}
GLOBAL_SEGMENT_SIZE_MB=${GLOBAL_SEGMENT_SIZE_MB:-0}
LOCAL_BUFFER_SIZE_MB=${LOCAL_BUFFER_SIZE_MB:-0}
PROTOCOL=${PROTOCOL:-nvlink,rdma}
CUDA_DEVICE=${CUDA_DEVICE:-0}
GET_WARMUP=${GET_WARMUP:-0}
GET_REPEAT=${GET_REPEAT:-1}
PYTHON_BIN=${PYTHON:-python3}

export REPO NODE_B_IP TRANSFER_PORT LOCAL_HOSTNAME SIZE_MB
export GLOBAL_SEGMENT_SIZE_MB LOCAL_BUFFER_SIZE_MB PROTOCOL CUDA_DEVICE
export GET_WARMUP GET_REPEAT

exec sh "$REPO/scripts/gb200_env_exec.sh" "$PYTHON_BIN" - <<'PY'
import ctypes
import gc
import hashlib
import os
import sys
import time

try:
    from mooncake.store import MooncakeDistributedStore
except ModuleNotFoundError:
    from store import MooncakeDistributedStore

MB = 1024 * 1024


def getenv_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def load_cuda():
    errors = []
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("failed to load CUDA driver: " + "; ".join(errors))


cuda = load_cuda()
cuda.cuInit.argtypes = [ctypes.c_uint]
cuda.cuInit.restype = ctypes.c_int
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuDeviceGet.restype = ctypes.c_int
cuda.cuDevicePrimaryCtxRetain.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int,
]
cuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
cuda.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
cuda.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
cuda.cuCtxSetCurrent.restype = ctypes.c_int
cuda.cuCtxSynchronize.argtypes = []
cuda.cuCtxSynchronize.restype = ctypes.c_int
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
cuda.cuMemAlloc_v2.restype = ctypes.c_int
cuda.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]
cuda.cuMemFree_v2.restype = ctypes.c_int
cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]
cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
if hasattr(cuda, "cuGetErrorName"):
    cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    cuda.cuGetErrorName.restype = ctypes.c_int


def cuda_name(rc):
    if hasattr(cuda, "cuGetErrorName"):
        name = ctypes.c_char_p()
        if cuda.cuGetErrorName(rc, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode()
    return f"CUDA_ERROR_{rc}"


def check_cuda(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: {cuda_name(rc)} ({rc})")


reader_start_ns = time.perf_counter_ns()
timings_ms = {}


def mark_latency(name, start_ns):
    timings_ms[name] = (time.perf_counter_ns() - start_ns) / 1_000_000


def timed_call(name, fn, *args):
    start_ns = time.perf_counter_ns()
    try:
        return fn(*args)
    finally:
        mark_latency(name, start_ns)


def mib_per_sec(byte_count, elapsed_ms):
    if elapsed_ms <= 0:
        return None
    return (byte_count / MB) / (elapsed_ms / 1000)


def summarize_ms(values):
    if not values:
        return {}
    return {
        "min_ms": min(values),
        "max_ms": max(values),
        "avg_ms": sum(values) / len(values),
    }


def expected_digest(size):
    pattern = bytes(((i * 131 + 17) & 0xFF) for i in range(256))
    chunk = pattern * 4096
    digest = hashlib.sha256()
    remaining = size
    while remaining >= len(chunk):
        digest.update(chunk)
        remaining -= len(chunk)
    if remaining:
        digest.update(
            (pattern * (remaining // len(pattern))) +
            pattern[: remaining % len(pattern)]
        )
    return digest.hexdigest()


def print_descriptors(label, replicas):
    print(label, "count", len(replicas), flush=True)
    for idx, replica in enumerate(replicas):
        if not replica.is_memory_replica():
            print(label, idx, "non_memory", flush=True)
            continue
        desc = replica.get_memory_descriptor().buffer_descriptor
        print(label, idx, {
            "transport_endpoint": desc.transport_endpoint,
            "size": desc.size,
            "buffer_address": desc.buffer_address,
            "memory_kind": desc.memory_kind,
            "scale_up_domain_id": desc.scale_up_domain_id,
            "selected_protocol": desc.selected_protocol,
        }, flush=True)


key = os.environ.get("KEY", "dual_nvlink_rdma_real_path")
size = getenv_int("SIZE_MB", 256) * MB
local_hostname = os.environ["LOCAL_HOSTNAME"]
metadata_server = os.environ["MC_METADATA_SERVER"]
master_server = os.environ["MASTER_SERVER"]
protocol = os.environ.get("PROTOCOL", "nvlink,rdma")
rdma_devices = os.environ.get("RDMA_DEVICES", "auto-discovery")
expected_path = os.environ.get("EXPECTED_PATH", "")
cuda_device = getenv_int("CUDA_DEVICE", 0)
get_warmup = getenv_int("GET_WARMUP", 0)
get_repeat = getenv_int("GET_REPEAT", 1)
if get_warmup < 0:
    raise RuntimeError("GET_WARMUP must be >= 0")
if get_repeat < 1:
    raise RuntimeError("GET_REPEAT must be >= 1")

print("reader_config", {
    "local_hostname": local_hostname,
    "metadata_server": metadata_server,
    "master_server": master_server,
    "protocol": protocol,
    "rdma_devices": rdma_devices,
    "size": size,
    "cuda_device": cuda_device,
    "expected_path": expected_path,
    "scale_up_domain_id": os.environ.get("MC_NVLINK_SCALE_UP_DOMAIN_ID", ""),
    "get_warmup": get_warmup,
    "get_repeat": get_repeat,
}, flush=True)

check_cuda(timed_call("cuda_init_ms", cuda.cuInit, 0), "cuInit")
device = ctypes.c_int()
check_cuda(
    timed_call("cuda_device_get_ms", cuda.cuDeviceGet,
               ctypes.byref(device), cuda_device),
    "cuDeviceGet",
)
ctx = ctypes.c_void_p()
# RDMA DMA-BUF registration uses CUDA primary context internally; allocate the
# destination HBM from the same context instead of a private driver context.
check_cuda(
    timed_call("cuda_primary_ctx_retain_ms", cuda.cuDevicePrimaryCtxRetain,
               ctypes.byref(ctx), device.value),
    "cuDevicePrimaryCtxRetain",
)
check_cuda(
    timed_call("cuda_ctx_set_current_ms", cuda.cuCtxSetCurrent, ctx),
    "cuCtxSetCurrent",
)
hbm_ptr = ctypes.c_ulonglong(0)
registered = False
store = None
selected_protocol = ""
warmup_get_into_ms = []
measured_get_into_ms = []

try:
    check_cuda(
        timed_call("cuda_mem_alloc_ms", cuda.cuMemAlloc_v2,
                   ctypes.byref(hbm_ptr), size),
        "cuMemAlloc_v2",
    )
    print("reader_hbm_alloc", {"ptr": hex(hbm_ptr.value), "bytes": size}, flush=True)

    store = MooncakeDistributedStore()
    setup_start_ns = time.perf_counter_ns()
    rc = store.setup(
        local_hostname,
        metadata_server,
        0,
        0,
        protocol,
        rdma_devices,
        master_server,
    )
    mark_latency("setup_ms", setup_start_ns)
    if rc != 0:
        raise RuntimeError(f"reader setup failed rc={rc}")

    print_descriptors("replicas_before_selection", store.get_replica_desc(key))
    selected = store.get_selected_replica_desc_for_buffer(key, hbm_ptr.value, size)
    print_descriptors("selected_for_hbm", selected)
    if not selected or not selected[0].is_memory_replica():
        raise RuntimeError("no memory replica selected for HBM destination")
    selected_desc = selected[0].get_memory_descriptor().buffer_descriptor
    selected_protocol = selected_desc.selected_protocol
    if expected_path and selected_protocol != expected_path:
        raise RuntimeError(
            f"selected protocol mismatch: expected={expected_path} actual={selected_protocol}"
        )

    register_start_ns = time.perf_counter_ns()
    rc = store.register_buffer(hbm_ptr.value, size)
    mark_latency("register_buffer_ms", register_start_ns)
    if rc != 0:
        raise RuntimeError(f"register_buffer failed rc={rc}")
    registered = True

    def run_get_into(phase, iteration):
        get_into_start_ns = time.perf_counter_ns()
        read_size = store.get_into(key, hbm_ptr.value, size)
        get_into_ms = (time.perf_counter_ns() - get_into_start_ns) / 1_000_000
        if read_size != size:
            raise RuntimeError(
                f"{phase} get_into returned {read_size}, expected {size}"
            )
        transfer_mib_s = mib_per_sec(size, get_into_ms)
        print("reader_iteration_latency", {
            "phase": phase,
            "iteration": iteration,
            "selected_protocol": selected_protocol,
            "bytes": size,
            "get_into_ms": round(get_into_ms, 3),
            "transfer_mib_s": (
                None if transfer_mib_s is None else round(transfer_mib_s, 2)
            ),
        }, flush=True)
        return get_into_ms

    for iteration in range(get_warmup):
        warmup_get_into_ms.append(run_get_into("warmup", iteration))

    for iteration in range(get_repeat):
        measured_get_into_ms.append(run_get_into("measured", iteration))

    timings_ms["get_into_ms"] = measured_get_into_ms[0]
    measured_summary = summarize_ms(measured_get_into_ms)
    warmup_summary = summarize_ms(warmup_get_into_ms)
    if warmup_get_into_ms:
        timings_ms["cold_get_into_ms"] = warmup_get_into_ms[0]
    else:
        timings_ms["cold_get_into_ms"] = measured_get_into_ms[0]
    timings_ms["measured_get_into_avg_ms"] = measured_summary["avg_ms"]
    timings_ms["measured_get_into_min_ms"] = measured_summary["min_ms"]
    timings_ms["measured_get_into_max_ms"] = measured_summary["max_ms"]

    check_cuda(
        timed_call("cuda_ctx_reset_current_ms", cuda.cuCtxSetCurrent, ctx),
        "cuCtxSetCurrent after get_into",
    )
    check_cuda(
        timed_call("cuda_sync_ms", cuda.cuCtxSynchronize),
        "cuCtxSynchronize after get_into",
    )
    host = (ctypes.c_ubyte * size)()
    check_cuda(
        timed_call("dtoh_ms", cuda.cuMemcpyDtoH_v2,
                   host, hbm_ptr.value, size),
        "cuMemcpyDtoH_v2",
    )
    hash_start_ns = time.perf_counter_ns()
    actual = hashlib.sha256(bytes(host)).hexdigest()
    expected = expected_digest(size)
    mark_latency("verify_hash_ms", hash_start_ns)
    if actual != expected:
        raise RuntimeError(f"payload sha256 mismatch: expected={expected} actual={actual}")

    transfer_mib_s = mib_per_sec(size, measured_summary["avg_ms"])
    print("reader_ok", {
        "key": key,
        "bytes": size,
        "selected_protocol": selected_protocol,
        "sha256": actual,
    }, flush=True)
    timings_ms["total_ms"] = (time.perf_counter_ns() - reader_start_ns) / 1_000_000
    print("reader_latency", {
        "selected_protocol": selected_protocol,
        "bytes": size,
        "warmup_reads": get_warmup,
        "measured_reads": get_repeat,
        "transfer_mib_s": (
            None if transfer_mib_s is None else round(transfer_mib_s, 2)
        ),
        "cold_get_into_ms": round(timings_ms["cold_get_into_ms"], 3),
        "warmup_get_into_ms": [
            round(value, 3) for value in warmup_get_into_ms
        ],
        "measured_get_into_ms": [
            round(value, 3) for value in measured_get_into_ms
        ],
        "measured_get_into_summary": {
            name: round(value, 3) for name, value in measured_summary.items()
        },
        "warmup_get_into_summary": {
            name: round(value, 3) for name, value in warmup_summary.items()
        },
        "latency_ms": {
            name: round(value, 3) for name, value in timings_ms.items()
        },
    }, flush=True)
finally:
    cleanup_start_ns = time.perf_counter_ns()
    try:
        if registered and store is not None:
            timed_call("cleanup_ctx_set_current_before_unregister_ms",
                       cuda.cuCtxSetCurrent, ctx)
            unregister_start_ns = time.perf_counter_ns()
            rc = store.unregister_buffer(hbm_ptr.value)
            mark_latency("unregister_buffer_ms", unregister_start_ns)
            if rc != 0:
                print(
                    f"warning: unregister_buffer failed rc={rc}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if store is not None:
            # Python binding exposes close(), which maps to RealClient teardown.
            close = getattr(store, "close", None)
            if callable(close):
                timed_call("cleanup_ctx_set_current_before_close_ms",
                           cuda.cuCtxSetCurrent, ctx)
                close_start_ns = time.perf_counter_ns()
                rc = close()
                mark_latency("close_ms", close_start_ns)
                if rc != 0:
                    print(
                        f"warning: close failed rc={rc}",
                        file=sys.stderr,
                        flush=True,
                    )
            store = None
            gc.collect()
        if hbm_ptr.value:
            timed_call("cleanup_ctx_set_current_before_free_ms",
                       cuda.cuCtxSetCurrent, ctx)
            timed_call("cuda_mem_free_ms", cuda.cuMemFree_v2, hbm_ptr.value)
        if ctx.value:
            timed_call("cuda_primary_ctx_release_ms",
                       cuda.cuDevicePrimaryCtxRelease, device.value)
        mark_latency("cleanup_total_ms", cleanup_start_ns)
        timings_ms["total_with_cleanup_ms"] = (
            time.perf_counter_ns() - reader_start_ns
        ) / 1_000_000
        print("reader_cleanup_latency", {
            "selected_protocol": selected_protocol,
            "bytes": size,
            "latency_ms": {
                name: round(value, 3)
                for name, value in timings_ms.items()
                if name.startswith("cleanup_") or
                name in (
                    "unregister_buffer_ms",
                    "close_ms",
                    "cuda_mem_free_ms",
                    "cuda_primary_ctx_release_ms",
                    "cleanup_total_ms",
                    "total_with_cleanup_ms",
                )
            },
        }, flush=True)
PY
