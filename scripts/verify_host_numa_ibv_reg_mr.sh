#!/usr/bin/env bash
set -euo pipefail

# Verify whether CUDA HOST_NUMA fabric VMM memory can be registered as an RDMA
# memory region with ibv_reg_mr.
#
# The test intentionally uses two registrations in one process:
#   1. ordinary CPU host memory, which must register successfully; this proves
#      the selected RDMA device and verbs stack are usable.
#   2. CU_MEM_LOCATION_TYPE_HOST_NUMA + CU_MEM_HANDLE_TYPE_FABRIC VMM memory,
#      matching Mooncake's HOST_NUMA fabric allocation path.

SIZE_MB="${SIZE_MB:-64}"
HOST_NUMA_NODE="${HOST_NUMA_NODE:-${MC_NVLINK_HOST_NUMA_NODE:-0}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
IB_DEV="${IB_DEV:-}"
WORKDIR="${WORKDIR:-/tmp/mooncake-host-numa-regmr}"
CXX="${CXX:-g++}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

mkdir -p "$WORKDIR"
SRC="$WORKDIR/verify_host_numa_ibv_reg_mr.cpp"
BIN="$WORKDIR/verify_host_numa_ibv_reg_mr"

cat >"$SRC" <<'CPP'
#include <cuda.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

static const char* cuda_name(CUresult rc) {
    const char* name = nullptr;
    if (cuGetErrorName(rc, &name) == CUDA_SUCCESS && name) return name;
    return "CUDA_ERROR_UNKNOWN";
}

static void die_cuda(CUresult rc, const char* what) {
    if (rc == CUDA_SUCCESS) return;
    fprintf(stderr, "%s failed: %s (%d)\n", what, cuda_name(rc), (int)rc);
    exit(2);
}

static size_t align_up(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

struct RegResult {
    bool ok;
    int err;
};

static RegResult try_reg_mr(ibv_pd* pd, const char* label, void* addr,
                            size_t length) {
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                 IBV_ACCESS_REMOTE_WRITE;
    errno = 0;
    ibv_mr* mr = ibv_reg_mr(pd, addr, length, access);
    int saved_errno = errno;
    if (mr) {
        printf("[%s] ibv_reg_mr OK addr=%p len=%zu lkey=0x%x rkey=0x%x\n",
               label, addr, length, mr->lkey, mr->rkey);
        if (ibv_dereg_mr(mr) != 0) {
            fprintf(stderr, "[%s] ibv_dereg_mr failed: errno=%d %s\n", label,
                    errno, strerror(errno));
        }
        return {true, 0};
    }

    printf("[%s] ibv_reg_mr FAIL addr=%p len=%zu errno=%d %s\n", label, addr,
           length, saved_errno, strerror(saved_errno));
    return {false, saved_errno};
}

static ibv_context* open_rdma_device(const char* requested_name) {
    int num_devices = 0;
    ibv_device** devices = ibv_get_device_list(&num_devices);
    if (!devices || num_devices <= 0) {
        fprintf(stderr, "ibv_get_device_list found no RDMA devices\n");
        exit(10);
    }

    ibv_device* selected = nullptr;
    if (requested_name && requested_name[0]) {
        for (int i = 0; i < num_devices; ++i) {
            if (strcmp(ibv_get_device_name(devices[i]), requested_name) == 0) {
                selected = devices[i];
                break;
            }
        }
        if (!selected) {
            fprintf(stderr, "requested IB_DEV=%s not found. Available:", requested_name);
            for (int i = 0; i < num_devices; ++i) {
                fprintf(stderr, " %s", ibv_get_device_name(devices[i]));
            }
            fprintf(stderr, "\n");
            ibv_free_device_list(devices);
            exit(11);
        }
    } else {
        selected = devices[0];
    }

    printf("Using RDMA device: %s\n", ibv_get_device_name(selected));
    ibv_context* ctx = ibv_open_device(selected);
    if (!ctx) {
        fprintf(stderr, "ibv_open_device failed: errno=%d %s\n", errno,
                strerror(errno));
        ibv_free_device_list(devices);
        exit(12);
    }
    ibv_free_device_list(devices);
    return ctx;
}

static bool all_devices_support_fabric() {
    int count = 0;
    CUresult rc = cuDeviceGetCount(&count);
    if (rc != CUDA_SUCCESS || count <= 0) {
        fprintf(stderr, "cuDeviceGetCount failed or found no devices: %s (%d)\n",
                cuda_name(rc), (int)rc);
        return false;
    }

    for (int i = 0; i < count; ++i) {
        CUdevice dev;
        int supported = 0;
        die_cuda(cuDeviceGet(&dev, i), "cuDeviceGet");
        rc = cuDeviceGetAttribute(&supported,
                                  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                  dev);
        if (rc != CUDA_SUCCESS || !supported) {
            fprintf(stderr, "CUDA device %d does not support FABRIC handles\n", i);
            return false;
        }
    }
    return true;
}

struct HostNumaAllocation {
    CUdeviceptr addr = 0;
    size_t size = 0;
};

static HostNumaAllocation alloc_host_numa_fabric(size_t requested_size,
                                                 int numa_node) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = numa_node;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t granularity = 0;
    die_cuda(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM),
             "cuMemGetAllocationGranularity");
    size_t size = align_up(std::max(requested_size, granularity), granularity);

    CUmemGenericAllocationHandle handle;
    die_cuda(cuMemCreate(&handle, size, &prop, 0), "cuMemCreate HOST_NUMA");

    CUdeviceptr addr = 0;
    CUresult rc = cuMemAddressReserve(&addr, size, granularity, 0, 0);
    if (rc != CUDA_SUCCESS) {
        cuMemRelease(handle);
        die_cuda(rc, "cuMemAddressReserve");
    }

    rc = cuMemMap(addr, size, 0, handle, 0);
    if (rc != CUDA_SUCCESS) {
        cuMemAddressFree(addr, size);
        cuMemRelease(handle);
        die_cuda(rc, "cuMemMap");
    }

    int device_count = 0;
    die_cuda(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    std::vector<CUmemAccessDesc> access(device_count);
    for (int i = 0; i < device_count; ++i) {
        access[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[i].location.id = i;
        access[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    rc = cuMemSetAccess(addr, size, access.data(), access.size());
    if (rc != CUDA_SUCCESS) {
        cuMemUnmap(addr, size);
        cuMemAddressFree(addr, size);
        cuMemRelease(handle);
        die_cuda(rc, "cuMemSetAccess");
    }

    cuMemRelease(handle);
    printf("Allocated HOST_NUMA fabric VMM addr=%p size=%zu granularity=%zu numa_node=%d\n",
           (void*)addr, size, granularity, numa_node);
    return {addr, size};
}

static void free_host_numa_fabric(const HostNumaAllocation& alloc) {
    if (!alloc.addr || !alloc.size) return;
    CUresult rc = cuMemUnmap(alloc.addr, alloc.size);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemUnmap failed during cleanup: %s (%d)\n",
                cuda_name(rc), (int)rc);
    }
    rc = cuMemAddressFree(alloc.addr, alloc.size);
    if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAddressFree failed during cleanup: %s (%d)\n",
                cuda_name(rc), (int)rc);
    }
}

int main(int argc, char** argv) {
    size_t size = 64ull * 1024 * 1024;
    int numa_node = 0;
    int cuda_device_ordinal = 0;
    const char* ib_dev = "";

    if (argc > 1) size = strtoull(argv[1], nullptr, 10);
    if (argc > 2) numa_node = atoi(argv[2]);
    if (argc > 3) cuda_device_ordinal = atoi(argv[3]);
    if (argc > 4) ib_dev = argv[4];

    ibv_context* rdma_ctx = open_rdma_device(ib_dev);
    ibv_pd* pd = ibv_alloc_pd(rdma_ctx);
    if (!pd) {
        fprintf(stderr, "ibv_alloc_pd failed: errno=%d %s\n", errno,
                strerror(errno));
        ibv_close_device(rdma_ctx);
        return 13;
    }

    void* host = nullptr;
    if (posix_memalign(&host, 4096, size) != 0 || !host) {
        fprintf(stderr, "posix_memalign failed\n");
        return 14;
    }
    memset(host, 0x5a, size);
    RegResult host_result = try_reg_mr(pd, "ordinary-host", host, size);
    free(host);
    if (!host_result.ok) {
        fprintf(stderr,
                "Baseline ordinary host memory registration failed; RDMA stack "
                "is not healthy enough to validate HOST_NUMA.\n");
        ibv_dealloc_pd(pd);
        ibv_close_device(rdma_ctx);
        return 20;
    }

    die_cuda(cuInit(0), "cuInit");
    CUdevice cuda_device;
    die_cuda(cuDeviceGet(&cuda_device, cuda_device_ordinal), "cuDeviceGet");
    CUcontext cuda_ctx;
    die_cuda(cuCtxCreate(&cuda_ctx, 0, cuda_device), "cuCtxCreate");

    CUdeviceptr dptr = 0;
    CUresult dalloc = cuMemAlloc(&dptr, size);
    if (dalloc == CUDA_SUCCESS) {
        RegResult device_result =
            try_reg_mr(pd, "cuda-device-gpudirect-optional", (void*)dptr, size);
        printf("[cuda-device-gpudirect-optional] result is informational; "
               "success depends on GPUDirect RDMA / nvidia-peermem support.\n");
        (void)device_result;
        cuMemFree(dptr);
    } else {
        printf("[cuda-device-gpudirect-optional] cuMemAlloc skipped: %s (%d)\n",
               cuda_name(dalloc), (int)dalloc);
    }

    if (!all_devices_support_fabric()) {
        fprintf(stderr, "HOST_NUMA fabric allocation is unsupported here; cannot validate ibv_reg_mr behavior.\n");
        cuCtxDestroy(cuda_ctx);
        ibv_dealloc_pd(pd);
        ibv_close_device(rdma_ctx);
        return 30;
    }

    HostNumaAllocation host_numa =
        alloc_host_numa_fabric(size, numa_node);
    RegResult host_numa_result =
        try_reg_mr(pd, "cuda-host-numa-fabric-vmm", (void*)host_numa.addr,
                   host_numa.size);
    free_host_numa_fabric(host_numa);

    cuCtxDestroy(cuda_ctx);
    ibv_dealloc_pd(pd);
    ibv_close_device(rdma_ctx);

    if (host_numa_result.ok) {
        printf("RESULT: HOST_NUMA fabric VMM memory DID register with ibv_reg_mr on this system.\n");
        return 40;
    }

    printf("RESULT: CONFIRMED - ordinary host memory registered, but HOST_NUMA fabric VMM memory failed ibv_reg_mr with errno=%d (%s).\n",
           host_numa_result.err, strerror(host_numa_result.err));
    return 0;
}
CPP

CUDA_INC="$CUDA_HOME/include"
CUDA_LIBS=()
if [ -d "$CUDA_HOME/lib64/stubs" ]; then
  CUDA_LIBS+=("-L$CUDA_HOME/lib64/stubs")
fi
if [ -d "$CUDA_HOME/lib64" ]; then
  CUDA_LIBS+=("-L$CUDA_HOME/lib64")
fi

echo "Compiling $BIN"
"$CXX" -std=c++17 -O2 -Wall -Wextra \
  -I"$CUDA_INC" \
  "$SRC" -o "$BIN" \
  "${CUDA_LIBS[@]}" -lcuda -libverbs

SIZE_BYTES=$((SIZE_MB * 1024 * 1024))
echo "Running: SIZE_MB=$SIZE_MB HOST_NUMA_NODE=$HOST_NUMA_NODE CUDA_DEVICE=$CUDA_DEVICE IB_DEV=${IB_DEV:-<auto>}"
if [ -n "$IB_DEV" ]; then
  "$BIN" "$SIZE_BYTES" "$HOST_NUMA_NODE" "$CUDA_DEVICE" "$IB_DEV"
else
  "$BIN" "$SIZE_BYTES" "$HOST_NUMA_NODE" "$CUDA_DEVICE"
fi
