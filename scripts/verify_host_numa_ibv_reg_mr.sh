#!/usr/bin/env bash
set -euo pipefail

# Verify whether CUDA HBM and HOST_NUMA fabric VMM memory can be registered as
# RDMA memory regions through both the legacy nvidia-peermem path and the modern
# DMA-BUF path used on GB200/GDR_C2C systems.

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
#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#ifndef EPROTONOSUPPORT
#define EPROTONOSUPPORT EOPNOTSUPP
#endif

using IbvRegDmabufMrFn = ibv_mr* (*)(ibv_pd*, uint64_t, size_t, uint64_t,
                                     int, int);

static const char* cuda_name(CUresult rc) {
    const char* name = nullptr;
    if (cuGetErrorName(rc, &name) == CUDA_SUCCESS && name) return name;
    return "CUDA_ERROR_UNKNOWN";
}

static const char* mem_type_name(CUmemorytype mem_type) {
    switch (mem_type) {
        case CU_MEMORYTYPE_HOST:
            return "HOST";
        case CU_MEMORYTYPE_DEVICE:
            return "DEVICE";
        case CU_MEMORYTYPE_ARRAY:
            return "ARRAY";
        case CU_MEMORYTYPE_UNIFIED:
            return "UNIFIED";
        default:
            return "UNKNOWN";
    }
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
    bool attempted = false;
    bool ok = false;
    int err = 0;
};

struct DmaBufResult {
    bool attempted = false;
    bool export_ok = false;
    bool reg_ok = false;
    int err = 0;
    CUresult cuda_err = CUDA_SUCCESS;
    const char* stage = "skipped";
};

struct TouchResult {
    bool attempted = false;
    bool ok = false;
    int signal = 0;
    size_t offset = 0;
};

static sigjmp_buf touch_jmp;
static volatile sig_atomic_t touch_active = 0;
static volatile sig_atomic_t touch_signal = 0;
static volatile size_t touch_offset = 0;

static void touch_signal_handler(int sig) {
    if (touch_active) {
        touch_signal = sig;
        siglongjmp(touch_jmp, 1);
    }
}

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
        return {true, true, 0};
    }

    printf("[%s] ibv_reg_mr FAIL addr=%p len=%zu errno=%d %s\n", label, addr,
           length, saved_errno, strerror(saved_errno));
    return {true, false, saved_errno};
}

static IbvRegDmabufMrFn load_ibv_reg_dmabuf_mr() {
    dlerror();
    void* sym = dlsym(RTLD_DEFAULT, "ibv_reg_dmabuf_mr");
    const char* err = dlerror();
    if (!sym) {
        printf("[verbs-dmabuf-symbol] SKIP ibv_reg_dmabuf_mr not found%s%s\n",
               err ? ": " : "", err ? err : "");
        return nullptr;
    }
    printf("[verbs-dmabuf-symbol] ibv_reg_dmabuf_mr found\n");
    return reinterpret_cast<IbvRegDmabufMrFn>(sym);
}

static bool probe_ibv_dmabuf(ibv_pd* pd, IbvRegDmabufMrFn reg_dmabuf_mr) {
    if (!reg_dmabuf_mr) return false;

    errno = 0;
    ibv_mr* mr = reg_dmabuf_mr(pd, 0ULL, 0ULL, 0ULL, -1, 0);
    int saved_errno = errno;
    if (mr) {
        printf("[verbs-dmabuf-probe] ibv_reg_dmabuf_mr dummy call unexpectedly "
               "succeeded; treating DMA-BUF as available\n");
        ibv_dereg_mr(mr);
        return true;
    }

    if (saved_errno == EOPNOTSUPP || saved_errno == EPROTONOSUPPORT) {
        printf("[verbs-dmabuf-probe] NOT_SUPPORTED errno=%d %s\n",
               saved_errno, strerror(saved_errno));
        return false;
    }

    printf("[verbs-dmabuf-probe] available; dummy fd=-1 failed with errno=%d "
           "%s\n",
           saved_errno, strerror(saved_errno));
    return true;
}

static DmaBufResult try_reg_dmabuf_mr(ibv_pd* pd,
                                      IbvRegDmabufMrFn reg_dmabuf_mr,
                                      bool verbs_dmabuf_supported,
                                      const char* label, CUdeviceptr ptr,
                                      size_t length) {
    if (!reg_dmabuf_mr) {
        printf("[%s] ibv_reg_dmabuf_mr SKIP: symbol missing\n", label);
        return {false, false, false, 0, CUDA_SUCCESS, "symbol"};
    }
    if (!verbs_dmabuf_supported) {
        printf("[%s] ibv_reg_dmabuf_mr SKIP: verbs DMA-BUF unsupported\n",
               label);
        return {false, false, false, 0, CUDA_SUCCESS, "verbs-probe"};
    }

    CUmemorytype mem_type;
    CUresult rc =
        cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr);
    if (rc == CUDA_SUCCESS) {
        printf("[%s] CUDA pointer memory_type=%s\n", label,
               mem_type_name(mem_type));
    } else {
        printf("[%s] cuPointerGetAttribute MEMORY_TYPE failed: %s (%d); "
               "continuing with cuMemGetAddressRange\n",
               label, cuda_name(rc), (int)rc);
    }

    CUdeviceptr alloc_base = 0;
    size_t alloc_size = 0;
    rc = cuMemGetAddressRange(&alloc_base, &alloc_size, ptr);
    if (rc != CUDA_SUCCESS) {
        printf("[%s] cuMemGetAddressRange FAIL addr=%p len=%zu cuda=%s (%d)\n",
               label, (void*)ptr, length, cuda_name(rc), (int)rc);
        return {true, false, false, 0, rc, "address-range"};
    }

    bool cuda_host_pointer = rc == CUDA_SUCCESS && mem_type == CU_MEMORYTYPE_HOST;
    CUdeviceptr export_base = alloc_base;
    size_t export_size = alloc_size;
    uint64_t offset = 0;
    if (cuda_host_pointer) {
        export_base = ptr;
        export_size = length;
        printf("[%s] CUDA HOST pointer: using original mapped range for "
               "DMA-BUF export addr=%p len=%zu\n",
               label, (void*)export_base, export_size);
    } else {
        offset = (uint64_t)(ptr - alloc_base);
        if (offset > alloc_size || length > alloc_size - offset) {
            printf("[%s] allocation range mismatch addr=%p len=%zu base=%p "
                   "alloc_size=%zu offset=%llu\n",
                   label, (void*)ptr, length, (void*)alloc_base, alloc_size,
                   (unsigned long long)offset);
            return {true, false, false, EINVAL, CUDA_SUCCESS, "address-range"};
        }
    }

    int dmabuf_fd = -1;
    rc = cuMemGetHandleForAddressRange(
        (void*)&dmabuf_fd, export_base, export_size,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
    if (rc != CUDA_SUCCESS) {
        printf("[%s] cuMemGetHandleForAddressRange DMA_BUF_FD FAIL base=%p "
               "alloc_size=%zu cuda=%s (%d)\n",
               label, (void*)export_base, export_size, cuda_name(rc), (int)rc);
        return {true, false, false, 0, rc, "dmabuf-export"};
    }

    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                 IBV_ACCESS_REMOTE_WRITE;
    errno = 0;
    ibv_mr* mr =
        reg_dmabuf_mr(pd, offset, length, (uintptr_t)ptr, dmabuf_fd, access);
    int saved_errno = errno;
    if (close(dmabuf_fd) != 0) {
        fprintf(stderr, "[%s] close(dmabuf_fd) failed: errno=%d %s\n", label,
                errno, strerror(errno));
    }

    if (mr) {
        printf("[%s] ibv_reg_dmabuf_mr OK addr=%p len=%zu base=%p "
               "alloc_size=%zu offset=%llu lkey=0x%x rkey=0x%x\n",
               label, (void*)ptr, length, (void*)export_base, export_size,
               (unsigned long long)offset, mr->lkey, mr->rkey);
        if (ibv_dereg_mr(mr) != 0) {
            fprintf(stderr, "[%s] ibv_dereg_mr failed: errno=%d %s\n", label,
                    errno, strerror(errno));
        }
        return {true, true, true, 0, CUDA_SUCCESS, "register"};
    }

    printf("[%s] ibv_reg_dmabuf_mr FAIL addr=%p len=%zu base=%p "
           "alloc_size=%zu offset=%llu errno=%d %s\n",
           label, (void*)ptr, length, (void*)export_base, export_size,
           (unsigned long long)offset, saved_errno, strerror(saved_errno));
    return {true, true, false, saved_errno, CUDA_SUCCESS, "register"};
}

static TouchResult try_cpu_touch(const char* label, CUdeviceptr ptr,
                                 size_t length) {
    struct sigaction action = {};
    struct sigaction old_sigsegv = {};
    struct sigaction old_sigbus = {};
    action.sa_handler = touch_signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;

    bool sigsegv_installed = sigaction(SIGSEGV, &action, &old_sigsegv) == 0;
    bool sigbus_installed = sigaction(SIGBUS, &action, &old_sigbus) == 0;
    if (!sigsegv_installed || !sigbus_installed) {
        int saved_errno = errno;
        if (sigsegv_installed) sigaction(SIGSEGV, &old_sigsegv, nullptr);
        if (sigbus_installed) sigaction(SIGBUS, &old_sigbus, nullptr);
        printf("[%s] CPU touch SKIP: sigaction failed errno=%d %s\n", label,
               saved_errno, strerror(saved_errno));
        return {false, false, 0, 0};
    }

    TouchResult result;
    result.attempted = true;
    touch_active = 1;
    touch_signal = 0;
    touch_offset = 0;

    if (sigsetjmp(touch_jmp, 1) == 0) {
        long page_size_long = sysconf(_SC_PAGESIZE);
        size_t page_size = page_size_long > 0 ? (size_t)page_size_long : 4096;
        volatile unsigned char* p =
            reinterpret_cast<volatile unsigned char*>((uintptr_t)ptr);
        for (size_t off = 0; off < length; off += page_size) {
            touch_offset = off;
            unsigned char value = p[off];
            p[off] = (unsigned char)(value ^ 0x5a);
            p[off] = value;
        }
        if (length > 0 && ((length - 1) % page_size) != 0) {
            touch_offset = length - 1;
            unsigned char value = p[length - 1];
            p[length - 1] = (unsigned char)(value ^ 0x5a);
            p[length - 1] = value;
        }
        result.ok = true;
        printf("[%s] CPU page touch OK addr=%p len=%zu page_size=%zu\n", label,
               (void*)ptr, length, page_size);
    } else {
        result.ok = false;
        result.signal = touch_signal;
        result.offset = (size_t)touch_offset;
        printf("[%s] CPU page touch FAIL addr=%p len=%zu signal=%d offset=%zu\n",
               label, (void*)ptr, length, result.signal, result.offset);
    }

    touch_active = 0;
    sigaction(SIGSEGV, &old_sigsegv, nullptr);
    sigaction(SIGBUS, &old_sigbus, nullptr);
    return result;
}

static void print_reg_summary(const char* label, const RegResult& result) {
    if (!result.attempted) {
        printf("  %-44s SKIP\n", label);
    } else if (result.ok) {
        printf("  %-44s OK\n", label);
    } else {
        printf("  %-44s FAIL errno=%d %s\n", label, result.err,
               strerror(result.err));
    }
}

static void print_dmabuf_summary(const char* label,
                                 const DmaBufResult& result) {
    if (!result.attempted) {
        printf("  %-44s SKIP stage=%s\n", label, result.stage);
    } else if (result.reg_ok) {
        printf("  %-44s OK\n", label);
    } else if (!result.export_ok) {
        if (result.err != 0) {
            printf("  %-44s FAIL stage=%s errno=%d %s\n", label,
                   result.stage, result.err, strerror(result.err));
        } else {
            printf("  %-44s FAIL stage=%s cuda=%s (%d)\n", label,
                   result.stage, cuda_name(result.cuda_err),
                   (int)result.cuda_err);
        }
    } else {
        printf("  %-44s FAIL stage=%s errno=%d %s\n", label, result.stage,
               result.err, strerror(result.err));
    }
}

static void print_touch_summary(const char* label, const TouchResult& result) {
    if (!result.attempted) {
        printf("  %-44s SKIP\n", label);
    } else if (result.ok) {
        printf("  %-44s OK\n", label);
    } else {
        printf("  %-44s FAIL signal=%d offset=%zu\n", label, result.signal,
               result.offset);
    }
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
            fprintf(stderr, "requested IB_DEV=%s not found. Available:",
                    requested_name);
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
            fprintf(stderr, "CUDA device %d does not support FABRIC handles\n",
                    i);
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
    std::vector<CUmemAccessDesc> access(device_count + 1);
    access[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    access[0].location.id = numa_node;
    access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (int i = 0; i < device_count; ++i) {
        access[i + 1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[i + 1].location.id = i;
        access[i + 1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    rc = cuMemSetAccess(addr, size, access.data(), access.size());
    if (rc != CUDA_SUCCESS) {
        cuMemUnmap(addr, size);
        cuMemAddressFree(addr, size);
        cuMemRelease(handle);
        die_cuda(rc, "cuMemSetAccess");
    }

    cuMemRelease(handle);
    printf("Allocated HOST_NUMA fabric VMM addr=%p size=%zu granularity=%zu "
           "numa_node=%d access_entries=%zu\n",
           (void*)addr, size, granularity, numa_node, access.size());
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
        ibv_dealloc_pd(pd);
        ibv_close_device(rdma_ctx);
        return 14;
    }
    memset(host, 0x5a, size);
    RegResult host_result =
        try_reg_mr(pd, "ordinary-host/legacy-ibv_reg_mr", host, size);
    free(host);
    if (!host_result.ok) {
        fprintf(stderr,
                "Baseline ordinary host memory registration failed; RDMA stack "
                "is not healthy enough to validate CUDA memory.\n");
        ibv_dealloc_pd(pd);
        ibv_close_device(rdma_ctx);
        return 20;
    }

    die_cuda(cuInit(0), "cuInit");
    CUdevice cuda_device;
    die_cuda(cuDeviceGet(&cuda_device, cuda_device_ordinal), "cuDeviceGet");
    CUcontext cuda_ctx;
    die_cuda(cuCtxCreate(&cuda_ctx, 0, cuda_device), "cuCtxCreate");

    int device_dmabuf_supported = 0;
    CUresult attr_rc =
        cuDeviceGetAttribute(&device_dmabuf_supported,
                             CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                             cuda_device);
    if (attr_rc == CUDA_SUCCESS) {
        printf("[cuda-device] CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=%d\n",
               device_dmabuf_supported);
    } else {
        printf("[cuda-device] CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED query "
               "failed: %s (%d)\n",
               cuda_name(attr_rc), (int)attr_rc);
    }

    IbvRegDmabufMrFn reg_dmabuf_mr = load_ibv_reg_dmabuf_mr();
    bool verbs_dmabuf_supported = probe_ibv_dmabuf(pd, reg_dmabuf_mr);

    RegResult hbm_legacy_result;
    DmaBufResult hbm_dmabuf_result;
    CUdeviceptr dptr = 0;
    CUresult dalloc = cuMemAlloc(&dptr, size);
    if (dalloc == CUDA_SUCCESS) {
        hbm_legacy_result =
            try_reg_mr(pd, "cuda-hbm/legacy-ibv_reg_mr", (void*)dptr, size);
        hbm_dmabuf_result =
            try_reg_dmabuf_mr(pd, reg_dmabuf_mr, verbs_dmabuf_supported,
                              "cuda-hbm/dmabuf", dptr, size);
        cuMemFree(dptr);
    } else {
        printf("[cuda-hbm] cuMemAlloc skipped: %s (%d)\n", cuda_name(dalloc),
               (int)dalloc);
    }

    if (!all_devices_support_fabric()) {
        fprintf(stderr,
                "HOST_NUMA fabric allocation is unsupported here; cannot "
                "validate HOST_NUMA registration behavior.\n");
        cuCtxDestroy(cuda_ctx);
        ibv_dealloc_pd(pd);
        ibv_close_device(rdma_ctx);
        return 30;
    }

    HostNumaAllocation host_numa = alloc_host_numa_fabric(size, numa_node);
    TouchResult host_numa_touch_result =
        try_cpu_touch("cuda-host-numa-fabric/cpu-touch", host_numa.addr,
                      host_numa.size);
    RegResult host_numa_legacy_result =
        try_reg_mr(pd, "cuda-host-numa-fabric/legacy-ibv_reg_mr",
                   (void*)host_numa.addr, host_numa.size);
    DmaBufResult host_numa_dmabuf_result =
        try_reg_dmabuf_mr(pd, reg_dmabuf_mr, verbs_dmabuf_supported,
                          "cuda-host-numa-fabric/dmabuf", host_numa.addr,
                          host_numa.size);
    free_host_numa_fabric(host_numa);

    cuCtxDestroy(cuda_ctx);
    ibv_dealloc_pd(pd);
    ibv_close_device(rdma_ctx);

    printf("\nSUMMARY:\n");
    print_reg_summary("ordinary host legacy ibv_reg_mr", host_result);
    print_reg_summary("HBM legacy ibv_reg_mr", hbm_legacy_result);
    print_dmabuf_summary("HBM DMA-BUF ibv_reg_dmabuf_mr",
                         hbm_dmabuf_result);
    print_touch_summary("HOST_NUMA CPU page touch", host_numa_touch_result);
    print_reg_summary("HOST_NUMA legacy ibv_reg_mr",
                      host_numa_legacy_result);
    print_dmabuf_summary("HOST_NUMA DMA-BUF ibv_reg_dmabuf_mr",
                         host_numa_dmabuf_result);

    if (host_numa_dmabuf_result.reg_ok) {
        printf("RESULT: HOST_NUMA fabric VMM memory can be registered through "
               "DMA-BUF on this system.\n");
    } else if (host_numa_dmabuf_result.attempted &&
               !host_numa_dmabuf_result.export_ok) {
        printf("RESULT: HOST_NUMA fabric VMM memory could not export a DMA-BUF "
               "fd on this system.\n");
    } else if (host_numa_dmabuf_result.attempted) {
        printf("RESULT: HOST_NUMA fabric VMM memory exported a DMA-BUF fd, but "
               "ibv_reg_dmabuf_mr failed on this system.\n");
    } else {
        printf("RESULT: HOST_NUMA DMA-BUF validation was skipped; see probe "
               "output above.\n");
    }
    printf("NOTE: legacy ibv_reg_mr failures for CUDA UVA are expected when "
           "nvidia-peermem is absent and do not disprove DMA-BUF GDR.\n");
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
  "${CUDA_LIBS[@]}" -lcuda -libverbs -ldl

SIZE_BYTES=$((SIZE_MB * 1024 * 1024))
echo "Running: SIZE_MB=$SIZE_MB HOST_NUMA_NODE=$HOST_NUMA_NODE CUDA_DEVICE=$CUDA_DEVICE IB_DEV=${IB_DEV:-<auto>}"
if [ -n "$IB_DEV" ]; then
  "$BIN" "$SIZE_BYTES" "$HOST_NUMA_NODE" "$CUDA_DEVICE" "$IB_DEV"
else
  "$BIN" "$SIZE_BYTES" "$HOST_NUMA_NODE" "$CUDA_DEVICE"
fi
