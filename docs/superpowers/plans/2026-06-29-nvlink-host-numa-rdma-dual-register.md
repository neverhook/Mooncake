# NVLink HOST_NUMA RDMA Dual Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an explicitly enabled mode where one Mooncake store memory segment can be registered for both NVLink HOST_NUMA fabric access and RDMA, while workers keep selecting the replica and protocol locally.

**Architecture:** Keep the master metadata contract unchanged: it returns all complete replicas. Extend transfer metadata and store descriptors so workers can see multi-protocol capability, HOST_NUMA memory kind, and scale-up domain. Add worker-side candidate selection that chooses local memcpy, same-domain NVLink, or RDMA, then submits the existing batched `TransferRequest` vector through the selected transport.

**Tech Stack:** C++17, Mooncake TransferEngine, Mooncake Store, CUDA driver VMM APIs, GoogleTest, CMake, optional GB200/NVL72 hardware-gated NVLink tests.

---

## File Structure

Modify these existing files:

- `mooncake-transfer-engine/include/transfer_metadata.h`
  - Add optional metadata fields to `TransferMetadata::BufferDesc` for memory kind and scale-up domain.
- `mooncake-transfer-engine/src/transfer_metadata.cpp`
  - Extend multi-protocol validation and JSON encode/decode for `nvlink,rdma`.
- `mooncake-transfer-engine/tests/transfer_metadata_test.cpp`
  - Add white-box encode/decode tests for `nvlink,rdma`.
- `mooncake-transfer-engine/include/config.h`
  - Add disabled-by-default NVLink HOST_NUMA config fields.
- `mooncake-transfer-engine/src/config.cpp`
  - Load the new config fields from environment variables.
- `mooncake-transfer-engine/tests/config_test.cpp`
  - Add config parsing tests.
- `mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h`
  - Add HOST_NUMA allocation and capability helpers.
- `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`
  - Implement HOST_NUMA VMM allocation, metadata tagging, import-domain checks, and safe cleanup.
- `mooncake-transfer-engine/tests/nvlink_transport_test.cpp`
  - Add a hardware-gated remote HOST_NUMA DRAM to local HBM read test.
- `mooncake-store/include/allocator.h`
  - Extend `AllocatedBuffer::Descriptor` with optional memory kind and scale-up domain metadata.
- `mooncake-store/src/allocator.cpp`
  - Populate the new descriptor fields from `AllocatedBuffer`.
- `mooncake-store/include/client_service.h`
  - Add a dual-protocol mount helper and local scale-up domain accessors.
- `mooncake-store/src/client_service.cpp`
  - Install both transports for dual registration and call `mp_registerLocalMemory`.
- `mooncake-store/include/transfer_task.h`
  - Add selected-protocol submission overloads.
- `mooncake-store/src/transfer_task.cpp`
  - Submit multi-protocol batches through `mp_submitTransfer`.
- `mooncake-store/src/real_client.cpp`
  - Replace the local anonymous `SelectBestReplica` flow with transfer-candidate selection.
- `mooncake-store/tests/transfer_task_test.cpp`
  - Add worker-side protocol selection and batching tests.
- `mooncake-store/tests/client_integration_test.cpp`
  - Add RDMA-only behavior regression coverage and a simulated dual-protocol descriptor test if the in-process harness can build it without CUDA.
- `mooncake-store/src/CMakeLists.txt`
  - Add the new store selector source file.

Create these new files:

- `mooncake-store/include/transfer_candidate_selector.h`
  - Defines `TransferCandidate`, `TransferCandidateContext`, and `SelectTransferCandidate`.
- `mooncake-store/src/transfer_candidate_selector.cpp`
  - Implements worker-side `(replica, protocol)` selection.
- `mooncake-store/tests/transfer_candidate_selector_test.cpp`
  - Focused tests for local, same-domain NVLink, cross-domain RDMA, and disk fallback selection.

Build directories used by verification:

- `build`
  - Normal unit-test build with `ENABLE_MULTI_PROTOCOL=ON`.
- `build-mnnvl`
  - CUDA/MNNVL build for hardware-gated NVLink tests.

---

## Task 1: Transfer Metadata Supports `nvlink,rdma`

**Files:**
- Modify: `mooncake-transfer-engine/include/transfer_metadata.h`
- Modify: `mooncake-transfer-engine/src/transfer_metadata.cpp`
- Modify: `mooncake-transfer-engine/tests/transfer_metadata_test.cpp`

- [ ] **Step 1: Write failing metadata encode/decode tests**

Add this test-only access shim at the top of `mooncake-transfer-engine/tests/transfer_metadata_test.cpp`, replacing the existing direct include of `transfer_metadata.h`:

```cpp
#define private public
#include "transfer_metadata.h"
#undef private
```

Add these tests inside `namespace mooncake`:

```cpp
TEST_F(TransferMetadataTest, EncodeDecodeNvlinkRdmaSegment) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    TransferMetadata::SegmentDesc desc;
    desc.name = "dual-segment";
    desc.protocol = "nvlink,rdma";
    desc.rdma_server_name = "10.0.0.1:12345";
    desc.tcp_data_port = 0;

    TransferMetadata::DeviceDesc device;
    device.name = "mlx5_0";
    device.lid = 1;
    device.gid = "0000:0000:0000:0000:0000:ffff:0a00:0001";
    desc.devices.push_back(device);

    TransferMetadata::BufferDesc nvlink_buffer;
    nvlink_buffer.name = "host_numa:0";
    nvlink_buffer.addr = 0x100000;
    nvlink_buffer.length = 4096;
    nvlink_buffer.protocol = "nvlink";
    nvlink_buffer.shm_name = "fabric-handle-bytes";
    nvlink_buffer.memory_kind = "HOST_NUMA";
    nvlink_buffer.scale_up_domain_id = "domain-a";
    desc.buffers.push_back(nvlink_buffer);

    TransferMetadata::BufferDesc rdma_buffer;
    rdma_buffer.name = "host_numa:0";
    rdma_buffer.addr = 0x100000;
    rdma_buffer.length = 4096;
    rdma_buffer.protocol = "rdma";
    rdma_buffer.lkey.push_back(11);
    rdma_buffer.rkey.push_back(22);
    desc.buffers.push_back(rdma_buffer);

    Json::Value encoded;
    ASSERT_EQ(metadata_client->encodeSegmentDesc(desc, encoded), 0);
    ASSERT_TRUE(encoded["protocol"].isArray());
    ASSERT_EQ(encoded["protocol"][0].asString(), "nvlink");
    ASSERT_EQ(encoded["protocol"][1].asString(), "rdma");

    auto decoded =
        metadata_client->decodeSegmentDesc(encoded, "dual-segment");
    ASSERT_NE(decoded, nullptr);
    EXPECT_EQ(decoded->protocol, "nvlink,rdma");
    ASSERT_EQ(decoded->buffers.size(), 2u);
    EXPECT_EQ(decoded->buffers[0].protocol, "nvlink");
    EXPECT_EQ(decoded->buffers[0].memory_kind, "HOST_NUMA");
    EXPECT_EQ(decoded->buffers[0].scale_up_domain_id, "domain-a");
    EXPECT_EQ(decoded->buffers[0].shm_name, "fabric-handle-bytes");
    EXPECT_EQ(decoded->buffers[1].protocol, "rdma");
    EXPECT_EQ(decoded->buffers[1].rkey[0], 22u);
#endif
}

TEST_F(TransferMetadataTest, RejectsUnsupportedMultiProtocolTriple) {
#ifndef ENABLE_MULTI_PROTOCOL
    GTEST_SKIP() << "ENABLE_MULTI_PROTOCOL is not compiled in";
#else
    TransferMetadata::SegmentDesc desc;
    desc.name = "bad-segment";
    desc.protocol = "nvlink,rdma,tcp";
    Json::Value encoded;
    EXPECT_EQ(metadata_client->encodeSegmentDesc(desc, encoded),
              ERR_INVALID_ARGUMENT);
#endif
}
```

- [ ] **Step 2: Run metadata test and verify it fails**

Run:

```bash
cmake -S . -B build -DBUILD_UNIT_TESTS=ON -DWITH_TE=ON -DWITH_STORE=ON -DENABLE_MULTI_PROTOCOL=ON -DUSE_TCP=ON
cmake --build build --target transfer_metadata_test -j
ctest --test-dir build -R '^transfer_metadata_test$' --output-on-failure
```

Expected: compile fails because `BufferDesc` has no `memory_kind` and `scale_up_domain_id`, or the test fails because `nvlink,rdma` is rejected.

- [ ] **Step 3: Add BufferDesc metadata fields**

In `mooncake-transfer-engine/include/transfer_metadata.h`, extend `BufferDesc`:

```cpp
        std::string shm_name;               // for nvlink and hip
        std::string memory_kind;            // optional, e.g. HOST_NUMA
        std::string scale_up_domain_id;     // optional NVLink scale-up domain
        uint64_t offset;                    // for cxl
```

- [ ] **Step 4: Extend multi-protocol validation**

In `mooncake-transfer-engine/src/transfer_metadata.cpp`, add a helper near `splitProtocols`:

```cpp
#ifdef ENABLE_MULTI_PROTOCOL
static bool isSupportedMultiProtocolPair(const std::vector<std::string>& protocols) {
    if (protocols.size() != 2) return false;
    bool has_cxl = false;
    bool has_tcp = false;
    bool has_rdma = false;
    bool has_nvlink = false;
    for (const auto& proto : protocols) {
        if (proto == "cxl")
            has_cxl = true;
        else if (proto == "tcp")
            has_tcp = true;
        else if (proto == "rdma")
            has_rdma = true;
        else if (proto == "nvlink")
            has_nvlink = true;
    }
    return (has_cxl && (has_tcp || has_rdma)) || (has_nvlink && has_rdma);
}
#endif
```

Replace both existing duplicated "CXL+TCP or CXL+RDMA" checks in `encodeSegmentDesc` and `decodeSegmentDesc` with calls to `isSupportedMultiProtocolPair(protocols)` or the equivalent vector built from JSON.

- [ ] **Step 5: Encode NVLink buffers in multi-protocol JSON**

In `encodeMultiProtocolSegmentDesc`, add this branch inside the buffer loop:

```cpp
        } else if (buffer.protocol == "nvlink") {
            bufferJSON["addr"] = static_cast<Json::UInt64>(buffer.addr);
            bufferJSON["shm_name"] = buffer.shm_name;
            if (!buffer.memory_kind.empty()) {
                bufferJSON["memory_kind"] = buffer.memory_kind;
            }
            if (!buffer.scale_up_domain_id.empty()) {
                bufferJSON["scale_up_domain_id"] = buffer.scale_up_domain_id;
            }
```

- [ ] **Step 6: Decode NVLink buffers in multi-protocol JSON**

In `decodeMultiProtocolSegmentDesc`, add this branch after the RDMA branch:

```cpp
        } else if (buffer_protocol == "nvlink") {
            TransferMetadata::BufferDesc buffer;
            buffer.name = bufferJSON["name"].asString();
            buffer.addr = bufferJSON["addr"].asUInt64();
            buffer.length = bufferJSON["length"].asUInt64();
            buffer.protocol = buffer_protocol;
            buffer.shm_name = bufferJSON["shm_name"].asString();
            if (bufferJSON.isMember("memory_kind")) {
                buffer.memory_kind = bufferJSON["memory_kind"].asString();
            }
            if (bufferJSON.isMember("scale_up_domain_id")) {
                buffer.scale_up_domain_id =
                    bufferJSON["scale_up_domain_id"].asString();
            }
            if (buffer.name.empty() || !buffer.addr || !buffer.length ||
                buffer.shm_name.empty()) {
                LOG(WARNING)
                    << "Corrupted segment descriptor, name " << segment_name
                    << " buffer_protocol " << buffer_protocol;
                return nullptr;
            }
            desc->buffers.push_back(buffer);
```

- [ ] **Step 7: Run metadata test and verify it passes**

Run:

```bash
cmake --build build --target transfer_metadata_test -j
ctest --test-dir build -R '^transfer_metadata_test$' --output-on-failure
```

Expected: `transfer_metadata_test` passes.

- [ ] **Step 8: Commit metadata support**

Run:

```bash
git add mooncake-transfer-engine/include/transfer_metadata.h mooncake-transfer-engine/src/transfer_metadata.cpp mooncake-transfer-engine/tests/transfer_metadata_test.cpp
git commit -m "feat: support nvlink rdma metadata segments"
```

---

## Task 2: Configuration for NVLink HOST_NUMA Mode

**Files:**
- Modify: `mooncake-transfer-engine/include/config.h`
- Modify: `mooncake-transfer-engine/src/config.cpp`
- Modify: `mooncake-transfer-engine/tests/config_test.cpp`

- [ ] **Step 1: Write failing config test**

Add this test to `mooncake-transfer-engine/tests/config_test.cpp`:

```cpp
TEST(ConfigTest, LoadsNvlinkHostNumaConfig) {
    setenv("MC_ENABLE_NVLINK_HOST_NUMA", "1", 1);
    setenv("MC_NVLINK_HOST_NUMA_STRICT", "true", 1);
    setenv("MC_NVLINK_HOST_NUMA_NODE", "2", 1);
    setenv("MC_NVLINK_SCALE_UP_DOMAIN_ID", "gb200-domain-a", 1);

    GlobalConfig config;
    loadGlobalConfig(config);

    EXPECT_TRUE(config.enable_nvlink_host_numa);
    EXPECT_TRUE(config.nvlink_host_numa_strict);
    EXPECT_EQ(config.nvlink_host_numa_node, 2);
    EXPECT_EQ(config.nvlink_scale_up_domain_id, "gb200-domain-a");

    unsetenv("MC_ENABLE_NVLINK_HOST_NUMA");
    unsetenv("MC_NVLINK_HOST_NUMA_STRICT");
    unsetenv("MC_NVLINK_HOST_NUMA_NODE");
    unsetenv("MC_NVLINK_SCALE_UP_DOMAIN_ID");
}
```

- [ ] **Step 2: Run config test and verify it fails**

Run:

```bash
cmake --build build --target config_test -j
ctest --test-dir build -R '^config_test$' --output-on-failure
```

Expected: compile fails because `GlobalConfig` does not have the new fields.

- [ ] **Step 3: Add config fields**

In `mooncake-transfer-engine/include/config.h`, add these fields after `ascend_store_te_init`:

```cpp
    bool enable_nvlink_host_numa = false;
    bool nvlink_host_numa_strict = false;
    int nvlink_host_numa_node = 0;
    std::string nvlink_scale_up_domain_id;
```

- [ ] **Step 4: Load config from env**

In `mooncake-transfer-engine/src/config.cpp`, add this helper near the top of the namespace:

```cpp
static bool isTruthyEnv(const char* value) {
    if (value == nullptr) return false;
    std::string text(value);
    for (auto& ch : text) ch = static_cast<char>(std::tolower(ch));
    return text == "1" || text == "true" || text == "yes" || text == "on";
}
```

Add this block in `loadGlobalConfig` after the existing ascend config loading block:

```cpp
    config.enable_nvlink_host_numa =
        isTruthyEnv(std::getenv("MC_ENABLE_NVLINK_HOST_NUMA"));
    config.nvlink_host_numa_strict =
        isTruthyEnv(std::getenv("MC_NVLINK_HOST_NUMA_STRICT"));
    if (const char* numa_node_env = std::getenv("MC_NVLINK_HOST_NUMA_NODE")) {
        int val = atoi(numa_node_env);
        if (val >= 0) {
            config.nvlink_host_numa_node = val;
        } else {
            LOG(WARNING) << "Ignore value from environment variable "
                         << "MC_NVLINK_HOST_NUMA_NODE";
        }
    }
    if (const char* domain_env = std::getenv("MC_NVLINK_SCALE_UP_DOMAIN_ID")) {
        config.nvlink_scale_up_domain_id = domain_env;
    }
```

If `std::tolower` is not already available in this file, add:

```cpp
#include <cctype>
```

- [ ] **Step 5: Run config test and verify it passes**

Run:

```bash
cmake --build build --target config_test -j
ctest --test-dir build -R '^config_test$' --output-on-failure
```

Expected: `config_test` passes.

- [ ] **Step 6: Commit config support**

Run:

```bash
git add mooncake-transfer-engine/include/config.h mooncake-transfer-engine/src/config.cpp mooncake-transfer-engine/tests/config_test.cpp
git commit -m "feat: add nvlink host numa config"
```

---

## Task 3: NVLink HOST_NUMA VMM Allocation and Registration

**Files:**
- Modify: `mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h`
- Modify: `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`
- Modify: `mooncake-transfer-engine/tests/nvlink_transport_test.cpp`

- [ ] **Step 1: Write hardware-gated failing test**

Add helpers and this test to `mooncake-transfer-engine/tests/nvlink_transport_test.cpp`:

```cpp
static bool canRunHostNumaFabricTest() {
#if defined(USE_CUDA) && defined(USE_MNNVL)
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return false;
    }
    int supported = 0;
    CUdevice dev;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return false;
    if (cuDeviceGetAttribute(&supported,
                             CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                             dev) != CUDA_SUCCESS) {
        return false;
    }
    return supported != 0;
#else
    return false;
#endif
}

TEST(NvlinkTransportTest, HostNumaFabricAllocationHasMetadata) {
    if (!canRunHostNumaFabricTest()) {
        GTEST_SKIP() << "HOST_NUMA fabric memory is not supported here";
    }
    setenv("MC_ENABLE_NVLINK_HOST_NUMA", "1", 1);
    setenv("MC_NVLINK_HOST_NUMA_NODE", "0", 1);
    setenv("MC_NVLINK_SCALE_UP_DOMAIN_ID", "test-domain", 1);

    void* buffer = NvlinkTransport::allocateHostNumaFabricMemory(4096, 0);
    ASSERT_NE(buffer, nullptr);

    auto engine = std::make_unique<TransferEngine>(false);
    ASSERT_EQ(engine->init(FLAGS_metadata_server, FLAGS_local_server_name), 0);
    ASSERT_NE(engine->installTransport(MNNVL_PROTOCOL, nullptr), nullptr);

    ASSERT_EQ(engine->registerLocalMemory(buffer, 4096, "host_numa:0"), 0);
    auto desc = engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->buffers.empty());
    EXPECT_EQ(desc->buffers.back().memory_kind, "HOST_NUMA");
    EXPECT_EQ(desc->buffers.back().scale_up_domain_id, "test-domain");
    EXPECT_FALSE(desc->buffers.back().shm_name.empty());

    engine->unregisterLocalMemory(buffer);
    NvlinkTransport::freeHostNumaFabricMemory(buffer);
    unsetenv("MC_ENABLE_NVLINK_HOST_NUMA");
    unsetenv("MC_NVLINK_HOST_NUMA_NODE");
    unsetenv("MC_NVLINK_SCALE_UP_DOMAIN_ID");
}
```

- [ ] **Step 2: Build test and verify it fails to compile**

Run:

```bash
cmake -S . -B build-mnnvl -DBUILD_UNIT_TESTS=ON -DWITH_TE=ON -DWITH_STORE=ON -DENABLE_MULTI_PROTOCOL=ON -DUSE_CUDA=ON -DUSE_MNNVL=ON -DUSE_TCP=ON
cmake --build build-mnnvl --target nvlink_transport_test -j
```

Expected: compile fails because `allocateHostNumaFabricMemory` and `freeHostNumaFabricMemory` are not defined.

- [ ] **Step 3: Add public HOST_NUMA helpers**

In `mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h`, add:

```cpp
    static bool supportHostNumaFabricMem();

    static void* allocateHostNumaFabricMemory(size_t length, int numa_node);

    static void freeHostNumaFabricMemory(void* addr);
```

- [ ] **Step 4: Implement HOST_NUMA allocation**

In `mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp`, add:

```cpp
bool NvlinkTransport::supportHostNumaFabricMem() {
    if (!supportFabricMem()) return false;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return false;
    }
    return true;
}

void* NvlinkTransport::allocateHostNumaFabricMemory(size_t size,
                                                    int numa_node) {
    if (!supportHostNumaFabricMem()) return nullptr;
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle handle;
    void* ptr = nullptr;

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = numa_node;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    CUresult result = cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (result != CUDA_SUCCESS) {
        LOG(ERROR) << "NvlinkTransport: HOST_NUMA granularity failed: "
                   << result;
        return nullptr;
    }
    size = (size + granularity - 1) & ~(granularity - 1);
    if (size == 0) size = granularity;

    result = cuMemCreate(&handle, size, &prop, 0);
    if (result != CUDA_SUCCESS) {
        LOG(ERROR) << "NvlinkTransport: HOST_NUMA cuMemCreate failed: "
                   << result;
        return nullptr;
    }
    result = cuMemAddressReserve((CUdeviceptr*)&ptr, size, granularity, 0, 0);
    if (result != CUDA_SUCCESS) {
        cuMemRelease(handle);
        return nullptr;
    }
    result = cuMemMap((CUdeviceptr)ptr, size, 0, handle, 0);
    if (result != CUDA_SUCCESS) {
        cuMemAddressFree((CUdeviceptr)ptr, size);
        cuMemRelease(handle);
        return nullptr;
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::vector<CUmemAccessDesc> access(device_count);
    for (int device_id = 0; device_id < device_count; ++device_id) {
        access[device_id].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[device_id].location.id = device_id;
        access[device_id].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    result = cuMemSetAccess((CUdeviceptr)ptr, size, access.data(),
                            access.size());
    if (result != CUDA_SUCCESS) {
        cuMemUnmap((CUdeviceptr)ptr, size);
        cuMemAddressFree((CUdeviceptr)ptr, size);
        cuMemRelease(handle);
        return nullptr;
    }
    return ptr;
}

void NvlinkTransport::freeHostNumaFabricMemory(void* ptr) {
    freePinnedLocalMemory(ptr);
}
```

Add `#include <vector>` if it is not already present in the `.cpp`.

- [ ] **Step 5: Tag NVLink metadata for HOST_NUMA**

In `NvlinkTransport::registerLocalMemory`, after the fabric path assigns
`desc.shm_name`, add:

```cpp
        if (globalConfig().enable_nvlink_host_numa) {
            desc.memory_kind = "HOST_NUMA";
            desc.scale_up_domain_id = globalConfig().nvlink_scale_up_domain_id;
        }
#ifdef ENABLE_MULTI_PROTOCOL
        desc.protocol = "nvlink";
#endif
```

Also set `desc.protocol = "nvlink"` in the cudaIpc path when `ENABLE_MULTI_PROTOCOL` is compiled.

- [ ] **Step 6: Add domain mismatch skip in import path**

In `NvlinkTransport::relocateSharedMemoryAddress`, before importing a fabric handle, add:

```cpp
                    if (!entry.scale_up_domain_id.empty() &&
                        entry.scale_up_domain_id !=
                            globalConfig().nvlink_scale_up_domain_id) {
                        LOG(WARNING)
                            << "NvlinkTransport: skip fabric import due to "
                            << "scale-up domain mismatch remote="
                            << entry.scale_up_domain_id << " local="
                            << globalConfig().nvlink_scale_up_domain_id;
                        return ERR_INVALID_ARGUMENT;
                    }
```

- [ ] **Step 7: Build and run hardware-gated NVLink test**

Run:

```bash
cmake --build build-mnnvl --target nvlink_transport_test -j
ctest --test-dir build-mnnvl -R '^nvlink_transport_test$' --output-on-failure
```

Expected on non-GB200 or non-fabric systems: test binary passes with the HOST_NUMA test skipped. Expected on GB200/NVL72 with fabric support: test passes and validates metadata.

- [ ] **Step 8: Commit NVLink HOST_NUMA helpers**

Run:

```bash
git add mooncake-transfer-engine/include/transport/nvlink_transport/nvlink_transport.h mooncake-transfer-engine/src/transport/nvlink_transport/nvlink_transport.cpp mooncake-transfer-engine/tests/nvlink_transport_test.cpp
git commit -m "feat: add nvlink host numa fabric helpers"
```

---

## Task 4: Store Descriptors and Dual-Protocol Mount

**Files:**
- Modify: `mooncake-store/include/allocator.h`
- Modify: `mooncake-store/src/allocator.cpp`
- Modify: `mooncake-store/include/client_service.h`
- Modify: `mooncake-store/src/client_service.cpp`
- Modify: `mooncake-store/tests/client_integration_test.cpp`

- [ ] **Step 1: Write descriptor propagation test**

Add a test to `mooncake-store/tests/client_integration_test.cpp`:

```cpp
TEST(AllocatedBufferDescriptorTest, CarriesProtocolMemoryKindAndDomain) {
    AllocatedBuffer::Descriptor desc;
    desc.size_ = 4096;
    desc.buffer_address_ = 0x100000;
    desc.protocol_ = "nvlink,rdma";
    desc.transport_endpoint_ = "host-a:12345";
    desc.memory_kind_ = "HOST_NUMA";
    desc.scale_up_domain_id_ = "domain-a";
    desc.selected_protocol_ = "nvlink";

    EXPECT_EQ(desc.protocol_, "nvlink,rdma");
    EXPECT_EQ(desc.memory_kind_, "HOST_NUMA");
    EXPECT_EQ(desc.scale_up_domain_id_, "domain-a");
    EXPECT_EQ(desc.selected_protocol_, "nvlink");
}
```

- [ ] **Step 2: Run store test and verify it fails**

Run:

```bash
cmake --build build --target client_integration_test -j
ctest --test-dir build -R '^client_integration_test$' --output-on-failure
```

Expected: compile fails because descriptor fields do not exist.

- [ ] **Step 3: Extend AllocatedBuffer descriptor**

In `mooncake-store/include/allocator.h`, extend `AllocatedBuffer::Descriptor`:

```cpp
        std::string protocol_;
        std::string transport_endpoint_;
        std::string memory_kind_;
        std::string scale_up_domain_id_;
        std::string selected_protocol_;
        YLT_REFL(Descriptor, size_, buffer_address_, protocol_,
                 transport_endpoint_, memory_kind_, scale_up_domain_id_,
                 selected_protocol_);
```

Add private fields to `AllocatedBuffer`:

```cpp
    std::string memory_kind_;
    std::string scale_up_domain_id_;
```

Add public setters:

```cpp
    void setMemoryAttributes(std::string memory_kind,
                             std::string scale_up_domain_id) {
        memory_kind_ = std::move(memory_kind);
        scale_up_domain_id_ = std::move(scale_up_domain_id);
    }
```

- [ ] **Step 4: Populate descriptor fields**

In `mooncake-store/src/allocator.cpp`, update `AllocatedBuffer::get_descriptor()` return value:

```cpp
    return {static_cast<uint64_t>(size()),
            reinterpret_cast<uintptr_t>(buffer_ptr_),
            this->protocol,
            endpoint,
            this->memory_kind_,
            this->scale_up_domain_id_,
            ""};
```

- [ ] **Step 5: Add registration-free mount helper**

In `mooncake-store/include/client_service.h`, add this private helper:

```cpp
    tl::expected<UUID, ErrorCode> MountSegmentAfterRegistrationLocked(
        const void* buffer, size_t size, const std::string& protocol);
```

Add this implementation to `mooncake-store/src/client_service.cpp`:

```cpp
tl::expected<UUID, ErrorCode> Client::MountSegmentAfterRegistrationLocked(
    const void* buffer, size_t size, const std::string& protocol) {
    Segment segment;
    segment.id = generate_uuid();
    segment.name = local_hostname_;
    segment.base = reinterpret_cast<uintptr_t>(buffer);
    segment.size = size;
    segment.protocol = protocol;
    if (metadata_connstring_ == P2PHANDSHAKE) {
        segment.te_endpoint = transfer_engine_->getLocalIpAndPort();
    } else {
        segment.te_endpoint = local_hostname_;
    }

    auto mount_result = master_client_.MountSegment(segment);
    if (!mount_result) {
        ErrorCode err = mount_result.error();
        LOG(ERROR) << "mount_segment_to_master_failed base=" << buffer
                   << " size=" << size << ", error=" << err;
        return tl::unexpected(err);
    }

    mounted_segments_[segment.id] = segment;
    return segment.id;
}
```

Keep the overlap checks and transfer-engine registration in
`MountSegmentAndGetId`.
After the move, `MountSegmentAndGetId` ends with:

```cpp
        int rc = transfer_engine_->registerLocalMemory((void*)buffer, size,
                                                       location, true, true);
        if (rc != 0) {
            LOG(ERROR) << "register_local_memory_failed base=" << buffer
                       << " size=" << size << ", error=" << rc;
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }

        auto mounted =
            MountSegmentAfterRegistrationLocked(buffer, size, protocol);
        if (!mounted) return mounted;
        segment_id = mounted.value();
```

- [ ] **Step 6: Add dual-protocol mount path**

In `mooncake-store/include/client_service.h`, add a public helper next to `MountSegmentAndGetId`:

```cpp
    tl::expected<UUID, ErrorCode> MountDualProtocolSegmentAndGetId(
        const void* buffer, size_t size, const std::string& location);
```

In `mooncake-store/src/client_service.cpp`, implement:

```cpp
tl::expected<UUID, ErrorCode> Client::MountDualProtocolSegmentAndGetId(
    const void* buffer, size_t size, const std::string& location) {
#ifndef ENABLE_MULTI_PROTOCOL
    LOG(ERROR) << "dual-protocol segment requires ENABLE_MULTI_PROTOCOL";
    return tl::unexpected(ErrorCode::INVALID_PARAMS);
#else
    auto check_result = CheckRegisterMemoryParams(buffer, size);
    if (!check_result) return tl::unexpected(check_result.error());

    std::unordered_map<std::string,
                       std::vector<TransferEngine::RegisteredBuffer>>
        buffer_map;
    buffer_map["nvlink"].emplace_back((void*)buffer, size, location, true, false);
    buffer_map["rdma"].emplace_back((void*)buffer, size, location, true, false);

    int rc = transfer_engine_->mp_registerLocalMemory(buffer_map);
    if (rc != 0) {
        if (globalConfig().nvlink_host_numa_strict) {
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
        rc = transfer_engine_->registerLocalMemory((void*)buffer, size,
                                                   location, true, true);
        if (rc != 0) return tl::unexpected(ErrorCode::INVALID_PARAMS);
        return MountSegmentAfterRegistrationLocked(buffer, size, "rdma");
    }

    auto mounted =
        MountSegmentAfterRegistrationLocked(buffer, size, "nvlink,rdma");
    if (!mounted) return mounted;
    int rc_update = transfer_engine_->getMetadata()->updateLocalSegmentDesc();
    if (rc_update != 0) {
        LOG(ERROR) << "update dual-protocol segment metadata failed rc="
                   << rc_update;
        return tl::unexpected(ErrorCode::INTERNAL_ERROR);
    }
    return mounted;
#endif
}
```

- [ ] **Step 7: Install both transports for dual mode**

In `Client::InitTransferEngine`, add a protocol branch:

```cpp
        } else if (protocol == "nvlink,rdma" || protocol == "rdma,nvlink") {
            Transport* nvlink_transport =
                transfer_engine_->installTransport("nvlink", nullptr);
            if (!nvlink_transport) {
                LOG(ERROR) << "Failed to install nvlink transport";
                return ErrorCode::INTERNAL_ERROR;
            }
            Transport* rdma_transport =
                transfer_engine_->installTransport("rdma", nullptr);
            if (!rdma_transport) {
                LOG(ERROR) << "Failed to install rdma transport";
                return ErrorCode::INTERNAL_ERROR;
            }
```

Place this branch after the RDMA branch so RDMA device discovery remains in one
code path. Extract the RDMA installation body into:

```cpp
auto install_rdma_transport = [&]() -> ErrorCode {
    if (!device_names.has_value() || device_names->empty()) {
        LOG(ERROR) << "RDMA protocol requires device names when auto "
                      "discovery is disabled";
        return ErrorCode::INVALID_PARAMS;
    }
    std::vector<std::string> devices =
        splitString(device_names.value(), ',', true);
    auto topology = transfer_engine_->getLocalTopology();
    if (topology) topology->discover(devices);
    Transport* transport = transfer_engine_->installTransport("rdma", nullptr);
    if (!transport) return ErrorCode::INTERNAL_ERROR;
    return ErrorCode::OK;
};
```

The existing `protocol == "rdma"` branch calls `install_rdma_transport()`. The
dual-protocol branch installs NVLink first, then calls
`install_rdma_transport()`.

- [ ] **Step 8: Run store tests**

Run:

```bash
cmake --build build --target client_integration_test -j
ctest --test-dir build -R '^client_integration_test$' --output-on-failure
```

Expected: descriptor test passes. Existing client integration tests pass for TCP/RDMA builds.

- [ ] **Step 9: Commit store descriptor and mount work**

Run:

```bash
git add mooncake-store/include/allocator.h mooncake-store/src/allocator.cpp mooncake-store/include/client_service.h mooncake-store/src/client_service.cpp mooncake-store/tests/client_integration_test.cpp
git commit -m "feat: add dual protocol store descriptors"
```

---

## Task 5: Worker Transfer Candidate Selector

**Files:**
- Create: `mooncake-store/include/transfer_candidate_selector.h`
- Create: `mooncake-store/src/transfer_candidate_selector.cpp`
- Create: `mooncake-store/tests/transfer_candidate_selector_test.cpp`
- Modify: `mooncake-store/src/CMakeLists.txt`
- Modify: `mooncake-store/tests/CMakeLists.txt`

- [ ] **Step 1: Write failing selector tests**

Create `mooncake-store/tests/transfer_candidate_selector_test.cpp`:

```cpp
#include "transfer_candidate_selector.h"

#include <gtest/gtest.h>

namespace mooncake {

static Replica::Descriptor MakeMemoryReplica(const std::string& protocol,
                                             const std::string& endpoint,
                                             const std::string& memory_kind,
                                             const std::string& domain) {
    AllocatedBuffer::Descriptor buffer;
    buffer.size_ = 4096;
    buffer.buffer_address_ = 0x100000;
    buffer.protocol_ = protocol;
    buffer.transport_endpoint_ = endpoint;
    buffer.memory_kind_ = memory_kind;
    buffer.scale_up_domain_id_ = domain;

    MemoryDescriptor memory;
    memory.buffer_descriptor = buffer;

    Replica::Descriptor replica;
    replica.id = 1;
    replica.status = ReplicaStatus::COMPLETE;
    replica.descriptor_variant = memory;
    return replica;
}

TEST(TransferCandidateSelectorTest, ChoosesSameDomainNvlink) {
    std::vector<Replica::Descriptor> replicas;
    replicas.push_back(MakeMemoryReplica("nvlink,rdma", "remote:12345",
                                         "HOST_NUMA", "domain-a"));

    TransferCandidateContext context;
    context.local_endpoints.insert("local:12345");
    context.enable_nvlink_host_numa = true;
    context.local_scale_up_domain_id = "domain-a";
    context.destination_is_hbm = true;

    auto candidate = SelectTransferCandidate(replicas, context);
    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(candidate->protocol, "nvlink");
}

TEST(TransferCandidateSelectorTest, ChoosesRdmaForCrossDomain) {
    std::vector<Replica::Descriptor> replicas;
    replicas.push_back(MakeMemoryReplica("nvlink,rdma", "remote:12345",
                                         "HOST_NUMA", "domain-b"));

    TransferCandidateContext context;
    context.enable_nvlink_host_numa = true;
    context.local_scale_up_domain_id = "domain-a";
    context.destination_is_hbm = true;

    auto candidate = SelectTransferCandidate(replicas, context);
    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(candidate->protocol, "rdma");
}

TEST(TransferCandidateSelectorTest, SkipsNvlinkForNonHbmDestination) {
    std::vector<Replica::Descriptor> replicas;
    replicas.push_back(MakeMemoryReplica("nvlink,rdma", "remote:12345",
                                         "HOST_NUMA", "domain-a"));

    TransferCandidateContext context;
    context.enable_nvlink_host_numa = true;
    context.local_scale_up_domain_id = "domain-a";
    context.destination_is_hbm = false;

    auto candidate = SelectTransferCandidate(replicas, context);
    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(candidate->protocol, "rdma");
}

}  // namespace mooncake
```

Add this target to `mooncake-store/tests/CMakeLists.txt`:

```cmake
add_store_test(transfer_candidate_selector_test
               transfer_candidate_selector_test.cpp)
```

- [ ] **Step 2: Run selector test and verify it fails**

Run:

```bash
cmake --build build --target transfer_candidate_selector_test -j
```

Expected: compile fails because `transfer_candidate_selector.h` does not exist.

- [ ] **Step 3: Add selector header**

Create `mooncake-store/include/transfer_candidate_selector.h`:

```cpp
#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "replica.h"

namespace mooncake {

struct TransferCandidateContext {
    std::unordered_set<std::string> local_endpoints;
    bool enable_nvlink_host_numa = false;
    bool destination_is_hbm = false;
    std::string local_scale_up_domain_id;
};

struct TransferCandidate {
    Replica::Descriptor replica;
    std::string protocol;
};

std::optional<TransferCandidate> SelectTransferCandidate(
    const std::vector<Replica::Descriptor>& replicas,
    const TransferCandidateContext& context);

}  // namespace mooncake
```

- [ ] **Step 4: Implement selector source**

Create `mooncake-store/src/transfer_candidate_selector.cpp`:

```cpp
#include "transfer_candidate_selector.h"

#include <sstream>

namespace mooncake {
namespace {

bool HasProtocol(const std::string& protocols, const std::string& expected) {
    std::stringstream ss(protocols);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item == expected) return true;
    }
    return protocols == expected;
}

bool IsSameDomainNvlinkCandidate(const AllocatedBuffer::Descriptor& buffer,
                                 const TransferCandidateContext& context) {
    return context.enable_nvlink_host_numa && context.destination_is_hbm &&
           HasProtocol(buffer.protocol_, "nvlink") &&
           buffer.memory_kind_ == "HOST_NUMA" &&
           !buffer.scale_up_domain_id_.empty() &&
           buffer.scale_up_domain_id_ == context.local_scale_up_domain_id;
}

}  // namespace

std::optional<TransferCandidate> SelectTransferCandidate(
    const std::vector<Replica::Descriptor>& replicas,
    const TransferCandidateContext& context) {
    for (const auto& replica : replicas) {
        if (replica.status != ReplicaStatus::COMPLETE) continue;
        if (!replica.is_memory_replica()) continue;
        const auto& buffer =
            replica.get_memory_descriptor().buffer_descriptor;
        if (context.local_endpoints.count(buffer.transport_endpoint_)) {
            return TransferCandidate{replica, "local"};
        }
    }

    for (const auto& replica : replicas) {
        if (replica.status != ReplicaStatus::COMPLETE) continue;
        if (!replica.is_memory_replica()) continue;
        const auto& buffer =
            replica.get_memory_descriptor().buffer_descriptor;
        if (IsSameDomainNvlinkCandidate(buffer, context)) {
            return TransferCandidate{replica, "nvlink"};
        }
    }

    for (const auto& replica : replicas) {
        if (replica.status != ReplicaStatus::COMPLETE) continue;
        if (!replica.is_memory_replica()) continue;
        const auto& buffer =
            replica.get_memory_descriptor().buffer_descriptor;
        if (HasProtocol(buffer.protocol_, "rdma")) {
            return TransferCandidate{replica, "rdma"};
        }
    }

    for (const auto& replica : replicas) {
        if (replica.status == ReplicaStatus::COMPLETE) {
            return TransferCandidate{replica, ""};
        }
    }
    return std::nullopt;
}

}  // namespace mooncake
```

- [ ] **Step 5: Add selector source to store library**

In `mooncake-store/src/CMakeLists.txt`, append `transfer_candidate_selector.cpp` to `MOONCAKE_STORE_SOURCES` near `transfer_task.cpp`:

```cmake
    transfer_task.cpp
    transfer_candidate_selector.cpp
    tenant_quota.cpp
```

- [ ] **Step 6: Run selector test**

Run:

```bash
cmake --build build --target transfer_candidate_selector_test -j
ctest --test-dir build -R '^transfer_candidate_selector_test$' --output-on-failure
```

Expected: selector tests pass.

- [ ] **Step 7: Commit selector**

Run:

```bash
git add mooncake-store/include/transfer_candidate_selector.h mooncake-store/src/transfer_candidate_selector.cpp mooncake-store/tests/transfer_candidate_selector_test.cpp mooncake-store/src/CMakeLists.txt mooncake-store/tests/CMakeLists.txt
git commit -m "feat: add store transfer candidate selector"
```

---

## Task 6: Submit Selected Protocol While Preserving Batch Semantics

**Files:**
- Modify: `mooncake-store/include/transfer_task.h`
- Modify: `mooncake-store/src/transfer_task.cpp`
- Modify: `mooncake-store/src/real_client.cpp`
- Modify: `mooncake-store/tests/transfer_task_test.cpp`

- [ ] **Step 1: Write failing batching test**

Add this lightweight test helper to `mooncake-store/tests/transfer_task_test.cpp`:

```cpp
TEST_F(TransferTaskTest, ProtocolSelectionDoesNotCollapseSlices) {
    std::vector<Slice> slices;
    char a[16] = {};
    char b[32] = {};
    char c[48] = {};
    slices.push_back(Slice{a, sizeof(a)});
    slices.push_back(Slice{b, sizeof(b)});
    slices.push_back(Slice{c, sizeof(c)});

    auto requests = TransferSubmitter::BuildTransferRequestsForTest(
        7, 0x1000, slices, TransferRequest::READ, 0);

    ASSERT_EQ(requests.size(), 3u);
    EXPECT_EQ(requests[0].source, a);
    EXPECT_EQ(requests[0].target_offset, 0x1000u);
    EXPECT_EQ(requests[1].source, b);
    EXPECT_EQ(requests[1].target_offset, 0x1010u);
    EXPECT_EQ(requests[2].source, c);
    EXPECT_EQ(requests[2].target_offset, 0x1030u);
}
```

- [ ] **Step 2: Run transfer task test and verify it fails**

Run:

```bash
cmake --build build --target transfer_task_test -j
ctest --test-dir build -R '^transfer_task_test$' --output-on-failure
```

Expected: compile fails because `BuildTransferRequestsForTest` is missing.

- [ ] **Step 3: Add request builder helper**

In `mooncake-store/include/transfer_task.h`, add a public static helper to `TransferSubmitter`:

```cpp
    static std::vector<TransferRequest> BuildTransferRequestsForTest(
        SegmentHandle segment, uint64_t base_address,
        const std::vector<Slice>& slices, TransferRequest::OpCode op_code,
        uint64_t src_offset);
```

In `mooncake-store/src/transfer_task.cpp`, implement it:

```cpp
std::vector<TransferRequest> TransferSubmitter::BuildTransferRequestsForTest(
    SegmentHandle segment, uint64_t base_address,
    const std::vector<Slice>& slices, TransferRequest::OpCode op_code,
    uint64_t src_offset) {
    std::vector<TransferRequest> requests;
    requests.reserve(slices.size());
    uint64_t offset = src_offset;
    for (const auto& slice : slices) {
        if (slice.ptr == nullptr) continue;
        TransferRequest request;
        request.opcode = op_code;
        request.source = static_cast<char*>(slice.ptr);
        request.target_id = segment;
        request.target_offset = base_address + offset;
        request.length = slice.size;
        offset += slice.size;
        requests.emplace_back(request);
    }
    return requests;
}
```

Update `submitTransferEngineOperation` to call this helper instead of building requests inline.

- [ ] **Step 4: Add selected-protocol submit overload**

In `mooncake-store/include/transfer_task.h`, change the private submit API:

```cpp
    std::optional<TransferFuture> submitTransfer(
        std::vector<TransferRequest>& requests,
        const std::string& selected_protocol = "");
```

In `mooncake-store/src/transfer_task.cpp`, update `submitTransfer`:

```cpp
std::optional<TransferFuture> TransferSubmitter::submitTransfer(
    std::vector<TransferRequest>& requests,
    const std::string& selected_protocol) {
    const size_t batch_size = requests.size();
    BatchID batch_id = engine_.allocateBatchID(batch_size);
    if (batch_id == INVALID_BATCH_ID) {
        LOG(ERROR) << "Failed to allocate batch ID";
        return std::nullopt;
    }

    Status s = Status::OK();
#ifdef ENABLE_MULTI_PROTOCOL
    if (!selected_protocol.empty() && selected_protocol != "local") {
        std::string proto = selected_protocol;
        s = engine_.mp_submitTransfer(batch_id, requests, proto);
    } else
#endif
    {
        s = engine_.submitTransfer(batch_id, requests);
    }

    if (!s.ok()) {
        LOG(ERROR) << "Failed to submit all transfers, error code is "
                   << s.code();
        engine_.freeBatchID(batch_id);
        return std::nullopt;
    }

    auto state = std::make_shared<TransferEngineOperationState>(
        engine_, batch_id, batch_size);
    return TransferFuture(state);
}
```

- [ ] **Step 5: Submit transfer-engine operations with descriptor-selected protocol**

In `mooncake-store/src/transfer_task.cpp`, update the end of
`submitTransferEngineOperation`:

```cpp
    std::vector<TransferRequest> requests = BuildTransferRequestsForTest(
        seg, base_address, slices, op_code, src_offset);
    return submitTransfer(requests, handle.selected_protocol_);
```

This keeps scattered slices as one request vector and changes only the
transport selected by the local worker.

- [ ] **Step 6: Pass selected protocol from RealClient**

In `mooncake-store/src/real_client.cpp`, include the selector:

```cpp
#include "transfer_candidate_selector.h"
```

Replace calls to the anonymous `SelectBestReplica` with `SelectTransferCandidate`. Build the context from existing local endpoints:

```cpp
TransferCandidateContext context;
context.local_endpoints = client_->GetLocalEndpoints();
context.enable_nvlink_host_numa = globalConfig().enable_nvlink_host_numa;
context.local_scale_up_domain_id = globalConfig().nvlink_scale_up_domain_id;
context.destination_is_hbm =
    gpu_staging::IsDevicePointer(buffer_handle->ptr(), nullptr);
auto candidate = SelectTransferCandidate(replica_list, context);
```

Before filtering the query result or building ranged-read metadata, copy the
selected protocol into the local replica descriptor:

```cpp
if (!candidate) {
    LOG(ERROR) << "No usable replica for key: " << key;
    return nullptr;
}
auto replica = candidate->replica;
if (replica.is_memory_replica()) {
    replica.get_memory_descriptor()
        .buffer_descriptor.selected_protocol_ = candidate->protocol;
}
```

For functions that return `tl::expected`, replace `return nullptr;` in the
snippet above with:

```cpp
return tl::unexpected(ErrorCode::INVALID_REPLICA);
```

- [ ] **Step 7: Run transfer task and selector tests**

Run:

```bash
cmake --build build --target transfer_task_test transfer_candidate_selector_test -j
ctest --test-dir build -R '^(transfer_task_test|transfer_candidate_selector_test)$' --output-on-failure
```

Expected: both tests pass.

- [ ] **Step 8: Commit selected protocol submission**

Run:

```bash
git add mooncake-store/include/transfer_task.h mooncake-store/src/transfer_task.cpp mooncake-store/src/real_client.cpp mooncake-store/tests/transfer_task_test.cpp
git commit -m "feat: submit selected memory transport protocol"
```

---

## Task 7: End-to-End Validation and Regression Suite

**Files:**
- Modify: `mooncake-transfer-engine/tests/nvlink_transport_test.cpp`
- Modify: `mooncake-store/tests/client_integration_test.cpp`
- Modify: `docs/superpowers/specs/2026-06-29-nvlink-host-numa-rdma-dual-register-design.md` only if validation finds a design mismatch.

- [ ] **Step 1: Add same-domain remote DRAM to local HBM test**

Extend the hardware-gated NVLink test in `nvlink_transport_test.cpp` with a second test named:

```cpp
TEST(NvlinkTransportTest, HostNumaRemoteDramToLocalHbmRead)
```

Use this structure:

```cpp
if (!canRunHostNumaFabricTest()) {
    GTEST_SKIP() << "HOST_NUMA fabric memory is not supported here";
}
setenv("MC_ENABLE_NVLINK_HOST_NUMA", "1", 1);
setenv("MC_NVLINK_SCALE_UP_DOMAIN_ID", "test-domain", 1);

const size_t kDataLength = 4096;
void* remote_host_numa =
    NvlinkTransport::allocateHostNumaFabricMemory(kDataLength, 0);
ASSERT_NE(remote_host_numa, nullptr);
void* local_hbm = allocateCudaBuffer(kDataLength, FLAGS_gpu_id);
ASSERT_NE(local_hbm, nullptr);

std::vector<char> expected(kDataLength, 'H');
ASSERT_EQ(cudaMemcpy(remote_host_numa, expected.data(), kDataLength,
                     cudaMemcpyHostToDevice),
          cudaSuccess);

auto server_engine = std::make_unique<TransferEngine>(false);
ASSERT_EQ(server_engine->init(FLAGS_metadata_server,
                              FLAGS_local_server_name),
          0);
ASSERT_NE(server_engine->installTransport(MNNVL_PROTOCOL, nullptr), nullptr);
ASSERT_EQ(server_engine->registerLocalMemory(remote_host_numa, kDataLength,
                                             "host_numa:0"),
          0);
auto segment_id = server_engine->openSegment(FLAGS_segment_id);
ASSERT_NE(segment_id, static_cast<uint64_t>(ERR_INVALID_ARGUMENT));

auto client_engine = std::make_unique<TransferEngine>(false);
ASSERT_EQ(client_engine->init(FLAGS_metadata_server, "cuda_client:12346"), 0);
ASSERT_NE(client_engine->installTransport(MNNVL_PROTOCOL, nullptr), nullptr);
ASSERT_EQ(client_engine->registerLocalMemory(local_hbm, kDataLength,
                                             "cuda:" +
                                                 std::to_string(FLAGS_gpu_id)),
          0);

auto batch_id = client_engine->allocateBatchID(1);
ASSERT_NE(batch_id, INVALID_BATCH_ID);
TransferRequest entry;
entry.opcode = TransferRequest::READ;
entry.length = kDataLength;
entry.source = local_hbm;
entry.target_id = segment_id;
entry.target_offset = reinterpret_cast<uint64_t>(remote_host_numa);
Status submit_status = client_engine->submitTransfer(batch_id, {entry});
ASSERT_TRUE(submit_status.ok());

TransferStatus status;
do {
    submit_status = client_engine->getTransferStatus(batch_id, 0, status);
    ASSERT_TRUE(submit_status.ok());
} while (status.s == TransferStatusEnum::WAITING);
ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
ASSERT_TRUE(client_engine->freeBatchID(batch_id).ok());

std::vector<char> actual(kDataLength);
ASSERT_EQ(cudaMemcpy(actual.data(), local_hbm, kDataLength,
                     cudaMemcpyDeviceToHost),
          cudaSuccess);
EXPECT_EQ(actual, expected);

client_engine->unregisterLocalMemory(local_hbm);
server_engine->unregisterLocalMemory(remote_host_numa);
freeCudaBuffer(local_hbm);
NvlinkTransport::freeHostNumaFabricMemory(remote_host_numa);
unsetenv("MC_ENABLE_NVLINK_HOST_NUMA");
unsetenv("MC_NVLINK_SCALE_UP_DOMAIN_ID");
```

When filling the host NUMA allocation, use `cudaMemcpyDefault` if the CUDA
runtime rejects `cudaMemcpyHostToDevice` for the UVA pointer.

- [ ] **Step 2: Run hardware-gated NVLink test**

Run:

```bash
cmake --build build-mnnvl --target nvlink_transport_test -j
ctest --test-dir build-mnnvl -R '^nvlink_transport_test$' --output-on-failure
```

Expected on unsupported hardware: test passes with skips. Expected on GB200/NVL72: HOST_NUMA read test passes.

- [ ] **Step 3: Run store regression tests**

Run:

```bash
cmake --build build --target transfer_metadata_test config_test transfer_candidate_selector_test transfer_task_test client_integration_test -j
ctest --test-dir build -R '^(transfer_metadata_test|config_test|transfer_candidate_selector_test|transfer_task_test|client_integration_test)$' --output-on-failure
```

Expected: all selected tests pass.

- [ ] **Step 4: Run formatting and diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` prints no output. `git status --short` shows only files intentionally changed for this feature.

- [ ] **Step 5: Commit validation fixes**

If Step 1 through Step 4 required code or test changes after Task 6, commit them:

```bash
git add mooncake-transfer-engine/tests/nvlink_transport_test.cpp mooncake-store/tests/client_integration_test.cpp docs/superpowers/specs/2026-06-29-nvlink-host-numa-rdma-dual-register-design.md
git commit -m "test: validate nvlink host numa rdma fallback"
```

If no files changed after validation, do not create an empty commit.

---

## Final Verification

Run these commands before claiming implementation is complete:

```bash
cmake -S . -B build -DBUILD_UNIT_TESTS=ON -DWITH_TE=ON -DWITH_STORE=ON -DENABLE_MULTI_PROTOCOL=ON -DUSE_TCP=ON
cmake --build build --target transfer_metadata_test config_test transfer_candidate_selector_test transfer_task_test client_integration_test -j
ctest --test-dir build -R '^(transfer_metadata_test|config_test|transfer_candidate_selector_test|transfer_task_test|client_integration_test)$' --output-on-failure
git diff --check
git status --short
```

For GB200/NVL72 validation:

```bash
cmake -S . -B build-mnnvl -DBUILD_UNIT_TESTS=ON -DWITH_TE=ON -DWITH_STORE=ON -DENABLE_MULTI_PROTOCOL=ON -DUSE_CUDA=ON -DUSE_MNNVL=ON -DUSE_TCP=ON
cmake --build build-mnnvl --target nvlink_transport_test -j
ctest --test-dir build-mnnvl -R '^nvlink_transport_test$' --output-on-failure
```

Expected final state:

- Existing RDMA/TCP store behavior is unchanged when `MC_ENABLE_NVLINK_HOST_NUMA` is unset.
- Multi-protocol metadata accepts `nvlink,rdma`.
- Dual-registered memory descriptors carry `HOST_NUMA` and scale-up domain metadata.
- Worker-side selection chooses NVLink only for same-domain HBM reads.
- Worker-side selection chooses RDMA for cross-domain reads.
- Scattered local slices remain a single batched request vector.
- HOST_NUMA NVLink tests skip cleanly when fabric-capable hardware is absent.
