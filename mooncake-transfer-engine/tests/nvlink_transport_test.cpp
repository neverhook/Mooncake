#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cuda_alike.h"
#include "config.h"
#include "error.h"
#include "transfer_engine.h"
#include "transfer_metadata.h"
#include "transport/nvlink_transport/nvlink_transport.h"
#include "transport/transport.h"

using namespace mooncake;

// Select protocol based on build configuration
#ifdef USE_HIP
#define MNNVL_PROTOCOL "hip"
#else
#define MNNVL_PROTOCOL "nvlink"
#endif

DEFINE_string(metadata_server, "127.0.0.1:2379", "etcd server host address");
DEFINE_string(local_server_name, "cuda_server:12345", "Local server name");
DEFINE_string(segment_id, "cuda_server:12345", "Segment ID to access data");
DEFINE_int32(gpu_id, 0, "GPU ID to use");

static void checkCudaError(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        LOG(ERROR) << message << " (Error code: " << result << " - "
                   << cudaGetErrorString(result) << ")";
        exit(EXIT_FAILURE);
    }
}

static void* allocateCudaBuffer(size_t size, int gpu_id) {
    checkCudaError(cudaSetDevice(gpu_id), "Failed to set device");
    void* d_buf = nullptr;
    checkCudaError(cudaMalloc(&d_buf, size),
                   "Failed to allocate device memory");
    return d_buf;
}

static void freeCudaBuffer(void* addr) {
    checkCudaError(cudaFree(addr), "Failed to free device memory");
}

namespace {

struct ConfigGuard {
    GlobalConfig& config;
    bool enable_nvlink_host_numa;
    std::string scale_up_domain_id;
    int host_numa_node;

    explicit ConfigGuard(GlobalConfig& config)
        : config(config),
          enable_nvlink_host_numa(config.enable_nvlink_host_numa),
          scale_up_domain_id(config.nvlink_scale_up_domain_id),
          host_numa_node(config.nvlink_host_numa_node) {}

    ~ConfigGuard() {
        config.enable_nvlink_host_numa = enable_nvlink_host_numa;
        config.nvlink_scale_up_domain_id = scale_up_domain_id;
        config.nvlink_host_numa_node = host_numa_node;
    }
};

class RegisteredMemoryGuard {
   public:
    RegisteredMemoryGuard(TransferEngine* engine, void* addr)
        : engine_(engine), addr_(addr) {}

    ~RegisteredMemoryGuard() {
        if (active_) engine_->unregisterLocalMemory(addr_);
    }

    RegisteredMemoryGuard(const RegisteredMemoryGuard&) = delete;
    RegisteredMemoryGuard& operator=(const RegisteredMemoryGuard&) = delete;

   private:
    TransferEngine* engine_;
    void* addr_;
    bool active_ = true;
};

struct CudaBufferDeleter {
    void operator()(void* addr) const {
        if (addr) freeCudaBuffer(addr);
    }
};

struct HostNumaFabricMemoryDeleter {
    void operator()(void* addr) const {
        if (addr) NvlinkTransport::freeHostNumaFabricMemory(addr);
    }
};

::testing::AssertionResult copyHostToCudaVisibleMemory(void* dst,
                                                       const void* src,
                                                       size_t length) {
    cudaError_t err = cudaMemcpy(dst, src, length, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        return ::testing::AssertionSuccess();
    }

    const char* host_to_device_error = cudaGetErrorString(err);
    cudaGetLastError();
    cudaError_t default_err = cudaMemcpy(dst, src, length, cudaMemcpyDefault);
    if (default_err == cudaSuccess) {
        return ::testing::AssertionSuccess();
    }

    return ::testing::AssertionFailure()
           << "cudaMemcpyHostToDevice failed for CUDA-visible memory: "
           << host_to_device_error << "; cudaMemcpyDefault also failed: "
           << cudaGetErrorString(default_err);
}

::testing::AssertionResult submitAndWaitForTransfer(
    TransferEngine* engine, const TransferRequest& entry) {
    constexpr auto kTransferWaitTimeout = std::chrono::seconds(30);
    auto batch_id = engine->allocateBatchID(1);
    if (batch_id == INVALID_BATCH_ID) {
        return ::testing::AssertionFailure() << "allocateBatchID failed";
    }
    Status s = engine->submitTransfer(batch_id, {entry});
    if (!s.ok()) {
        Status free_status = engine->freeBatchID(batch_id);
        return ::testing::AssertionFailure()
               << "submitTransfer failed: " << s.ToString()
               << "; freeBatchID status: " << free_status.ToString();
    }

    TransferStatus status;
    const auto deadline = std::chrono::steady_clock::now() + kTransferWaitTimeout;
    do {
        s = engine->getTransferStatus(batch_id, 0, status);
        if (!s.ok()) {
            Status free_status = engine->freeBatchID(batch_id);
            return ::testing::AssertionFailure()
                   << "getTransferStatus failed: " << s.ToString()
                   << "; freeBatchID status: " << free_status.ToString();
        }
        if (status.s == TransferStatusEnum::WAITING) {
            if (std::chrono::steady_clock::now() >= deadline) {
                Status free_status = engine->freeBatchID(batch_id);
                return ::testing::AssertionFailure()
                       << "transfer timed out after "
                       << std::chrono::duration_cast<std::chrono::seconds>(
                              kTransferWaitTimeout)
                              .count()
                       << "s; freeBatchID status: "
                       << free_status.ToString();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (status.s == TransferStatusEnum::WAITING);

    Status free_status = engine->freeBatchID(batch_id);
    if (!free_status.ok()) {
        return ::testing::AssertionFailure()
               << "freeBatchID failed: " << free_status.ToString();
    }
    if (status.s != TransferStatusEnum::COMPLETED) {
        return ::testing::AssertionFailure()
               << "transfer ended with status " << status.s;
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

TEST(NvlinkTransportTest, HostNumaFabricRegistrationMetadata) {
#if !defined(USE_CUDA) || !defined(USE_MNNVL)
    GTEST_SKIP() << "CUDA MNNVL support is not compiled in";
#else
    if (!NvlinkTransport::supportHostNumaFabricMem()) {
        GTEST_SKIP() << "HOST_NUMA fabric memory is not supported";
    }

    auto& config = globalConfig();
    ConfigGuard config_guard(config);
    config.enable_nvlink_host_numa = true;
    config.nvlink_scale_up_domain_id = "test-domain";
    config.nvlink_host_numa_node = 0;

    void* host_numa_buffer =
        NvlinkTransport::allocateHostNumaFabricMemory(4096, 0);
    if (!host_numa_buffer) {
        GTEST_SKIP() << "failed to allocate HOST_NUMA fabric memory";
    }
    std::unique_ptr<void, decltype(&NvlinkTransport::freeHostNumaFabricMemory)>
        buffer_guard(host_numa_buffer,
                     &NvlinkTransport::freeHostNumaFabricMemory);

    auto engine = std::make_unique<TransferEngine>(false);
    ASSERT_EQ(engine->init(FLAGS_metadata_server,
                           "cuda_host_numa_server:12347"),
              0);

    Transport* transport = engine->installTransport(MNNVL_PROTOCOL, nullptr);
    ASSERT_NE(transport, nullptr);

    int rc = engine->registerLocalMemory(host_numa_buffer, 4096, "host_numa:0");
    ASSERT_EQ(rc, 0);
    RegisteredMemoryGuard registration_guard(engine.get(), host_numa_buffer);

    auto segment_desc = engine->getMetadata()->getSegmentDescByID(
        LOCAL_SEGMENT_ID, false);
    ASSERT_NE(segment_desc, nullptr);
    ASSERT_EQ(segment_desc->buffers.size(), 1u);
    const auto& buffer = segment_desc->buffers[0];
    EXPECT_FALSE(buffer.shm_name.empty());
    EXPECT_EQ(buffer.memory_kind, "HOST_NUMA");
    EXPECT_EQ(buffer.scale_up_domain_id, "test-domain");
#ifdef ENABLE_MULTI_PROTOCOL
    EXPECT_EQ(buffer.protocol, "nvlink");
#endif
#endif
}

TEST(NvlinkTransportTest, HostNumaAccessIncludesHostNumaAndCudaDevices) {
#if !defined(USE_CUDA) || !defined(USE_MNNVL)
    GTEST_SKIP() << "CUDA MNNVL support is not compiled in";
#else
    constexpr int kDeviceCount = 4;
    constexpr int kNumaNode = 2;

    auto access_desc =
        NvlinkTransport::buildHostNumaAccessDescsForTest(kDeviceCount,
                                                         kNumaNode);

    ASSERT_EQ(access_desc.size(), static_cast<size_t>(kDeviceCount + 1));
    EXPECT_EQ(access_desc[0].location.type, CU_MEM_LOCATION_TYPE_HOST_NUMA);
    EXPECT_EQ(access_desc[0].location.id, kNumaNode);
    EXPECT_EQ(access_desc[0].flags, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);

    for (int device_id = 0; device_id < kDeviceCount; ++device_id) {
        const auto& device_access = access_desc[device_id + 1];
        EXPECT_EQ(device_access.location.type, CU_MEM_LOCATION_TYPE_DEVICE);
        EXPECT_EQ(device_access.location.id, device_id);
        EXPECT_EQ(device_access.flags, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    }
#endif
}

TEST(NvlinkTransportTest, DeviceVmmRegistrationDoesNotAdvertiseHostNuma) {
#if !defined(USE_CUDA) || !defined(USE_MNNVL)
    GTEST_SKIP() << "CUDA MNNVL support is not compiled in";
#else
    if (!NvlinkTransport::supportHostNumaFabricMem()) {
        GTEST_SKIP() << "fabric memory is not supported";
    }

    auto& config = globalConfig();
    ConfigGuard config_guard(config);
    config.enable_nvlink_host_numa = true;
    config.nvlink_scale_up_domain_id = "test-domain";
    config.nvlink_host_numa_node = 0;

    checkCudaError(cudaSetDevice(FLAGS_gpu_id), "Failed to set device");
    void* device_vmm_buffer = NvlinkTransport::allocatePinnedLocalMemory(4096);
    if (!device_vmm_buffer) {
        GTEST_SKIP() << "failed to allocate device VMM fabric memory";
    }
    std::unique_ptr<void, decltype(&NvlinkTransport::freePinnedLocalMemory)>
        buffer_guard(device_vmm_buffer,
                     &NvlinkTransport::freePinnedLocalMemory);

    auto engine = std::make_unique<TransferEngine>(false);
    ASSERT_EQ(engine->init(FLAGS_metadata_server,
                           "cuda_device_vmm_server:12348"),
              0);

    Transport* transport = engine->installTransport(MNNVL_PROTOCOL, nullptr);
    ASSERT_NE(transport, nullptr);

    int rc =
        engine->registerLocalMemory(device_vmm_buffer, 4096, "cuda_vmm:0");
    ASSERT_EQ(rc, 0);
    RegisteredMemoryGuard registration_guard(engine.get(), device_vmm_buffer);

    auto segment_desc = engine->getMetadata()->getSegmentDescByID(
        LOCAL_SEGMENT_ID, false);
    ASSERT_NE(segment_desc, nullptr);
    ASSERT_EQ(segment_desc->buffers.size(), 1u);
    const auto& buffer = segment_desc->buffers[0];
    EXPECT_FALSE(buffer.shm_name.empty());
    EXPECT_TRUE(buffer.memory_kind.empty());
    EXPECT_TRUE(buffer.scale_up_domain_id.empty());
#ifdef ENABLE_MULTI_PROTOCOL
    EXPECT_EQ(buffer.protocol, "nvlink");
#endif

#endif
}

TEST(NvlinkTransportTest, HostNumaRemoteDramToLocalHbmRead) {
#if !defined(USE_CUDA) || !defined(USE_MNNVL)
    GTEST_SKIP() << "CUDA MNNVL support is not compiled in";
#else
    if (!NvlinkTransport::supportHostNumaFabricMem()) {
        GTEST_SKIP() << "HOST_NUMA fabric memory is not supported";
    }

    constexpr size_t kDataLength = 1 << 20;
    int gpu_id = FLAGS_gpu_id;

    auto& config = globalConfig();
    ConfigGuard config_guard(config);
    config.enable_nvlink_host_numa = true;
    config.nvlink_scale_up_domain_id = "test-domain";
    config.nvlink_host_numa_node = 0;

    void* host_numa_buffer = NvlinkTransport::allocateHostNumaFabricMemory(
        kDataLength, config.nvlink_host_numa_node);
    if (!host_numa_buffer) {
        GTEST_SKIP() << "failed to allocate HOST_NUMA fabric memory";
    }
    std::unique_ptr<void, HostNumaFabricMemoryDeleter> host_numa_guard(
        host_numa_buffer);

    std::vector<uint8_t> expected(kDataLength);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<uint8_t>((i * 131 + 17) & 0xff);
    }
    auto fill_result = copyHostToCudaVisibleMemory(
        host_numa_buffer, expected.data(), expected.size());
    if (!fill_result) {
        GTEST_SKIP() << fill_result.message();
    }

    cudaError_t cuda_err = cudaSetDevice(gpu_id);
    if (cuda_err != cudaSuccess) {
        GTEST_SKIP() << "failed to set CUDA device " << gpu_id << ": "
                     << cudaGetErrorString(cuda_err);
    }

    void* local_hbm_buffer = nullptr;
    cuda_err = cudaMalloc(&local_hbm_buffer, kDataLength);
    if (cuda_err != cudaSuccess) {
        GTEST_SKIP() << "failed to allocate local HBM buffer: "
                     << cudaGetErrorString(cuda_err);
    }
    std::unique_ptr<void, CudaBufferDeleter> local_hbm_guard(local_hbm_buffer);

    cuda_err = cudaMemset(local_hbm_buffer, 0, kDataLength);
    ASSERT_EQ(cuda_err, cudaSuccess)
        << "failed to initialize local HBM buffer: "
        << cudaGetErrorString(cuda_err);

    const std::string server_name = "127.0.0.1:12349";
    const std::string client_name = "127.0.0.1:12350";

    auto server_engine = std::make_unique<TransferEngine>(false);
    if (server_engine->init(P2PHANDSHAKE, server_name) != 0) {
        GTEST_SKIP() << "server TransferEngine P2PHANDSHAKE init failed";
    }
    Transport* server_transport =
        server_engine->installTransport(MNNVL_PROTOCOL, nullptr);
    ASSERT_NE(server_transport, nullptr);

    int rc =
        server_engine->registerLocalMemory(host_numa_buffer, kDataLength,
                                           "host_numa:0");
    ASSERT_EQ(rc, 0);
    RegisteredMemoryGuard server_registration_guard(server_engine.get(),
                                                    host_numa_buffer);

    auto client_engine = std::make_unique<TransferEngine>(false);
    if (client_engine->init(P2PHANDSHAKE, client_name) != 0) {
        GTEST_SKIP() << "client TransferEngine P2PHANDSHAKE init failed";
    }
    Transport* client_transport =
        client_engine->installTransport(MNNVL_PROTOCOL, nullptr);
    ASSERT_NE(client_transport, nullptr);

    rc = client_engine->registerLocalMemory(
        local_hbm_buffer, kDataLength, "cuda:" + std::to_string(gpu_id));
    ASSERT_EQ(rc, 0);
    RegisteredMemoryGuard client_registration_guard(client_engine.get(),
                                                    local_hbm_buffer);

    const std::string server_endpoint = server_engine->getLocalIpAndPort();
    ASSERT_FALSE(server_endpoint.empty());
    auto segment_id = client_engine->openSegment(server_endpoint);
    ASSERT_NE(segment_id, static_cast<SegmentHandle>(ERR_INVALID_ARGUMENT));
    auto segment_desc =
        client_engine->getMetadata()->getSegmentDescByID(segment_id, false);
    ASSERT_NE(segment_desc, nullptr);
    ASSERT_EQ(segment_desc->protocol, "nvlink");
    ASSERT_EQ(segment_desc->buffers.size(), 1u);
    const auto& remote_buffer = segment_desc->buffers[0];
    EXPECT_EQ(remote_buffer.memory_kind, "HOST_NUMA");
    EXPECT_EQ(remote_buffer.scale_up_domain_id, "test-domain");

    TransferRequest entry;
    entry.opcode = TransferRequest::READ;
    entry.length = kDataLength;
    entry.source = local_hbm_buffer;
    entry.target_id = segment_id;
    entry.target_offset = remote_buffer.addr;
    ASSERT_TRUE(submitAndWaitForTransfer(client_engine.get(), entry));

    std::vector<uint8_t> actual(kDataLength);
    cuda_err = cudaMemcpy(actual.data(), local_hbm_buffer, kDataLength,
                          cudaMemcpyDeviceToHost);
    ASSERT_EQ(cuda_err, cudaSuccess)
        << "failed to copy local HBM buffer back to host: "
        << cudaGetErrorString(cuda_err);
    EXPECT_EQ(actual, expected);
#endif
}

TEST(NvlinkTransportTest, WriteAndRead) {
    const size_t kDataLength = 4096000;
    int gpu_id = FLAGS_gpu_id;

    // Server (target) setup
    auto server_engine = std::make_unique<TransferEngine>(false);
    server_engine->init(FLAGS_metadata_server, FLAGS_local_server_name);

    // Install MNNVL transport (nvlink or hip) on server
    Transport* server_transport =
        server_engine->installTransport(MNNVL_PROTOCOL, nullptr);
    ASSERT_NE(server_transport, nullptr);

    void* server_buffer = allocateCudaBuffer(kDataLength * 2, gpu_id);
    int rc = server_engine->registerLocalMemory(server_buffer, kDataLength * 2,
                                                "cuda:0");
    ASSERT_EQ(rc, 0);

    auto segment_id = server_engine->openSegment(FLAGS_segment_id);

    // Client (initiator) setup
    auto client_engine = std::make_unique<TransferEngine>(false);
    client_engine->init(FLAGS_metadata_server, "cuda_client:12346");

    // Install MNNVL transport (nvlink or hip) on client
    Transport* client_transport =
        client_engine->installTransport(MNNVL_PROTOCOL, nullptr);
    ASSERT_NE(client_transport, nullptr);

    void* client_buffer = allocateCudaBuffer(kDataLength * 2, gpu_id);
    rc = client_engine->registerLocalMemory(client_buffer, kDataLength * 2,
                                            "cuda:" + std::to_string(gpu_id));
    ASSERT_EQ(rc, 0);

    // Write: client -> server
    {
        // Fill client buffer with data
        std::vector<char> host_data(kDataLength, 'A');
        checkCudaError(cudaMemcpy(client_buffer, host_data.data(), kDataLength,
                                  cudaMemcpyHostToDevice),
                       "Memcpy to client_buffer");

        auto batch_id = client_engine->allocateBatchID(1);
        TransferRequest entry;
        entry.opcode = TransferRequest::WRITE;
        entry.length = kDataLength;
        entry.source = client_buffer;
        entry.target_id = segment_id;
        entry.target_offset = (uint64_t)server_buffer;
        Status s = client_engine->submitTransfer(batch_id, {entry});
        ASSERT_TRUE(s.ok());

        // Wait for completion
        TransferStatus status;
        do {
            s = client_engine->getTransferStatus(batch_id, 0, status);
            ASSERT_TRUE(s.ok());
        } while (status.s == TransferStatusEnum::WAITING);

        ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
        s = client_engine->freeBatchID(batch_id);
        ASSERT_TRUE(s.ok());
    }

    // Read: server -> client
    {
        auto batch_id = client_engine->allocateBatchID(1);
        TransferRequest entry;
        entry.opcode = TransferRequest::READ;
        entry.length = kDataLength;
        entry.source = (char*)client_buffer + kDataLength;
        entry.target_id = segment_id;
        entry.target_offset = (uint64_t)server_buffer;
        Status s = client_engine->submitTransfer(batch_id, {entry});
        ASSERT_TRUE(s.ok());

        // Wait for completion
        TransferStatus status;
        do {
            s = client_engine->getTransferStatus(batch_id, 0, status);
            ASSERT_TRUE(s.ok());
        } while (status.s == TransferStatusEnum::WAITING);

        ASSERT_EQ(status.s, TransferStatusEnum::COMPLETED);
        s = client_engine->freeBatchID(batch_id);
        ASSERT_TRUE(s.ok());
    }

    // Check data
    std::vector<char> host_check(kDataLength);
    checkCudaError(
        cudaMemcpy(host_check.data(), (char*)client_buffer + kDataLength,
                   kDataLength, cudaMemcpyDeviceToHost),
        "Memcpy from client_buffer");
    for (size_t i = 0; i < kDataLength; ++i) {
        ASSERT_EQ(host_check[i], 'A');
    }

    // Cleanup
    client_engine->unregisterLocalMemory(client_buffer);
    freeCudaBuffer(client_buffer);
    server_engine->unregisterLocalMemory(server_buffer);
    freeCudaBuffer(server_buffer);
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
