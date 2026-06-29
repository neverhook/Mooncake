#include "client_service.h"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <unordered_set>
#include <vector>

#include "replica.h"
#include "transfer_metadata.h"
#include "types.h"

namespace mooncake {
namespace {

class ClientReadSelectionTestAdapter : public Client {
   public:
    using Client::Client;
    using Client::HasVerifiedNvlinkHostNumaBuffer;
    using Client::SelectReadReplicaForTransfer;
};

Replica::Descriptor MemoryReplica(const std::string& endpoint,
                                  const std::string& protocol,
                                  const std::string& memory_kind,
                                  const std::string& scale_up_domain) {
    AllocatedBuffer::Descriptor buffer;
    buffer.size_ = 4096;
    buffer.buffer_address_ = 0x100000;
    buffer.protocol_ = protocol;
    buffer.transport_endpoint_ = endpoint;
    buffer.memory_kind_ = memory_kind;
    buffer.scale_up_domain_id_ = scale_up_domain;
    return Replica::Descriptor{1, MemoryDescriptor{buffer},
                               ReplicaStatus::COMPLETE};
}

TransferMetadata::BufferDesc NvlinkBuffer(uintptr_t addr, uint64_t length,
                                          const std::string& memory_kind,
                                          const std::string& domain) {
    TransferMetadata::BufferDesc buffer;
    buffer.addr = addr;
    buffer.length = length;
#ifdef ENABLE_MULTI_PROTOCOL
    buffer.protocol = "nvlink";
#endif
    buffer.memory_kind = memory_kind;
    buffer.scale_up_domain_id = domain;
    return buffer;
}

}  // namespace

TEST(ClientReadSelectionTest, SelectReadReplicaSetsNvlinkForDeviceRead) {
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote-rdma", "rdma", "", ""),
        MemoryReplica("remote-dual", "nvlink,rdma", "HOST_NUMA",
                      "domain-a")};
    std::unordered_set<std::string> local_endpoints;
    std::vector<Slice> slices = {{reinterpret_cast<void*>(0x1), 4096}};

    auto selected = ClientReadSelectionTestAdapter::SelectReadReplicaForTransfer(
        replicas, local_endpoints, /*destination_is_device=*/true,
        /*enable_nvlink_host_numa=*/true, "domain-a");

    ASSERT_TRUE(selected.has_value());
    const auto& buffer =
        selected->get_memory_descriptor().buffer_descriptor;
    EXPECT_EQ(buffer.transport_endpoint_, "remote-dual");
    EXPECT_EQ(buffer.selected_protocol_, "nvlink");
}

TEST(ClientReadSelectionTest, SelectReadReplicaFallsBackToRdmaForHostRead) {
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote-dual", "nvlink,rdma", "HOST_NUMA",
                      "domain-a")};
    std::unordered_set<std::string> local_endpoints;
    std::vector<Slice> slices = {{reinterpret_cast<void*>(0x1), 4096}};

    auto selected = ClientReadSelectionTestAdapter::SelectReadReplicaForTransfer(
        replicas, local_endpoints, /*destination_is_device=*/false,
        /*enable_nvlink_host_numa=*/true, "domain-a");

    ASSERT_TRUE(selected.has_value());
    const auto& buffer =
        selected->get_memory_descriptor().buffer_descriptor;
    EXPECT_EQ(buffer.transport_endpoint_, "remote-dual");
    EXPECT_EQ(buffer.selected_protocol_, "rdma");
}

TEST(ClientReadSelectionTest, VerifiesCoveringNvlinkHostNumaDescriptor) {
    TransferMetadata::SegmentDesc desc;
    desc.buffers.push_back(
        NvlinkBuffer(0x100000, 0x4000, "HOST_NUMA", "domain-a"));

    EXPECT_TRUE(ClientReadSelectionTestAdapter::HasVerifiedNvlinkHostNumaBuffer(
        desc, reinterpret_cast<void*>(0x101000), 0x1000, "domain-a"));
}

TEST(ClientReadSelectionTest, RejectsNonHostNumaOrMismatchedDescriptor) {
    TransferMetadata::SegmentDesc desc;
    desc.buffers.push_back(NvlinkBuffer(0x100000, 0x4000, "", "domain-a"));
    desc.buffers.push_back(
        NvlinkBuffer(0x200000, 0x1000, "HOST_NUMA", "domain-a"));
    desc.buffers.push_back(
        NvlinkBuffer(0x300000, 0x4000, "HOST_NUMA", "domain-b"));

    EXPECT_FALSE(ClientReadSelectionTestAdapter::HasVerifiedNvlinkHostNumaBuffer(
        desc, reinterpret_cast<void*>(0x101000), 0x1000, "domain-a"));
    EXPECT_FALSE(ClientReadSelectionTestAdapter::HasVerifiedNvlinkHostNumaBuffer(
        desc, reinterpret_cast<void*>(0x201000), 0x1000, "domain-a"));
    EXPECT_FALSE(ClientReadSelectionTestAdapter::HasVerifiedNvlinkHostNumaBuffer(
        desc, reinterpret_cast<void*>(0x301000), 0x1000, "domain-a"));
}

}  // namespace mooncake
