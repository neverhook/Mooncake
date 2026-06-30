#include "transfer_candidate_selector.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace mooncake {
namespace {

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

Replica::Descriptor DiskReplica() {
    return Replica::Descriptor{2, DiskDescriptor{"/tmp/object", 4096},
                               ReplicaStatus::COMPLETE};
}

TransferCandidateContext GpuSameDomainContext() {
    TransferCandidateContext context;
    context.local_endpoints = {"local:1234"};
    context.enable_nvlink_host_numa = true;
    context.local_scale_up_domain_id = "domain-a";
    context.destination_is_device = true;
    return context;
}

}  // namespace

TEST(TransferCandidateSelectorTest, ChoosesLocalMemoryWithoutSelectedProtocol) {
    auto context = GpuSameDomainContext();
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote:1234", "nvlink,rdma", "HOST_NUMA", "domain-a"),
        MemoryReplica("local:1234", "rdma", "", "")};

    auto candidate = SelectTransferCandidate(replicas, context);

    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(candidate->replica.get_memory_descriptor()
                  .buffer_descriptor.transport_endpoint_,
              "local:1234");
    EXPECT_TRUE(candidate->selected_protocol.empty());
}

TEST(TransferCandidateSelectorTest, ChoosesSameDomainHostNumaNvlink) {
    auto context = GpuSameDomainContext();
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote-a:1234", "rdma", "HOST_NUMA", "domain-a"),
        MemoryReplica("remote-b:1234", "nvlink,rdma", "HOST_NUMA",
                      "domain-a")};

    auto candidate = SelectTransferCandidate(replicas, context);

    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(candidate->replica.get_memory_descriptor()
                  .buffer_descriptor.transport_endpoint_,
              "remote-b:1234");
    EXPECT_EQ(candidate->selected_protocol, "nvlink");
}

TEST(TransferCandidateSelectorTest, FallsBackToRdmaWhenNvlinkConditionsFail) {
    auto context = GpuSameDomainContext();
    context.destination_is_device = false;
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote-a:1234", "nvlink,rdma", "HOST_NUMA",
                      "domain-a"),
        MemoryReplica("remote-b:1234", "rdma", "HOST_NUMA", "domain-b")};

    auto candidate = SelectTransferCandidate(replicas, context);

    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(candidate->replica.get_memory_descriptor()
                  .buffer_descriptor.transport_endpoint_,
              "remote-a:1234");
    EXPECT_EQ(candidate->selected_protocol, "rdma");
}

TEST(TransferCandidateSelectorTest, EmptyDomainDoesNotSelectNvlink) {
    auto context = GpuSameDomainContext();
    context.local_scale_up_domain_id = "";
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote:1234", "nvlink,rdma", "HOST_NUMA", "")};

    auto candidate = SelectTransferCandidate(replicas, context);

    ASSERT_TRUE(candidate.has_value());
    EXPECT_EQ(candidate->replica.get_memory_descriptor()
                  .buffer_descriptor.transport_endpoint_,
              "remote:1234");
    EXPECT_EQ(candidate->selected_protocol, "rdma");
}

TEST(TransferCandidateSelectorTest, ChoosesOrdinaryRemoteNvlinkMemory) {
    auto context = GpuSameDomainContext();
    context.enable_nvlink_host_numa = false;
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote-hbm:1234", "nvlink", "", ""),
        DiskReplica()};

    auto candidate = SelectTransferCandidate(replicas, context);

    ASSERT_TRUE(candidate.has_value());
    ASSERT_TRUE(candidate->replica.is_memory_replica());
    EXPECT_EQ(candidate->replica.get_memory_descriptor()
                  .buffer_descriptor.transport_endpoint_,
              "remote-hbm:1234");
    EXPECT_TRUE(candidate->selected_protocol.empty());
}

TEST(TransferCandidateSelectorTest, UsesDiskFallbackAfterMemoryCandidates) {
    auto context = GpuSameDomainContext();
    std::vector<Replica::Descriptor> replicas = {DiskReplica()};

    auto candidate = SelectTransferCandidate(replicas, context);

    ASSERT_TRUE(candidate.has_value());
    EXPECT_TRUE(candidate->replica.is_disk_replica());
    EXPECT_TRUE(candidate->selected_protocol.empty());
}

TEST(TransferCandidateSelectorTest, DoesNotFallbackToRemoteNvlinkOnlyMemory) {
    auto context = GpuSameDomainContext();
    context.enable_nvlink_host_numa = false;
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplica("remote:1234", "nvlink", "HOST_NUMA", "domain-a"),
        DiskReplica()};

    auto candidate = SelectTransferCandidate(replicas, context);

    ASSERT_TRUE(candidate.has_value());
    EXPECT_TRUE(candidate->replica.is_disk_replica());
    EXPECT_TRUE(candidate->selected_protocol.empty());
}

}  // namespace mooncake
