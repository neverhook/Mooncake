// transfer_task_test.cpp
#include "transfer_task.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace mooncake {
namespace {

class CountingOperationState : public OperationState {
   public:
    CountingOperationState(ErrorCode result, std::atomic<int>* wait_count)
        : configured_result_(result), wait_count_(wait_count) {}

    bool is_completed() override { return false; }

    void wait_for_completion() override {
        wait_count_->fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = configured_result_;
        cv_.notify_all();
    }

    TransferStrategy get_strategy() const override {
        return TransferStrategy::TRANSFER_ENGINE;
    }

   private:
    ErrorCode configured_result_;
    std::atomic<int>* wait_count_;
};

Replica::Descriptor MemoryReplicaForTransferTaskTest(
    const std::string& selected_protocol, uint64_t remote_address,
    uint64_t size, const std::string& protocol = "") {
    AllocatedBuffer::Descriptor buffer;
    buffer.size_ = size;
    buffer.buffer_address_ = remote_address;
    buffer.protocol_ = protocol;
    buffer.transport_endpoint_ = "endpoint";
    buffer.selected_protocol_ = selected_protocol;
    return Replica::Descriptor{1, MemoryDescriptor{buffer},
                               ReplicaStatus::COMPLETE};
}

}  // namespace

// Test fixture for TransferTask tests
// TODO: Currently, this test does not cover TransferSubmitter and
// TransferEngine integration. Will add more tests in the future.
class TransferTaskTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize glog for logging
        google::InitGoogleLogging("TransferTaskTest");
        FLAGS_logtostderr = 1;  // Output logs to stderr
    }

    void TearDown() override {
        // Cleanup glog
        google::ShutdownGoogleLogging();
    }
};

// Test basic MemcpyOperation functionality
TEST_F(TransferTaskTest, MemcpyOperationBasic) {
    const size_t data_size = 1024;
    std::vector<char> src_data(data_size, 'A');
    std::vector<char> dest_data(data_size, 'B');

    // Create memcpy operation
    MemcpyOperation op(dest_data.data(), src_data.data(), data_size);

    // Verify operation parameters
    EXPECT_EQ(op.dest, dest_data.data());
    EXPECT_EQ(op.src, src_data.data());
    EXPECT_EQ(op.size, data_size);

    // Perform memcpy manually to test
    std::memcpy(op.dest, op.src, op.size);

    // Verify data was copied correctly
    EXPECT_EQ(dest_data, src_data);
    for (size_t i = 0; i < data_size; ++i) {
        EXPECT_EQ(dest_data[i], 'A');
    }
}

// Test MemcpyOperationState functionality
TEST_F(TransferTaskTest, MemcpyOperationState) {
    auto state = std::make_shared<MemcpyOperationState>();

    // Initially not completed
    EXPECT_FALSE(state->is_completed());
    EXPECT_EQ(state->get_strategy(), TransferStrategy::LOCAL_MEMCPY);

    // Set completed with success
    state->set_completed(ErrorCode::OK);
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);
}

// Test MemcpyWorkerPool basic functionality
TEST_F(TransferTaskTest, MemcpyWorkerPoolBasic) {
    MemcpyWorkerPool pool;

    const size_t data_size = 512;
    std::vector<char> src_data(data_size, 'X');
    std::vector<char> dest_data(data_size, 'Y');

    auto state = std::make_shared<MemcpyOperationState>();

    // Create memcpy operations
    std::vector<MemcpyOperation> operations;
    operations.emplace_back(dest_data.data(), src_data.data(), data_size);

    // Create and submit task
    MemcpyTask task(std::move(operations), state);
    pool.submitTask(std::move(task));

    // Wait for completion
    state->wait_for_completion();

    // Verify completion and result
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);

    // Verify data was copied correctly
    for (size_t i = 0; i < data_size; ++i) {
        EXPECT_EQ(dest_data[i], 'X');
    }
}

// Test multiple memcpy operations in one task
TEST_F(TransferTaskTest, MemcpyWorkerPoolMultipleOperations) {
    MemcpyWorkerPool pool;

    const size_t num_ops = 3;
    const size_t data_size = 256;

    std::vector<std::vector<char>> src_buffers(num_ops);
    std::vector<std::vector<char>> dest_buffers(num_ops);

    // Initialize source buffers with different patterns
    for (size_t i = 0; i < num_ops; ++i) {
        src_buffers[i].resize(data_size, 'A' + i);
        dest_buffers[i].resize(data_size, 'Z');
    }

    auto state = std::make_shared<MemcpyOperationState>();

    // Create multiple memcpy operations
    std::vector<MemcpyOperation> operations;
    for (size_t i = 0; i < num_ops; ++i) {
        operations.emplace_back(dest_buffers[i].data(), src_buffers[i].data(),
                                data_size);
    }

    // Create and submit task
    MemcpyTask task(std::move(operations), state);
    pool.submitTask(std::move(task));

    // Wait for completion
    state->wait_for_completion();

    // Verify completion and result
    EXPECT_TRUE(state->is_completed());
    EXPECT_EQ(state->get_result(), ErrorCode::OK);

    // Verify all data was copied correctly
    for (size_t i = 0; i < num_ops; ++i) {
        for (size_t j = 0; j < data_size; ++j) {
            EXPECT_EQ(dest_buffers[i][j], 'A' + i);
        }
    }
}

// Test the locality decision used by TransferSubmitter::isLocalTransfer.
// Same-host different-process pairs share an IP but have distinct ports;
// they must NOT be treated as locally addressable, otherwise memcpy in the
// caller process would dereference a virtual address belonging to a peer
// process and segfault.
TEST_F(TransferTaskTest, IsSameProcessEndpoint) {
    // Empty inputs -> not same-process (cannot prove locality).
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint("", ""));
    EXPECT_FALSE(
        TransferSubmitter::isSameProcessEndpoint("", "192.168.1.10:12345"));
    EXPECT_FALSE(
        TransferSubmitter::isSameProcessEndpoint("192.168.1.10:12345", ""));

    // Identical ip:port -> same process.
    EXPECT_TRUE(TransferSubmitter::isSameProcessEndpoint("192.168.1.10:12345",
                                                         "192.168.1.10:12345"));

    // Same host, different port -> different process, NOT local.
    // This is the regression case fixed by this change.
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint(
        "192.168.1.10:12345", "192.168.1.10:12346"));

    // Different hosts -> not local.
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint(
        "192.168.1.10:12345", "192.168.1.11:12345"));

    // Hostname endpoints (non-P2P metadata mode) compare as full strings.
    EXPECT_TRUE(TransferSubmitter::isSameProcessEndpoint("host-a", "host-a"));
    EXPECT_FALSE(TransferSubmitter::isSameProcessEndpoint("host-a", "host-b"));
}

TEST_F(TransferTaskTest, BuildTransferRequestsPreservesSliceVectorShape) {
    std::vector<char> first(16);
    std::vector<char> second(32);
    std::vector<Slice> slices = {{first.data(), first.size()},
                                 {second.data(), second.size()}};

    auto requests = TransferSubmitter::BuildTransferRequests(
        42, 0x100000, slices, TransferRequest::READ, 128);

    ASSERT_EQ(requests.size(), 2);
    EXPECT_EQ(requests[0].opcode, TransferRequest::READ);
    EXPECT_EQ(requests[0].source, first.data());
    EXPECT_EQ(requests[0].target_id, 42);
    EXPECT_EQ(requests[0].target_offset, 0x100000 + 128);
    EXPECT_EQ(requests[0].length, first.size());
    EXPECT_EQ(requests[1].source, second.data());
    EXPECT_EQ(requests[1].target_id, 42);
    EXPECT_EQ(requests[1].target_offset, 0x100000 + 128 + first.size());
    EXPECT_EQ(requests[1].length, second.size());
}

TEST_F(TransferTaskTest, ResolveSelectedProtocolDefaultsDualProtocolToRdma) {
    AllocatedBuffer::Descriptor handle;
    handle.protocol_ = "nvlink,rdma";

    EXPECT_EQ(TransferSubmitter::ResolveSelectedProtocol(handle), "rdma");

    handle.selected_protocol_ = "nvlink";
    EXPECT_EQ(TransferSubmitter::ResolveSelectedProtocol(handle), "nvlink");
}

TEST_F(TransferTaskTest, BuildBatchTransferGroupsSplitsBySelectedProtocol) {
    std::vector<char> rdma_first(16);
    std::vector<char> nvlink(32);
    std::vector<char> rdma_second(8);
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplicaForTransferTaskTest("rdma", 0x100000, rdma_first.size()),
        MemoryReplicaForTransferTaskTest("nvlink", 0x200000, nvlink.size()),
        MemoryReplicaForTransferTaskTest("rdma", 0x300000, rdma_second.size())};
    std::vector<std::vector<Slice>> all_slices = {
        {{rdma_first.data(), rdma_first.size()}},
        {{nvlink.data(), nvlink.size()}},
        {{rdma_second.data(), rdma_second.size()}}};
    std::vector<SegmentHandle> segments = {11, 22, 33};

    auto groups = TransferSubmitter::BuildBatchTransferGroups(
        replicas, all_slices, segments, TransferRequest::READ);

    ASSERT_EQ(groups.size(), 2);
    std::unordered_map<std::string, const TransferRequestGroup*> by_protocol;
    for (const auto& group : groups) {
        by_protocol.emplace(group.selected_protocol, &group);
    }
    ASSERT_TRUE(by_protocol.count("rdma"));
    ASSERT_TRUE(by_protocol.count("nvlink"));
    const auto& rdma_group = *by_protocol.at("rdma");
    const auto& nvlink_group = *by_protocol.at("nvlink");
    ASSERT_EQ(rdma_group.requests.size(), 2);
    ASSERT_EQ(nvlink_group.requests.size(), 1);
    EXPECT_EQ(rdma_group.requests[0].target_id, 11);
    EXPECT_EQ(rdma_group.requests[0].target_offset, 0x100000);
    EXPECT_EQ(rdma_group.requests[1].target_id, 33);
    EXPECT_EQ(rdma_group.requests[1].target_offset, 0x300000);
    EXPECT_EQ(nvlink_group.requests[0].target_id, 22);
    EXPECT_EQ(nvlink_group.requests[0].target_offset, 0x200000);
}

TEST_F(TransferTaskTest, BuildBatchTransferGroupsDefaultsWriteDualProtocolToRdma) {
    std::vector<char> data(16);
    std::vector<Replica::Descriptor> replicas = {
        MemoryReplicaForTransferTaskTest("", 0x100000, data.size(),
                                         "nvlink,rdma")};
    std::vector<std::vector<Slice>> all_slices = {
        {{data.data(), data.size()}}};
    std::vector<SegmentHandle> segments = {11};

    auto groups = TransferSubmitter::BuildBatchTransferGroups(
        replicas, all_slices, segments, TransferRequest::WRITE);

    ASSERT_EQ(groups.size(), 1);
    EXPECT_EQ(groups[0].selected_protocol, "rdma");
    ASSERT_EQ(groups[0].requests.size(), 1);
    EXPECT_EQ(groups[0].requests[0].opcode, TransferRequest::WRITE);
    EXPECT_EQ(groups[0].requests[0].target_id, 11);
    EXPECT_EQ(groups[0].requests[0].target_offset, 0x100000);
}

TEST_F(TransferTaskTest, AggregateTransferFutureWaitsForEveryChild) {
    std::atomic<int> ok_wait_count{0};
    std::atomic<int> failed_wait_count{0};
    std::vector<TransferFuture> futures;
    futures.emplace_back(std::make_shared<CountingOperationState>(
        ErrorCode::OK, &ok_wait_count));
    futures.emplace_back(std::make_shared<CountingOperationState>(
        ErrorCode::TRANSFER_FAIL, &failed_wait_count));

    auto aggregate =
        TransferSubmitter::AggregateTransferFutures(std::move(futures));

    EXPECT_EQ(aggregate.get(), ErrorCode::TRANSFER_FAIL);
    EXPECT_EQ(ok_wait_count.load(), 1);
    EXPECT_EQ(failed_wait_count.load(), 1);
}

TEST_F(TransferTaskTest, DrainSubmittedFuturesWaitsForEveryChild) {
    std::atomic<int> ok_wait_count{0};
    std::atomic<int> failed_wait_count{0};
    std::vector<TransferFuture> futures;
    futures.emplace_back(std::make_shared<CountingOperationState>(
        ErrorCode::OK, &ok_wait_count));
    futures.emplace_back(std::make_shared<CountingOperationState>(
        ErrorCode::TRANSFER_FAIL, &failed_wait_count));

    EXPECT_EQ(TransferSubmitter::DrainSubmittedFutures(futures),
              ErrorCode::TRANSFER_FAIL);
    EXPECT_EQ(ok_wait_count.load(), 1);
    EXPECT_EQ(failed_wait_count.load(), 1);
}

TEST_F(TransferTaskTest, AggregateTransferFutureConcurrentWaitCollectsOnce) {
    std::atomic<int> wait_count{0};
    std::vector<TransferFuture> futures;
    futures.emplace_back(std::make_shared<CountingOperationState>(
        ErrorCode::OK, &wait_count));
    auto aggregate = TransferSubmitter::AggregateTransferFutures(
        std::move(futures));
    ErrorCode first = ErrorCode::INTERNAL_ERROR;
    ErrorCode second = ErrorCode::INTERNAL_ERROR;

    std::thread t1([&] { first = aggregate.get(); });
    std::thread t2([&] { second = aggregate.get(); });
    t1.join();
    t2.join();

    EXPECT_EQ(first, ErrorCode::OK);
    EXPECT_EQ(second, ErrorCode::OK);
    EXPECT_EQ(wait_count.load(), 1);
}

// Test TransferStrategy enum and stream operator
TEST_F(TransferTaskTest, TransferStrategyEnum) {
    // Test enum values
    EXPECT_EQ(static_cast<int>(TransferStrategy::LOCAL_MEMCPY), 0);
    EXPECT_EQ(static_cast<int>(TransferStrategy::TRANSFER_ENGINE), 1);
    EXPECT_EQ(static_cast<int>(TransferStrategy::FILE_READ), 2);
    EXPECT_EQ(static_cast<int>(TransferStrategy::EMPTY), 3);
    EXPECT_EQ(static_cast<int>(TransferStrategy::SPDK_NVMF), 4);

    // Test stream operator
    std::ostringstream oss;
    oss << TransferStrategy::LOCAL_MEMCPY;
    EXPECT_EQ(oss.str(), "LOCAL_MEMCPY");

    oss.str("");
    oss << TransferStrategy::TRANSFER_ENGINE;
    EXPECT_EQ(oss.str(), "TRANSFER_ENGINE");

    oss.str("");
    oss << TransferStrategy::SPDK_NVMF;
    EXPECT_EQ(oss.str(), "SPDK_NVMF");

    oss.str("");
    oss << TransferStrategy::FILE_READ;
    EXPECT_EQ(oss.str(), "FILE_READ");

    oss.str("");
    oss << TransferStrategy::EMPTY;
    EXPECT_EQ(oss.str(), "EMPTY");
}

}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
