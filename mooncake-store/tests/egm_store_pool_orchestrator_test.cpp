#include "egm_store_pool_orchestrator.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "egm_store_pool.h"

namespace mooncake {
namespace {

class FakeAllocation final : public EgmStorePoolAllocation {
   public:
    FakeAllocation(void* base, size_t length) : base_(base), length_(length) {}

    void* base() const override { return base_; }
    size_t length() const override { return length_; }

   private:
    void* base_;
    size_t length_;
};

class FakeOperations final : public EgmStorePoolOperations {
   public:
    tl::expected<std::unique_ptr<EgmStorePoolAllocation>, ErrorCode> Allocate(
        const EgmStorePoolAllocationRequest& request) override {
        ++allocate_calls;
        if (fail_allocate_call == allocate_calls) {
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        const uintptr_t address = 0x10000000 + request.plan_index * 0x01000000;
        live_bases.insert(reinterpret_cast<void*>(address));
        return std::unique_ptr<EgmStorePoolAllocation>(new FakeAllocation(
            reinterpret_cast<void*>(address), request.length));
    }

    tl::expected<UUID, ErrorCode> MountGlobal(void* base, size_t) override {
        ++mount_calls;
        registered_bases.insert(base);
        if (fail_mount_call == mount_calls) {
            return tl::make_unexpected(ErrorCode::RPC_FAIL);
        }
        UUID id = generate_uuid();
        mounted[id] = base;
        return id;
    }

    tl::expected<void, ErrorCode> UnmountGlobal(
        const UUID& segment_id) override {
        ++unmount_calls;
        if (fail_unmount) {
            return tl::make_unexpected(ErrorCode::RPC_FAIL);
        }
        auto it = mounted.find(segment_id);
        if (it == mounted.end()) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        registered_bases.erase(it->second);
        mounted.erase(it);
        return {};
    }

    tl::expected<void, ErrorCode> UnregisterIfPresent(void* base) override {
        ++unregister_calls;
        registered_bases.erase(base);
        return {};
    }

    tl::expected<void, ErrorCode> Destroy(
        std::unique_ptr<EgmStorePoolAllocation>& allocation) override {
        ++destroy_calls;
        if (fail_destroy_call == destroy_calls) {
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        live_bases.erase(allocation->base());
        allocation.reset();
        return {};
    }

    size_t allocate_calls = 0;
    size_t mount_calls = 0;
    size_t unmount_calls = 0;
    size_t unregister_calls = 0;
    size_t destroy_calls = 0;
    size_t fail_allocate_call = 0;
    size_t fail_mount_call = 0;
    size_t fail_destroy_call = 0;
    bool fail_unmount = false;
    std::set<void*> live_bases;
    std::set<void*> registered_bases;
    std::map<UUID, void*> mounted;
};

EgmStorePoolPlan TwoChunkPlan() {
    EgmStorePoolPlan plan;
    plan.requested_total = 2 * 0x01000000;
    plan.effective_total = plan.requested_total;
    plan.common_alignment = 0x01000000;
    plan.chunks = {
        {.node_id = 0, .chunk_bytes = 0x01000000, .plan_index = 0},
        {.node_id = 1, .chunk_bytes = 0x01000000, .plan_index = 1},
    };
    return plan;
}

TEST(EgmStorePoolOrchestratorTest, PublishesAndCleansInOneTransaction) {
    FakeOperations operations;
    std::vector<EgmStorePoolGlobalRecord> records;

    ASSERT_TRUE(
        SetupEgmStorePoolOrchestration(operations, TwoChunkPlan(), records));
    EXPECT_EQ(records.size(), 2);
    EXPECT_EQ(operations.live_bases.size(), 2);
    EXPECT_EQ(operations.mounted.size(), 2);

    EXPECT_TRUE(CleanupEgmStorePoolOrchestration(operations, records));
    EXPECT_TRUE(records.empty());
    EXPECT_TRUE(operations.live_bases.empty());
    EXPECT_TRUE(operations.registered_bases.empty());
    EXPECT_TRUE(operations.mounted.empty());
}

TEST(EgmStorePoolOrchestratorTest,
     AllocationFailureOccursBeforeAnyPublication) {
    FakeOperations operations;
    operations.fail_allocate_call = 2;
    std::vector<EgmStorePoolGlobalRecord> records;

    auto setup =
        SetupEgmStorePoolOrchestration(operations, TwoChunkPlan(), records);
    ASSERT_FALSE(setup);
    EXPECT_EQ(setup.error().stage, EgmStorePoolOrchestrationStage::kAllocation);
    EXPECT_EQ(operations.mount_calls, 0);
    EXPECT_TRUE(CleanupEgmStorePoolOrchestration(operations, records));
    EXPECT_TRUE(operations.live_bases.empty());
}

TEST(EgmStorePoolOrchestratorTest,
     MountFailureRollsBackPublishedAndCurrentRegistrations) {
    FakeOperations operations;
    operations.fail_mount_call = 2;
    std::vector<EgmStorePoolGlobalRecord> records;

    auto setup =
        SetupEgmStorePoolOrchestration(operations, TwoChunkPlan(), records);
    ASSERT_FALSE(setup);
    EXPECT_EQ(setup.error().stage, EgmStorePoolOrchestrationStage::kMount);

    EXPECT_TRUE(CleanupEgmStorePoolOrchestration(operations, records));
    EXPECT_EQ(operations.unmount_calls, 1);
    EXPECT_EQ(operations.unregister_calls, 1);
    EXPECT_TRUE(operations.live_bases.empty());
    EXPECT_TRUE(operations.registered_bases.empty());
}

TEST(EgmStorePoolOrchestratorTest, FailedUnmountRetainsOwnershipForRetry) {
    FakeOperations operations;
    std::vector<EgmStorePoolGlobalRecord> records;
    ASSERT_TRUE(
        SetupEgmStorePoolOrchestration(operations, TwoChunkPlan(), records));

    operations.fail_unmount = true;
    EXPECT_FALSE(CleanupEgmStorePoolOrchestration(operations, records));
    EXPECT_EQ(records.size(), 2);
    EXPECT_EQ(operations.destroy_calls, 0);

    operations.fail_unmount = false;
    EXPECT_TRUE(CleanupEgmStorePoolOrchestration(operations, records));
    EXPECT_TRUE(records.empty());
}

TEST(EgmStorePoolOrchestratorTest,
     PartialDestroyFailureRetriesOnlyRemainingOwnership) {
    FakeOperations operations;
    std::vector<EgmStorePoolGlobalRecord> records;
    ASSERT_TRUE(
        SetupEgmStorePoolOrchestration(operations, TwoChunkPlan(), records));

    operations.fail_destroy_call = 2;
    EXPECT_FALSE(CleanupEgmStorePoolOrchestration(operations, records));
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(operations.destroy_calls, 2);
    EXPECT_EQ(operations.live_bases.size(), 1);

    EXPECT_TRUE(CleanupEgmStorePoolOrchestration(operations, records));
    EXPECT_EQ(operations.destroy_calls, 3);
    EXPECT_TRUE(records.empty());
    EXPECT_TRUE(operations.live_bases.empty());
}

}  // namespace
}  // namespace mooncake
