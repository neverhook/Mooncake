#include "egm_store_pool_orchestrator.h"

#include <glog/logging.h>

#include <cstdint>
#include <utility>

#include "egm_store_pool.h"

namespace mooncake {
namespace {

tl::unexpected<EgmStorePoolOrchestrationFailure> Failure(
    ErrorCode error, EgmStorePoolOrchestrationStage stage) {
    return tl::make_unexpected(
        EgmStorePoolOrchestrationFailure{.error = error, .stage = stage});
}

}  // namespace

tl::expected<void, EgmStorePoolOrchestrationFailure>
SetupEgmStorePoolOrchestration(
    EgmStorePoolOperations& operations, const EgmStorePoolPlan& plan,
    std::vector<EgmStorePoolGlobalRecord>& global_records) {
    if (!global_records.empty()) {
        LOG(ERROR) << "EGM Store Pool orchestration refused non-empty state";
        return Failure(ErrorCode::INVALID_PARAMS,
                       EgmStorePoolOrchestrationStage::kAllocation);
    }

    // Allocate every chunk before publishing the first one. This keeps an
    // allocation failure invisible to Consumers and gives every later failure
    // one uniform rollback path.
    global_records.reserve(plan.chunks.size());
    for (const auto& chunk : plan.chunks) {
        EgmStorePoolAllocationRequest request{
            .numa_node = chunk.node_id,
            .length = chunk.chunk_bytes,
            .required_va_alignment = plan.common_alignment,
            .plan_index = chunk.plan_index,
        };
        auto allocation = operations.Allocate(request);
        if (!allocation) {
            LOG(ERROR) << "EGM Store Pool allocation failed for NUMA node "
                       << chunk.node_id << ", chunk=" << chunk.plan_index;
            return Failure(allocation.error(),
                           EgmStorePoolOrchestrationStage::kAllocation);
        }

        EgmStorePoolGlobalRecord record;
        record.allocation = std::move(*allocation);
        record.numa_node = chunk.node_id;
        record.plan_index = chunk.plan_index;
        global_records.push_back(std::move(record));

        const auto& stored = global_records.back();
        if (!stored.allocation || stored.allocation->base() == nullptr ||
            stored.allocation->length() != chunk.chunk_bytes ||
            plan.common_alignment == 0 ||
            reinterpret_cast<uintptr_t>(stored.allocation->base()) %
                    plan.common_alignment !=
                0) {
            LOG(ERROR) << "EGM Store Pool allocation violated the plan for "
                          "NUMA node "
                       << chunk.node_id << ", chunk=" << chunk.plan_index;
            return Failure(ErrorCode::INVALID_PARAMS,
                           EgmStorePoolOrchestrationStage::kAllocation);
        }
    }

    for (auto& record : global_records) {
        record.registration_attempted = true;
        auto mounted = operations.MountGlobal(record.allocation->base(),
                                              record.allocation->length());
        if (!mounted) {
            LOG(ERROR) << "EGM Store Pool publication failed for NUMA node "
                       << record.numa_node << ", chunk=" << record.plan_index;
            return Failure(mounted.error(),
                           EgmStorePoolOrchestrationStage::kMount);
        }
        record.mounted_segment_id = *mounted;
    }
    return {};
}

bool CleanupEgmStorePoolOrchestration(
    EgmStorePoolOperations& operations,
    std::vector<EgmStorePoolGlobalRecord>& global_records) {
    bool publication_cleanup_succeeded = true;

    for (auto it = global_records.rbegin(); it != global_records.rend(); ++it) {
        if (!it->mounted_segment_id) continue;
        auto unmounted = operations.UnmountGlobal(*it->mounted_segment_id);
        if (!unmounted) {
            publication_cleanup_succeeded = false;
            LOG(ERROR) << "EGM Store Pool unmount failed for NUMA node "
                       << it->numa_node << ", chunk=" << it->plan_index;
            continue;
        }
        it->mounted_segment_id.reset();
        it->registration_attempted = false;
    }

    // A failed MountGlobal can leave a TE registration without a Master
    // segment. The operation is idempotent so it also covers the case where
    // MountGlobal already compensated the registration internally.
    for (auto it = global_records.rbegin(); it != global_records.rend(); ++it) {
        if (it->mounted_segment_id || !it->registration_attempted) continue;
        auto unregistered =
            operations.UnregisterIfPresent(it->allocation->base());
        if (!unregistered) {
            publication_cleanup_succeeded = false;
            LOG(ERROR) << "EGM Store Pool registration rollback failed for "
                          "NUMA node "
                       << it->numa_node << ", chunk=" << it->plan_index;
            continue;
        }
        it->registration_attempted = false;
    }

    if (!publication_cleanup_succeeded) {
        LOG(ERROR) << "EGM Store Pool cleanup retained VMM ownership for retry";
        return false;
    }

    bool destroy_succeeded = true;
    for (auto it = global_records.rbegin(); it != global_records.rend(); ++it) {
        // A previous cleanup attempt may already have released this owner
        // before a later release failed. Keep successful progress and retry
        // only the allocations that still exist.
        if (!it->allocation) continue;
        auto destroyed = operations.Destroy(it->allocation);
        if (!destroyed || it->allocation) {
            destroy_succeeded = false;
            LOG(ERROR) << "EGM Store Pool VMM release failed for NUMA node "
                       << it->numa_node << ", chunk=" << it->plan_index;
        }
    }
    if (!destroy_succeeded) return false;

    global_records.clear();
    return true;
}

void AbandonEgmStorePoolOwnership(
    std::vector<EgmStorePoolGlobalRecord>& global_records) noexcept {
    for (auto& record : global_records) {
        (void)record.allocation.release();
    }
    global_records.clear();
}

}  // namespace mooncake
