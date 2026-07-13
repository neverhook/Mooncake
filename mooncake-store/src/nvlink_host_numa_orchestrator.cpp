#include "nvlink_host_numa_orchestrator.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <utility>

#include "nvlink_host_numa.h"

namespace mooncake {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t ElapsedMicros(Clock::time_point start) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                             Clock::now() - start)
                             .count();
    return static_cast<uint64_t>(std::max<int64_t>(elapsed, 0));
}

tl::unexpected<NvlinkHostNumaOrchestrationFailure> Failure(
    ErrorCode error, NvlinkHostNumaOrchestrationStage stage) {
    return tl::make_unexpected(
        NvlinkHostNumaOrchestrationFailure{.error = error, .stage = stage});
}

}  // namespace

tl::expected<void, NvlinkHostNumaOrchestrationFailure>
SetupNvlinkHostNumaOrchestration(
    NvlinkHostNumaOperations& operations, const NvlinkHostNumaPlan& plan,
    int local_numa_node, size_t local_buffer_size,
    std::vector<NvlinkHostNumaGlobalRecord>& global_records,
    std::optional<NvlinkHostNumaLocalRecord>& local_record,
    bool& allocator_installed, const NvlinkHostNumaStageObserver& observer) {
    if (!global_records.empty() || local_record || allocator_installed) {
        LOG(ERROR) << "NVLink HOST_NUMA orchestration refused non-empty state";
        return Failure(ErrorCode::INVALID_PARAMS,
                       NvlinkHostNumaOrchestrationStage::kAllocation);
    }

    auto observe = [&](NvlinkHostNumaOrchestrationStage stage,
                       Clock::time_point start, bool success) {
        if (observer) observer(stage, ElapsedMicros(start), success);
    };

    // Complete all allocations before the first allocator/TE/Master
    // publication. Records take ownership immediately so every later failure
    // is handled by the same explicit destroy path.
    auto stage_start = Clock::now();
    global_records.reserve(plan.chunks.size());
    if (local_buffer_size > 0) {
        NvlinkHostNumaAllocationRequest request;
        request.role = NvlinkHostNumaAllocationRole::kLocal;
        request.numa_node = local_numa_node;
        request.requested_length = local_buffer_size;
        request.fabric_exportable = false;

        auto allocation = operations.Allocate(request);
        if (!allocation) {
            LOG(ERROR) << "NVLink HOST_NUMA local allocation failed";
            observe(NvlinkHostNumaOrchestrationStage::kAllocation, stage_start,
                    false);
            return Failure(allocation.error(),
                           NvlinkHostNumaOrchestrationStage::kAllocation);
        }
        NvlinkHostNumaLocalRecord record;
        record.allocation = std::move(*allocation);
        local_record.emplace(std::move(record));
        if (!local_record->allocation ||
            local_record->allocation->base() == nullptr ||
            local_record->allocation->length() < local_buffer_size) {
            LOG(ERROR) << "NVLink HOST_NUMA local allocation violated plan";
            observe(NvlinkHostNumaOrchestrationStage::kAllocation, stage_start,
                    false);
            return Failure(ErrorCode::INVALID_PARAMS,
                           NvlinkHostNumaOrchestrationStage::kAllocation);
        }
    }

    for (const auto& chunk : plan.chunks) {
        NvlinkHostNumaAllocationRequest request;
        request.role = NvlinkHostNumaAllocationRole::kGlobal;
        request.numa_node = chunk.node_id;
        request.requested_length = chunk.chunk_bytes;
        request.fabric_exportable = true;
        request.required_va_alignment = plan.common_alignment;
        request.plan_index = chunk.plan_index;

        auto allocation = operations.Allocate(request);
        if (!allocation) {
            LOG(ERROR) << "NVLink HOST_NUMA global allocation failed for node "
                       << chunk.node_id << ", chunk=" << chunk.plan_index;
            observe(NvlinkHostNumaOrchestrationStage::kAllocation, stage_start,
                    false);
            return Failure(allocation.error(),
                           NvlinkHostNumaOrchestrationStage::kAllocation);
        }

        NvlinkHostNumaGlobalRecord record;
        record.allocation = std::move(*allocation);
        record.numa_node = chunk.node_id;
        record.plan_index = chunk.plan_index;
        global_records.push_back(std::move(record));
        auto& stored = global_records.back();

        const bool invalid =
            !stored.allocation || stored.allocation->base() == nullptr ||
            stored.allocation->length() != chunk.chunk_bytes ||
            plan.common_alignment == 0 ||
            reinterpret_cast<uintptr_t>(stored.allocation->base()) %
                    plan.common_alignment !=
                0;
        if (invalid) {
            LOG(ERROR) << "NVLink HOST_NUMA global allocation violated plan "
                          "for node "
                       << chunk.node_id << ", chunk=" << chunk.plan_index;
            observe(NvlinkHostNumaOrchestrationStage::kAllocation, stage_start,
                    false);
            return Failure(ErrorCode::INVALID_PARAMS,
                           NvlinkHostNumaOrchestrationStage::kAllocation);
        }
    }
    observe(NvlinkHostNumaOrchestrationStage::kAllocation, stage_start, true);

    stage_start = Clock::now();
    auto installed = operations.InstallAllocatorView(
        local_record ? local_record->allocation.get() : nullptr,
        local_buffer_size);
    if (!installed) {
        LOG(ERROR) << "NVLink HOST_NUMA allocator view installation failed";
        observe(NvlinkHostNumaOrchestrationStage::kRegistration, stage_start,
                false);
        return Failure(installed.error(),
                       NvlinkHostNumaOrchestrationStage::kRegistration);
    }
    allocator_installed = true;

    if (local_record) {
        local_record->registration_attempted = true;
        auto registered = operations.RegisterLocal(
            local_record->allocation->base(), local_buffer_size, false);
        if (!registered) {
            LOG(ERROR) << "NVLink HOST_NUMA local registration failed";
            observe(NvlinkHostNumaOrchestrationStage::kRegistration,
                    stage_start, false);
            return Failure(registered.error(),
                           NvlinkHostNumaOrchestrationStage::kRegistration);
        }
        local_record->registered = true;
    }
    observe(NvlinkHostNumaOrchestrationStage::kRegistration, stage_start, true);

    stage_start = Clock::now();
    for (auto& record : global_records) {
        record.registration_attempted = true;
        auto mounted = operations.MountGlobal(record.allocation->base(),
                                              record.allocation->length());
        if (!mounted) {
            LOG(ERROR) << "NVLink HOST_NUMA mount failed for node "
                       << record.numa_node << ", chunk=" << record.plan_index
                       << ", bytes=" << record.allocation->length();
            observe(NvlinkHostNumaOrchestrationStage::kMount, stage_start,
                    false);
            return Failure(mounted.error(),
                           NvlinkHostNumaOrchestrationStage::kMount);
        }
        record.mounted_segment_id = *mounted;
    }
    observe(NvlinkHostNumaOrchestrationStage::kMount, stage_start, true);
    return {};
}

bool CleanupNvlinkHostNumaOrchestration(
    NvlinkHostNumaOperations& operations,
    std::vector<NvlinkHostNumaGlobalRecord>& global_records,
    std::optional<NvlinkHostNumaLocalRecord>& local_record,
    bool& allocator_installed) {
    bool publication_cleanup_succeeded = true;
    auto record_cleanup = [&](bool success, const char* operation) {
        if (!success) {
            publication_cleanup_succeeded = false;
            LOG(ERROR) << "NVLink HOST_NUMA " << operation << " cleanup failed";
        }
    };

    // Successful Master publications are reversed strictly by UUID and in
    // reverse plan order. A repeated Provider name is never used as identity.
    for (auto it = global_records.rbegin(); it != global_records.rend(); ++it) {
        if (!it->mounted_segment_id) continue;
        auto unmounted = operations.UnmountGlobal(*it->mounted_segment_id);
        record_cleanup(unmounted.has_value(), "global unmount");
        if (unmounted) {
            it->mounted_segment_id.reset();
            it->registration_attempted = false;
        }
    }

    // MountGlobal owns the first current-chunk compensation attempt. This is
    // the orchestrator's idempotent fallback, including the harmless case in
    // which the inner attempt already removed the TE registration.
    for (auto it = global_records.rbegin(); it != global_records.rend(); ++it) {
        if (it->mounted_segment_id || !it->registration_attempted) continue;
        auto unregistered =
            operations.UnregisterIfPresent(it->allocation->base(), true);
        record_cleanup(unregistered.has_value(),
                       "current registration fallback");
        if (unregistered) it->registration_attempted = false;
    }

    if (local_record && local_record->registration_attempted) {
        auto unregistered = operations.UnregisterIfPresent(
            local_record->allocation->base(), false);
        record_cleanup(unregistered.has_value(), "local unregister");
        if (unregistered) {
            local_record->registration_attempted = false;
            local_record->registered = false;
        }
    }

    if (!publication_cleanup_succeeded) {
        LOG(ERROR) << "NVLink HOST_NUMA publication cleanup is incomplete; "
                      "retaining allocator view and VMM ownership for retry";
        return false;
    }

    // Drop all aliases before destroying any VMM allocation. This ordering is
    // also used when only a local workspace or only global Provider chunks
    // exist.
    if (allocator_installed) {
        operations.ReleaseAllocatorView();
        allocator_installed = false;
    }

    bool destroy_succeeded = true;
    auto destroy = [&](std::unique_ptr<NvlinkHostNumaAllocation>& allocation,
                       const char* role) {
        if (!allocation) return;
        auto result = operations.Destroy(allocation);
        if (!result || allocation) {
            destroy_succeeded = false;
            LOG(ERROR) << "NVLink HOST_NUMA " << role
                       << " allocation destroy failed";
        }
    };

    for (auto it = global_records.rbegin(); it != global_records.rend(); ++it) {
        destroy(it->allocation, "global");
    }
    if (local_record) destroy(local_record->allocation, "local");

    if (!destroy_succeeded) {
        LOG(ERROR) << "NVLink HOST_NUMA allocation destruction is incomplete; "
                      "retaining ownership records for retry";
        return false;
    }

    local_record.reset();
    global_records.clear();
    return true;
}

}  // namespace mooncake
