#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "types.h"

namespace mooncake {

struct EgmStorePoolPlan;

class EgmStorePoolAllocation {
   public:
    virtual ~EgmStorePoolAllocation() = default;

    virtual void* base() const = 0;
    virtual size_t length() const = 0;
};

struct EgmStorePoolAllocationRequest {
    int numa_node = -1;
    size_t length = 0;
    size_t required_va_alignment = 0;
    size_t plan_index = 0;
};

// Internal dependency boundary for the global-pool publication transaction.
// Production delegates to NvlinkVmmAllocation and Client; unit tests use a
// recording fake without adding test hooks to either production class.
class EgmStorePoolOperations {
   public:
    virtual ~EgmStorePoolOperations() = default;

    virtual tl::expected<std::unique_ptr<EgmStorePoolAllocation>, ErrorCode>
    Allocate(const EgmStorePoolAllocationRequest& request) = 0;
    virtual tl::expected<UUID, ErrorCode> MountGlobal(void* base,
                                                      size_t length) = 0;
    virtual tl::expected<void, ErrorCode> UnmountGlobal(
        const UUID& segment_id) = 0;
    virtual tl::expected<void, ErrorCode> UnregisterIfPresent(void* base) = 0;
    virtual tl::expected<void, ErrorCode> Destroy(
        std::unique_ptr<EgmStorePoolAllocation>& allocation) = 0;
};

struct EgmStorePoolGlobalRecord {
    std::unique_ptr<EgmStorePoolAllocation> allocation;
    int numa_node = -1;
    size_t plan_index = 0;
    // Set before MountGlobal because TE registration may have completed even
    // when the later Master publication fails.
    bool registration_attempted = false;
    std::optional<UUID> mounted_segment_id;
};

enum class EgmStorePoolOrchestrationStage { kAllocation, kMount };

struct EgmStorePoolOrchestrationFailure {
    ErrorCode error = ErrorCode::INTERNAL_ERROR;
    EgmStorePoolOrchestrationStage stage =
        EgmStorePoolOrchestrationStage::kAllocation;
};

// All VMM allocations are completed before the first TE/Master publication.
// On failure, ownership remains in global_records for the caller to clean up.
tl::expected<void, EgmStorePoolOrchestrationFailure>
SetupEgmStorePoolOrchestration(
    EgmStorePoolOperations& operations, const EgmStorePoolPlan& plan,
    std::vector<EgmStorePoolGlobalRecord>& global_records);

// Reverse publication before releasing VMM ownership. Failed cleanup retains
// the exact remaining state and can be retried by the same RealClient.
bool CleanupEgmStorePoolOrchestration(
    EgmStorePoolOperations& operations,
    std::vector<EgmStorePoolGlobalRecord>& global_records);

// Last-resort destructor path. The VMM owners are intentionally leaked so a
// stale published descriptor cannot point at a recycled virtual address.
void AbandonEgmStorePoolOwnership(
    std::vector<EgmStorePoolGlobalRecord>& global_records) noexcept;

}  // namespace mooncake
