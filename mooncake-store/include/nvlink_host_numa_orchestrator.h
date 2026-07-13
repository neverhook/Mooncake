#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "types.h"

namespace mooncake {

struct NvlinkHostNumaPlan;

// Type-erased allocation ownership lets the production CUDA VMM path and the
// ordinary CPU failure tests exercise the exact same orchestration state
// machine. Destroy is deliberately an operation rather than an implicit
// unique_ptr reset so failures retain ownership for an explicit cleanup retry.
class NvlinkHostNumaAllocation {
   public:
    virtual ~NvlinkHostNumaAllocation() = default;

    virtual void* base() const = 0;
    virtual size_t length() const = 0;
    virtual size_t granularity() const = 0;
};

enum class NvlinkHostNumaAllocationRole { kLocal, kGlobal };

struct NvlinkHostNumaAllocationRequest {
    NvlinkHostNumaAllocationRole role = NvlinkHostNumaAllocationRole::kGlobal;
    int numa_node = -1;
    size_t requested_length = 0;
    bool fabric_exportable = false;
    size_t required_va_alignment = 0;
    size_t plan_index = 0;
};

// Internal dependency boundary for the publication transaction. Production
// delegates to NvlinkVmmAllocation and Client; tests use a recording fake.
class NvlinkHostNumaOperations {
   public:
    virtual ~NvlinkHostNumaOperations() = default;

    virtual tl::expected<std::unique_ptr<NvlinkHostNumaAllocation>, ErrorCode>
    Allocate(const NvlinkHostNumaAllocationRequest& request) = 0;

    virtual tl::expected<void, ErrorCode> InstallAllocatorView(
        NvlinkHostNumaAllocation* local_allocation,
        size_t configured_local_length) = 0;
    virtual void ReleaseAllocatorView() = 0;

    virtual tl::expected<void, ErrorCode> RegisterLocal(
        void* base, size_t length, bool remote_accessible) = 0;
    virtual tl::expected<UUID, ErrorCode> MountGlobal(void* base,
                                                      size_t length) = 0;
    virtual tl::expected<void, ErrorCode> UnmountGlobal(
        const UUID& segment_id) = 0;
    virtual tl::expected<void, ErrorCode> UnregisterIfPresent(
        void* base, bool update_metadata) = 0;

    virtual tl::expected<void, ErrorCode> Destroy(
        std::unique_ptr<NvlinkHostNumaAllocation>& allocation) = 0;
};

struct NvlinkHostNumaGlobalRecord {
    std::unique_ptr<NvlinkHostNumaAllocation> allocation;
    int numa_node = -1;
    size_t plan_index = 0;
    bool registration_attempted = false;
    std::optional<UUID> mounted_segment_id;
};

struct NvlinkHostNumaLocalRecord {
    std::unique_ptr<NvlinkHostNumaAllocation> allocation;
    // Set before RegisterLocal so a side-effect-then-fail implementation is
    // still compensated by the idempotent unregister path.
    bool registration_attempted = false;
    bool registered = false;
};

enum class NvlinkHostNumaOrchestrationStage {
    kAllocation,
    kRegistration,
    kMount,
};

struct NvlinkHostNumaOrchestrationFailure {
    ErrorCode error = ErrorCode::INTERNAL_ERROR;
    NvlinkHostNumaOrchestrationStage stage =
        NvlinkHostNumaOrchestrationStage::kAllocation;
};

using NvlinkHostNumaStageObserver =
    std::function<void(NvlinkHostNumaOrchestrationStage stage,
                       uint64_t duration_us, bool success)>;

// Allocate every local/global range before installing the allocator view or
// publishing anything. On failure, ownership remains in the supplied records;
// the caller must run CleanupNvlinkHostNumaOrchestration, as RealClient does
// through its setup rollback guard.
tl::expected<void, NvlinkHostNumaOrchestrationFailure>
SetupNvlinkHostNumaOrchestration(
    NvlinkHostNumaOperations& operations, const NvlinkHostNumaPlan& plan,
    int local_numa_node, size_t local_buffer_size,
    std::vector<NvlinkHostNumaGlobalRecord>& global_records,
    std::optional<NvlinkHostNumaLocalRecord>& local_record,
    bool& allocator_installed,
    const NvlinkHostNumaStageObserver& observer = {});

// Reverse publication first. Allocator views and VMM ownership are released
// only after every unmount/unregister succeeds. Failed cleanup retains the
// exact remaining state and is safe to retry.
bool CleanupNvlinkHostNumaOrchestration(
    NvlinkHostNumaOperations& operations,
    std::vector<NvlinkHostNumaGlobalRecord>& global_records,
    std::optional<NvlinkHostNumaLocalRecord>& local_record,
    bool& allocator_installed);

}  // namespace mooncake
