#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ylt/util/tl/expected.hpp>

#include "types.h"

namespace mooncake {

template <typename T>
using NvlinkHostNumaResult = tl::expected<T, std::string>;

struct NvlinkHostNumaOptions {
    bool enabled = false;
    bool auto_nodes = true;
    std::vector<int> nodes;
};

struct NvlinkHostNumaNodePlan {
    int node_id = -1;
    size_t allocation_granularity = 0;
    size_t effective_bytes = 0;
};

struct NvlinkHostNumaChunkPlan {
    int node_id = -1;
    size_t chunk_bytes = 0;
    size_t plan_index = 0;
};

struct NvlinkHostNumaPlan {
    size_t requested_total = 0;
    size_t effective_total = 0;
    size_t common_alignment = 0;
    std::vector<NvlinkHostNumaNodePlan> nodes;
    std::vector<NvlinkHostNumaChunkPlan> chunks;
};

// Environment-dependent discovery is kept behind this interface so parsing,
// discovery policy, and local-node selection can run in ordinary CPU tests.
class NvlinkHostNumaEnvironment {
   public:
    virtual ~NvlinkHostNumaEnvironment() = default;

    virtual NvlinkHostNumaResult<std::vector<int>> VisibleCudaDevices()
        const = 0;
    virtual NvlinkHostNumaResult<std::string> PciBdfForCudaDevice(
        int device_id) const = 0;
    virtual NvlinkHostNumaResult<int> ReadPciNumaNode(
        const std::string& pci_bdf) const = 0;
    virtual bool IsNumaNodeOnline(int node_id) const = 0;
    virtual NvlinkHostNumaResult<int> CurrentCpu() const = 0;
    virtual NvlinkHostNumaResult<int> NumaNodeForCpu(int cpu_id) const = 0;
};

// Parse only the ConfigDict-only HOST_NUMA controls. The node expression is
// deliberately ignored when it cannot affect a nonzero global pool.
NvlinkHostNumaResult<NvlinkHostNumaOptions> ParseNvlinkHostNumaOptions(
    const ConfigDict& config, size_t global_segment_size);

// Resolve the sorted global NUMA node set. Returns an empty set when the
// feature is disabled or global_segment_size is zero.
NvlinkHostNumaResult<std::vector<int>> DiscoverNvlinkHostNumaNodes(
    const NvlinkHostNumaOptions& options, size_t global_segment_size,
    const NvlinkHostNumaEnvironment& environment);

// Resolve setup CPU locality for a nonzero HOST_NUMA local workspace.
// Disabled/zero-size configurations return std::nullopt without discovery.
NvlinkHostNumaResult<std::optional<int>> ResolveNvlinkHostNumaLocalNode(
    const NvlinkHostNumaOptions& options, size_t local_buffer_size,
    const NvlinkHostNumaEnvironment& environment);

// Build a deterministic NUMA/chunk plan. node_granularities may arrive in any
// order; the result is sorted by node ID. store_alignment defaults to the
// Cachelib slab size in the implementation so each Store segment is mountable.
NvlinkHostNumaResult<NvlinkHostNumaPlan> PlanNvlinkHostNumaCapacity(
    size_t requested_total,
    const std::vector<std::pair<int, size_t>>& node_granularities,
    size_t max_mr_size, size_t store_alignment = 0);

std::unique_ptr<NvlinkHostNumaEnvironment>
CreateProductionNvlinkHostNumaEnvironment();

}  // namespace mooncake
