// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "real_client.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "client_buffer.h"
#include "config.h"
#include "egm_store_pool.h"
#include "memory_location.h"
#include "transport/nvlink_transport/nvlink_vmm_allocation.h"

namespace mooncake {

namespace {

class ProductionEgmStorePoolAllocation final : public EgmStorePoolAllocation {
   public:
    explicit ProductionEgmStorePoolAllocation(
        std::unique_ptr<NvlinkVmmAllocation> owner)
        : owner_(std::move(owner)) {}

    void *base() const override { return owner_->base(); }
    size_t length() const override { return owner_->length(); }
    size_t granularity() const override { return owner_->granularity(); }
    Status Release() { return owner_->Release(); }

   private:
    std::unique_ptr<NvlinkVmmAllocation> owner_;
};

}  // namespace

class ProductionEgmStorePoolOperations final : public EgmStorePoolOperations {
   public:
    ProductionEgmStorePoolOperations(
        RealClient &owner,
        std::function<std::optional<ErrorCode>(const Segment &)> master_failure)
        : owner_(owner), master_failure_(std::move(master_failure)) {}

    tl::expected<std::unique_ptr<EgmStorePoolAllocation>, ErrorCode> Allocate(
        const EgmStorePoolAllocationRequest &request) override {
        NvlinkVmmAllocation::Options options;
        options.location_type = NvlinkVmmAllocation::LocationType::HOST_NUMA;
        options.location_id = request.numa_node;
        options.requested_length = request.requested_length;
        options.fabric_exportable = request.fabric_exportable;
        options.required_va_alignment = request.required_va_alignment;
        options.access_observer = [this](uint64_t duration_us, bool success) {
            if (owner_.client_) {
                owner_.client_->ObserveEgmStorePoolStage(
                    EgmStorePoolStage::kAccess, duration_us, success);
            }
        };

        std::unique_ptr<NvlinkVmmAllocation> allocation;
        Status status = NvlinkVmmAllocation::Create(options, allocation);
        if (!status.ok() || allocation == nullptr) {
            LOG(ERROR) << "EGM Store Pool "
                       << (request.role == EgmStorePoolAllocationRole::kLocal
                               ? "local"
                               : "global")
                       << " allocation failed for node " << request.numa_node
                       << ", chunk=" << request.plan_index << ": " << status;
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        return std::unique_ptr<EgmStorePoolAllocation>(
            new ProductionEgmStorePoolAllocation(std::move(allocation)));
    }

    tl::expected<void, ErrorCode> InstallAllocatorView(
        EgmStorePoolAllocation *local_allocation,
        size_t configured_local_length) override {
        try {
            std::shared_ptr<ClientBufferAllocator> allocator;
            if (local_allocation != nullptr) {
                allocator = ClientBufferAllocator::create(
                    local_allocation->base(), configured_local_length,
                    "nvlink");
            } else {
                allocator = ClientBufferAllocator::create(size_t{0}, "nvlink");
            }
            owner_.publish_client_buffer_allocator(std::move(allocator));
        } catch (const std::exception &error) {
            LOG(ERROR) << "EGM Store Pool allocator view creation failed: "
                       << error.what();
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        return {};
    }

    tl::expected<void, ErrorCode> ReleaseAllocatorView() override {
        return owner_.release_egm_store_pool_allocator_view();
    }

    tl::expected<void, ErrorCode> RegisterLocal(
        void *base, size_t length, bool remote_accessible) override {
        if (!owner_.client_) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        auto registered = owner_.client_->RegisterLocalMemory(
            base, length, kWildcardLocation, remote_accessible, false);
        if (!registered) return registered;

        std::unique_lock<std::shared_mutex> lock(
            owner_.registered_buffer_mutex_);
        owner_.local_buffer_region_ = RealClient::WritableBufferRegion{
            .base = base,
            .size = length,
            .offset = 0,
        };
        return {};
    }

    tl::expected<UUID, ErrorCode> MountGlobal(void *base,
                                              size_t length) override {
        if (!owner_.client_) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        return owner_.client_->MountSegmentAndGetIdImpl(
            base, length, "nvlink", kWildcardLocation, master_failure_);
    }

    tl::expected<void, ErrorCode> UnmountGlobal(
        const UUID &segment_id) override {
        if (!owner_.client_) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        return owner_.client_->UnmountSegmentById(segment_id);
    }

    tl::expected<void, ErrorCode> UnregisterIfPresent(
        void *base, bool update_metadata) override {
        if (!owner_.client_) {
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        return owner_.client_->UnregisterLocalMemoryIfPresent(base,
                                                              update_metadata);
    }

    tl::expected<void, ErrorCode> Destroy(
        std::unique_ptr<EgmStorePoolAllocation> &allocation) override {
        auto *production_allocation =
            dynamic_cast<ProductionEgmStorePoolAllocation *>(allocation.get());
        if (production_allocation == nullptr) {
            LOG(ERROR) << "EGM Store Pool destroy received an unknown "
                          "allocation implementation";
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        try {
            Status status = production_allocation->Release();
            if (!status.ok()) {
                LOG(ERROR) << "EGM Store Pool VMM release failed; retaining "
                              "ownership for cleanup retry: "
                           << status;
                return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
            }
        } catch (const std::exception &error) {
            LOG(ERROR) << "EGM Store Pool VMM release threw; retaining "
                          "ownership for cleanup retry: "
                       << error.what();
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        } catch (...) {
            LOG(ERROR) << "EGM Store Pool VMM release threw an unknown "
                          "exception; retaining ownership for cleanup retry";
            return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
        }
        allocation.reset();
        return {};
    }

   private:
    RealClient &owner_;
    std::function<std::optional<ErrorCode>(const Segment &)> master_failure_;
};

namespace {

EgmStorePoolStage ToMetricStage(EgmStorePoolOrchestrationStage stage) {
    switch (stage) {
        case EgmStorePoolOrchestrationStage::kAllocation:
            return EgmStorePoolStage::kAllocation;
        case EgmStorePoolOrchestrationStage::kRegistration:
            return EgmStorePoolStage::kRegistration;
        case EgmStorePoolOrchestrationStage::kMount:
            return EgmStorePoolStage::kMount;
    }
    return EgmStorePoolStage::kAllocation;
}

}  // namespace

void RealClient::prepare_egm_store_pool_runtime() {
    // Reserve the destructor's fail-closed ownership handoff before any CUDA
    // resource exists. EGM Store Pool setup refuses to start if this allocation
    // was unavailable, so teardown never depends on allocating under failure.
    egm_store_pool_quarantine_node_ =
        PrepareEgmStorePoolProcessQuarantineNode();
}

void RealClient::quarantine_egm_store_pool_on_destroy(bool teardown_succeeded) {
    // Include partial setup state even if the feature flag was cleared.
    const bool retained_state = egm_store_pool_enabled_ ||
                                !egm_store_pool_globals_.empty() ||
                                egm_store_pool_local_.has_value() ||
                                egm_store_pool_allocator_installed_;
    if (teardown_succeeded || !retained_state) return;

    // Cleanup could not prove that every published descriptor was removed.
    // Deliberately quarantine the VMM mappings until process exit instead of
    // letting member destruction recycle their virtual addresses while a stale
    // Master/TE descriptor may still exist.
    const auto ownership = GetEgmStorePoolOwnershipStats(
        egm_store_pool_globals_, egm_store_pool_local_);
    if (QuarantineEgmStorePoolOwnership(
            std::move(egm_store_pool_quarantine_node_), egm_store_pool_globals_,
            egm_store_pool_local_)) {
        const auto process_stats = GetEgmStorePoolProcessQuarantineStats();
        if (client_) {
            client_->SetEgmStorePoolProcessQuarantine(process_stats.clients,
                                                      process_stats.allocations,
                                                      process_stats.bytes);
        }
        LOG(ERROR)
            << "EGM Store Pool cleanup remained incomplete during "
               "destruction; quarantined allocations="
            << ownership.allocations << " bytes=" << ownership.bytes
            << ". This process is unhealthy and refuses new EGM Store Pool "
               "setup; restart is required";
        return;
    }

    LOG(ERROR) << "EGM Store Pool process quarantine node is missing; "
                  "terminating rather than recycling a potentially published "
                  "VMM address";
    std::terminate();
}

tl::expected<void, ErrorCode> RealClient::setup_egm_store_pool(
    const EgmStorePoolOptions &options, size_t global_segment_size,
    size_t local_buffer_size,
    const EgmStorePoolRuntimeDependencies &dependencies) {
    auto observe_stage = [this](EgmStorePoolStage stage,
                                std::chrono::steady_clock::time_point start,
                                bool success) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        client_->ObserveEgmStorePoolStage(
            stage, static_cast<uint64_t>(std::max<int64_t>(elapsed, 0)),
            success);
    };

    const auto process_quarantine = GetEgmStorePoolProcessQuarantineStats();
    if (client_) {
        client_->SetEgmStorePoolProcessQuarantine(
            process_quarantine.clients, process_quarantine.allocations,
            process_quarantine.bytes);
    }
    if (process_quarantine.clients != 0) {
        LOG(ERROR)
            << "EGM Store Pool setup refused because a destroyed "
               "client left process-quarantined VMM ownership: clients="
            << process_quarantine.clients
            << " allocations=" << process_quarantine.allocations
            << " bytes=" << process_quarantine.bytes
            << "; restart the process before enabling EGM Store Pool again";
        observe_stage(EgmStorePoolStage::kPreflight,
                      std::chrono::steady_clock::now(), false);
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }
    if (!egm_store_pool_quarantine_node_) {
        LOG(ERROR) << "EGM Store Pool setup cannot reserve its fail-closed "
                      "process quarantine node";
        observe_stage(EgmStorePoolStage::kPreflight,
                      std::chrono::steady_clock::now(), false);
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    auto stage_start = std::chrono::steady_clock::now();
    if (client_->IsUsingTent()) {
        LOG(ERROR) << "EGM Store Pool V1 requires legacy NvlinkTransport; "
                      "TENT is not supported";
        observe_stage(EgmStorePoolStage::kPreflight, stage_start, false);
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (!client_->IsNvlinkFabricTransportReady()) {
        LOG(ERROR) << "EGM Store Pool V1 requires NvlinkTransport as the only "
                      "installed transport with Fabric memory enabled; check "
                      "MC_MS_AUTO_DISC/MC_FORCE_MNNVL and MC_USE_NVLINK_IPC";
        observe_stage(EgmStorePoolStage::kPreflight, stage_start, false);
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    Status capability = NvlinkVmmAllocation::CheckStrictFabricCapability();
    if (!capability.ok()) {
        LOG(ERROR) << "EGM Store Pool Fabric preflight failed: " << capability;
        observe_stage(EgmStorePoolStage::kPreflight, stage_start, false);
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }
    observe_stage(EgmStorePoolStage::kPreflight, stage_start, true);

    auto environment = CreateProductionEgmStorePoolEnvironment();
    stage_start = std::chrono::steady_clock::now();
    auto global_nodes =
        DiscoverEgmStorePoolNodes(options, global_segment_size, *environment);
    if (!global_nodes) {
        LOG(ERROR) << "EGM Store Pool node discovery failed: "
                   << global_nodes.error();
        observe_stage(EgmStorePoolStage::kDiscovery, stage_start, false);
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    std::vector<std::pair<int, size_t>> node_granularities;
    node_granularities.reserve(global_nodes->size());
    for (int node : *global_nodes) {
        size_t granularity = 0;
        Status status = NvlinkVmmAllocation::GetAllocationGranularity(
            NvlinkVmmAllocation::LocationType::HOST_NUMA, node, true,
            granularity);
        if (!status.ok()) {
            LOG(ERROR) << "EGM Store Pool granularity query failed for node "
                       << node << ": " << status;
            observe_stage(EgmStorePoolStage::kDiscovery, stage_start, false);
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        node_granularities.emplace_back(node, granularity);
    }
    observe_stage(EgmStorePoolStage::kDiscovery, stage_start, true);

    EgmStorePoolPlan plan;
    stage_start = std::chrono::steady_clock::now();
    if (global_segment_size > 0) {
        auto planned =
            PlanEgmStorePoolCapacity(global_segment_size, node_granularities,
                                     globalConfig().max_mr_size);
        if (!planned) {
            LOG(ERROR) << "EGM Store Pool capacity planning failed: "
                       << planned.error();
            observe_stage(EgmStorePoolStage::kPlanning, stage_start, false);
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        plan = std::move(*planned);
    }
    observe_stage(EgmStorePoolStage::kPlanning, stage_start, true);

    int local_numa_node = -1;
    int setup_cpu = -1;
    stage_start = std::chrono::steady_clock::now();
    if (local_buffer_size > 0) {
        auto cpu = environment->CurrentCpu();
        if (!cpu) {
            LOG(ERROR) << "EGM Store Pool setup CPU resolution failed: "
                       << cpu.error();
            observe_stage(EgmStorePoolStage::kDiscovery, stage_start, false);
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        setup_cpu = *cpu;
        auto node = environment->NumaNodeForCpu(setup_cpu);
        if (!node || *node < 0 || !environment->IsNumaNodeOnline(*node)) {
            LOG(ERROR) << "EGM Store Pool local node resolution failed for "
                          "setup CPU "
                       << setup_cpu
                       << (node ? ": invalid/offline node " +
                                      std::to_string(*node)
                                : ": " + node.error());
            observe_stage(EgmStorePoolStage::kDiscovery, stage_start, false);
            return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
        }
        local_numa_node = *node;
    }
    observe_stage(EgmStorePoolStage::kDiscovery, stage_start, true);

    int current_cuda_device = -1;
#ifdef USE_CUDA
    if (cudaGetDevice(&current_cuda_device) != cudaSuccess) {
        current_cuda_device = -1;
    }
#endif
    LOG(INFO) << "EGM Store Pool locality setup_cpu=" << setup_cpu
              << " local_numa=" << local_numa_node
              << " current_cuda_device=" << current_cuda_device;

    client_->ObserveEgmStorePoolCapacity(global_segment_size,
                                         plan.effective_total);
    for (const auto &node : plan.nodes) {
        const uint64_t chunk_count = static_cast<uint64_t>(std::count_if(
            plan.chunks.begin(), plan.chunks.end(),
            [&](const auto &chunk) { return chunk.node_id == node.node_id; }));
        client_->ObserveEgmStorePoolNode(
            node.node_id, static_cast<uint64_t>(node.effective_bytes),
            chunk_count);
    }

    ProductionEgmStorePoolOperations operations(
        *this, dependencies.master_mount_failure);
    auto orchestrated = SetupEgmStorePoolOrchestration(
        operations, plan, local_numa_node, local_buffer_size,
        egm_store_pool_globals_, egm_store_pool_local_,
        egm_store_pool_allocator_installed_,
        [this](EgmStorePoolOrchestrationStage stage, uint64_t duration_us,
               bool success) {
            client_->ObserveEgmStorePoolStage(ToMetricStage(stage), duration_us,
                                              success);
        });
    if (!orchestrated) {
        return tl::make_unexpected(orchestrated.error().error);
    }

    LOG(INFO) << "EGM Store Pool publication complete: requested="
              << global_segment_size << " effective=" << plan.effective_total
              << " chunks=" << plan.chunks.size() << " setup_cpu=" << setup_cpu
              << " local_numa=" << local_numa_node;
    return {};
}

bool RealClient::cleanup_egm_store_pool(bool rollback) {
    const auto cleanup_start = std::chrono::steady_clock::now();
    ProductionEgmStorePoolOperations operations(*this, {});
    const bool success = CleanupEgmStorePoolOrchestration(
        operations, egm_store_pool_globals_, egm_store_pool_local_,
        egm_store_pool_allocator_installed_);
    const auto pending = GetEgmStorePoolOwnershipStats(egm_store_pool_globals_,
                                                       egm_store_pool_local_);
    egm_store_pool_cleanup_pending_.store(!success, std::memory_order_release);
    if (client_) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - cleanup_start)
                .count();
        client_->ObserveEgmStorePoolStage(
            rollback ? EgmStorePoolStage::kRollback
                     : EgmStorePoolStage::kTeardown,
            static_cast<uint64_t>(std::max<int64_t>(elapsed, 0)), success);
        if (rollback) {
            client_->ObserveEgmStorePoolRollback(success);
        }
        client_->ObserveEgmStorePoolCleanup(success, pending.allocations,
                                            pending.bytes);
    }
    LOG(INFO) << "EGM Store Pool " << (rollback ? "rollback" : "teardown")
              << " completed, success=" << success;
    egm_store_pool_enabled_ = !success;
    return success;
}

void RealClient::publish_client_buffer_allocator(
    std::shared_ptr<ClientBufferAllocator> allocator) {
    std::atomic_store_explicit(&client_buffer_allocator_, std::move(allocator),
                               std::memory_order_release);
}

std::optional<BufferHandle> RealClient::allocate_client_buffer(size_t size) {
    auto allocator = SnapshotClientBufferAllocator();
    if (!allocator) return std::nullopt;
    // The snapshot remains alive through allocate(), and a successful handle
    // takes its own shared_ptr lease before the local snapshot is released.
    return allocator->allocate(size);
}

tl::expected<void, ErrorCode> RealClient::release_egm_store_pool_allocator_view(
    const std::function<void()> &after_exchange_for_test) {
    auto owner = std::atomic_exchange_explicit(
        &client_buffer_allocator_, std::shared_ptr<ClientBufferAllocator>{},
        std::memory_order_acq_rel);

    if (after_exchange_for_test) {
        try {
            after_exchange_for_test();
        } catch (...) {
            std::atomic_store_explicit(&client_buffer_allocator_, owner,
                                       std::memory_order_release);
            throw;
        }
    }

    // exchange() is the release linearization point. A reader either acquired
    // a lease before it (and is counted here), or observes null afterwards.
    if (owner && owner.use_count() > 1) {
        const long outstanding_holders = owner.use_count() - 1;
        std::atomic_store_explicit(&client_buffer_allocator_, owner,
                                   std::memory_order_release);
        LOG(ERROR) << "EGM Store Pool allocator view has "
                   << outstanding_holders
                   << " outstanding holder(s); cleanup must be retried after "
                      "in-flight allocator users and BufferHandles release it";
        return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
    }

    {
        std::unique_lock<std::shared_mutex> lock(registered_buffer_mutex_);
        local_buffer_region_.reset();
    }
    return {};
}

}  // namespace mooncake
