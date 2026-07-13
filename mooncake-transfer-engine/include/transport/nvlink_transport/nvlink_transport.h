// Copyright 2024 KVCache.AI

#ifndef NVLINK_TRANSPORT_H_
#define NVLINK_TRANSPORT_H_

#include "cuda_alike.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/hash_utils.h"
#include "topology.h"
#include "transfer_metadata.h"
#include "transport/transport.h"

namespace mooncake {

class TransferMetadata;

class NvlinkVmmAllocation {
   public:
    enum class LocationType { DEVICE, HOST_NUMA };

    struct Options {
        LocationType location_type = LocationType::DEVICE;
        int location_id = 0;
        size_t requested_length = 0;
        bool fabric_exportable = false;
        // Zero selects the CUDA allocation granularity. Store global chunks can
        // request a stronger alignment (for example Cachelib slab alignment).
        size_t required_va_alignment = 0;
        // Optional bounded observability hook for the access-grant portion of
        // Create(). It is invoked once when that stage is reached.
        std::function<void(uint64_t duration_us, bool success)> access_observer;
    };

#if defined(USE_MNNVL) && defined(USE_CUDA)
    struct DriverApi {
        std::function<CUresult(int*)> device_get_count;
        std::function<CUresult(CUdevice*, int)> device_get;
        std::function<CUresult(int*, CUdevice_attribute, CUdevice)>
            device_get_attribute;
        std::function<CUresult(size_t*, const CUmemAllocationProp*,
                               CUmemAllocationGranularity_flags)>
            mem_get_allocation_granularity;
        std::function<CUresult(CUmemGenericAllocationHandle*, size_t,
                               const CUmemAllocationProp*, unsigned long long)>
            mem_create;
        std::function<CUresult(CUdeviceptr*, size_t, size_t, CUdeviceptr,
                               unsigned long long)>
            mem_address_reserve;
        std::function<CUresult(CUdeviceptr, size_t, size_t,
                               CUmemGenericAllocationHandle,
                               unsigned long long)>
            mem_map;
        std::function<CUresult(CUdeviceptr, size_t, const CUmemAccessDesc*,
                               size_t)>
            mem_set_access;
        std::function<CUresult(CUdeviceptr, size_t)> mem_unmap;
        std::function<CUresult(CUdeviceptr, size_t)> mem_address_free;
        std::function<CUresult(CUmemGenericAllocationHandle)> mem_release;
        std::function<CUresult(CUmemGenericAllocationHandle*, void*)>
            mem_retain_allocation_handle;
        std::function<CUresult(CUmemAllocationProp*,
                               CUmemGenericAllocationHandle)>
            mem_get_allocation_properties_from_handle;
        std::function<CUresult(CUdeviceptr*, size_t*, CUdeviceptr)>
            mem_get_address_range;
        std::function<CUresult(void*, CUmemGenericAllocationHandle,
                               CUmemAllocationHandleType, unsigned long long)>
            mem_export_to_shareable_handle;
        std::function<CUresult(CUmemGenericAllocationHandle*, void*,
                               CUmemAllocationHandleType)>
            mem_import_from_shareable_handle;
    };
#endif

    NvlinkVmmAllocation(const NvlinkVmmAllocation&) = delete;
    NvlinkVmmAllocation& operator=(const NvlinkVmmAllocation&) = delete;

#if defined(USE_MNNVL) && defined(USE_CUDA)
    NvlinkVmmAllocation(NvlinkVmmAllocation&& other) noexcept;
    NvlinkVmmAllocation& operator=(NvlinkVmmAllocation&& other) noexcept;
    ~NvlinkVmmAllocation();

    static Status Create(const Options& options,
                         std::unique_ptr<NvlinkVmmAllocation>& allocation);
    static Status GetAllocationGranularity(LocationType location_type,
                                           int location_id,
                                           bool fabric_exportable,
                                           size_t& granularity);
    static Status CheckStrictFabricCapability();

    // The explicit adapter overloads keep CUDA failure tests independent of
    // Fabric hardware. Production callers use the overloads above.
    static Status CreateWithDriverApi(
        const Options& options, const DriverApi& api,
        std::unique_ptr<NvlinkVmmAllocation>& allocation);
    static Status GetAllocationGranularityWithDriverApi(
        LocationType location_type, int location_id, bool fabric_exportable,
        const DriverApi& api, size_t& granularity);
    static Status CheckStrictFabricCapabilityWithDriverApi(
        const DriverApi& api);
#else
    NvlinkVmmAllocation(NvlinkVmmAllocation&&) noexcept = default;
    NvlinkVmmAllocation& operator=(NvlinkVmmAllocation&&) noexcept = default;
    ~NvlinkVmmAllocation() = default;

    static Status Create(const Options&,
                         std::unique_ptr<NvlinkVmmAllocation>& allocation) {
        allocation.reset();
        return Status::NotSupportedTransport(
            "NVLink HOST_NUMA VMM requires USE_MNNVL and USE_CUDA");
    }
    static Status GetAllocationGranularity(LocationType, int, bool,
                                           size_t& granularity) {
        granularity = 0;
        return Status::NotSupportedTransport(
            "NVLink HOST_NUMA VMM requires USE_MNNVL and USE_CUDA");
    }
    static Status CheckStrictFabricCapability() {
        return Status::NotSupportedTransport(
            "NVLink HOST_NUMA VMM requires USE_MNNVL and USE_CUDA");
    }
#endif

    void* base() const { return base_; }
    size_t length() const { return length_; }
    size_t granularity() const { return granularity_; }
    size_t va_alignment() const { return va_alignment_; }
    LocationType location_type() const { return location_type_; }
    int location_id() const { return location_id_; }
    bool fabric_exportable() const { return fabric_exportable_; }

   private:
    friend class NvlinkTransport;
    friend class NvlinkTransportTestPeer;
    NvlinkVmmAllocation() = default;

#if defined(USE_MNNVL) && defined(USE_CUDA)
    static DriverApi ProductionDriverApi();
    static bool RegisterOwnedRange(void* base, size_t length);
    static void UnregisterOwnedRange(void* base, size_t length);
    static bool IsExactOwnedRange(void* base, size_t length);
    void reset() noexcept;
#endif

    void* base_ = nullptr;
    size_t length_ = 0;
    size_t granularity_ = 0;
    size_t va_alignment_ = 0;
    LocationType location_type_ = LocationType::DEVICE;
    int location_id_ = 0;
    bool fabric_exportable_ = false;
    bool mapped_ = false;
    bool address_reserved_ = false;
    bool handle_owned_ = false;
    uint64_t allocation_handle_ = 0;
#if defined(USE_MNNVL) && defined(USE_CUDA)
    bool owned_range_registered_ = false;
    DriverApi driver_api_;
#endif
};

class NvlinkTransport : public Transport {
   public:
    NvlinkTransport();

    ~NvlinkTransport();

    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;

    Status submitTransferTask(
        const std::vector<TransferTask*>& task_list) override;

    Status getTransferStatus(BatchID batch_id, size_t task_id,
                             TransferStatus& status) override;

    void finalizeTransferResult(TransferTask& task, bool success) override;

    void appendMetrics(std::string& output) override;

    static void* allocatePinnedLocalMemory(size_t length);

    static void freePinnedLocalMemory(void* addr);

    [[nodiscard]] bool isFabricMemoryEnabled() const { return use_fabric_mem_; }

   protected:
    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> meta,
                std::shared_ptr<Topology> topo) override;

    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location, bool remote_accessible,
                            bool update_metadata = true) override;

    int unregisterLocalMemory(void* addr, bool update_metadata = true) override;

    int registerLocalMemoryBatch(const std::vector<BufferEntry>& buffer_list,
                                 const std::string& location) override;

    int unregisterLocalMemoryBatch(
        const std::vector<void*>& addr_list) override;

    int relocateSharedMemoryAddress(uint64_t& dest_addr, uint64_t length,
                                    uint64_t target_id);

    const char* getName() const override { return "nvlink"; }

   private:
    friend class NvlinkTransportTestPeer;

    enum class ConsumerFailureStage {
        IMPORT,
        RESERVE,
        MAP,
        SET_ACCESS,
        COPY,
    };

    struct ConsumerMetrics;

    void observeCacheLookup(bool hit);
    void observeLazyImportLatency(uint64_t duration_us);
    void observeConsumerFailure(ConsumerFailureStage stage);
    void observeTransferResult(TransferRequest::OpCode operation, bool success);
    bool observeTransferResultOnce(TransferTask& task, bool success);
    void finalizeSubmissionFailure(TransferTask& task, bool copy_failure);

    std::atomic_bool running_;

    enum class OpenedMappingKind { IPC, FABRIC };

    struct OpenedShmEntry {
        void* shm_addr = nullptr;
        uint64_t length = 0;
        OpenedMappingKind kind = OpenedMappingKind::IPC;
    };

    struct LocalRegistration {
        void* requested_addr = nullptr;
        uint64_t requested_length = 0;
        void* mapped_base = nullptr;
        uint64_t mapped_length = 0;
        bool remote_accessible = false;
        bool published = false;
        bool retained_handle_owned = false;
        uint64_t retained_handle = 0;
    };

    std::unordered_map<std::pair<uint64_t, uint64_t>, OpenedShmEntry, PairHash>
        remap_entries_;
    RWSpinlock remap_lock_;
    bool use_fabric_mem_;

    std::mutex register_mutex_;
    std::unordered_map<void*, LocalRegistration> local_registrations_;
#if defined(USE_MNNVL) && defined(USE_CUDA)
    NvlinkVmmAllocation::DriverApi fabric_driver_api_;
#endif

    std::function<int(const BufferDesc&, bool)> add_buffer_for_testing_;
    std::function<int(void*, bool)> remove_buffer_for_testing_;
    std::function<std::shared_ptr<SegmentDesc>(uint64_t)>
        get_segment_for_testing_;
    std::unique_ptr<ConsumerMetrics> consumer_metrics_;
};

}  // namespace mooncake

#endif  // NVLINK_TRANSPORT_H_
