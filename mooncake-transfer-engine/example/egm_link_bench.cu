// Copyright 2024 KVCache.AI

#include <cuda.h>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#if __has_include(<jsoncpp/json/json.h>)
#include <jsoncpp/json/json.h>
#else
#include <json/json.h>
#endif

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "common/serialization.h"
#include "egm_validation_cuda.h"
#include "transfer_metadata.h"
#include "transport/nvlink_transport/nvlink_host_numa_allocation.h"

DEFINE_string(metadata_server, "",
              "HTTP metadata endpoint ending in /metadata");
DEFINE_string(segment_name, "", "Provider Transfer Engine segment name");
DEFINE_string(devices, "0,1,2,3", "Comma-separated CUDA devices");
DEFINE_string(payload_sizes,
              "134217728,536870912,1073741824,2147483648,4294967296",
              "Comma-separated aggregate bytes per device");
DEFINE_string(stream_counts, "1,2,4,8", "Comma-separated CUDA stream counts");
DEFINE_string(paths, "H2D,D2H,H2H_LOCAL_TO_REMOTE,H2H_REMOTE_TO_LOCAL",
              "Comma-separated raw transfer paths");
DEFINE_bool(run_latency, true, "Run SM load/store latency samples");
DEFINE_int32(warmups, 3, "Warmup operations excluded from results");
DEFINE_int32(samples, 10, "Steady samples per case");
DEFINE_double(min_window_seconds, 1.0,
              "Minimum bytes-in-flight window per sample");
DEFINE_uint64(latency_accesses, 262144,
              "Dependent accesses per latency sample");
DEFINE_int32(latency_samples, 30, "Latency samples per memory kind");
DEFINE_string(latency_working_sets, "2097152,268435456,1073741824",
              "Comma-separated latency working-set bytes");
DEFINE_string(run_id, "", "Validation run ID");
DEFINE_string(source_sha, "", "Validation source SHA");

namespace mooncake::validation {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kAlignment = 2 * 1024 * 1024;
constexpr double kC2cReferenceGbPerSecond = 450.0;

std::mutex output_mutex;

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

void checkDriver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name == nullptr ? "unknown" : name) + " " +
                             (description == nullptr ? "" : description));
}

std::vector<uint64_t> parseUnsignedList(const std::string& value,
                                        const char* name,
                                        bool allow_zero = false) {
    std::vector<uint64_t> values;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) continue;
        size_t parsed = 0;
        uint64_t number = std::stoull(item, &parsed);
        if (parsed != item.size() || (!allow_zero && number == 0))
            throw std::invalid_argument(std::string(name) + " contains " +
                                        item);
        values.push_back(number);
    }
    if (values.empty())
        throw std::invalid_argument(std::string(name) + " must not be empty");
    return values;
}

std::vector<int> parseDevices(const std::string& value) {
    std::vector<uint64_t> parsed = parseUnsignedList(value, "devices", true);
    std::vector<int> devices;
    std::set<int> seen;
    for (uint64_t number : parsed) {
        if (number > static_cast<uint64_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("CUDA device ID is too large");
        int device = static_cast<int>(number);
        if (!seen.insert(device).second)
            throw std::invalid_argument("CUDA device IDs must be unique");
        devices.push_back(device);
    }
    return devices;
}

void emit(Json::Value value) {
    value["run_id"] = FLAGS_run_id;
    value["source_sha"] = FLAGS_source_sha;
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cout << Json::writeString(builder, value) << std::endl;
}

class RemoteArena {
   public:
    RemoteArena(const TransferMetadata::SegmentDesc& segment, size_t length,
                const std::vector<int>& devices) {
        std::vector<std::pair<CUmemFabricHandle, size_t>> handles;
        size_t available = 0;
        for (const auto& buffer : segment.buffers) {
            if (buffer.shm_name.empty()) continue;
            std::vector<unsigned char> decoded;
            deserializeBinaryData(buffer.shm_name, decoded);
            if (decoded.size() != sizeof(CUmemFabricHandle)) continue;
            CUmemFabricHandle fabric_handle;
            std::memcpy(&fabric_handle, decoded.data(), sizeof(fabric_handle));
            handles.emplace_back(fabric_handle, buffer.length);
            available += buffer.length;
            if (available >= length) break;
        }
        if (available < length)
            throw std::runtime_error(
                "Provider EGM capacity is smaller than raw benchmark "
                "requirement");

        length_ = available;
        checkDriver(cuMemAddressReserve(&base_, length_, kAlignment, 0, 0),
                    "cuMemAddressReserve(remote EGM arena)");
        size_t offset = 0;
        try {
            for (const auto& [fabric_handle, mapped_length] : handles) {
                CUmemGenericAllocationHandle handle;
                checkDriver(
                    cuMemImportFromShareableHandle(
                        &handle, const_cast<CUmemFabricHandle*>(&fabric_handle),
                        CU_MEM_HANDLE_TYPE_FABRIC),
                    "cuMemImportFromShareableHandle");
                CUresult map_result =
                    cuMemMap(base_ + offset, mapped_length, 0, handle, 0);
                checkDriver(map_result, "cuMemMap(remote EGM chunk)");
                mapped_lengths_.push_back(mapped_length);
                offset += mapped_length;
                CUresult release_result = cuMemRelease(handle);
                checkDriver(release_result, "cuMemRelease(imported handle)");
            }
            std::vector<CUmemAccessDesc> access;
            for (int device : devices) {
                CUmemAccessDesc desc{};
                desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                desc.location.id = device;
                desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                access.push_back(desc);
            }
            checkDriver(
                cuMemSetAccess(base_, length_, access.data(), access.size()),
                "cuMemSetAccess(remote EGM arena)");
        } catch (...) {
            release();
            throw;
        }
    }

    RemoteArena(const RemoteArena&) = delete;
    RemoteArena& operator=(const RemoteArena&) = delete;
    ~RemoteArena() { release(); }

    void* at(size_t offset) const {
        if (offset >= length_)
            throw std::out_of_range("remote EGM arena offset");
        return reinterpret_cast<void*>(base_ + offset);
    }

   private:
    void release() {
        size_t offset = 0;
        for (size_t mapped_length : mapped_lengths_) {
            CUresult result = cuMemUnmap(base_ + offset, mapped_length);
            if (result != CUDA_SUCCESS)
                LOG(ERROR) << "cuMemUnmap(remote EGM chunk) failed: " << result;
            offset += mapped_length;
        }
        mapped_lengths_.clear();
        if (base_ != 0) {
            CUresult result = cuMemAddressFree(base_, length_);
            if (result != CUDA_SUCCESS)
                LOG(ERROR) << "cuMemAddressFree(remote EGM arena) failed: "
                           << result;
            base_ = 0;
            length_ = 0;
        }
    }

    CUdeviceptr base_ = 0;
    size_t length_ = 0;
    std::vector<size_t> mapped_lengths_;
};

struct DeviceContext {
    int device;
    int numa_node;
    size_t remote_offset;
    void* hbm = nullptr;
    std::unique_ptr<NvlinkHostNumaAllocation> local_egm;
    std::vector<cudaStream_t> streams;

    DeviceContext(int device_id, size_t offset, size_t length,
                  size_t max_streams)
        : device(device_id), numa_node(-1), remote_offset(offset) {
        checkCuda(cudaSetDevice(device), "cudaSetDevice");
        CUdevice cuda_device;
        checkDriver(cuDeviceGet(&cuda_device, device), "cuDeviceGet");
        checkDriver(
            cuDeviceGetAttribute(&numa_node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                                 cuda_device),
            "cuDeviceGetAttribute(HOST_NUMA_ID)");
        checkCuda(cudaMalloc(&hbm, length), "cudaMalloc(raw HBM)");
        Status status = NvlinkHostNumaAllocation::Create(numa_node, length,
                                                         kAlignment, local_egm);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        for (size_t index = 0; index < max_streams; ++index) {
            cudaStream_t stream;
            checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags");
            streams.push_back(stream);
        }
    }

    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    ~DeviceContext() {
        cudaSetDevice(device);
        for (auto stream : streams) cudaStreamDestroy(stream);
        if (hbm != nullptr) cudaFree(hbm);
        if (local_egm) {
            Status status = local_egm->Release();
            if (!status.ok())
                LOG(ERROR) << "local EGM release failed: " << status;
        }
    }
};

enum class Path { H2D, D2H, H2H_LOCAL_TO_REMOTE, H2H_REMOTE_TO_LOCAL };
enum class Engine { CE, SM };

std::vector<Path> parsePaths(const std::string& value) {
    std::vector<Path> paths;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item == "H2D")
            paths.push_back(Path::H2D);
        else if (item == "D2H")
            paths.push_back(Path::D2H);
        else if (item == "H2H_LOCAL_TO_REMOTE")
            paths.push_back(Path::H2H_LOCAL_TO_REMOTE);
        else if (item == "H2H_REMOTE_TO_LOCAL")
            paths.push_back(Path::H2H_REMOTE_TO_LOCAL);
        else
            throw std::invalid_argument("unsupported raw path: " + item);
    }
    if (paths.empty()) throw std::invalid_argument("paths must not be empty");
    return paths;
}

const char* pathName(Path path) {
    switch (path) {
        case Path::H2D:
            return "EGM_H2D";
        case Path::D2H:
            return "EGM_D2H";
        case Path::H2H_LOCAL_TO_REMOTE:
            return "EGM_H2H_LOCAL_TO_REMOTE";
        case Path::H2H_REMOTE_TO_LOCAL:
            return "EGM_H2H_REMOTE_TO_LOCAL";
    }
    return "unknown";
}

const char* engineName(Engine engine) {
    return engine == Engine::CE ? "CE" : "SM";
}

std::pair<void*, void*> endpoints(DeviceContext& context, RemoteArena& remote,
                                  Path path) {
    void* remote_address = remote.at(context.remote_offset);
    switch (path) {
        case Path::H2D:
            return {remote_address, context.hbm};
        case Path::D2H:
            return {context.hbm, remote_address};
        case Path::H2H_LOCAL_TO_REMOTE:
            return {context.local_egm->base(), remote_address};
        case Path::H2H_REMOTE_TO_LOCAL:
            return {remote_address, context.local_egm->base()};
    }
    throw std::logic_error("unknown path");
}

void fillAndClear(DeviceContext& context, void* source, void* destination,
                  size_t length, uint64_t seed) {
    int result = egmValidationFill(source, length, seed, context.device);
    if (result != cudaSuccess)
        throw std::runtime_error("egmValidationFill failed: " +
                                 std::to_string(result));
    checkCuda(cudaMemset(destination, 0, length),
              "cudaMemset(raw destination)");
}

void verify(DeviceContext& context, const void* destination, size_t length,
            uint64_t seed) {
    uint64_t mismatches = 0;
    int result = egmValidationVerify(destination, length, seed, context.device,
                                     &mismatches);
    if (result != cudaSuccess || mismatches != 0)
        throw std::runtime_error(
            "raw transfer verification failed: CUDA=" + std::to_string(result) +
            " mismatches=" + std::to_string(mismatches));
}

double submitCopy(DeviceContext& context, void* source, void* destination,
                  size_t length, size_t stream_count, uint64_t loops,
                  Engine engine) {
    checkCuda(cudaSetDevice(context.device), "cudaSetDevice(copy)");
    size_t partition =
        ((length / stream_count + kAlignment - 1) / kAlignment) * kAlignment;
    std::vector<cudaEvent_t> starts(stream_count), stops(stream_count);
    for (size_t index = 0; index < stream_count; ++index) {
        checkCuda(cudaEventCreate(&starts[index]), "cudaEventCreate(start)");
        checkCuda(cudaEventCreate(&stops[index]), "cudaEventCreate(stop)");
        checkCuda(cudaEventRecord(starts[index], context.streams[index]),
                  "cudaEventRecord(start)");
    }
    for (uint64_t loop = 0; loop < loops; ++loop) {
        for (size_t index = 0; index < stream_count; ++index) {
            size_t offset = index * partition;
            if (offset >= length) continue;
            size_t bytes = std::min(partition, length - offset);
            void* src = static_cast<uint8_t*>(source) + offset;
            void* dst = static_cast<uint8_t*>(destination) + offset;
            if (engine == Engine::CE) {
                checkCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault,
                                          context.streams[index]),
                          "cudaMemcpyAsync(raw EGM)");
            } else {
                int result = egmValidationCopySmAsync(
                    dst, src, bytes, context.device,
                    reinterpret_cast<void*>(context.streams[index]));
                if (result != cudaSuccess)
                    throw std::runtime_error(
                        "egmValidationCopySmAsync failed: " +
                        std::to_string(result));
            }
        }
    }
    for (size_t index = 0; index < stream_count; ++index)
        checkCuda(cudaEventRecord(stops[index], context.streams[index]),
                  "cudaEventRecord(stop)");
    float maximum_ms = 0;
    for (size_t index = 0; index < stream_count; ++index) {
        checkCuda(cudaEventSynchronize(stops[index]), "cudaEventSynchronize");
        float elapsed_ms = 0;
        checkCuda(
            cudaEventElapsedTime(&elapsed_ms, starts[index], stops[index]),
            "cudaEventElapsedTime");
        maximum_ms = std::max(maximum_ms, elapsed_ms);
        cudaEventDestroy(starts[index]);
        cudaEventDestroy(stops[index]);
    }
    return static_cast<double>(maximum_ms) / 1000.0;
}

struct Sample {
    std::string group;
    std::string path;
    std::string engine;
    int device;
    size_t payload;
    size_t streams;
    int sample;
    uint64_t loops;
    int64_t started_ns;
    int64_t ended_ns;
    double duration_seconds;
    double gb_per_second;
};

void runDeviceMatrix(DeviceContext& context, RemoteArena& remote,
                     const std::string& group, std::barrier<>& barrier,
                     const std::vector<Path>& paths,
                     const std::vector<uint64_t>& payloads,
                     const std::vector<uint64_t>& stream_counts,
                     std::vector<Sample>& output, std::mutex& output_lock) {
    for (Path path : paths) {
        for (Engine engine : {Engine::CE, Engine::SM}) {
            for (uint64_t payload : payloads) {
                for (uint64_t stream_count : stream_counts) {
                    auto [source, destination] =
                        endpoints(context, remote, path);
                    uint64_t seed = static_cast<uint64_t>(context.device + 1)
                                        << 56 ^
                                    payload ^ stream_count ^
                                    static_cast<uint64_t>(path) << 32;
                    fillAndClear(context, source, destination, payload, seed);
                    barrier.arrive_and_wait();
                    double probe = submitCopy(context, source, destination,
                                              payload, stream_count, 1, engine);
                    barrier.arrive_and_wait();
                    Json::Value probe_record;
                    probe_record["event"] = "raw_lazy_init_probe";
                    probe_record["group"] = group;
                    probe_record["path"] = pathName(path);
                    probe_record["engine"] = engineName(engine);
                    probe_record["device"] = context.device;
                    probe_record["bytes"] = Json::UInt64(payload);
                    probe_record["streams"] = Json::UInt64(stream_count);
                    probe_record["duration_seconds"] = probe;
                    emit(probe_record);
                    double steady_estimate = probe;
                    for (int warmup = 0; warmup < FLAGS_warmups; ++warmup) {
                        barrier.arrive_and_wait();
                        steady_estimate =
                            submitCopy(context, source, destination, payload,
                                       stream_count, 1, engine);
                        barrier.arrive_and_wait();
                    }
                    uint64_t loops = std::max<uint64_t>(
                        1, static_cast<uint64_t>(
                               std::ceil(FLAGS_min_window_seconds * 1.10 /
                                         std::max(steady_estimate, 1e-6))));
                    loops = std::min<uint64_t>(loops, 20000);
                    for (int sample = 0; sample < FLAGS_samples; ++sample) {
                        barrier.arrive_and_wait();
                        auto started = Clock::now();
                        double elapsed =
                            submitCopy(context, source, destination, payload,
                                       stream_count, loops, engine);
                        auto ended = Clock::now();
                        barrier.arrive_and_wait();
                        double bytes = static_cast<double>(payload) * loops;
                        Sample record{
                            group,
                            pathName(path),
                            engineName(engine),
                            context.device,
                            static_cast<size_t>(payload),
                            static_cast<size_t>(stream_count),
                            sample,
                            loops,
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                started.time_since_epoch())
                                .count(),
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                ended.time_since_epoch())
                                .count(),
                            elapsed,
                            bytes / elapsed / 1e9,
                        };
                        {
                            std::lock_guard<std::mutex> lock(output_lock);
                            output.push_back(record);
                        }
                        Json::Value value;
                        value["event"] = "raw_bandwidth_sample";
                        value["group"] = group;
                        value["path"] = record.path;
                        value["engine"] = record.engine;
                        value["device"] = record.device;
                        value["bytes"] = Json::UInt64(record.payload);
                        value["streams"] = Json::UInt64(record.streams);
                        value["sample"] = record.sample;
                        value["loops"] = Json::UInt64(record.loops);
                        value["duration_seconds"] = record.duration_seconds;
                        value["bandwidth_gb_s"] = record.gb_per_second;
                        value["bandwidth_gib_s"] =
                            record.gb_per_second * 1e9 /
                            static_cast<double>(1ULL << 30);
                        value["c2c_450gb_s_utilization"] =
                            record.gb_per_second / kC2cReferenceGbPerSecond;
                        value["route_verification"] =
                            "PENDING_EXTERNAL_COUNTERS";
                        emit(value);
                    }
                    verify(context, destination, payload, seed);
                    Json::Value correctness;
                    correctness["event"] = "raw_correctness";
                    correctness["status"] = "PASS";
                    correctness["group"] = group;
                    correctness["path"] = pathName(path);
                    correctness["engine"] = engineName(engine);
                    correctness["device"] = context.device;
                    correctness["bytes"] = Json::UInt64(payload);
                    correctness["streams"] = Json::UInt64(stream_count);
                    emit(correctness);
                }
            }
        }
    }
}

void emitAggregateSamples(const std::vector<Sample>& samples) {
    using Key =
        std::tuple<std::string, std::string, std::string, size_t, size_t, int>;
    std::set<Key> keys;
    for (const auto& sample : samples)
        keys.emplace(sample.group, sample.path, sample.engine, sample.payload,
                     sample.streams, sample.sample);
    for (const auto& key : keys) {
        std::vector<const Sample*> matches;
        for (const auto& sample : samples) {
            if (Key(sample.group, sample.path, sample.engine, sample.payload,
                    sample.streams, sample.sample) == key)
                matches.push_back(&sample);
        }
        int64_t started = std::numeric_limits<int64_t>::max();
        int64_t ended = 0;
        double bytes = 0;
        Json::Value devices(Json::arrayValue);
        for (const Sample* sample : matches) {
            started = std::min(started, sample->started_ns);
            ended = std::max(ended, sample->ended_ns);
            bytes += static_cast<double>(sample->payload) * sample->loops;
            devices.append(sample->device);
        }
        double duration = static_cast<double>(ended - started) / 1e9;
        Json::Value value;
        value["event"] = "raw_aggregate_sample";
        value["group"] = std::get<0>(key);
        value["path"] = std::get<1>(key);
        value["engine"] = std::get<2>(key);
        value["bytes_per_device"] = Json::UInt64(std::get<3>(key));
        value["streams"] = Json::UInt64(std::get<4>(key));
        value["sample"] = std::get<5>(key);
        value["devices"] = devices;
        value["duration_seconds"] = duration;
        value["aggregate_bandwidth_gb_s"] = bytes / duration / 1e9;
        emit(value);
    }
}

void emitLatency(DeviceContext& context, RemoteArena& remote,
                 const std::vector<uint64_t>& working_sets) {
    checkCuda(cudaSetDevice(context.device), "cudaSetDevice(latency)");
    int clock_khz = 0;
    checkCuda(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate,
                                     context.device),
              "cudaDeviceGetAttribute(clock rate)");
    for (uint64_t length : working_sets) {
        for (const auto& [kind, address] :
             std::vector<std::pair<std::string, void*>>{
                 {"HBM", context.hbm},
                 {"LOCAL_EGM", context.local_egm->base()},
                 {"REMOTE_EGM", remote.at(context.remote_offset)},
             }) {
            for (int sample = 0; sample < FLAGS_latency_samples; ++sample) {
                int result = egmValidationInitPointerChain(
                    address, length, 8191, context.device);
                if (result != cudaSuccess)
                    throw std::runtime_error(
                        "pointer-chain initialization failed");
                uint64_t load_cycles = 0;
                result =
                    egmValidationLoadLatency(address, FLAGS_latency_accesses,
                                             context.device, &load_cycles);
                if (result != cudaSuccess)
                    throw std::runtime_error("load latency kernel failed");
                uint64_t store_cycles = 0;
                result = egmValidationStoreFenceLatency(
                    address, length, FLAGS_latency_accesses, context.device,
                    &store_cycles);
                if (result != cudaSuccess)
                    throw std::runtime_error("store latency kernel failed");
                Json::Value value;
                value["event"] = "memory_latency_sample";
                value["device"] = context.device;
                value["memory_kind"] = kind;
                value["working_set_bytes"] = Json::UInt64(length);
                value["sample"] = sample;
                value["accesses"] = Json::UInt64(FLAGS_latency_accesses);
                value["gpu_clock_khz"] = clock_khz;
                value["load_cycles_per_op"] =
                    static_cast<double>(load_cycles) / FLAGS_latency_accesses;
                value["load_ns_per_op"] =
                    static_cast<double>(load_cycles) * 1e6 /
                    (static_cast<double>(FLAGS_latency_accesses) * clock_khz);
                value["store_system_fence_cycles_per_op"] =
                    static_cast<double>(store_cycles) / FLAGS_latency_accesses;
                value["store_system_fence_completion_ns_per_op"] =
                    static_cast<double>(store_cycles) * 1e6 /
                    (static_cast<double>(FLAGS_latency_accesses) * clock_khz);
                emit(value);
            }
        }
        for (const auto& [direction, source, destination] :
             std::vector<std::tuple<std::string, void*, void*>>{
                 {"LOCAL_TO_REMOTE", context.local_egm->base(),
                  remote.at(context.remote_offset)},
                 {"REMOTE_TO_LOCAL", remote.at(context.remote_offset),
                  context.local_egm->base()},
             }) {
            bool supported = true;
            for (int sample = 0; sample < FLAGS_latency_samples; ++sample) {
                int result = egmValidationInitPointerChain(source, length, 8191,
                                                           context.device);
                if (result != cudaSuccess) {
                    supported = false;
                    break;
                }
                uint64_t cycles = 0;
                result = egmValidationLoadStoreFenceLatency(
                    source, destination, length, FLAGS_latency_accesses,
                    context.device, &cycles);
                if (result != cudaSuccess) {
                    supported = false;
                    break;
                }
                Json::Value value;
                value["event"] = "egm_h2h_latency_sample";
                value["device"] = context.device;
                value["direction"] = direction;
                value["working_set_bytes"] = Json::UInt64(length);
                value["sample"] = sample;
                value["accesses"] = Json::UInt64(FLAGS_latency_accesses);
                value["gpu_clock_khz"] = clock_khz;
                value["load_store_system_fence_cycles_per_op"] =
                    static_cast<double>(cycles) / FLAGS_latency_accesses;
                value["load_store_system_fence_ns_per_op"] =
                    static_cast<double>(cycles) * 1e6 /
                    (static_cast<double>(FLAGS_latency_accesses) * clock_khz);
                emit(value);
            }
            if (!supported) {
                Json::Value value;
                value["event"] = "egm_h2h_latency_status";
                value["status"] = "UNSUPPORTED_OR_ROUTE_UNVERIFIED";
                value["device"] = context.device;
                value["direction"] = direction;
                value["working_set_bytes"] = Json::UInt64(length);
                emit(value);
            }
        }
    }
}

}  // namespace

int run() {
    if (FLAGS_metadata_server.empty() || FLAGS_segment_name.empty() ||
        FLAGS_run_id.empty() || FLAGS_source_sha.empty())
        throw std::invalid_argument(
            "metadata_server, segment_name, run_id, and source_sha are "
            "required");
    if (FLAGS_warmups < 0 || FLAGS_samples <= 0 || FLAGS_latency_samples <= 0 ||
        FLAGS_min_window_seconds <= 0)
        throw std::invalid_argument(
            "invalid benchmark repetition configuration");
    checkDriver(cuInit(0), "cuInit");
    std::vector<int> devices = parseDevices(FLAGS_devices);
    std::vector<uint64_t> payloads =
        parseUnsignedList(FLAGS_payload_sizes, "payload_sizes");
    std::vector<uint64_t> stream_counts =
        parseUnsignedList(FLAGS_stream_counts, "stream_counts");
    std::vector<uint64_t> working_sets =
        parseUnsignedList(FLAGS_latency_working_sets, "latency_working_sets");
    std::vector<Path> paths = parsePaths(FLAGS_paths);
    size_t maximum_payload =
        *std::max_element(payloads.begin(), payloads.end());
    size_t maximum_working_set =
        *std::max_element(working_sets.begin(), working_sets.end());
    maximum_payload = std::max(maximum_payload, maximum_working_set);
    if (maximum_payload % kAlignment != 0)
        throw std::invalid_argument(
            "payloads and working sets must be 2 MiB aligned");
    size_t maximum_streams =
        *std::max_element(stream_counts.begin(), stream_counts.end());

    TransferMetadata metadata(FLAGS_metadata_server);
    auto segment = metadata.getSegmentDescByName(FLAGS_segment_name, true);
    if (!segment)
        throw std::runtime_error("Provider segment metadata not found");
    RemoteArena remote(*segment, maximum_payload * devices.size(), devices);
    std::vector<std::unique_ptr<DeviceContext>> contexts;
    for (size_t index = 0; index < devices.size(); ++index)
        contexts.push_back(std::make_unique<DeviceContext>(
            devices[index], index * maximum_payload, maximum_payload,
            maximum_streams));

    Json::Value start;
    start["event"] = "raw_benchmark_start";
    start["segment_name"] = FLAGS_segment_name;
    start["devices"] = FLAGS_devices;
    start["payload_sizes"] = FLAGS_payload_sizes;
    start["stream_counts"] = FLAGS_stream_counts;
    start["paths"] = FLAGS_paths;
    emit(start);

    std::vector<std::vector<size_t>> groups;
    for (size_t index = 0; index < contexts.size(); ++index)
        groups.push_back({index});
    for (size_t group_size : {size_t{2}, size_t{4}}) {
        if (group_size <= contexts.size()) {
            std::vector<size_t> indices(group_size);
            std::iota(indices.begin(), indices.end(), 0);
            groups.push_back(std::move(indices));
        }
    }
    for (const auto& indices : groups) {
        std::vector<Sample> samples;
        std::mutex samples_mutex;
        std::barrier barrier(static_cast<std::ptrdiff_t>(indices.size()));
        std::vector<std::thread> workers;
        std::string group = std::to_string(indices.size()) + "GPU";
        if (indices.size() == 1)
            group +=
                "-device" + std::to_string(contexts[indices.front()]->device);
        for (size_t index : indices) {
            workers.emplace_back(runDeviceMatrix, std::ref(*contexts[index]),
                                 std::ref(remote), group, std::ref(barrier),
                                 std::cref(paths), std::cref(payloads),
                                 std::cref(stream_counts), std::ref(samples),
                                 std::ref(samples_mutex));
        }
        for (auto& worker : workers) worker.join();
        emitAggregateSamples(samples);
    }
    if (FLAGS_run_latency) {
        for (auto& context : contexts)
            emitLatency(*context, remote, working_sets);
    }

    Json::Value gate;
    gate["event"] = "raw_benchmark_gate";
    gate["status"] = "PASS";
    gate["h2h_semantics"] = "host-submitted, GPU-CE/SM-executed EGM H2H";
    gate["route_verification"] = "PENDING_EXTERNAL_COUNTERS";
    gate["paths"] = FLAGS_paths;
    emit(gate);
    return 0;
}

}  // namespace mooncake::validation

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    try {
        return mooncake::validation::run();
    } catch (const std::exception& error) {
        Json::Value value;
        value["event"] = "raw_benchmark_gate";
        value["status"] = "FAIL";
        value["run_id"] = FLAGS_run_id;
        value["source_sha"] = FLAGS_source_sha;
        value["error"] = error.what();
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        std::cout << Json::writeString(builder, value) << std::endl;
        return 1;
    }
}
