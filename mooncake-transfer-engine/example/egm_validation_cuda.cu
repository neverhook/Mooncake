// Copyright 2024 KVCache.AI

#include "egm_validation_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace {

constexpr uint64_t kPatternMultiplier = 0x9e3779b97f4a7c15ULL;
constexpr size_t kPointerLineSize = 128;

__device__ __forceinline__ uint64_t patternWord(uint64_t index, uint64_t seed) {
    uint64_t value = seed ^ (index * kPatternMultiplier);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__global__ void fillKernel(uint64_t* address, size_t words, uint64_t seed) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < words;
         index += gridDim.x * blockDim.x) {
        address[index] = patternWord(index, seed);
    }
}

__global__ void verifyKernel(const uint64_t* address, size_t words,
                             uint64_t seed, unsigned long long* mismatches) {
    uint64_t local = 0;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < words;
         index += gridDim.x * blockDim.x) {
        local += address[index] != patternWord(index, seed);
    }
    if (local != 0) atomicAdd(mismatches, local);
}

__global__ void copySmKernel(uint64_t* destination, const uint64_t* source,
                             size_t words) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < words;
         index += gridDim.x * blockDim.x) {
        destination[index] = source[index];
    }
}

__global__ void initPointerChainKernel(uint8_t* address, size_t lines,
                                       uint64_t stride_lines) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < lines;
         index += gridDim.x * blockDim.x) {
        size_t next = (index + stride_lines) % lines;
        *reinterpret_cast<uint64_t**>(address + index * kPointerLineSize) =
            reinterpret_cast<uint64_t*>(address + next * kPointerLineSize);
    }
}

__global__ void loadLatencyKernel(const uint64_t* address, uint64_t accesses,
                                  uint64_t* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const volatile uint64_t* current = address;
    uint64_t start = clock64();
    for (uint64_t index = 0; index < accesses; ++index) {
        current = reinterpret_cast<const volatile uint64_t*>(*current);
    }
    uint64_t stop = clock64();
    output[0] = stop - start;
    output[1] = reinterpret_cast<uint64_t>(current);
}

__global__ void storeFenceLatencyKernel(uint8_t* address, size_t lines,
                                        uint64_t accesses, uint64_t* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    constexpr uint64_t stride = 8191;
    uint64_t start = clock64();
    for (uint64_t index = 0; index < accesses; ++index) {
        size_t line = (index * stride) & (lines - 1);
        *reinterpret_cast<volatile uint64_t*>(address +
                                              line * kPointerLineSize) = index;
        __threadfence_system();
    }
    uint64_t stop = clock64();
    output[0] = stop - start;
    output[1] = address[0];
}

__global__ void loadStoreFenceLatencyKernel(const uint64_t* source,
                                            uint8_t* destination,
                                            size_t destination_lines,
                                            uint64_t accesses,
                                            uint64_t* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const volatile uint64_t* current = source;
    uint64_t start = clock64();
    for (uint64_t index = 0; index < accesses; ++index) {
        current = reinterpret_cast<const volatile uint64_t*>(*current);
        size_t line = (reinterpret_cast<uint64_t>(current) >> 7) &
                      (destination_lines - 1);
        *reinterpret_cast<volatile uint64_t*>(destination +
                                              line * kPointerLineSize) =
            reinterpret_cast<uint64_t>(current);
        __threadfence_system();
    }
    uint64_t stop = clock64();
    output[0] = stop - start;
    output[1] = reinterpret_cast<uint64_t>(current);
}

int setDeviceAndValidate(const void* address, size_t length, int device) {
    if (address == nullptr || length == 0 || device < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaSetDevice(device));
}

int copyCycles(uint64_t* device_output, uint64_t* cycles) {
    uint64_t output[2] = {};
    cudaError_t error = cudaMemcpy(output, device_output, sizeof(output),
                                   cudaMemcpyDeviceToHost);
    if (error == cudaSuccess && cycles != nullptr) *cycles = output[0];
    return static_cast<int>(error);
}

template <typename Launch>
int runLatency(Launch launch, uint64_t* cycles) {
    if (cycles == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    uint64_t* output = nullptr;
    cudaError_t error =
        cudaMalloc(reinterpret_cast<void**>(&output), 2 * sizeof(uint64_t));
    if (error != cudaSuccess) return static_cast<int>(error);
    launch(output);
    error = cudaGetLastError();
    if (error == cudaSuccess)
        error = static_cast<cudaError_t>(copyCycles(output, cycles));
    cudaError_t free_error = cudaFree(output);
    return static_cast<int>(error == cudaSuccess ? free_error : error);
}

}  // namespace

extern "C" int egmValidationFill(void* address, size_t length, uint64_t seed,
                                 int device) {
    int result = setDeviceAndValidate(address, length, device);
    if (result != cudaSuccess) return result;
    if (length % sizeof(uint64_t) != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    size_t words = length / sizeof(uint64_t);
    fillKernel<<<std::min<size_t>(65535, (words + 255) / 256), 256>>>(
        static_cast<uint64_t*>(address), words, seed);
    return static_cast<int>(cudaDeviceSynchronize());
}

extern "C" int egmValidationVerify(const void* address, size_t length,
                                   uint64_t seed, int device,
                                   uint64_t* mismatches) {
    int result = setDeviceAndValidate(address, length, device);
    if (result != cudaSuccess || length % sizeof(uint64_t) != 0 ||
        mismatches == nullptr)
        return result == cudaSuccess ? static_cast<int>(cudaErrorInvalidValue)
                                     : result;
    unsigned long long* device_mismatches = nullptr;
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&device_mismatches),
                                   sizeof(*device_mismatches));
    if (error != cudaSuccess) return static_cast<int>(error);
    error = cudaMemset(device_mismatches, 0, sizeof(*device_mismatches));
    size_t words = length / sizeof(uint64_t);
    if (error == cudaSuccess) {
        verifyKernel<<<std::min<size_t>(65535, (words + 255) / 256), 256>>>(
            static_cast<const uint64_t*>(address), words, seed,
            device_mismatches);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess)
        error = cudaMemcpy(mismatches, device_mismatches, sizeof(*mismatches),
                           cudaMemcpyDeviceToHost);
    cudaError_t free_error = cudaFree(device_mismatches);
    return static_cast<int>(error == cudaSuccess ? free_error : error);
}

extern "C" int egmValidationCopySm(void* destination, const void* source,
                                   size_t length, int device) {
    int result =
        egmValidationCopySmAsync(destination, source, length, device, nullptr);
    if (result != cudaSuccess) return result;
    return static_cast<int>(cudaDeviceSynchronize());
}

extern "C" int egmValidationCopySmAsync(void* destination, const void* source,
                                        size_t length, int device,
                                        void* stream) {
    int result = setDeviceAndValidate(destination, length, device);
    if (result != cudaSuccess || source == nullptr ||
        length % sizeof(uint64_t) != 0)
        return result == cudaSuccess ? static_cast<int>(cudaErrorInvalidValue)
                                     : result;
    size_t words = length / sizeof(uint64_t);
    copySmKernel<<<std::min<size_t>(65535, (words + 255) / 256), 256, 0,
                   static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint64_t*>(destination),
        static_cast<const uint64_t*>(source), words);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int egmValidationInitPointerChain(void* address, size_t length,
                                             uint64_t stride_lines,
                                             int device) {
    int result = setDeviceAndValidate(address, length, device);
    size_t lines = length / kPointerLineSize;
    if (result != cudaSuccess || lines < 2 || stride_lines == 0)
        return result == cudaSuccess ? static_cast<int>(cudaErrorInvalidValue)
                                     : result;
    while (std::gcd(stride_lines, static_cast<uint64_t>(lines)) != 1)
        ++stride_lines;
    initPointerChainKernel<<<std::min<size_t>(65535, (lines + 255) / 256),
                             256>>>(static_cast<uint8_t*>(address), lines,
                                    stride_lines);
    return static_cast<int>(cudaDeviceSynchronize());
}

extern "C" int egmValidationLoadLatency(const void* address, uint64_t accesses,
                                        int device, uint64_t* cycles) {
    int result = setDeviceAndValidate(address, sizeof(uint64_t), device);
    if (result != cudaSuccess || accesses == 0) return result;
    return runLatency(
        [&](uint64_t* output) {
            loadLatencyKernel<<<1, 1>>>(static_cast<const uint64_t*>(address),
                                        accesses, output);
        },
        cycles);
}

extern "C" int egmValidationStoreFenceLatency(void* address, size_t length,
                                              uint64_t accesses, int device,
                                              uint64_t* cycles) {
    int result = setDeviceAndValidate(address, length, device);
    size_t lines = length / kPointerLineSize;
    if (result != cudaSuccess || accesses == 0 || lines == 0 ||
        (lines & (lines - 1)) != 0)
        return result == cudaSuccess ? static_cast<int>(cudaErrorInvalidValue)
                                     : result;
    return runLatency(
        [&](uint64_t* output) {
            storeFenceLatencyKernel<<<1, 1>>>(static_cast<uint8_t*>(address),
                                              lines, accesses, output);
        },
        cycles);
}

extern "C" int egmValidationLoadStoreFenceLatency(const void* source,
                                                  void* destination,
                                                  size_t destination_length,
                                                  uint64_t accesses, int device,
                                                  uint64_t* cycles) {
    int result = setDeviceAndValidate(destination, destination_length, device);
    size_t lines = destination_length / kPointerLineSize;
    if (result != cudaSuccess || source == nullptr || accesses == 0 ||
        lines == 0 || (lines & (lines - 1)) != 0)
        return result == cudaSuccess ? static_cast<int>(cudaErrorInvalidValue)
                                     : result;
    return runLatency(
        [&](uint64_t* output) {
            loadStoreFenceLatencyKernel<<<1, 1>>>(
                static_cast<const uint64_t*>(source),
                static_cast<uint8_t*>(destination), lines, accesses, output);
        },
        cycles);
}
