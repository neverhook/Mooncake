// Copyright 2024 KVCache.AI

#ifndef EGM_VALIDATION_CUDA_H_
#define EGM_VALIDATION_CUDA_H_

#include <cstddef>
#include <cstdint>

// Validation-only CUDA helpers shared by the GB200 raw-link executable and
// the Python Store harness. Every function returns a cudaError_t-compatible
// integer and synchronizes before returning.
extern "C" {

int egmValidationFill(void* address, size_t length, uint64_t seed, int device);
int egmValidationVerify(const void* address, size_t length, uint64_t seed,
                        int device, uint64_t* mismatches);
int egmValidationCopySm(void* destination, const void* source, size_t length,
                        int device);
int egmValidationCopySmAsync(void* destination, const void* source,
                             size_t length, int device, void* stream);
int egmValidationInitPointerChain(void* address, size_t length,
                                  uint64_t stride_lines, int device);
int egmValidationLoadLatency(const void* address, uint64_t accesses, int device,
                             uint64_t* cycles);
int egmValidationStoreFenceLatency(void* address, size_t length,
                                   uint64_t accesses, int device,
                                   uint64_t* cycles);
int egmValidationLoadStoreFenceLatency(const void* source, void* destination,
                                       size_t destination_length,
                                       uint64_t accesses, int device,
                                       uint64_t* cycles);
}

#endif  // EGM_VALIDATION_CUDA_H_
