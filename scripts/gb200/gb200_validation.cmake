# Validation-only targets injected through CMAKE_PROJECT_INCLUDE by
# egm_store_build.sh. Keeping them here avoids modifying production CMake files.

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR
   AND NOT DEFINED MOONCAKE_GB200_VALIDATION_TARGETS_SCHEDULED)
  enable_language(CUDA)
  set(MOONCAKE_GB200_VALIDATION_TARGETS_SCHEDULED
      TRUE
      CACHE INTERNAL "")

  function(mooncake_add_gb200_validation_targets)
    set(validation_workspace
        "${CMAKE_SOURCE_DIR}/mooncake-transfer-engine/example")

    add_library(egm_validation_cuda SHARED
                "${validation_workspace}/egm_validation_cuda.cu")
    target_include_directories(egm_validation_cuda
                               PUBLIC "${validation_workspace}")
    target_link_libraries(egm_validation_cuda PUBLIC CUDA::cudart)
    set_target_properties(
      egm_validation_cuda
      PROPERTIES CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_EXTENSIONS OFF)

    add_executable(egm_link_bench "${validation_workspace}/egm_link_bench.cu")
    target_include_directories(egm_link_bench PRIVATE "${validation_workspace}")
    target_link_libraries(
      egm_link_bench
      PRIVATE transfer_engine
              egm_validation_cuda
              CUDA::cuda_driver
              CUDA::cudart
              JsonCpp::JsonCpp
              gflags::gflags
              glog::glog)
    set_target_properties(
      egm_link_bench
      PROPERTIES CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_EXTENSIONS OFF)

    if(TORCH_CUDA_ARCH_LIST)
      set(validation_cuda_architectures "")
      foreach(architecture IN LISTS TORCH_CUDA_ARCH_LIST)
        string(REPLACE "." "" architecture "${architecture}")
        list(APPEND validation_cuda_architectures "${architecture}")
      endforeach()
      set_target_properties(
        egm_validation_cuda egm_link_bench
        PROPERTIES CUDA_ARCHITECTURES "${validation_cuda_architectures}")
    else()
      # GB200 is Blackwell compute capability 10.0.
      set_target_properties(egm_validation_cuda egm_link_bench
                            PROPERTIES CUDA_ARCHITECTURES "100")
    endif()
  endfunction()

  cmake_language(DEFER CALL mooncake_add_gb200_validation_targets)
endif()
