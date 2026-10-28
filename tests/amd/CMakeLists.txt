cmake_minimum_required(VERSION 3.24)
project(strata_amd_expert_probe LANGUAGES C CXX)
if(WIN32)
  message(FATAL_ERROR "The experimental Vulkan executor requires the POSIX GpuCache")
endif()

# Standalone diagnostic project: the production build graph and defaults are untouched.
option(STRATA_AMD_PROBE_VULKAN "Link the Vulkan diagnostic backend" ON)
if(NOT STRATA_GGML_DIR OR NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "Pass STRATA_GGML_DIR pointing to the retained pinned llama.cpp source")
endif()
set(STRATA_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_HIP OFF CACHE BOOL "" FORCE)
set(STRATA_HIP_GFX906 OFF CACHE BOOL "" FORCE)
set(STRATA_ENABLE_SYCL OFF CACHE BOOL "" FORCE)
set(STRATA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(STRATA_NATIVE_EXPERTS ON CACHE BOOL "" FORCE)
set(GGML_VULKAN ${STRATA_AMD_PROBE_VULKAN} CACHE BOOL "" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)
add_subdirectory(../.. strata EXCLUDE_FROM_ALL)
# Diagnostic ISA witness inherits the actual ggml-cpu C flags, including -march=native.
# No ggml source, kernel or architecture option is changed.
target_sources(ggml-cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/native_reference_isa.c)

# This opt-in target is intentionally absent from the production build graph.
add_library(strata_vulkan_executor STATIC ../../src/worker/vulkan_executor.cpp)
target_compile_features(strata_vulkan_executor PUBLIC cxx_std_17)
target_compile_definitions(strata_vulkan_executor PUBLIC STRATA_AMD_EXECUTOR_HOST_TEST=1)
target_link_libraries(strata_vulkan_executor PUBLIC strata_gpu_assets ggml-base ggml-cpu)
if(STRATA_AMD_PROBE_VULKAN)
  target_compile_definitions(strata_vulkan_executor PRIVATE STRATA_AMD_PROBE_VULKAN=1)
  target_link_libraries(strata_vulkan_executor PRIVATE ggml-vulkan)
endif()
add_executable(strata-vulkan-q8-expert vulkan_q8_expert.cpp)
target_compile_features(strata-vulkan-q8-expert PRIVATE cxx_std_17)
target_link_libraries(strata-vulkan-q8-expert PRIVATE strata_vulkan_executor strata_gpu_protocol strata_worker_tier strata_kernels_cpu ggml-cpu ggml-base)
if(STRATA_AMD_PROBE_VULKAN)
  target_compile_definitions(strata-vulkan-q8-expert PRIVATE STRATA_AMD_PROBE_VULKAN=1)
  target_link_libraries(strata-vulkan-q8-expert PRIVATE ggml-vulkan)
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(strata-vulkan-q8-expert PRIVATE Threads::Threads)
enable_testing()
add_test(NAME amd_q8_expert_host COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test_vulkan_q8_expert.py $<TARGET_FILE:strata-vulkan-q8-expert>)
set_tests_properties(amd_q8_expert_host PROPERTIES TIMEOUT 120)

add_test(NAME amd_vulkan_adapter_host COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test_vulkan_executor.py $<TARGET_FILE:strata-vulkan-q8-expert>)
set_tests_properties(amd_vulkan_adapter_host PROPERTIES TIMEOUT 120)

# Explicit opt-in Vulkan worker binary; not added to the production build graph.
add_executable(strata-vulkan-expert-worker ../../src/worker/vulkan_worker_main.cpp)
target_include_directories(strata-vulkan-expert-worker PRIVATE ../../src/worker)
target_link_libraries(strata-vulkan-expert-worker PRIVATE strata_vulkan_executor strata_gpu_protocol Threads::Threads)

add_test(NAME amd_vulkan_loopback_host COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test_vulkan_loopback.py $<TARGET_FILE:strata-vulkan-q8-expert>)
set_tests_properties(amd_vulkan_loopback_host PROPERTIES TIMEOUT 120)

# Existing CUDA-contract host regression suite, with no CUDA execution compiled in.
add_executable(amd_gpu_worker_regression ../worker/gpu_worker_test.cpp)
target_include_directories(amd_gpu_worker_regression PRIVATE ../../src/worker)
target_link_libraries(amd_gpu_worker_regression PRIVATE strata_gpu_protocol Threads::Threads)
add_test(NAME amd_gpu_worker_regression COMMAND amd_gpu_worker_regression)
set_tests_properties(amd_gpu_worker_regression PROPERTIES TIMEOUT 120)

add_executable(amd_gpu_group_regression ../worker/gpu_worker_group_test.cpp)
target_include_directories(amd_gpu_group_regression PRIVATE ../../src/worker)
target_link_libraries(amd_gpu_group_regression PRIVATE strata_gpu_protocol Threads::Threads)
add_test(NAME amd_gpu_group_regression COMMAND amd_gpu_group_regression)
set_tests_properties(amd_gpu_group_regression PROPERTIES TIMEOUT 120)
if(STRATA_WORKER_TLS)
  target_compile_definitions(strata-vulkan-q8-expert PRIVATE STRATA_TEST_TLS=1)
  target_compile_definitions(amd_gpu_worker_regression PRIVATE STRATA_TEST_TLS=1)
  target_compile_definitions(amd_gpu_group_regression PRIVATE STRATA_TEST_TLS=1)
  add_executable(amd_tls_transport_regression ../worker/tls_transport_test.cpp)
  target_include_directories(amd_tls_transport_regression PRIVATE ../../src/worker)
  target_link_libraries(amd_tls_transport_regression PRIVATE strata_expert_worker Threads::Threads)
  add_test(NAME amd_tls_transport_regression COMMAND amd_tls_transport_regression)
  set_tests_properties(amd_tls_transport_regression PROPERTIES TIMEOUT 60)
endif()
