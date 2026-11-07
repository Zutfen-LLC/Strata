# Standalone only, gated behind the existing strict FMA observation build.
function(strata_native_shared_enable out)
  set(shader "${out}/native_shared.spv")
  add_custom_command(OUTPUT "${out}/strata_native_shared_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_shared.comp" -o "${shader}"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${shader}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${shader}" "-DOUTPUT=${out}/strata_native_shared_spv.hpp"
      -DARRAY_NAME=strata_native_shared_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_shared.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_shared_shader DEPENDS "${out}/strata_native_shared_spv.hpp")
  add_custom_command(OUTPUT "${out}/strata_qsa_indexer_bf16_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      -DSTRATA_QSA_INDEXER_BF16_MATRIX=1 "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_shared.comp" -o "${out}/qsa_indexer_bf16.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/qsa_indexer_bf16.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/qsa_indexer_bf16.spv" "-DOUTPUT=${out}/strata_qsa_indexer_bf16_spv.hpp"
      -DARRAY_NAME=strata_qsa_indexer_bf16_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_shared.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_qsa_indexer_bf16_shader DEPENDS "${out}/strata_qsa_indexer_bf16_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_shared_shader)
  add_dependencies(ggml-vulkan strata_qsa_indexer_bf16_shader)
  add_executable(native_shared_component tests/kernels/native_shared_component.cpp)
  target_include_directories(native_shared_component PRIVATE src/core)
  target_link_libraries(native_shared_component PRIVATE ggml-vulkan ggml-base)
  add_executable(native_qsa_indexer_bf16_component tests/kernels/native_qsa_indexer_bf16_component.cpp)
  target_include_directories(native_qsa_indexer_bf16_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_qsa_indexer_bf16_component PRIVATE ggml-vulkan ggml-base)
  if(STRATA_BUILD_TESTS)
    add_executable(native_shared_effective_shader_test tests/kernels/native_shared_effective_shader_test.cpp)
    target_include_directories(native_shared_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_shared_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan)
    add_dependencies(native_shared_effective_shader_test strata_native_shared_shader)
    add_test(NAME native_shared_effective_shader_test COMMAND native_shared_effective_shader_test)
    add_executable(native_shared_dispatch_test tests/kernels/native_shared_dispatch_test.cpp)
    target_include_directories(native_shared_dispatch_test PRIVATE src/core "${out}" "${vk_source}/..")
    target_link_libraries(native_shared_dispatch_test PRIVATE ggml-base)
    add_dependencies(native_shared_dispatch_test strata_native_shared_shader)
    add_dependencies(native_shared_dispatch_test strata_qsa_indexer_bf16_shader)
    add_test(NAME native_shared_dispatch_test COMMAND native_shared_dispatch_test)
    add_executable(native_ple_key_dispatch_test tests/kernels/native_ple_key_dispatch_test.cpp)
    target_include_directories(native_ple_key_dispatch_test PRIVATE src/core "${vk_source}/..")
    target_link_libraries(native_ple_key_dispatch_test PRIVATE ggml-base)
    add_test(NAME native_ple_key_dispatch_test COMMAND native_ple_key_dispatch_test)
    add_test(NAME native_qsa_indexer_bf16_contract_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_qsa_indexer_bf16_contract.py"
      "$<TARGET_FILE:native_qsa_indexer_bf16_input_test>" "$<TARGET_FILE:native_qsa_indexer_bf16_component>")
    add_test(NAME native_shared_shader_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_shared_shader.py"
      "${Vulkan_GLSLC_EXECUTABLE}" "${STRATA_SPIRV_DIS}" "${STRATA_SPIRV_VAL}")
  endif()
endfunction()
