# Standalone observation only; no portable-model option or default integration.
function(strata_native_hc_enable out)
  set(shader "${out}/native_hc.spv")
  add_custom_command(OUTPUT "${out}/strata_native_hc_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_hc.comp" -o "${shader}"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${shader}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${shader}" "-DOUTPUT=${out}/strata_native_hc_spv.hpp"
      -DARRAY_NAME=strata_native_hc_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_hc.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_hc_shader DEPENDS "${out}/strata_native_hc_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_hc_shader)
  add_custom_command(OUTPUT "${out}/strata_native_hc_mix_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_hc_mix.comp" -o "${out}/native_hc_mix.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_hc_mix.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_hc_mix.spv" "-DOUTPUT=${out}/strata_native_hc_mix_spv.hpp"
      -DARRAY_NAME=strata_native_hc_mix_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_hc_mix.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_hc_mix_shader DEPENDS "${out}/strata_native_hc_mix_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_hc_mix_shader)
  add_executable(native_hc_mix_component tests/kernels/native_hc_mix_component.cpp)
  target_include_directories(native_hc_mix_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_hc_mix_component PRIVATE ggml-vulkan ggml-base)
  add_executable(native_hc_component tests/kernels/native_hc_component.cpp)
  target_include_directories(native_hc_component PRIVATE src/core)
  target_link_libraries(native_hc_component PRIVATE ggml-vulkan ggml-base)
  if(STRATA_BUILD_TESTS)
    add_test(NAME native_hc_mix_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_mix.py")
    add_test(NAME native_hc_compare_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_compare.py")
    add_test(NAME native_hc_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_numeric.py")
    add_executable(native_hc_mix_effective_shader_test tests/kernels/native_hc_mix_effective_shader_test.cpp)
    target_include_directories(native_hc_mix_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_hc_mix_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_hc_mix_effective_shader_test strata_native_hc_mix_shader)
    add_test(NAME native_hc_mix_effective_shader_test COMMAND native_hc_mix_effective_shader_test)
    add_executable(native_hc_effective_shader_test tests/kernels/native_hc_effective_shader_test.cpp)
    target_include_directories(native_hc_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_hc_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_hc_effective_shader_test strata_native_hc_shader)
    add_test(NAME native_hc_effective_shader_test COMMAND native_hc_effective_shader_test)
    add_executable(native_hc_dispatch_test tests/kernels/native_hc_dispatch_test.cpp)
    target_include_directories(native_hc_dispatch_test PRIVATE src/core "${out}" "${vk_source}/..")
    target_link_libraries(native_hc_dispatch_test PRIVATE ggml-base)
    add_dependencies(native_hc_dispatch_test strata_native_hc_shader)
    add_test(NAME native_hc_dispatch_test COMMAND native_hc_dispatch_test)
    add_test(NAME native_hc_shader_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_shader.py"
      "${Vulkan_GLSLC_EXECUTABLE}" "${STRATA_SPIRV_DIS}" "${STRATA_SPIRV_VAL}")
  endif()
endfunction()
