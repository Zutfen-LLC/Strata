# Standalone observation only, inside the existing explicit strict-FMA build.
function(strata_native_moe_combine_enable out)
  set(shader "${out}/native_moe_combine.spv")
  execute_process(COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_moe_combine.comp" -o "${shader}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "native MoE combine strict compiler unavailable: ${output}${error}")
  endif()
  execute_process(COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${shader}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "native MoE combine strict SPIR-V validation failed: ${output}${error}")
  endif()
  execute_process(COMMAND "${STRATA_SPIRV_DIS}" "${shader}" RESULT_VARIABLE result OUTPUT_VARIABLE assembly ERROR_VARIABLE error)
  if(NOT result EQUAL 0 OR NOT assembly MATCHES "OpFmaKHR" OR NOT assembly MATCHES "OpCapability FMAKHR" OR NOT assembly MATCHES "SPV_KHR_fma")
    message(FATAL_ERROR "native MoE combine requires verified OpFmaKHR/FMAKHR/SPV_KHR_fma: ${error}")
  endif()
  add_custom_command(OUTPUT "${out}/strata_native_moe_combine_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_moe_combine.comp" -o "${shader}"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${shader}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${shader}" "-DOUTPUT=${out}/strata_native_moe_combine_spv.hpp"
      -DARRAY_NAME=strata_native_moe_combine_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_moe_combine.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_moe_combine_shader DEPENDS "${out}/strata_native_moe_combine_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_moe_combine_shader)
  add_executable(native_moe_combine_component tests/kernels/native_moe_combine_component.cpp)
  target_include_directories(native_moe_combine_component PRIVATE src/core)
  target_link_libraries(native_moe_combine_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_moe_combine_component PRIVATE -fno-fast-math -ffp-contract=off -frounding-math)
  else()
    target_compile_options(native_moe_combine_component PRIVATE /fp:strict)
  endif()
  if(STRATA_BUILD_TESTS)
    add_executable(native_moe_combine_shader_test tests/kernels/native_moe_combine_shader_test.cpp)
    target_include_directories(native_moe_combine_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_moe_combine_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan)
    add_dependencies(native_moe_combine_shader_test strata_native_moe_combine_shader)
    add_test(NAME native_moe_combine_shader_test COMMAND native_moe_combine_shader_test "${out}/native_moe_combine_effective")
  endif()
endfunction()
