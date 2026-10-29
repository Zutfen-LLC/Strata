function(strata_enable_native_stages)
  find_package(Vulkan REQUIRED)
  find_package(SPIRV-Headers CONFIG REQUIRED)
  include(cmake/strata_logits_guard.cmake)
  strata_enable_logits_guard()
  set(out "${CMAKE_CURRENT_BINARY_DIR}/strata-logits-guard")
  # Reuse the exact pin gate and single build-tree copy; original source untouched.
  file(APPEND "${out}/ggml-vulkan-guard.cpp" "#include \"strata_vk_native_stages.inc\"\n")
  add_custom_command(OUTPUT "${out}/strata_native_stages_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_q8_stages.comp" -o "${out}/native_stages.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_stages.spv" "-DOUTPUT=${out}/strata_native_stages_spv.hpp"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_q8_stages.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_stages_shader DEPENDS "${out}/strata_native_stages_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_stages_shader)
  add_executable(native_q8_vulkan_component tests/kernels/native_q8_vulkan_component.cpp)
  target_include_directories(native_q8_vulkan_component PRIVATE src/core)
  target_link_libraries(native_q8_vulkan_component PRIVATE ggml-vulkan ggml-base)
  if(STRATA_BUILD_TESTS)
    add_executable(native_q8_spec_test tests/kernels/native_q8_spec_test.cpp)
    if(MSVC)
      target_compile_options(native_q8_spec_test PRIVATE /fp:strict)
    else()
      target_compile_options(native_q8_spec_test PRIVATE -ffp-contract=off -fno-fast-math)
    endif()
    add_test(NAME native_q8_spec_test COMMAND native_q8_spec_test)
    add_executable(native_q8_shader_test tests/kernels/native_q8_shader_test.cpp)
    target_include_directories(native_q8_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_q8_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan)
    add_dependencies(native_q8_shader_test strata_native_stages_shader)
    add_test(NAME native_q8_shader_test COMMAND native_q8_shader_test)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME native_q8_tools_test COMMAND "${Python3_EXECUTABLE}" -m unittest discover
      -s "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels" -p test_native_q8_tools.py)
  endif()
endfunction()
