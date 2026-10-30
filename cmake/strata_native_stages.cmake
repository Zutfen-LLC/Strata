function(strata_enable_native_stages)
  find_package(Vulkan REQUIRED)
  find_package(SPIRV-Headers CONFIG REQUIRED)
  include(cmake/strata_logits_guard.cmake)
  strata_enable_logits_guard()
  set(out "${CMAKE_CURRENT_BINARY_DIR}/strata-logits-guard")
  # Observation only: copy the pin-checked helper verbatim, rename its entry,
  # add a receipt argument and validate its FINAL shader before module creation.
  # The original helper and strict call site remain untouched.
  get_target_property(vk_source ggml-vulkan SOURCE_DIR)
  file(READ "${vk_source}/ggml-vulkan.cpp" original)
  string(FIND "${original}" "static void ggml_vk_create_pipeline_func(" begin)
  string(FIND "${original}" "\nvoid ggml_vk_destroy_pipeline(" end)
  if(begin LESS 0 OR end LESS begin)
    message(FATAL_ERROR "native observation: pinned pipeline helper not found")
  endif()
  math(EXPR length "${end} - ${begin}")
  string(SUBSTRING "${original}" ${begin} ${length} helper)
  string(REPLACE "static void ggml_vk_create_pipeline_func(" "static void strata_vk_observation_create_pipeline_func(" helper "${helper}")
  string(REPLACE "uint32_t required_subgroup_size) {" "uint32_t required_subgroup_size, const char* diagnostic_spv_path) {" helper "${helper}")
  set(anchor "    pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);")
  string(FIND "${helper}" "${anchor}" hook)
  if(hook LESS 0)
    message(FATAL_ERROR "native observation: final shader hook missing")
  endif()
  string(REPLACE "${anchor}" "    native_q8_validate_observation_effective(static_cast<const uint32_t*>(spv_data), spv_size/4, shader_module_create_info.pCode, shader_module_create_info.codeSize/4, device->float_controls_rte_fp16, device->float_controls_denorm_preserve_fp16, diagnostic_spv_path);\n${anchor}" helper "${helper}")
  file(WRITE "${out}/strata_native_observe_pipeline.inc" "// Generated from the exact pinned MIT-licensed GGML helper.\n${helper}")
  # Host-only fixture runs the exact pinned FP16 transformation block, not an
  # independently reimplemented approximation of its effective module.
  string(FIND "${original}" "    if (device->float_controls_rte_fp16 || device->float_controls_denorm_preserve_fp16) {" fp_begin)
  string(FIND "${original}" "\n#if defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)" fp_end)
  if(fp_begin LESS 0 OR fp_end LESS fp_begin)
    message(FATAL_ERROR "native observation: pinned FP16 transformation block missing")
  endif()
  math(EXPR fp_length "${fp_end} - ${fp_begin}")
  string(SUBSTRING "${original}" ${fp_begin} ${fp_length} fp_block)
  file(WRITE "${out}/strata_native_observe_test_helper.hpp"
    "// Generated host-only fixture of the exact pinned FP16 patch.\ninline std::vector<uint32_t> native_q8_test_helper_effective(const std::vector<uint32_t>& input, bool rte16, bool preserve16) {\nstruct Flags {bool float_controls_rte_fp16; bool float_controls_denorm_preserve_fp16;} flags{rte16,preserve16};auto* device=&flags;\nconst void* spv_data=input.data();size_t spv_size=input.size()*4;\nvk::ShaderModuleCreateInfo shader_module_create_info({},spv_size,input.data());std::vector<uint32_t> spirv;const auto CEIL_DIV=[](size_t a,size_t b){return (a+b-1)/b;};\n${fp_block}\nreturn {shader_module_create_info.pCode,shader_module_create_info.pCode+shader_module_create_info.codeSize/4};\n}\n")

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
  target_link_libraries(native_q8_vulkan_component PRIVATE SPIRV-Headers::SPIRV-Headers)
  target_link_libraries(native_q8_vulkan_component PRIVATE ggml-vulkan ggml-base)
  add_custom_command(OUTPUT "${out}/strata_q8_gu_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_q8_gu.comp" -o "${out}/native_q8_gu.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_q8_gu.spv" "-DOUTPUT=${out}/strata_q8_gu_spv.hpp"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_q8_gu_embed.cmake"
    DEPENDS src/kernels/vulkan/native_q8_gu.comp src/kernels/vulkan/native_q8_dot.glsl cmake/strata_q8_gu_embed.cmake VERBATIM)
  add_custom_target(strata_q8_gu_shader DEPENDS "${out}/strata_q8_gu_spv.hpp")
  add_dependencies(ggml-vulkan strata_q8_gu_shader)
  add_executable(native_q8_gu_component tests/kernels/native_q8_gu_component.cpp)
  target_include_directories(native_q8_gu_component PRIVATE src/core)
  target_link_libraries(native_q8_gu_component PRIVATE ggml-vulkan ggml-base)
  add_custom_command(OUTPUT "${out}/strata_q8_down_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_q8_down.comp" -o "${out}/native_q8_down.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_q8_down.spv" "-DOUTPUT=${out}/strata_q8_down_spv.hpp"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_q8_down_embed.cmake"
    DEPENDS src/kernels/vulkan/native_q8_down.comp src/kernels/vulkan/native_q8_dot.glsl cmake/strata_q8_down_embed.cmake VERBATIM)
  add_custom_target(strata_q8_down_shader DEPENDS "${out}/strata_q8_down_spv.hpp")
  add_dependencies(ggml-vulkan strata_q8_down_shader)
  add_executable(native_q8_down_component tests/kernels/native_q8_down_component.cpp)
  target_include_directories(native_q8_down_component PRIVATE src/core)
  target_link_libraries(native_q8_down_component PRIVATE ggml-vulkan ggml-base)
  add_executable(native_q8_composed_component tests/kernels/native_q8_composed_component.cpp)
  target_include_directories(native_q8_composed_component PRIVATE src/core)
  target_link_libraries(native_q8_composed_component PRIVATE ggml-vulkan ggml-base)
  add_executable(native_q8_geometry_component tests/kernels/native_q8_geometry_component.cpp)
  target_include_directories(native_q8_geometry_component PRIVATE src/core)
  target_link_libraries(native_q8_geometry_component PRIVATE ggml-vulkan ggml-base)
  if(STRATA_BUILD_TESTS)
    add_executable(native_q8_gu_shader_test tests/kernels/native_q8_gu_shader_test.cpp)
    target_include_directories(native_q8_gu_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_q8_gu_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan)
    add_dependencies(native_q8_gu_shader_test strata_q8_gu_shader)
    add_test(NAME native_q8_gu_shader_test COMMAND native_q8_gu_shader_test)
    add_executable(native_q8_down_shader_test tests/kernels/native_q8_down_shader_test.cpp)
    target_include_directories(native_q8_down_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_q8_down_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan)
    add_dependencies(native_q8_down_shader_test strata_q8_down_shader)
    add_test(NAME native_q8_down_shader_test COMMAND native_q8_down_shader_test)
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
