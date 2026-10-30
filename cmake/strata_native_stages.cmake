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
  include(cmake/strata_native_q8_integer_dot.cmake)
  strata_native_q8_dot_patch("${original}" dot_patched)
  file(WRITE "${out}/ggml-vulkan-guard.cpp" "${dot_patched}\n#include \"strata_vk_logits_guard.inc\"\n#include \"strata_vk_native_stages.inc\"\n")
  # Host test executes the actual patched feature-chain block with a mock
  # feature query and creation success/failure; no device or alternative chain.
  set(dot_begin_text "        VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR shader_integer_dot_product_features {};")
  string(FIND "${dot_patched}" "${dot_begin_text}" dot_begin)
  string(FIND "${dot_patched}" "\n        VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE" dot_end)
  if(dot_begin LESS 0 OR dot_end LESS dot_begin)
    message(FATAL_ERROR "native Q8 dot: feature-chain block missing")
  endif()
  math(EXPR dot_length "${dot_end} - ${dot_begin}")
  string(SUBSTRING "${dot_patched}" ${dot_begin} ${dot_length} dot_block)
  file(WRITE "${out}/strata_native_q8_dot_test.hpp"
    "// Generated from the exact patched feature-chain block.\ninline DotReceipt dot_feature_chain(bool request, bool acceleration, bool support, bool create_success) {\nstruct Diagnostic {NativeQ8IntegerDotState strata_q8_dot;} diagnostic;auto* strata_q8_device=&diagnostic;diagnostic.strata_q8_dot.requested=request;\nVkPhysicalDeviceFeatures2 root{};root.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;auto* last_struct=reinterpret_cast<VkBaseOutStructure*>(&root);std::vector<const char*> device_extensions;\n${dot_block}\nconst bool chained=root.pNext==&shader_integer_dot_product_features;\nif(chained)shader_integer_dot_product_features.shaderIntegerDotProduct=support?VK_TRUE:VK_FALSE;\nif(create_success)strata_q8_device->strata_q8_dot.logical_device_created(shader_integer_dot_product_features.shaderIntegerDotProduct != VK_FALSE);\nreturn {diagnostic.strata_q8_dot,chained,request&&acceleration,device_extensions.size()};\n}\n")
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
    add_executable(native_q8_integer_dot_test tests/kernels/native_q8_integer_dot_test.cpp)
    target_include_directories(native_q8_integer_dot_test PRIVATE src/core "${out}")
    target_link_libraries(native_q8_integer_dot_test PRIVATE Vulkan::Vulkan)
    add_test(NAME native_q8_integer_dot_test COMMAND native_q8_integer_dot_test)
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
