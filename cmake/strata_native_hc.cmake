# Standalone observation only; no portable-model option or default integration.
function(strata_native_hc_enable out)
  add_custom_command(OUTPUT "${out}/strata_native_ple_grouped_norm_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_ple_grouped_norm_t1.comp" -o "${out}/native_ple_grouped_norm_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_ple_grouped_norm_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_ple_grouped_norm_t1.spv" "-DOUTPUT=${out}/strata_native_ple_grouped_norm_t1_spv.hpp"
      -DARRAY_NAME=strata_native_ple_grouped_norm_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_ple_grouped_norm_t1.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_ple_grouped_norm_t1_shader DEPENDS "${out}/strata_native_ple_grouped_norm_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_ple_grouped_norm_t1_shader)
  add_executable(native_ple_grouped_norm_component tests/kernels/native_ple_grouped_norm_component.cpp)
  target_include_directories(native_ple_grouped_norm_component PRIVATE src/core)
  target_link_libraries(native_ple_grouped_norm_component PRIVATE ggml-vulkan ggml-base)
  add_custom_command(OUTPUT "${out}/strata_native_ple_gate_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_ple_gate_t1.comp" -o "${out}/native_ple_gate_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_ple_gate_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_ple_gate_t1.spv" "-DOUTPUT=${out}/strata_native_ple_gate_t1_spv.hpp"
      -DARRAY_NAME=strata_native_ple_gate_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_ple_gate_t1.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_ple_gate_t1_shader DEPENDS "${out}/strata_native_ple_gate_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_ple_gate_t1_shader)
  add_executable(native_component_host_bound_test tests/kernels/native_component_host_bound_test.cpp)
  add_dependencies(native_component_host_bound_test strata_native_ple_gate_t1_shader)
  target_include_directories(native_component_host_bound_test PRIVATE src/core "${out}")
  target_link_libraries(native_component_host_bound_test PRIVATE ggml-vulkan ggml-base)
  add_test(NAME native_component_host_bound_test COMMAND native_component_host_bound_test)
  add_executable(native_ple_gate_component tests/kernels/native_ple_gate_component.cpp)
  target_include_directories(native_ple_gate_component PRIVATE src/core)
  target_link_libraries(native_ple_gate_component PRIVATE ggml-vulkan ggml-base)
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
    DEPENDS src/kernels/vulkan/native_hc_mix.comp src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_hc_mix_shader DEPENDS "${out}/strata_native_hc_mix_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_hc_mix_shader)
  add_custom_command(OUTPUT "${out}/strata_native_head_hc_mix_rn_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O -DSTRATA_HC_HEAD_DIV_RN=1
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_hc_mix.comp" -o "${out}/native_head_hc_mix_rn.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_head_hc_mix_rn.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_head_hc_mix_rn.spv" "-DOUTPUT=${out}/strata_native_head_hc_mix_rn_spv.hpp"
      -DARRAY_NAME=strata_native_head_hc_mix_rn_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_hc_mix.comp src/kernels/vulkan/native_f32_div_rn.glsl src/kernels/vulkan/native_head_hc_silu_tail.glsl src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_head_hc_mix_rn_shader DEPENDS "${out}/strata_native_head_hc_mix_rn_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_head_hc_mix_rn_shader)
  add_custom_command(OUTPUT "${out}/strata_native_gdn_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_gdn_t1.comp" -o "${out}/native_gdn_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_gdn_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_gdn_t1.spv" "-DOUTPUT=${out}/strata_native_gdn_t1_spv.hpp"
      -DARRAY_NAME=strata_native_gdn_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_gdn_t1.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_gdn_t1_shader DEPENDS "${out}/strata_native_gdn_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_gdn_t1_shader)
  add_executable(native_gdn_t1_component tests/kernels/native_gdn_t1_component.cpp)
  target_include_directories(native_gdn_t1_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_gdn_t1_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_gdn_t1_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_custom_command(OUTPUT "${out}/strata_native_ple_conv_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_ple_conv_t1.comp" -o "${out}/native_ple_conv_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_ple_conv_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_ple_conv_t1.spv" "-DOUTPUT=${out}/strata_native_ple_conv_t1_spv.hpp"
      -DARRAY_NAME=strata_native_ple_conv_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_ple_conv_t1.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_ple_conv_t1_shader DEPENDS "${out}/strata_native_ple_conv_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_ple_conv_t1_shader)
  add_executable(native_ple_conv_component tests/kernels/native_ple_conv_component.cpp)
  target_include_directories(native_ple_conv_component PRIVATE src/core)
  target_link_libraries(native_ple_conv_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_ple_conv_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_custom_command(OUTPUT "${out}/strata_native_gdn_conv_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_gdn_conv_t1.comp" -o "${out}/native_gdn_conv_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_gdn_conv_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_gdn_conv_t1.spv" "-DOUTPUT=${out}/strata_native_gdn_conv_t1_spv.hpp"
      -DARRAY_NAME=strata_native_gdn_conv_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_gdn_conv_t1.comp src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_gdn_conv_t1_shader DEPENDS "${out}/strata_native_gdn_conv_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_gdn_conv_t1_shader)
  add_executable(native_gdn_conv_t1_component tests/kernels/native_gdn_conv_t1_component.cpp)
  target_include_directories(native_gdn_conv_t1_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_gdn_conv_t1_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_gdn_conv_t1_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_custom_command(OUTPUT "${out}/strata_native_gdn_log1p_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_gdn_log1p.comp" -o "${out}/native_gdn_log1p.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_gdn_log1p.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_gdn_log1p.spv" "-DOUTPUT=${out}/strata_native_gdn_log1p_spv.hpp"
      -DARRAY_NAME=strata_native_gdn_log1p_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_gdn_log1p.comp src/kernels/vulkan/native_gdn_log1p.glsl cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_gdn_log1p_shader DEPENDS "${out}/strata_native_gdn_log1p_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_gdn_log1p_shader)
  add_executable(native_gdn_log1p_component tests/kernels/native_gdn_log1p_component.cpp)
  target_include_directories(native_gdn_log1p_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_gdn_log1p_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_gdn_log1p_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_custom_command(OUTPUT "${out}/strata_native_llvm_log1p_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_llvm_log1p.comp" -o "${out}/native_llvm_log1p.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_llvm_log1p.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_llvm_log1p.spv" "-DOUTPUT=${out}/strata_native_llvm_log1p_spv.hpp"
      -DARRAY_NAME=strata_native_llvm_log1p_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_llvm_log1p.comp src/kernels/vulkan/native_llvm_log1p.glsl cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_llvm_log1p_shader DEPENDS "${out}/strata_native_llvm_log1p_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_llvm_log1p_shader)
  add_executable(native_llvm_log1p_component tests/kernels/native_llvm_log1p_component.cpp)
  target_include_directories(native_llvm_log1p_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_llvm_log1p_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_llvm_log1p_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_custom_command(OUTPUT "${out}/strata_native_gdn_ab_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_gdn_ab.comp" -o "${out}/native_gdn_ab.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_gdn_ab.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_gdn_ab.spv" "-DOUTPUT=${out}/strata_native_gdn_ab_spv.hpp"
      -DARRAY_NAME=strata_native_gdn_ab_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_gdn_ab.comp src/kernels/vulkan/native_llvm_log1p.glsl src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_gdn_ab_shader DEPENDS "${out}/strata_native_gdn_ab_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_gdn_ab_shader)
  add_executable(native_gdn_ab_component tests/kernels/native_gdn_ab_component.cpp)
  target_include_directories(native_gdn_ab_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_gdn_ab_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_gdn_ab_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_custom_command(OUTPUT "${out}/strata_native_qsa_norm_rope_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_qsa_norm_rope.comp" -o "${out}/native_qsa_norm_rope.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_qsa_norm_rope.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_qsa_norm_rope.spv" "-DOUTPUT=${out}/strata_native_qsa_norm_rope_spv.hpp"
      -DARRAY_NAME=strata_native_qsa_norm_rope_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_qsa_norm_rope.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_qsa_norm_rope_shader DEPENDS "${out}/strata_native_qsa_norm_rope_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_qsa_norm_rope_shader)
  add_custom_command(OUTPUT "${out}/strata_native_qsa_indexer_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_qsa_indexer_t1.comp" -o "${out}/native_qsa_indexer_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_qsa_indexer_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_qsa_indexer_t1.spv" "-DOUTPUT=${out}/strata_native_qsa_indexer_t1_spv.hpp"
      -DARRAY_NAME=strata_native_qsa_indexer_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_qsa_indexer_t1.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_qsa_indexer_t1_shader DEPENDS "${out}/strata_native_qsa_indexer_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_qsa_indexer_t1_shader)
  add_custom_command(OUTPUT "${out}/strata_native_qsa_attn_chunk_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_qsa_attn_chunk_t1.comp" -o "${out}/native_qsa_attn_chunk_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_qsa_attn_chunk_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_qsa_attn_chunk_t1.spv" "-DOUTPUT=${out}/strata_native_qsa_attn_chunk_t1_spv.hpp" -DARRAY_NAME=strata_native_qsa_attn_chunk_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_qsa_attn_chunk_t1.comp src/kernels/vulkan/native_f32_div_rn.glsl src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_qsa_attn_chunk_t1_shader DEPENDS "${out}/strata_native_qsa_attn_chunk_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_qsa_attn_chunk_t1_shader)
  add_custom_command(OUTPUT "${out}/strata_native_qsa_attn_merge_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_qsa_attn_merge_t1.comp" -o "${out}/native_qsa_attn_merge_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_qsa_attn_merge_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_qsa_attn_merge_t1.spv" "-DOUTPUT=${out}/strata_native_qsa_attn_merge_t1_spv.hpp" -DARRAY_NAME=strata_native_qsa_attn_merge_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_qsa_attn_merge_t1.comp src/kernels/vulkan/native_f32_div_rn.glsl src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_qsa_attn_merge_t1_shader DEPENDS "${out}/strata_native_qsa_attn_merge_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_qsa_attn_merge_t1_shader)
  add_custom_command(OUTPUT "${out}/strata_native_qsa_attn_gate_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_qsa_attn_gate_t1.comp" -o "${out}/native_qsa_attn_gate_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_qsa_attn_gate_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_qsa_attn_gate_t1.spv" "-DOUTPUT=${out}/strata_native_qsa_attn_gate_t1_spv.hpp" -DARRAY_NAME=strata_native_qsa_attn_gate_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_qsa_attn_gate_t1.comp src/kernels/vulkan/native_f32_div_rn.glsl src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_qsa_attn_gate_t1_shader DEPENDS "${out}/strata_native_qsa_attn_gate_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_qsa_attn_gate_t1_shader)
  add_custom_command(OUTPUT "${out}/strata_native_qsa_scores_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_qsa_scores_t1.comp" -o "${out}/native_qsa_scores_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_qsa_scores_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_qsa_scores_t1.spv" "-DOUTPUT=${out}/strata_native_qsa_scores_t1_spv.hpp"
      -DARRAY_NAME=strata_native_qsa_scores_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_qsa_scores_t1.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_qsa_scores_t1_shader DEPENDS "${out}/strata_native_qsa_scores_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_qsa_scores_t1_shader)
  add_custom_command(OUTPUT "${out}/strata_native_qsa_select_t1_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_qsa_select_t1.comp" -o "${out}/native_qsa_select_t1.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_qsa_select_t1.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_qsa_select_t1.spv" "-DOUTPUT=${out}/strata_native_qsa_select_t1_spv.hpp"
      -DARRAY_NAME=strata_native_qsa_select_t1_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_qsa_select_t1.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_qsa_select_t1_shader DEPENDS "${out}/strata_native_qsa_select_t1_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_qsa_select_t1_shader)
  add_executable(native_qsa_attention_component tests/kernels/native_qsa_attention_component.cpp)
  target_include_directories(native_qsa_attention_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_qsa_attention_component PRIVATE ggml-vulkan ggml-base)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(native_qsa_attention_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_executable(native_qsa_select_component tests/kernels/native_qsa_select_component.cpp)
  target_include_directories(native_qsa_select_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_qsa_select_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_qsa_select_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_executable(native_qsa_indexer_component tests/kernels/native_qsa_indexer_component.cpp)
  target_include_directories(native_qsa_indexer_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_qsa_indexer_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_qsa_indexer_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_executable(native_qsa_norm_rope_component tests/kernels/native_qsa_norm_rope_component.cpp)
  if(STRATA_BUILD_TESTS)
    add_executable(native_ple_grouped_norm_reference_test tests/kernels/native_ple_grouped_norm_reference_test.cpp)
    if(MSVC)
      target_compile_options(native_ple_grouped_norm_reference_test PRIVATE /fp:strict)
    else()
      target_compile_options(native_ple_grouped_norm_reference_test PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME native_ple_grouped_norm_reference_test COMMAND native_ple_grouped_norm_reference_test)
    add_executable(native_ple_grouped_norm_dispatch_test tests/kernels/native_ple_grouped_norm_dispatch_test.cpp)
    target_include_directories(native_ple_grouped_norm_dispatch_test PRIVATE src/core "${out}" "${vk_source}/..")
    target_link_libraries(native_ple_grouped_norm_dispatch_test PRIVATE ggml-base)
    add_dependencies(native_ple_grouped_norm_dispatch_test strata_native_ple_grouped_norm_t1_shader)
    add_test(NAME native_ple_grouped_norm_dispatch_test COMMAND native_ple_grouped_norm_dispatch_test)
    add_executable(native_ple_grouped_norm_effective_shader_test tests/kernels/native_ple_grouped_norm_effective_shader_test.cpp)
    target_include_directories(native_ple_grouped_norm_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_ple_grouped_norm_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan)
    add_dependencies(native_ple_grouped_norm_effective_shader_test strata_native_ple_grouped_norm_t1_shader)
    add_test(NAME native_ple_grouped_norm_effective_shader_test COMMAND native_ple_grouped_norm_effective_shader_test)
    add_test(NAME native_ple_grouped_norm_startup_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_ple_grouped_norm_startup.py" "$<TARGET_FILE:native_ple_grouped_norm_component>" "${CMAKE_CURRENT_BINARY_DIR}")
    add_executable(native_ple_gate_reference_test tests/kernels/native_ple_gate_reference_test.cpp)
    if(MSVC)
      target_compile_options(native_ple_gate_reference_test PRIVATE /fp:strict)
    else()
      target_compile_options(native_ple_gate_reference_test PRIVATE -fno-fast-math -ffp-contract=off)
    endif()
    add_test(NAME native_ple_gate_reference_test COMMAND native_ple_gate_reference_test)
    add_executable(native_ple_gate_dispatch_test tests/kernels/native_ple_gate_dispatch_test.cpp)
    target_include_directories(native_ple_gate_dispatch_test PRIVATE src/core "${out}" "${vk_source}/..")
    target_link_libraries(native_ple_gate_dispatch_test PRIVATE ggml-base)
    add_dependencies(native_ple_gate_dispatch_test strata_native_ple_gate_t1_shader)
    add_test(NAME native_ple_gate_dispatch_test COMMAND native_ple_gate_dispatch_test)
    add_executable(native_ple_gate_effective_shader_test tests/kernels/native_ple_gate_effective_shader_test.cpp)
    target_include_directories(native_ple_gate_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_ple_gate_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan)
    add_dependencies(native_ple_gate_effective_shader_test strata_native_ple_gate_t1_shader)
    add_test(NAME native_ple_gate_effective_shader_test COMMAND native_ple_gate_effective_shader_test)
    add_test(NAME native_ple_gate_startup_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_ple_gate_startup.py" "$<TARGET_FILE:native_ple_gate_component>" "${CMAKE_CURRENT_BINARY_DIR}")
    if(STRATA_SPIRV_DIS)
      add_test(NAME native_ple_gate_shader_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_ple_gate_shader.py"
        --shader "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_ple_gate_t1.comp"
        --glslc "${Vulkan_GLSLC_EXECUTABLE}" --validator "${STRATA_SPIRV_VAL}"
        --disassembler "${STRATA_SPIRV_DIS}" --scratch "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    add_test(NAME native_qsa_norm_rope_output_contract_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation-harness/test_qsa_producer_contract.py"
      "$<TARGET_FILE:native_qsa_norm_rope_input_test>" "$<TARGET_FILE:native_qsa_norm_rope_component>")
  endif()

  target_include_directories(native_qsa_norm_rope_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_qsa_norm_rope_component PRIVATE ggml-vulkan ggml-base)
  if(NOT MSVC)
    target_compile_options(native_qsa_norm_rope_component PRIVATE -fno-fast-math -ffp-contract=off)
  endif()
  add_executable(native_head_hc_component tests/kernels/native_head_hc_component.cpp)
  target_include_directories(native_head_hc_component PRIVATE include src/core tests/kernels)
  target_link_libraries(native_head_hc_component PRIVATE ggml-vulkan ggml-base)
  add_executable(native_hc_mix_component tests/kernels/native_hc_mix_component.cpp)
  target_include_directories(native_hc_mix_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_hc_mix_component PRIVATE ggml-vulkan ggml-base)
  add_executable(native_hc_component tests/kernels/native_hc_component.cpp)
  target_include_directories(native_hc_component PRIVATE src/core)
  target_link_libraries(native_hc_component PRIVATE ggml-vulkan ggml-base)
  add_custom_command(OUTPUT "${out}/strata_native_hc_pending_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_hc_pending.comp" -o "${out}/native_hc_pending.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_hc_pending.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_hc_pending.spv" "-DOUTPUT=${out}/strata_native_hc_pending_spv.hpp"
      -DARRAY_NAME=strata_native_hc_pending_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_hc_pending.comp src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_hc_pending_shader DEPENDS "${out}/strata_native_hc_pending_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_hc_pending_shader)
  add_custom_command(OUTPUT "${out}/strata_native_hc_write_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -DSTRATA_HC_FINAL_WRITE=1 -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_hc_pending.comp" -o "${out}/native_hc_write.spv"
    COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${out}/native_hc_write.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/native_hc_write.spv" "-DOUTPUT=${out}/strata_native_hc_write_spv.hpp"
      -DARRAY_NAME=strata_native_hc_write_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_hc_pending.comp src/core/native_hc_mix_limits.h cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_hc_write_shader DEPENDS "${out}/strata_native_hc_write_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_hc_write_shader)
  add_executable(native_hc_write_component tests/kernels/native_hc_write_component.cpp)
  target_include_directories(native_hc_write_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_hc_write_component PRIVATE ggml-vulkan ggml-base)
  add_executable(native_hc_pending_component tests/kernels/native_hc_pending_component.cpp)
  target_include_directories(native_hc_pending_component PRIVATE src/core tests/kernels)
  target_link_libraries(native_hc_pending_component PRIVATE ggml-vulkan ggml-base)
  if(STRATA_BUILD_TESTS)
    add_executable(native_gdn_t1_effective_shader_test tests/kernels/native_gdn_t1_effective_shader_test.cpp)
    target_include_directories(native_gdn_t1_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_gdn_t1_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_gdn_t1_effective_shader_test strata_native_gdn_t1_shader)
    add_test(NAME native_gdn_t1_effective_shader_test COMMAND native_gdn_t1_effective_shader_test)
    add_executable(native_gdn_conv_t1_effective_shader_test tests/kernels/native_gdn_conv_t1_effective_shader_test.cpp)
    target_include_directories(native_gdn_conv_t1_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_gdn_conv_t1_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_gdn_conv_t1_effective_shader_test strata_native_gdn_conv_t1_shader)
    add_test(NAME native_gdn_conv_t1_effective_shader_test COMMAND native_gdn_conv_t1_effective_shader_test)
    add_executable(native_qsa_attention_effective_shader_test tests/kernels/native_qsa_attention_effective_shader_test.cpp)
    target_include_directories(native_qsa_attention_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_qsa_attention_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_qsa_attention_effective_shader_test strata_native_qsa_attn_chunk_t1_shader strata_native_qsa_attn_merge_t1_shader strata_native_qsa_attn_gate_t1_shader)
    add_test(NAME native_qsa_attention_effective_shader_test COMMAND native_qsa_attention_effective_shader_test)
    add_test(NAME native_qsa_attention_output_contract_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation-harness/test_qsa_attention_output_contract.py" "$<TARGET_FILE:native_qsa_attention_input_test>" "$<TARGET_FILE:native_qsa_attention_component>")
    add_executable(native_qsa_select_effective_shader_test tests/kernels/native_qsa_select_effective_shader_test.cpp)
    target_include_directories(native_qsa_select_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_qsa_select_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_qsa_select_effective_shader_test strata_native_qsa_scores_t1_shader strata_native_qsa_select_t1_shader)
    add_test(NAME native_qsa_select_effective_shader_test COMMAND native_qsa_select_effective_shader_test)
    add_test(NAME native_qsa_select_output_contract_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation-harness/test_qsa_select_output_contract.py"
      "$<TARGET_FILE:native_qsa_select_input_test>" "$<TARGET_FILE:native_qsa_select_component>")
    add_executable(native_qsa_indexer_effective_shader_test tests/kernels/native_qsa_indexer_effective_shader_test.cpp)
    target_include_directories(native_qsa_indexer_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_qsa_indexer_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_qsa_indexer_effective_shader_test strata_native_qsa_indexer_t1_shader)
    add_test(NAME native_qsa_indexer_effective_shader_test COMMAND native_qsa_indexer_effective_shader_test)
    add_test(NAME native_qsa_indexer_output_contract_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation-harness/test_qsa_indexer_output_contract.py"
      "$<TARGET_FILE:native_qsa_indexer_input_test>" "$<TARGET_FILE:native_qsa_indexer_component>")
    add_executable(native_qsa_norm_rope_effective_shader_test tests/kernels/native_qsa_norm_rope_effective_shader_test.cpp)
    target_include_directories(native_qsa_norm_rope_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_qsa_norm_rope_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_qsa_norm_rope_effective_shader_test strata_native_qsa_norm_rope_shader)
    add_test(NAME native_qsa_norm_rope_effective_shader_test COMMAND native_qsa_norm_rope_effective_shader_test)
    add_executable(native_gdn_log1p_effective_shader_test tests/kernels/native_gdn_log1p_effective_shader_test.cpp)
    target_include_directories(native_gdn_log1p_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_gdn_log1p_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_gdn_log1p_effective_shader_test strata_native_gdn_log1p_shader)
    add_test(NAME native_gdn_log1p_effective_shader_test COMMAND native_gdn_log1p_effective_shader_test)
    add_executable(native_llvm_log1p_effective_shader_test tests/kernels/native_llvm_log1p_effective_shader_test.cpp)
    target_include_directories(native_llvm_log1p_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_llvm_log1p_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_llvm_log1p_effective_shader_test strata_native_llvm_log1p_shader)
    add_test(NAME native_llvm_log1p_effective_shader_test COMMAND native_llvm_log1p_effective_shader_test)
    add_executable(native_gdn_ab_effective_shader_test tests/kernels/native_gdn_ab_effective_shader_test.cpp)
    target_include_directories(native_gdn_ab_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_gdn_ab_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_gdn_ab_effective_shader_test strata_native_gdn_ab_shader)
    add_test(NAME native_gdn_ab_effective_shader_test COMMAND native_gdn_ab_effective_shader_test)
  endif()
  if(STRATA_BUILD_TESTS)
    add_executable(native_ple_conv_dispatch_test tests/kernels/native_ple_conv_dispatch_test.cpp)
    target_include_directories(native_ple_conv_dispatch_test PRIVATE src/core "${out}" "${vk_source}/..")
    target_link_libraries(native_ple_conv_dispatch_test PRIVATE ggml-base)
    add_dependencies(native_ple_conv_dispatch_test strata_native_ple_conv_t1_shader)
    add_test(NAME native_ple_conv_dispatch_test COMMAND native_ple_conv_dispatch_test)
    add_executable(native_ple_conv_effective_shader_test tests/kernels/native_ple_conv_effective_shader_test.cpp)
    target_include_directories(native_ple_conv_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_ple_conv_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_ple_conv_effective_shader_test strata_native_ple_conv_t1_shader)
    add_test(NAME native_ple_conv_effective_shader_test COMMAND native_ple_conv_effective_shader_test)
    add_test(NAME native_ple_conv_observation_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_ple_conv_observation.py"
      --shader "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_ple_conv_t1.comp"
      --glslc "${Vulkan_GLSLC_EXECUTABLE}" --validator "${STRATA_SPIRV_VAL}"
      --disassembler "${STRATA_SPIRV_DIS}" --scratch "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME native_hc_write_fixture_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_write_fixture.py" "$<TARGET_FILE:native_hc_write_component>")
    add_test(NAME native_hc_write_extract_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_write_extract.py")
    add_executable(native_hc_write_input_test tests/kernels/native_hc_write_input_test.cpp)
    target_include_directories(native_hc_write_input_test PRIVATE src/core tests/kernels)
    target_link_libraries(native_hc_write_input_test PRIVATE ggml-base)
    if(NOT MSVC)
      target_compile_options(native_hc_write_input_test PRIVATE -fno-fast-math -ffp-contract=off -frounding-math)
    endif()
    add_test(NAME native_hc_write_input_test COMMAND native_hc_write_input_test)
    add_test(NAME native_hc_write_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_write.py")
    add_executable(native_hc_write_effective_shader_test tests/kernels/native_hc_write_effective_shader_test.cpp)
    target_include_directories(native_hc_write_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_hc_write_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_hc_write_effective_shader_test strata_native_hc_write_shader)
    add_test(NAME native_hc_write_effective_shader_test COMMAND native_hc_write_effective_shader_test)
    add_test(NAME native_hc_pending_fixture_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_pending_fixture.py" "$<TARGET_FILE:native_hc_pending_component>")
    add_test(NAME native_hc_pending_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_pending.py")
    add_executable(native_hc_pending_effective_shader_test tests/kernels/native_hc_pending_effective_shader_test.cpp)
    target_include_directories(native_hc_pending_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_hc_pending_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_hc_pending_effective_shader_test strata_native_hc_pending_shader)
    add_test(NAME native_hc_pending_effective_shader_test COMMAND native_hc_pending_effective_shader_test)
    add_test(NAME native_hc_mix_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_mix.py")
    add_test(NAME native_hc_compare_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_compare.py")
    add_test(NAME native_hc_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_hc_numeric.py")
    add_executable(native_hc_mix_effective_shader_test tests/kernels/native_hc_mix_effective_shader_test.cpp)
    target_include_directories(native_hc_mix_effective_shader_test PRIVATE src/core "${out}")
    target_link_libraries(native_hc_mix_effective_shader_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_hc_mix_effective_shader_test strata_native_hc_mix_shader)
    add_test(NAME native_hc_mix_effective_shader_test COMMAND native_hc_mix_effective_shader_test)
    add_executable(native_head_hc_mix_rn_test tests/kernels/native_head_hc_mix_rn_test.cpp)
    target_include_directories(native_head_hc_mix_rn_test PRIVATE src/core "${out}")
    target_link_libraries(native_head_hc_mix_rn_test PRIVATE SPIRV-Headers::SPIRV-Headers Vulkan::Vulkan ggml-base)
    add_dependencies(native_head_hc_mix_rn_test strata_native_head_hc_mix_rn_shader)
    add_test(NAME native_head_hc_mix_rn_test COMMAND native_head_hc_mix_rn_test)
    add_test(NAME native_head_hc_fixture_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_fixture.py" "$<TARGET_FILE:native_head_hc_fixture_tool>" "$<TARGET_FILE:native_head_hc_component>")
    add_test(NAME native_head_hc_wrapper_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_wrapper.py" "$<TARGET_FILE:native_head_hc_fixture_tool>" "${STRATA_GGML_DIR}/ggml")
    add_test(NAME native_head_hc_exact_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_numeric.py" "$<TARGET_FILE:native_head_hc_fixture_tool>")
    add_test(NAME native_head_hc_contract_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_contract.py" "$<TARGET_FILE:native_head_hc_fixture_tool>")
    add_test(NAME native_head_hc_rne_operands_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_rne.py")
    add_test(NAME native_head_hc_overflow_numeric_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_overflow.py")
    add_test(NAME native_head_hc_tail_fixture_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_tail_fixture.py" "$<TARGET_FILE:native_hc_mix_component>")
    add_test(NAME native_head_hc_tail_compare_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_tail_compare.py")
    add_test(NAME native_head_hc_primitive_report_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_primitive_report.py")
    add_test(NAME native_head_hc_primitive_fake_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_head_hc_primitive_fake.py")
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
