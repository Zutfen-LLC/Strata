# Uses the existing explicit strict observation build. Router arithmetic itself
# has no FMA; do not inject one or claim Exp/FDiv bit identity across vendors.
function(strata_native_router_enable out)
  find_program(STRATA_ROUTER_SPIRV_VAL spirv-val REQUIRED)
  find_program(STRATA_ROUTER_SPIRV_DIS spirv-dis REQUIRED)
  set(shader "${out}/native_router.spv")
  add_custom_command(OUTPUT "${out}/strata_native_router_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_router.comp" -o "${shader}"
    COMMAND "${STRATA_ROUTER_SPIRV_VAL}" --target-env vulkan1.2 "${shader}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${shader}" "-DOUTPUT=${out}/strata_native_router_spv.hpp"
      -DARRAY_NAME=strata_native_router_spv -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
    DEPENDS src/kernels/vulkan/native_router.comp cmake/strata_native_stages_embed.cmake VERBATIM)
  add_custom_target(strata_native_router_shader DEPENDS "${out}/strata_native_router_spv.hpp")
  add_dependencies(ggml-vulkan strata_native_router_shader)
  add_executable(native_router_component tests/kernels/native_router_component.cpp)
  target_include_directories(native_router_component PRIVATE src/core)
  target_link_libraries(native_router_component PRIVATE ggml-vulkan ggml-base)
  if(STRATA_BUILD_TESTS)
    add_test(NAME native_router_shader_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/test_native_router_shader.py"
      "${Vulkan_GLSLC_EXECUTABLE}" "${STRATA_ROUTER_SPIRV_DIS}" "${STRATA_ROUTER_SPIRV_VAL}")
  endif()
endfunction()
