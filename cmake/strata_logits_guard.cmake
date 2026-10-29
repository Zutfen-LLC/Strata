# One opt-in portable head bridge, not a backend patch/extension framework.
# Exact private-source hashes are required because the local include uses the
# pinned backend's internal pipeline/queue/buffer contracts. No input is edited.
function(strata_logits_guard_validate GGML_ROOT)
  set(paths src/ggml-vulkan/ggml-vulkan.cpp src/ggml-vulkan/ggml-vulkan-common.h
      src/ggml-vulkan/ggml-vulkan-types.h src/ggml-vulkan/ggml-vulkan-push-constants.h
      src/ggml-backend-impl.h src/ggml-impl.h include/ggml.h
      src/ggml-vulkan/ggml-vulkan-buffers.cpp include/ggml-backend.h include/ggml-alloc.h)
  set(hashes cbacf747bee9f72cab18706acd1f96e5f4a87abaa178f297388e04c7217e8343
      6ee618834ccff51a9d7ee71e26e776016dc8e408366a6d3335feeb685abef619
      019ba142b5821e6ce0d78b619e0f07ff6dcf1b764172589e6859e9bc20648494
      45cbd017baafef4b1a497619ad3f368d34355e0d65cae7af3005bd24d7322405
      683e0583268e91e1f50b4ffc416b6523e0cec2dfdb3984c89792bda29e32566b
      43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3
      12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9
      d821d92595086242e762c4bb4aa981203f4787740bc0862de87f03ec13599c4e
      46d84cb998105f871240864fd0f55446939a2fe86c5c281afa63a010fb1f65a2
      94e4cd069b9313b2ceb35dacec901981e0bb478d8bb31035b7126be091998c23)
  list(LENGTH paths n)
  math(EXPR last "${n} - 1")
  foreach(i RANGE 0 ${last})
    list(GET paths ${i} path)
    list(GET hashes ${i} expected)
    if(NOT EXISTS "${GGML_ROOT}/${path}")
      message(FATAL_ERROR "integer logits guard: incompatible pinned source; missing ${path}")
    endif()
    file(SHA256 "${GGML_ROOT}/${path}" actual)
    if(NOT actual STREQUAL expected)
      message(FATAL_ERROR "integer logits guard: incompatible pinned source ${path}; expected ${expected}, got ${actual}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GGML_ROOT}/${path}")
  endforeach()
endfunction()
function(strata_enable_logits_guard)
  get_target_property(vk_source ggml-vulkan SOURCE_DIR)
  get_filename_component(ggml_root "${vk_source}/../.." ABSOLUTE)
  strata_logits_guard_validate("${ggml_root}")
  set(out "${CMAKE_CURRENT_BINARY_DIR}/strata-logits-guard")
  file(MAKE_DIRECTORY "${out}")
  file(READ "${vk_source}/ggml-vulkan.cpp" original)
  file(WRITE "${out}/ggml-vulkan-guard.cpp" "${original}\n#include \"strata_vk_logits_guard.inc\"\n")
  get_target_property(sources ggml-vulkan SOURCES)
  set(replaced 0)
  set(kept "")
  foreach(source IN LISTS sources)
    get_filename_component(absolute "${source}" ABSOLUTE BASE_DIR "${vk_source}")
    if(absolute STREQUAL "${vk_source}/ggml-vulkan.cpp")
      math(EXPR replaced "${replaced} + 1")
    else()
      list(APPEND kept "${source}")
    endif()
  endforeach()
  if(NOT replaced EQUAL 1)
    message(FATAL_ERROR "integer logits guard: expected exactly one pinned Vulkan translation unit")
  endif()
  set_property(TARGET ggml-vulkan PROPERTY SOURCES "${kept};${out}/ggml-vulkan-guard.cpp")
  target_include_directories(ggml-vulkan PRIVATE "${vk_source}" "${CMAKE_CURRENT_SOURCE_DIR}/src/core" "${out}")
  add_custom_command(OUTPUT "${out}/strata_logits_guard_spv.hpp"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/logits_nonfinite.comp" -o "${out}/logits_nonfinite.spv"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${out}/logits_nonfinite.spv" "-DOUTPUT=${out}/strata_logits_guard_spv.hpp"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_logits_guard_embed.cmake"
    DEPENDS src/kernels/vulkan/logits_nonfinite.comp cmake/strata_logits_guard_embed.cmake VERBATIM)
  add_custom_target(strata_logits_guard_shader DEPENDS "${out}/strata_logits_guard_spv.hpp")
  add_dependencies(ggml-vulkan strata_logits_guard_shader)
  if(STRATA_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME portable_logits_guard_build_test COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/test_logits_guard_build.py" "${ggml_root}" "${out}" "${CMAKE_COMMAND}")
  endif()
endfunction()
