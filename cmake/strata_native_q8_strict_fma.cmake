# Only the explicit strict diagnostic build invokes this module. The existing
# native-stage pin gate runs first. Never patch upstream files or default shaders.
function(strata_native_q8_strict_fma_patch original output)
  set(patched "${original}")
  # The pinned factory publishes its shared_ptr before querying features.
  # Strict admission introduces exceptions there, so remove failed objects from
  # that cache and never invoke Vulkan destruction without a logical device.
  strata_native_q8_dot_replace(patched "        vk_instance.devices[idx] = device;"
    "        vk_instance.devices[idx] = device;\n        try {")
  set(factory_tail "        return device;\n    }\n\n    return vk_instance.devices[idx];\n}")
  set(safe_tail "        } catch (...) {\n            vk_instance.devices[idx].reset();\n            throw;\n        }\n${factory_tail}")
  strata_native_q8_dot_replace(patched "${factory_tail}" "${safe_tail}")
  strata_native_q8_dot_replace(patched "vk_device_struct::~vk_device_struct() {"
    "vk_device_struct::~vk_device_struct() {\n    if (!device) return; // strict diagnostic: no logical device, no Vulkan resources")
  set(query "        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);")
  set(fma_chain [=[        VkPhysicalDeviceShaderFmaFeaturesKHR strata_fma_features{};
        strata_fma_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR;
        const bool strata_fma_extension = std::any_of(ext_props.begin(), ext_props.end(), [](const auto& ext) {
            return strcmp(ext.extensionName, VK_KHR_SHADER_FMA_EXTENSION_NAME) == 0;
        });
        if (strata_fma_extension) {
            last_struct->pNext = reinterpret_cast<VkBaseOutStructure*>(&strata_fma_features);
            last_struct = reinterpret_cast<VkBaseOutStructure*>(&strata_fma_features);
            device_extensions.push_back(VK_KHR_SHADER_FMA_EXTENSION_NAME);
        }
]=])
  set(fma_admit [=[        strata_q8_device->strata_q8_fma.queried(strata_fma_extension, strata_fma_features.shaderFmaFloat32 != VK_FALSE);
        // Request only FP32; queried support for other widths is not admission.
        strata_fma_features.shaderFmaFloat16 = VK_FALSE;
        strata_fma_features.shaderFmaFloat64 = VK_FALSE;
]=])
  strata_native_q8_dot_replace(patched "${query}" "${fma_chain}${query}\n${fma_admit}")
  set(create "        device->device = device->physical_device.createDevice(device_create_info);")
  strata_native_q8_dot_replace(patched "${create}" "${create}\n        strata_q8_device->strata_q8_fma.logical_device_created();")
  set(${output} "${patched}" PARENT_SCOPE)
  # Host fixture executes these exact inserted blocks, with the query/creation
  # boundaries mocked. It observes the same extension list and pNext chain.
  set(STRATA_FMA_CHAIN "${fma_chain}" PARENT_SCOPE)
  set(STRATA_FMA_ADMIT "${fma_admit}" PARENT_SCOPE)
  # Exercise the actual allocation/cache/rollback/destructor blocks with mocked
  # Vulkan boundaries. The rest of initialization is irrelevant to this failure.
  string(FIND "${patched}" "vk_device ggml_vk_get_device(size_t idx) {" factory_begin)
  string(FIND "${patched}" "        device->memory_logger =" factory_body)
  math(EXPR factory_prefix_length "${factory_body} - ${factory_begin}")
  string(SUBSTRING "${patched}" ${factory_begin} ${factory_prefix_length} factory_prefix)
  set(STRATA_FMA_FACTORY_PREFIX "${factory_prefix}" PARENT_SCOPE)
  set(STRATA_FMA_FACTORY_TAIL "${safe_tail}" PARENT_SCOPE)
  string(FIND "${patched}" "vk_device_struct::~vk_device_struct() {" destructor_begin)
  string(FIND "${patched}" "\nvoid vk_perf_logger::print_timings" destructor_end)
  math(EXPR destructor_length "${destructor_end} - ${destructor_begin}")
  string(SUBSTRING "${patched}" ${destructor_begin} ${destructor_length} destructor)
  set(STRATA_FMA_DESTRUCTOR "${destructor}" PARENT_SCOPE)
  set(STRATA_FMA_CREATE "${create}\n        strata_q8_device->strata_q8_fma.logical_device_created();" PARENT_SCOPE)
endfunction()

function(strata_native_q8_strict_fma_enable out)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${Vulkan_INCLUDE_DIRS}")
  check_cxx_source_compiles("#include <vulkan/vulkan.h>\nint main(){VkPhysicalDeviceShaderFmaFeaturesKHR f{}; f.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FMA_FEATURES_KHR; f.shaderFmaFloat32=VK_TRUE;return 0;}" STRATA_HAS_SHADER_FMA_HEADERS)
  if(NOT STRATA_HAS_SHADER_FMA_HEADERS)
    message(FATAL_ERROR "native Q8 strict FMA requires Vulkan headers declaring VkPhysicalDeviceShaderFmaFeaturesKHR and VK_KHR_shader_fma")
  endif()
  find_program(STRATA_SPIRV_VAL spirv-val REQUIRED)
  find_program(STRATA_SPIRV_DIS spirv-dis REQUIRED)
  # Compile the actual consumers now: an unavailable compiler/validator extension
  # is a configure error with its exact diagnostic, never an opcode workaround.
  foreach(stage gu down)
    set(shader "${out}/native_q8_${stage}_strict_fma.spv")
    execute_process(COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O
      -DSTRATA_NATIVE_Q8_STRICT_FMA=1 "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_q8_${stage}.comp" -o "${shader}"
      RESULT_VARIABLE compiled OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT compiled EQUAL 0)
      message(FATAL_ERROR "native Q8 strict FMA requires glslc GL_EXT_spirv_intrinsics/SPV_KHR_fma support: ${output}${error}")
    endif()
    execute_process(COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${shader}"
      RESULT_VARIABLE validated OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT validated EQUAL 0)
      message(FATAL_ERROR "native Q8 strict FMA requires spirv-val SPV_KHR_fma support: ${output}${error}")
    endif()
    execute_process(COMMAND "${STRATA_SPIRV_DIS}" "${shader}" RESULT_VARIABLE disassembled OUTPUT_VARIABLE assembly ERROR_VARIABLE error)
    if(NOT disassembled EQUAL 0 OR NOT assembly MATCHES "OpFmaKHR" OR NOT assembly MATCHES "OpCapability FMAKHR" OR NOT assembly MATCHES "SPV_KHR_fma")
      message(FATAL_ERROR "native Q8 strict FMA: compiler/disassembler did not confirm OpFmaKHR/FMAKHR/SPV_KHR_fma: ${error}")
    endif()
    add_custom_command(OUTPUT "${out}/strata_q8_${stage}_strict_fma_spv.hpp"
      COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -fshader-stage=compute --target-env=vulkan1.2 -O -DSTRATA_NATIVE_Q8_STRICT_FMA=1
        "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_q8_${stage}.comp" -o "${shader}"
      COMMAND "${STRATA_SPIRV_VAL}" --target-env vulkan1.2 "${shader}"
      COMMAND "${CMAKE_COMMAND}" "-DINPUT=${shader}" "-DOUTPUT=${out}/strata_q8_${stage}_strict_fma_spv.hpp"
        "-DARRAY_NAME=strata_q8_${stage}_strict_fma_spv" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/strata_native_stages_embed.cmake"
      DEPENDS src/kernels/vulkan/native_q8_${stage}.comp src/kernels/vulkan/native_q8_dot.glsl src/kernels/vulkan/native_q8_abi.h cmake/strata_native_stages_embed.cmake VERBATIM)
    add_custom_target(strata_q8_${stage}_strict_fma_shader DEPENDS "${out}/strata_q8_${stage}_strict_fma_spv.hpp")
    add_dependencies(ggml-vulkan strata_q8_${stage}_strict_fma_shader)
  endforeach()
  target_compile_definitions(ggml-vulkan PRIVATE STRATA_NATIVE_Q8_STRICT_FMA=1)
endfunction()
