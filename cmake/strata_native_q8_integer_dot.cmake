# Only the OFF-by-default standalone native-stage build invokes this patch.
# Never modify upstream source or GGML's acceleration/performance flag.
function(strata_native_q8_dot_replace variable old replacement)
  string(REPLACE "${old}" "" removed "${${variable}}")
  string(LENGTH "${${variable}}" before)
  string(LENGTH "${removed}" after)
  string(LENGTH "${old}" length)
  math(EXPR matches "(${before} - ${after}) / ${length}")
  if(NOT matches EQUAL 1)
    message(FATAL_ERROR "native Q8 dot: expected exactly one pinned patch anchor, found ${matches}: ${old}")
  endif()
  string(REPLACE "${old}" "${replacement}" result "${${variable}}")
  set(${variable} "${result}" PARENT_SCOPE)
endfunction()
function(strata_native_q8_dot_patch original output)
  string(SHA256 digest "${original}")
  if(NOT digest STREQUAL "cbacf747bee9f72cab18706acd1f96e5f4a87abaa178f297388e04c7217e8343")
    message(FATAL_ERROR "native Q8 dot: incompatible pinned backend source")
  endif()
  set(patched "${original}")
  strata_native_q8_dot_replace(patched "vk_device ggml_vk_get_device(size_t idx) {"
    "#include \"native_q8_vk_device.hpp\"\nvk_device ggml_vk_get_device(size_t idx) {")
  strata_native_q8_dot_replace(patched "        vk_device device = std::make_shared<vk_device_struct>();"
    "        auto strata_q8_device = std::make_shared<StrataVkQ8Device>();\n        vk_device device = strata_q8_device;")
  set(performance "        device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;")
  strata_native_q8_dot_replace(patched "${performance}"
    "        strata_q8_device->strata_q8_dot.requested = device->integer_dot_product;\n${performance}")
  strata_native_q8_dot_replace(patched "        if (device->integer_dot_product) {\n            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;"
    "        if (strata_q8_device->strata_q8_dot.requested) {\n            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;")
  set(create "        device->device = device->physical_device.createDevice(device_create_info);")
  strata_native_q8_dot_replace(patched "${create}"
    "${create}\n        strata_q8_device->strata_q8_dot.logical_device_created(shader_integer_dot_product_features.shaderIntegerDotProduct != VK_FALSE);")
  set(${output} "${patched}" PARENT_SCOPE)
endfunction()
