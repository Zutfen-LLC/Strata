function(strata_llvm_log1p_oracle out)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_llvm_log1p_oracle.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/vulkan/native_llvm_log1p.glsl")
  execute_process(COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_llvm_log1p_oracle.py"
    "${CMAKE_CURRENT_SOURCE_DIR}" "${out}" COMMAND_ERROR_IS_FATAL ANY)
endfunction()
