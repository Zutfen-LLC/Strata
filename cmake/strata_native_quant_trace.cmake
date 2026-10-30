# Separate instrumented build output. The checked-in strict/observation source
# is never edited. No arithmetic, precise qualifiers or compilation flags change.
function(strata_native_quant_trace_source base stores output)
  string(SHA256 digest "${base}")
  if(NOT digest STREQUAL "c23db4f75b3b09a5bf5675f5a91753578255ecc0614f0822de46d72052044256")
    message(FATAL_ERROR "Q8 quant trace: incompatible original shader")
  endif()
  set(anchor "    float clamped=rounded>127.0?127.0:rounded< -127.0?-127.0:rounded;")
  string(REPLACE "${anchor}" "${stores}${anchor}" result "${base}")
  set(${output} "${result}" PARENT_SCOPE)
endfunction()
