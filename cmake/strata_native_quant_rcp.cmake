# Observation candidate only. Keep the original and RN modules unchanged.
# The full F32 scale is clamped before reciprocal and is rounded to half only
# when the original packed metadata store calls half_rne.
function(strata_native_quant_rcp_source base output)
  string(SHA256 digest "${base}")
  if(NOT digest STREQUAL "c23db4f75b3b09a5bf5675f5a91753578255ecc0614f0822de46d72052044256")
    message(FATAL_ERROR "Q8 reciprocal observation: incompatible original shader")
  endif()
  set(scale_old "precise float scale=maxima[lane]/127.0;")
  set(ratio_old "precise float ratio=xi/d;")
  string(FIND "${base}" "${scale_old}" scale_at)
  string(FIND "${base}" "${ratio_old}" ratio_at)
  if(scale_at LESS 0 OR ratio_at LESS 0)
    message(FATAL_ERROR "Q8 reciprocal observation: quantizer anchors missing")
  endif()
  string(REPLACE "${scale_old}" "precise float scale=maxima[lane]*uintBitsToFloat(0x3c010204u);" result "${base}")
  string(REPLACE "${ratio_old}" "precise float reciprocal=1.0/d;\n    precise float ratio=xi*reciprocal;" result "${result}")
  # The reciprocal-only module also accepts a row in workgroup Y.  Y=0 is
  # exactly the legacy T1 address calculation; X remains one warp per block.
  string(FIND "${result}" "else xi=gate[p.gate_bias+i];" input_row_at)
  string(FIND "${result}" "p.packed_bias+gl_WorkGroupID.x*9u" packed_row_at)
  if(input_row_at LESS 0 OR packed_row_at LESS 0)
    message(FATAL_ERROR "Q8 reciprocal observation: token row anchors missing")
  endif()
  string(REPLACE "else xi=gate[p.gate_bias+i];" "else xi=gate[p.gate_bias+gl_WorkGroupID.y*p.elements+i];" result "${result}")
  string(REPLACE "p.packed_bias+gl_WorkGroupID.x*9u" "p.packed_bias+(gl_WorkGroupID.y*(p.elements/32u)+gl_WorkGroupID.x)*9u" result "${result}")
  set(${output} "${result}" PARENT_SCOPE)
endfunction()
