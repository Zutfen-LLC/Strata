# Q8 producer boundary observation

This four-block fixture and two small components compare actual production Q8_1 input producers. They do not establish general numerical parity, authorize a device or change admission. The CUDA expert selector calls `quantize_q8_1_rows`; its projection selector calls `native_quantize_q8_1`. Vulkan's expert selector uses the worker's RN observation; its `gdn-projection` selector uses the GDN input RCP observation. Other model projections may select a different existing variant and are not covered by this pairing. The Vulkan result records an effective SPIR-V file and leaves FP32 denormal behavior unspecified.

Inputs are 128 finite FP32 values (512 bytes), in four blocks. The boundary block's maximum is 254 times the least normal FP32, keeping its nominal scale normal while exposing subnormal values to quantization. The suite does not cover an all-subnormal block whose scale may underflow; the CUDA conversion may then enter an unsupported NaN-to-integer domain. Do not infer a pass for it. Each result is 144 bytes: four 36-byte Q8_1 blocks, each two little-endian binary16 fields (scale and original-input sum), then 32 signed quantized bytes. The comparator displays the exact bytes, refuses malformed files and exits nonzero on any difference.

Build CUDA target `native_q8_boundary_cuda` with the existing production CUDA build recipe and `STRATA_BUILD_TESTS=ON`; it links the actual `strata_kernels` library and its per-source compiler options. Build Vulkan target `native_q8_boundary_vulkan` with `STRATA_VULKAN_NATIVE_STAGES=ON`, `STRATA_VULKAN_NATIVE_Q8_STRICT_FMA=ON`, and the existing pinned GGML and SPIR-V dependencies. Use the existing private Vulkan runtime and select the exact intended physical device. CUDA and Vulkan device indices are independently enumerated: check each index/name before running. Running these commands uses the selected GPU; this task has not run them.

```
python3 tools/native_q8_boundary_fixture.py /path/to/new-fixture
build/native_q8_boundary_cuda CUDA_INDEX CUDA_NAME expert /path/to/new-fixture/all.f32le /path/to/new-cuda-expert.bin
build/native_q8_boundary_vulkan VULKAN_INDEX VULKAN_NAME expert /path/to/new-fixture/all.f32le /path/to/new-vulkan-expert.bin
python3 tools/native_q8_boundary_compare.py --producer expert --cuda /path/to/new-cuda-expert.bin --vulkan /path/to/new-vulkan-expert.bin
build/native_q8_boundary_cuda CUDA_INDEX CUDA_NAME projection /path/to/new-fixture/all.f32le /path/to/new-cuda-projection.bin
build/native_q8_boundary_vulkan VULKAN_INDEX VULKAN_NAME gdn-projection /path/to/new-fixture/all.f32le /path/to/new-vulkan-gdn.bin
python3 tools/native_q8_boundary_compare.py --producer gdn-projection --cuda /path/to/new-cuda-projection.bin --vulkan /path/to/new-vulkan-gdn.bin
```

Use separate new output filenames. The programs refuse an existing result and report execution or device-admission failure. Keep each producer's result separately, including failures; do not substitute a result from another producer. To examine the `STRATA_VERIFY_QDEDUP` byte-identity assumption, compare the two CUDA result files using the same comparator, clearly labeling this as CUDA producer-to-producer rather than Vulkan evidence. This is a direct producer check, not a full model dedup execution.
