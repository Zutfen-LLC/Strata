# Layer-1 PLE value projection

`--native-ple-value-t1 PREFIX` is an experimental, default-off replacement for
the value projection in the serial layer-1 PLE path. It requires a strict
accelerator, resident attention weights, eager T1 execution and a fresh receipt
prefix. The Q8_0 `ple_value.weight` role is converted to BF16 using the same
Q8_0 dequantization and nearest-even BF16 conversion as the native compatible
pack. The original F32 gathered n-gram embedding feeds the existing BF16/F32
MMVF matrix shader with 2,560 output rows; the PLE graph consumes its actual
output bank. This corresponds to the native `ple_set_native_bf16` value path.

The key projection and all PLE post-projection operations remain on their
existing portable paths. The previous layer's already materialized residual is
unchanged. PLE token history and the normalized convolution history retain their
existing transaction and reset behavior. This option does not enable native
GDN or HC at layer 1.

The device result is checked against an ordered host control. It rejects
nonfinite and subnormal BF16 weights or F32 inputs and intermediate results;
that is a bounded diagnostic domain, not a statement about CUDA's full domain.
The control retains about 12.5 MiB of host BF16 weights and performs a full
host matrix-vector operation on every invocation, so the option is not a
performance path. The device BF16 role replaces the Q8_0 role; allocations
and the host control envelope are separately admitted.

`portable_ple_native_value_test` checks the actual graph bank, host-backed
failure/rollback and reset behavior, and the layer-1 option scope.
`native_qsa_indexer_bf16_input_test` also checks dense and cancellation
sentinels at the PLE geometry; `native_shared_dispatch_test` checks the fixed
2,560-row Vulkan dispatch contract. These tests are not a physical CUDA/Vulkan
comparison. A native-first comparison with actual PLE role and embedding data
and a bounded model run remain required before claiming hardware or model parity.
