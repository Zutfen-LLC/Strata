# Standalone native T1 GDN a/b preparation

Specification: `Verifier::record_window()` calls `gdn_ab_multi()` in
`src/kernels/cuda/verify_kernels.cu`. This component ports the ordinary T1,
N2560/H48 BF16 method, not GGML matrix multiplication or a new reduction:

- Decode native BF16 row-major alpha/beta matrices without a new conversion.
- Each of 32 lanes keeps one accumulator across 10 chunks of 8 explicit FMA, then
  reduces with XOR 16/8/4/2/1. HC's per-chunk accumulator is a different method.
- Alpha adds dt, selects strictly v>20 ? v : log1pf(__expf(v)), then multiplies
  ssm_a. Equality at 20 still takes exp/log1pf. No extra decay exp is inserted.
- Beta uses 1/(1+__expf(-dot)), retaining its natural denominator-one and Inf tails.

Vulkan adapts only buffers/submission and warp exchange through shared memory.
It uses the existing tested LLVM libc accurate non-FMA/RNE primitive, with the
same physical/logical FP64, RTE64/signed-zero and strict FP32 FMA requirements.
Unsupported features refuse; no fallback. The CUDA reference calls the unchanged
production launcher and separately checks a source-order diagnostic control.
The diagnostic outputs are not instrumentation inside that production kernel.

The unchanged numerical policy checks the actual exp inputs pointwise, log1pf
within one adjacent F32 step of RNE (and Vulkan exact original LLVM source),
observed sigmoid division within its existing bound, and exact local dt/scale/
denominator operations. There is no final gate/beta epsilon or fitted cutoff.
Positive v>20 stores zero only as an explicitly unused exp diagnostic, never as
an exponential result. Actual exp underflow to zero is admitted by the existing
underflow allowance. Preserved subnormal softplus-exp inputs remain unsupported
by the selected primitive domain. Nonfinite/subnormal dot/scale arithmetic,
nonzero gate multiplication underflow, and the existing negative sigmoid
transition class refuse. Status bits 1/2/4/8 identify arithmetic, beta, log-input
or gate-underflow refusal; current outputs remain available for diagnosis.

Inputs are immutable. The standalone component has no recurrent-state commit;
it does not claim accepted-prefix/window scheduling parity or model integration.
Gate/beta are separate outputs. Packed diagnostic storage is bounded; tensor
admission is 2 MiB plus 64 KiB metadata. Input/output alias, invalid input, budget
refusal, retry, repeat and canaries are checked. Existing lifecycle/resource/
pressure/cleanup guards remain. Defaults and all product model paths are unchanged.

Four locally generated fixtures cover zero projections, strict 20 neighbors and
native exp/sigmoid tails, ordinary dense BF16 values, and independent continuous-
FMA/XOR/fused-rounding traps. All source data are synthetic; no role extraction
or model capture is needed. Host controls and module validation are not CUDA
compilation or GPU evidence. Native02 must validate the actual unchanged launcher
first, then Vulkan05 validates the corresponding candidate. Any refusal preserves
outputs and stops. A separately bounded integration follows only after component
evidence; qualified=false and full-model parity remain unclaimed.
