# Native QSA indexer append and pooling: ordinary T1

The specification is `Verifier::record_window()`'s `idx_append` and unchanged
`native_qsa_indexer_append()` / `append_body<false>` in
`src/kernels/cuda/native_qsa_indexer.cu`. This standalone component changes no
model option, default, scores, selection, attention or window scheduler.

Raw keys round to IEEE binary16 RNE and expand to F32 before entering the native
three-row tail. A completed block adds tail0, tail1, tail2 and incoming in that
order, then FMA(.25,sum,+0). Position zero repeats its incoming key four times
and retains that normalized/rotated spare in `dead`. The native tail is retained,
not cleared after completion. Completed pools append the persistent spare and
update block_pos to the first absolute position of that block. The spare always
rotates at zero, including nonzero sequence bases.

The port uses the existing integer FP16 converter from `native_q8_stages.comp`,
with integer exact expansion, rather than relying on implementation-dependent
packing/denormal treatment. Weighted normalization uses the native 256-thread
padded XOR reduction and multiplication association; rotation retains the native
rounded second product and FMA. The unchanged CUDA launcher supplies actual state;
a separate diagnostic kernel exposes primitive values without replacing it.

Inputs and candidate state are disjoint. The kernel never commits state. The
probe commits a candidate only after status/control success, tests reset/replay,
invalid geometry/alias noncommit and retry, and checks all inactive state words.
All buffers, status/alignment and metadata remain subject to explicit admission.

This first component supports only text T1, width128/block4/n_rot64, epsilon1e-6,
None scaling/base1e7 and capacity/absolute positions within256. It refuses table
RoPE, MRoPE, other geometry, nonfinite or FP16-overflow raw inputs, and unqualified
nonfinite/subnormal arithmetic intermediates. FP16 subnormal and zero results
are supported: their F32 expansions are normal values or signed zero. No input
is clipped. There is no model integration or native accepted-prefix qualification.

Acceptance reuses the frozen QSA normalization/RoPE contract in
[native-qsa-norm-rope-t1.md](native-qsa-norm-rope-t1.md): exact binary16 conversion,
ordered pooling, basic operations and state transitions; deliberate rsqrt/pow/
trig bounds and the existing native RoPE row-amplitude control. Cancellation uses
that amplitude control, not relative error around zero. Cross-backend bits remain
diagnostic. Independent pinned tie/subnormal/zero vectors and state expectations
supplement generated own-primitive checks. Incorrect conversion, pool and
inactive state mutations must reject, even if two backends agree on them.

Local tests/builds establish source/plumbing and effective-module protections,
not actual CUDA/Vulkan execution. The component handoff first builds/calls the
unchanged CUDA launcher, then Vulkan with the same generated fixtures. Model
wiring follows component success, retaining the existing FFN-success transaction
and original placement/admission; it is not silently enabled by component success.
