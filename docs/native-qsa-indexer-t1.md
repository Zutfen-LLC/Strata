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

## Default-disabled eager model seam

`--qsa-indexer-resident-weights PREFIX` reserves the same F32 k_norm role and
candidate banks in both arms. `--native-qsa-indexer-t1` enables only the on arm.
The actual completed raw-key projection is a retained graph output. Its native
candidate supplies existing score/selection consumers; no score or attention
algorithm changes. The three-row retained tail and full packed native state
publish only after successful FFN. Reset clears both; failed component or FFN
work leaves committed KV/indexer state unchanged. Paid calls remain counted.

The resident role replaces its original allocation. Before/candidate banks,
diagnostic/status alignment and metadata use the existing graph admission. A
separate 1MiB host allowance per layer covers the fixed state, detached candidate,
readback/serialization vectors and 32KiB GGML contexts (each state is at most
35,332 bytes); it is also enforced by the layer state budget and charged once
in host accounting. Existing generic KV/trace envelopes and physical reserve
remain. The original native planner decides fit; prior placement is not assumed.

Initial layer3 positions0..7 retain exact current raw/gamma/before operands,
candidate and diagnostic values. Accepted-state bytes link each candidate to
FFN-success publication. Within-arm reset hashes cover full native state, including
the persistent spare and inactive capacity; they are not cross-arm parity claims.
Only eager serial text/context<=256 is supported; production/chunks/table modes
refuse. Local coordinator tests use actual graph producers/consumers and host
component mocks; physical component and model validation remain separate.

Hermes reports the component at `ba070a1072f4d9adc4172f2d0480a14bff75fc88` passes the
bounded native/Vulkan five-case comparison: exact own-input state/order and
numerical controls, repeat/reset, Vulkan immutable-input/canary/noncommit/retry
and three-FMA/nine-protected-operation module checks. Exit/reap and three baseline
cleanup observations pass. Native has five in-loop guard samples; Vulkan has none,
so no continuous/per-child sampling claim follows. This is component evidence,
not model, native-window, quality or performance qualification.

## Bounded model result

Hermes reports the comparison at `f68c187ed5e3d991f14cc491f90705b4e61421b4`
passes 23 tests, fresh native planning/admission with matched ownership and two
reset-separated established 25-token/GEN3 requests per arm. On performs 28 actual
indexer calls per ordinary QSA layer/request; control performs none. Earlier
native QSA normalization stays fixed. Layer3 positions0..7 own-input state/math,
candidate-to-FFN-commit-to-next-input links, 24 new protected module uses and prior
HC/GDN/QSA checks pass. Within-arm head/route/full-state reset is exact. Retained
cross-arm routes match and matching-history logits at positions24..27 are bit
identical; this is a diagnostic, not full native parity. All three processes
exit/reap and restore baseline across three observations; guard sampling is
discrete. qualified=false; native scores/selection/attention, accepted-prefix/
window, quality and performance qualification remain separate.
