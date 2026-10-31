# Experimental layer-0 serial HC norm/injection

`--native-hc-norm-inject-layer0 RECEIPT_PREFIX` enables only the attention HC
normalization and BF16 injection at layer 0. Other layers, HC down/up/gating and
mixed arithmetic retain their existing portable implementations. This is partial
HC integration, not full-HC/model parity or a production/performance path.
The default is disabled.

The mode requires the existing strict-FMA native-stage build, a strict selected
Vulkan accelerator, resident attention/dense weights and a private receipt
prefix with an existing parent directory. GDN production policy, PLE, chunk
planning/execution and CPU fallback refuse explicitly. The model option requires
serial eager GDN; QSA's independent execution policy is unchanged. The lower
`PortableGdnOptions::native_hc_norm_inject` API also restricts the layer to 0.
Existing HC mode, feature/ABI and normal-domain refusals remain active.

The selected GGUF injection Q8_0 role is converted by the existing checked
original-packer BF16 conversion. Its 81,920-byte I32-packed allocation replaces
the 43,520-byte Q8 allocation; it is not an extra parallel weight bank. Norm
weights remain the original resident F32 role. Actual aligned allocations stay
inside the existing weight cap and are included in memory usage.

A serial owner retains four disjoint F32 tensors: residual and normalized vectors
(40,960 bytes each), RS and injection (16 bytes each). The aligned allocation plus
component ownership/status is admitted inside the existing graph cap and
reported once, including when graph scratch is shared. A separate conservative
1 MiB host bound covers explicit metadata, normal-domain preflight and optional
trace copies. Backend/driver workspace bounds are retained. No reserve/cap is
relaxed and no residency or expert selection policy changes.

Residual/normalized leaves retain the original 3D/2D geometry and direct backing.
Their F32 type and contiguous strides preserve the existing eligibility inputs
for the remaining Q8 down/up path. A simple contiguous reshape alone would not
change the pinned MMVQ selector; no such arithmetic change is claimed. The main graph borrows the allocated owner tensors, marked as inputs/outputs to
forbid liveness reuse. At runtime the actual current residual is uploaded first;
the component then runs; only afterward does the existing GDN graph consume the
normalized vector and injection. No kernel executes at graph construction.
Normalized values feed the current GGML down/up/mixed path, and injection feeds
the existing residual scatter. The synchronous component's bounded host-domain
preflight is retained as a refusal check, never a CPU numerical fallback.

State candidates, FFN dispatch and commit order are unchanged. A pre-graph HC
failure cannot publish token/state/route data. Existing eager rollback and
resident-state copies/reset remain. Model failure keeps its existing poison
semantics. `native_hc_layer0_calls_delta` reports actual successful component
calls, including work before a later failed forward; reset does not erase that
execution counter. The flag also emits `full_hc_parity=false`.

Optional eager `PortableGdnTrace` fields read the retained input, normalized,
RS and injection only after existing synchronous graph completion. They allow
upload/liveness/consumption diagnosis without adding an intermediate callback
or state commit. Traces and receipt directories remain private operator data.

The coordinator regression uses the real graph/owner with a mocked component:
it independently perturbs normalized/injection leaves to prove consumption and
checks upload order, shared-plan reuse, resident/eager reset/replay, conversion
replacement accounting, budget/mode refusals and failed-call rollback. It is not
GPU arithmetic evidence. Existing GDN/shared/observer/model regressions validate
the disabled path; numerical component validation remains separate under
[native-hc-normal-v1](native-hc-numerical-contract.md).

`portable_gdn_native_hc_probe OPERANDS FRESH_OUTPUT EXPECT_DEVICE` is a bounded
physical seam test. It accepts only the three fixed-size HC operand files, uses
the existing synthetic full-GDN fixture for every other role, warms shared graph
scratch, tests native reset/repeat and provider-failure rollback, and runs two
disabled controls. It exports the live post-graph HC fields and exact fixture for
comparison with an already sealed component/native reference. No model loads or
new raw-role extraction are necessary. Device-specific execution, admission and
cleanup must use the reviewed operator harness. Physical seam qualification is
pending; unit tests or prior standalone acceptance do not substitute for it.
