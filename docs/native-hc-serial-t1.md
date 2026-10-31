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
cleanup must use the reviewed operator harness. The bounded physical coordinator gate passed on Vulkan0/V340: initial HC
values matched the sealed same-backend component, the native numerical contract
passed, repeat/reset/disabled/rollback checks passed and replacement accounting
matched. These operator-produced results were independently checked from sealed
receipts; only two guard samples were retained and the executable PID was not
separately retained. This is not model or full-HC qualification.

For the next model diagnostic, each enabled owner writes only its first successful
attention graph's input, RS, normalized and injection values inside its private
receipt directory. No weights are exported. The bounded copies fit the existing
host allowance; failure refuses rather than publishing a partial forward. The
snapshot is not repeated after reset. `--dump-executed-logits` also supports stdio
head-elision requests, each in a fresh `request-N` directory, so a repeated request
can check reset behavior without another model load. Neither diagnostic changes
arithmetic, selection, state commits or the default-disabled path.


`--validate-options` uses the real CLI parser and the same mode predicates as
session creation, before Vulkan initialization or model access. It explicitly
reports that physical features, metadata and memory admission were not checked.
Preparation must validate every intended arm with the actual product, including
its resident-weight options; a fake child cannot establish those preconditions.
The absent flag does not imply resident GDN weights: both residency booleans
default to false. `--gdn-resident-weights` retains attention weights across all
36 GDN layers, including the existing PLE layer. It is not a layer-0-only switch.
Changing that mode requires a matched off/on residency configuration and a new
valid resource/ownership plan; prior nonresident controls do not qualify it.


`--head-shape-admission` is a separate default-off admission experiment. The
head's original 4 GiB graph cap was introduced with the bounded model assembly;
it is a policy ceiling, not an allocator measurement. This option builds the
same graph metadata without reading weight payloads or allocating scheduler
buffers. It sums every node and source-copy allocation with the larger of the
CPU/device default-buffer padding, charges original input storage on both
backends and takes no liveness or resident-copy reuse credit. Original immutable
weights remain separately charged. CPU workspace, the existing conversion and
other-workspace allowances, and aligned integer-guard storage plus its existing
1 MiB resource allowance are also retained.

Only the existing pinned Vulkan contiguous Q8/F32 T1 output-head contract with
unambiguous physical PCI/storage-range proof and no CPU fallback earns the
smaller bound. Unknown backends, geometry/range contracts or fallback retain the
full graph cap. Malformed tensor metadata refuses. The estimator never raises
the graph cap or device limit and never subtracts the physical free reserve.
Construction passes the admitted budget into the existing scheduler before
allocation and checks actual allocation bookkeeping against it afterwards.
Driver/pipeline private costs remain conservative allowances, not portable exact
measurements. Physical reserve and the existing process admission remain separate.

A valid mode or a metadata head estimate does not freeze expert ownership.
Residency changes require the native post-dense capacity calculation and helper
reselection from remaining ranked experts. Old nonresident controls remain valid
only for their original configuration. A new matched resident off/on pair must
share the residency/admission flags and compare actual ownership before GEN.
The layer-0 HC component borrows only its own resident norm/injection roles;
all-GDN residency is the currently available model ownership option. Narrowing
that option would be a separate policy change, not part of head accounting.

`portable_head_admission_test --vulkan INDEX` checks full-vocabulary metadata,
physical direct-Q8 proof and exact/short-room refusal without uploading weights.
`portable_head_test --vulkan INDEX` checks actual small-head scheduler/guard
allocation bookkeeping, resident and host weights, metadata-sized budgets and
exact same-backend outputs. These are component gates, not real-model resident
admission, native full-HC, quality or performance qualification.
