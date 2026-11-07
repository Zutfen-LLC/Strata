# Strata method: CUDA/HIP to Vulkan migration map

Tracking: [umbrella #65](https://github.com/Zutfen-LLC/Strata/issues/65), with
[GDN #66](https://github.com/Zutfen-LLC/Strata/issues/66),
[QSA #67](https://github.com/Zutfen-LLC/Strata/issues/67),
[scheduling #68](https://github.com/Zutfen-LLC/Strata/issues/68),
[prefill #69](https://github.com/Zutfen-LLC/Strata/issues/69) and
[product breadth #70](https://github.com/Zutfen-LLC/Strata/issues/70).
This updates the existing roadmap rather than introducing a parallel plan.

## Bounded prefill ownership prerequisite

For open [prefill #69](https://github.com/Zutfen-LLC/Strata/issues/69),
`native_hc_rows.hpp` is a standalone host-value transaction for one layer and
one sequence: two consecutive rows or a one-row tail. It copies position, token
and residual/block/injection values; layer 0 and layer 1 PLE accept materialized
residuals only. Publication, abort, reset and same-layer continuation are
explicit; borrowed references expire on mutation. This scaffold does not itself
validate producer ownership or execute arithmetic.

The separate standalone `portable_native_hc_composed.hpp` snapshots a typed,
packed four-role bank and accepts live producer-issued GDN or QSA pending-tuple
snapshots, checking producer, boundary, position and generation. Its T2/T1 owner
stages rows, composes pending, norm/injection and mix, and publishes only after
both synchronous consumers succeed. Consumers read independent host copies;
failed candidate views and superseded/reset views are invalidated, but external
consumer side effects cannot be rolled back. The backend is borrowed and must
outlive the owner and role bank; this is not an asynchronous device consumer or
model admission. The copied roles have a 13,230,080-byte payload before backend
alignment, additional to the source bank; the owner budget includes the copy,
scratch and wrapper allocations, not the still-resident source. The type-based
source-device check relies on GGML Vulkan's per-device default buffer type and
ordinary Vulkan allocation; it is not a general device-identity guarantee for
arbitrary backends. The composed pending path explicitly refuses layer 0 and the
layer 1 PLE boundary rather than substituting a pending tuple there.

The actual pending and mix Vulkan wrappers support one or two rows, with span,
non-overlap, device, offset and dispatch-limit checks and per-row domain checks
where applicable. CPU mock dispatch tests exercise wrapper and composed-owner
control/lifetime paths, including GDN/QSA snapshots; compiled SPIR-V validation
and disassembly checks inspect row indexing. These checks do not establish
physical device execution or numerical parity. The standalone owner is not wired
into model chunk execution: all existing model native chunk refusals remain.
Generic chunk work (PR76/77/78/79/82) does not establish native chunk parity.
PLE and GDN/QSA causal work, fixed local/remote/CPU routing, batched matrix and
expert execution, admission and model failure/reset qualification, and physical
GPU validation remain. No independent-request batching, speculation, full model
parity or throughput result is implied.

## Bounded 48-layer row executor

`PortableNativeModelRows` is an internal, synchronous borrowed-owner chain from
checked layer 0 token embedding through typed PLE and all ordinary GDN and QSA
layers to the layer 47 terminal and actual head. Assembly starts at position
zero. The caller owns the roles, providers, backends and separately admitted
component resources, and must keep them alive and exclusive. A failed batch
poisons the chain; earlier layers may already have committed and ordered reset
and replay are required. This is not a GGUF-to-model factory or public native
chunk admission; existing model chunk refusals are unchanged.

A CPU fixture with synthetic, bounded component mocks compares three T1 calls,
T2 plus T1, and T1 plus T2 for three tokens. It compares logits, routes, PLE
history and all GDN/QSA states, checks batch dispatch and preflight, and tests
provider and terminal failures followed by reset/replay. A changed prefix and a
synthetic GDN output perturbation also change downstream state and logits; the
bounded mock arithmetic is not a numerical oracle. Its F32 vocabulary has 16
entries. Failed-reset injection and isolated terminal-capacity rejection are
not covered by this fixture. These are orchestration checks, not physical GPU
or native numerical parity, production resource admission, quality or
performance proof.

## Bounded terminal composition

The standalone layer 47 terminal accepts live layer 46 FFN snapshots. It runs
QSA and named FFN HC roles, batches the FFN rows, and uses the distinct native
final-write owner. Checked device row views feed the existing serial
`PortableHead` once per row; this is not a batched head. CPU component mocks
exercise a bounded 42–47 chain, T2 plus T1 versus three T1 calls, and failure
rollback. The terminal fixture uses the F32 vocabulary graph; separate head
tests cover the native Q8 vocabulary component.

Publication waits for QSA source revalidation. Head counters and receipts can
record work done before a later failure. Earlier layers commit sequentially,
not atomically across the chain; reset and replay them in order after failure.
Borrowed owners and their backends must outlive the terminal and remain idle
while attached. Full 48-layer production construction, admission and physical
backend validation remain open. The Q8_1/F32 row convention does not establish
native-prefill FP16/BF16 GEMM equivalence, router or expert-reduction parity,
full-model numerical parity, quality or throughput.

## Internal GDN FP16 projection method

The internal GDN owners can select an FP16-operand method for QKV, gate and
output projections. It decodes Q8_0 signed codes and half scales to RN-FP16
weights, and separately rounds the projection activation to FP16 before
widening it to the F32 right-hand operand. Each T2 projection uses a two-column
GGML operation; T1 has a separate output. The original HC F32 mixed input still
feeds the GDN core, and the output projection reads the recurrence result.
The Q8_1 method remains the default. This selection does not admit public native
prompt chunks.

CPU tests execute the selected graphs at the native shapes and exercise one
mapped, named layer through its owning composition, with mocked HC, convolution,
alpha/beta and recurrence stages. The mixed input reaching the core and the
projection consumers are checked separately. The selected method has not been
run on a physical GPU or compared against native vendor reductions. Its
conservative requested budgets count source and uploaded raw roles, decoded
weights, graph tensors, metadata and bounded conversion/readback envelopes;
shared buffers and host allowances can be counted more than once. They are not
physical residency limits and exclude backend-private workspace, driver and
allocator overhead. Alpha/beta still needs the native separate BF16 products and
gate epilogue; HC producer, recurrence and gating methods also remain to be
qualified. No full-model numerical parity or performance claim follows.

## Bounded BF16 router operand method

An internal FFN owner can opt into a K2560, N512, T1/T2 router projection
with BF16 weights and BF16-rounded activations widened exactly to F32. It
accepts BF16 weights or exactly promoted BF16 F32 weights; it rejects inexact
F32 and nonfinite weights, nonfinite activations and activation BF16 overflow.
Shared work and the expert provider keep the original F32 activations. BF16X2
and the FP16 prompt variant are not implemented. The existing F32 method stays
the default, and public prompt chunk admission is unchanged.

Source inspection of the local GGML Vulkan backend indicates that T2 selects
one batched BF16-weight/F32-input matrix-vector dispatch, with F32 shader
accumulation, rather than a tiled GEMM. BF16 weights are not in its Q8_1 RHS
pipeline set. CPU graph and mock-owner tests are not GPU dispatch measurements.
The operand method does not establish vendor GEMM reduction identity, router
or full-model numerical parity, throughput, or quality. The all-layer model
owner selects it only through an explicit internal constructor option.

The FFN host envelope includes 512 KiB projection metadata and 64 KiB for
bounded conversion and readback scratch. Its tensor budget counts the copied
weight, the two T2 input banks, and separate T1 and T2 output tensors. T1
input and router-operand views reuse the T2 banks. These requested envelopes
are not a physical cap on backend-private graph workspace or driver memory.

## Standalone post-attention FFN rows

The standalone FFN composition can consume a live PLE/GDN or QSA transaction
for two rows or a one-row tail. It selects the layer's named FFN HC roles,
converts the Q8_0 roles to BF16, and copies them into its own HC bank. A single
GGML matrix projection feeds the native router; the existing shared and combine
owners surround one fixed expert-provider batch. The resulting deferred tuples
can be consumed by the next ordinary HC boundary after commit. Caller-retained
snapshots own their tuples; reset or destruction invalidates their provenance.
The upstream owner, row owner and backend are borrowed and must outlive the
composition. This is not full-model chunk admission.

CPU tests exercise the owner chain with low-level component mocks, a grouped
CPU provider, and socket/local ownership mocks. The connected test uses synthetic
named FFN views and zero-valued layer-3 fixture experts. GGUF role lookup is
checked separately. They do not establish GPU
arithmetic. The router matrix does not prove native MMVQ reduction equivalence.
Native helper partial weighted reductions differ from portable full unweighted
expert rows combined in original top-10 order. Q8_1/F32 row conventions do not
prove native-prefill FP16/BF16 GEMM equivalence. No throughput or default-path
claim follows from this slice.

## HC norm/injection checkpoint (2026-10-31)

[PR #97](https://github.com/Zutfen-LLC/Strata/pull/97) defines the standalone
BF16 HC norm/injection method and versioned `native-hc-normal-v1` acceptance.
Limited component acceptance applies only to its fixed geometry, selected
method and normal-or-zero domain. Exact comparisons remain diagnostics;
independent local operation checks and derived division/rsqrt bounds are
separate from full-model behavior. It does not establish universal intrinsic
bit identity, subnormal behavior, HC down/up/mixed or model parity.

[Draft PR #98](https://github.com/Zutfen-LLC/Strata/pull/98), source
`3d7eb782241cd45e202c87c152c2554eeb55dc27`, adds only an experimental layer-0
serial eager integration. It uploads the actual residual before native
norm/injection and lets the original graph borrow retained outputs. The BF16
injection allocation replaces Q8 storage; graph/state admission, leases and
commit rules stay explicit. The default is disabled, and chunks/PLE/GDN
production modes refuse. The bounded physical coordinator gate passed at
`3669d38323d817d7d7c3736570e47bea0b4ffc48`: retained initial HC boundaries,
repeat/reset/disabled/rollback and replacement accounting passed. This is an
operator-produced result independently checked from sealed receipts. Only two
guard samples and the timeout-wrapper PID were retained; no continuous
observation or separate executable-PID proof is claimed. The bounded matched resident off/on diagnostic has now completed; initial HC
contract checks and within-arm reset/replay checks pass. Generated tokens agree
for the retained short request, while routes and final heads differ. This is
limited partial-seam evidence, not full-model numerical or quality acceptance.
Bounded per-request head and initial HC diagnostics preserve those distinctions
without changing numerical methods.
Source now shares actual CLI/session mode checks through `--validate-options`,
which does not imply backend or memory admission. Enabled layer-0 HC requires
resident input roles; the existing minimum model option retains attention
weights across all GDN layers. Both arms of a resident comparison must use that
same ownership policy and fresh post-dense native capacity sizing.

`--head-shape-admission` is a separate default-off accounting experiment. It
bounds the unchanged head graph from metadata, allocator padding and node/source
copies without liveness savings, retaining workspace/guard allowances. Only the
existing proven Vulkan T1 direct-Q8 contract can replace the broad head graph
preflight; unknown contracts keep the full cap. Construction enforces the admitted
envelope before allocation, then checks actual bookkeeping. Caps, physical
reserve, native quota/rank/helper exclusion and arithmetic stay unchanged.
Small physical head bookkeeping/full-shape metadata and matched resident
admission have passed their bounded gates. Both resident arms use the same
head/admission and weight policy; native post-dense sizing freezes common
ownership before GEN. These gates do not qualify other shapes/backends.
[Draft PR #99](https://github.com/Zutfen-LLC/Strata/pull/99), source
`df488f25812a9a5cac94eabe6630bfa7bcae75b0`, adds a standalone T1 BF16
HC down/up/activation/mixed component with identical normalized operands and
native lane/reduction/stream order. Its bounded zero/sparse/dense synthetic
component gate passed: mixed outputs are exact across the two backends, while
one fixture's low projection/activation output differs within the deliberate
intrinsic contract. Exact local source-order checks remain separate from those
bounds. Operator-produced receipts were independently checked offline. This is
synthetic component acceptance only, not real-role or full-HC/model acceptance.
The scope and native unobserved-intermediate limit are documented in the
[pinned component contract](https://github.com/Zutfen-LLC/Strata/blob/df488f25812a9a5cac94eabe6630bfa7bcae75b0/docs/native-hc-mix-component.md).
No additional synthetic campaign is required before bounded real-role validation;
actual role identity/conversion and representative ranges remain the next gate.
The two authorized real-role extractions are complete; original compat-BF16
conversion matches both native roles byte-for-byte. An identical retained
normalized operand exists, but its scaled down preactivations exceed the initial
component domain. This is a qualification guard, not a native model limit.
A specification-derived normal-domain prerequisite is implemented at
[`73aeec1f`](https://github.com/Zutfen-LLC/Strata/blob/73aeec1f953b87f14a587c2d35d920262c811c3d/docs/native-hc-mix-component.md).
It derives exp/division limits, rejects quotient underflow/nonfinite results,
preserves the native operation/reduction order, and retains exact local checks
and the existing intrinsic error policies. Independent boundary/adversarial
regressions and offline real-input domain preflight pass. The bounded wider-domain/real-role
component gate passed both backends' separate own-input arithmetic/intrinsic
contracts. Cross-backend differences remain diagnostics; differing low operands
prevent up-only attribution. Native sampling included case-loop observations;
Vulkan sampling was baseline-only, so neither continuous nor per-child workload
coverage follows. No more extraction or full-model capture is needed.
A separately opt-in [serial layer0 composed attention read](https://github.com/Zutfen-LLC/Strata/pull/100)
is implemented with existing components at the actual residual upload seam.
The bounded coordinator gate has passed both component contracts and
consumer-liveness/reset/rollback checks with approved HC operands and synthetic
remaining GDN/FFN data. It does not qualify full real-weight GDN or the model;
its discrete samples do not establish continuous workload coverage. The bounded
matched-model diagnostic has completed the initial component contracts, finite
head checks and exact within-arm reset/replay checks. Its comparator retains
route/head differences as diagnostics, without a final-logit tolerance or a
quality/performance claim. The comparison uses an authenticated historical
norm/injection-only control, not a freshly executed default-off control or proof
of new-image bit equivalence. Fresh post-dense native planning preserves the
retained eager host-state policy (resident GDN weights do not imply resident
GDN state). Historical controls are reused only with compatible frozen
input/state/ownership/method/runtime bindings, and never relabeled new-image
default-off equivalence. Conversion
replaces resident banks, and aligned scratch/host bounds remain charged to
existing admission. The old partial flag and defaults retain their behavior.
Pending writes, FFN HC, other shapes/native modes and full-model acceptance remain
separate. Defaults stay disabled and PRs remain drafts.

The next bounded HC method gap is the native pending-write/read transition.
At [`f1561f31`](https://github.com/Zutfen-LLC/Strata/blob/f1561f313963746d172cebff11cc043947d814a9/include/strata/kernels/fused_gr.hpp),
`FusedGrArgs::apply` folds the previous block output and injection into the
residual with FMA before normalization, then materializes that same residual
in the up pass. The selected multi-read implementation preserves this order;
the current Vulkan component explicitly refuses pending writes. A standalone
T1 same-input component using deterministic residual/block-output/injection
operands is the next implementation gate, before changing graph/state lifetime
or adding FFN roles. Existing approved attention roles suffice for that mechanical
check; they do not qualify FFN role identity or native GDN output arithmetic.
Retained composed-read diagnostics cannot attribute downstream differences
without equal upstream mixed operands. No further harness redesign is a
prerequisite for this bounded component work.
[Draft PR #101](https://github.com/Zutfen-LLC/Strata/pull/101), source
`a47bcea56b0f8cd07ff4e2b57657000a47b27e16`, implements this standalone
T1 pending sigmoid/FMA, norm/injection and down/up/mixed composition. The actual
native reference selects `apply=true`; Vulkan retains a disjoint candidate and
leaves the original pending tuple unchanged. Existing model flags and state
commit behavior are untouched. Exact own-gate FMA/cancellation checks precede
the existing own-input v1/v2 contracts; no fitted cross-backend epsilon is added.
Host/build checks remain separate from physical qualification. Actual CUDA
compilation and the bounded synthetic native/Vulkan component gate have now
completed both own-input contracts. Cross-backend raw differences remain
diagnostics, not a reason to relabel the component as full-HC/model parity.
Model pending lifetime and FFN role integration remain separate. See the
[pending contract](https://github.com/Zutfen-LLC/Strata/blob/a47bcea56b0f8cd07ff4e2b57657000a47b27e16/docs/native-hc-pending-t1.md).
[Draft PR #102](https://github.com/Zutfen-LLC/Strata/pull/102), source
`ae72f3943a0c2013e49d2b5f07c80d1ba90c7559`, adds only an explicit diagnostic
pending tuple at the serial attention coordinator. R-prime feeds BOTH the HC
read and graph hidden consumers; public state still commits after FFN/provider
success. The source preserves original operands and separates paid counters,
charges the extra stable storage/host envelope and refuses the legacy observer
schema that lacks this tuple. Host/resident state, private/shared scratch,
rollback/reset/retry and materialized-candidate controls are tested; a deliberate
old-R graph mutant fails. The bounded physical coordinator gate now passes
pending arithmetic, own-input norm/mix contracts and exact available
materialized-candidate component arrays. Whole residual/state/history equality
is enforced by probe source assertions; whole state arrays are not exported.
The compact evidence binds omitted binaries/fixtures by hash and does not
independently establish complete shared-library closure or inherited loader
mapping. This is a synthetic coordinator gate, not an actual model pending write.
No model CLI flag or inferred prior write is added. Ordinary layer 0 attention
has no previous native write; real producer/half identity and FFN roles must be
bound before model carry. Existing approved attention operands plus synthetic
block output suffice for this diagnostic gate; no new extraction is needed.
See the [coordinator scope](https://github.com/Zutfen-LLC/Strata/blob/ae72f3943a0c2013e49d2b5f07c80d1ba90c7559/docs/native-hc-pending-serial-t1.md).

The next essential product seam is the actual layer-0 GDN→FFN HC read. Native
[`verify.cpp`](https://github.com/Zutfen-LLC/Strata/blob/ae72f3943a0c2013e49d2b5f07c80d1ba90c7559/src/core/verify.cpp#L1907)
calls `gr_read_group(1, true, inj_, inj2_)` after producing the real block output.
It consumes the original residual, actual projected block output and attention
injection with **FFN** HC roles. Portable
[`portable_gdn.cpp`](https://github.com/Zutfen-LLC/Strata/blob/ae72f3943a0c2013e49d2b5f07c80d1ba90c7559/src/core/portable_gdn.cpp#L214)
currently materializes that residual before the GGML FFN HC read in
[`portable_ffn.cpp`](https://github.com/Zutfen-LLC/Strata/blob/ae72f3943a0c2013e49d2b5f07c80d1ba90c7559/src/core/portable_ffn.cpp#L297).
The bounded next change should connect that real producer tuple to the FFN read,
preserving FMA-before-normalization, ownership, admission and whole-layer commit.
A new diagnostic flag is not a substitute for that connection.

Synthetic FFN fixtures and the existing loader support implementation without
new extraction. Real same-input numerical qualification additionally needs
validated `blk.0.hc_ffn_{norm,down,up,inject}.weight` roles and a matched native
position-0/layer-0 FFN input tuple. Attention-role approval and model provenance
alone do not establish those bytes or operands. Obtain the scoped provenance
and any necessary extraction authorization before publishing new raw FFN data;
reuse existing dump seams where suitable. Recompute actual residency/scheduler
costs through the existing post-dense planner rather than reusing historical
expert counts or treating the component replacement delta as the full GPU cost.

[Draft PR #103](https://github.com/Zutfen-LLC/Strata/pull/103), source
`f7c5441c2dfe45ecd1b7c60b28d5dd293dff9562`, implements that actual producer
connection as an opt-in layer-0 T1 C++ path. The projected graph output and
attention injection are retained through readback before shared-arena reuse;
FFN roles use the original BF16 conversion and existing pending/norm/mix owners.
The same candidate feeds FFN HC and the final residual consumer, with unchanged
whole-layer publication. Host tests inspect actual graph producer/consumer
identity, exercise rollback/reset and compare an independent materialized
control. An old-R final-consumer mutant fails; corrected source and default
controls pass. A bounded physical synthetic coordinator gate now records successful
pending/norm/mix contracts, materialized control, reset/rollback/disabled behavior
and lifecycle cleanup. The compact evidence supports independent receipt and
matching-module validation; omitted output arrays prevent independent numerical
replay, whole-state arrays are not exported, and runtime library closure is not
independently captured. No continuous monitoring claim is made. Original packer
conversion now reproduces all four extracted real FFN role payloads byte for
byte; this establishes role identity only. A standalone real FFN pending-read
gate now uses an actual matched native tuple
and all four conversion-verified roles. Both unchanged own-input numerical
contracts and actual effective modules independently replay successfully. The
backends retain different pending gates/residuals and downstream operands; this
is bounded component acceptance, not cross-backend bit identity or model parity.
There is
no new model CLI flag, and the final FFN write/next-layer carry still uses the
previous portable method.
[Draft PR #104](https://github.com/Zutfen-LLC/Strata/pull/104), source
`fcdee6895c24b9551842b16ca4c35f3d9b4fb22d`, supplies the missing native
pre-read tuple snapshot. STRL004 copies original R, actual GDN BO and attention
injection before the destructive half-1 read, using the existing stream and
completion boundary inside the unchanged observer reservation. Portable/legacy
captures keep their older schema and cannot supply missing BO by reconstruction.
Host serialization, CUDA-mock, portable enabled/disabled and reader checks pass;
actual native compilation and one guarded capture now supply the original tuple.
The compact tuple/provenance and lifecycle receipts are verified, but the full
sealed trace is not exported in that packet for independent decoder replay.
Tuple and exact role conversion now support the independently replayed standalone
real FFN component gate described above. The actual model option/wiring remains
missing: explicit matched layer-0 FFN residency, complete admission and native
planner ownership must accompany the existing C++ producer/consumer seam.
Final FFN write/next-layer carry remains separate unfinished work.
[Draft PR #105](https://github.com/Zutfen-LLC/Strata/pull/105), source
`6f08323d35a1e2040866631ad0cafca656386ca9`, now wires the default-disabled
actual FFN-read model option. Explicit layer-0 residency includes FFN weights
without changing attention-only GDN residency, other layers, the head or expert
placement. Complete actual allocation accounting and conservative host bounds
remain admitted; actual FFN paid-work counters survive reset/provider rollback.
Focused parser, actual producer/consumer, accounting/refusal, rollback and CPU
model disabled-output regressions pass. The corrected deployment continuation
passes planning-only entrypoints with the actual product parser and real
process policy against synthetic providers. A bounded matched resident model
gate now passes complete planner admission,
actual FFN participation, own-input initial numerical contracts and exact
within-arm head/route/receipt replay. Independent offline replay reproduces the
sealed checker and validates actual modules. Initial attention receipts match
across arms; generated IDs match while routes and heads differ. These are partial
integration diagnostics, not full state/quality/performance or full-HC parity.
Executable/library payloads remain host-bound; whole-state arrays and earlier
full native capture originals are not independently replayed here.
The remaining final-write seam is the GGML post multiply/add in
`src/core/portable_ffn.cpp`; original native code materializes before layer-1 PLE
and otherwise preserves its pending/control/head boundaries. A write-only
candidate must preserve the selected native postops' scale/sigmoid/FMA order.
The standalone T1 final-write component now passes bounded synthetic and
identical-input real-operand backend-local contracts with actual native/Vulkan
execution and effective-module validation. Cross-backend differences remain;
no output tolerance was fitted and component evidence is not model parity.
The default-disabled `--native-hc-ffn-write-layer0` model/layer seam now
materializes the actual final FFN producer before the next layer/PLE consumer.
It replaces the unused post graph, reuses retained inputs, preserves transaction
commit/rollback and charges its storage/host bound with separate paid counters.
Local actual-consumer/admission/parser/default-output regressions pass; its
bounded physical coordinator now passes actual producer/consumer, reset/rollback,
disabled repeats, own-input numerical contracts and effective-module checks.
A fresh same-image matched-model diagnostic now passes actual final-write
participation, initial own-input numerical checks and exact within-arm reset
heads/routes/receipts under the frozen native ownership. Generated IDs agree,
while later routes and retained heads differ; this is bounded integration
evidence, not model numerical, quality or full-HC parity. Processes exit and
restore measured baselines; whole-state arrays and executable/runtime payloads
remain outside the compact result. Generic cross-layer pending/control/head
integration remains separate. See the
[bounded implementation](https://github.com/Zutfen-LLC/Strata/blob/a2d36ce3d7153533ee9b3eee80cc38d044885f9a/docs/native-hc-ffn-write-layer0.md).
The standalone real replay uses the recovered actual post-combine FFN block
output; HC mixed and GDN BO are not substituted. The coordinator does not prove
model PLE parity: whole-state arrays are source asserted, runtime payloads stay
host bound and monitoring is discrete. Fresh same-image control/on model work
uses one native planning load followed automatically by fixed-ownership arms,
with actual per-arm admission under the same caps and reserve. No default is
enabled. The next concrete method gap is one ordinary inter-layer pending
transition: retain the layer-1 FFN producer tuple and fold its write into layer-2
attention HC read, as native `verify.cpp` does for non-PLE/control boundaries.
[Draft PR #108](https://github.com/Zutfen-LLC/Strata/pull/108) now implements
that default-disabled ordinary layer1-to-layer2 carry. Local actual two-layer
coordinators, shared-scratch lifetime, independent FMA/materialized control,
rollback/reset, accounting/refusal, actual CLI and CPU model default-output
regressions pass. Layer1 PLE and its upstream producer methods are preserved;
carrying their actual tuple does not qualify those methods as native-equivalent.
The operator-reported small synthetic Vulkan probe now passes shared-scratch
tuple consumption, materialized control, reset/refusal/retry, own-input numerical
contracts, effective-module checks and baseline cleanup. Actual layer1 PLE
and real layer2 roles were not exercised by that synthetic fixture. Existing layer-0 roles
suffice for its clearly labelled development fixtures. The integrated model owner records its actual pending tuple and HC outputs.
The fresh same-image actual-model own-input check now passes pending FMA,
normalization/mix, paid-work and exact within-arm reset/replay checks, using
actual layer2 roles from the same GGUF. Generated IDs agree across arms while
routes and retained heads differ. The retained-data checker completed offline
after correcting a blanket rejection of lifecycle events; the live monitor,
numerical criteria and original STOP remain unchanged. Existing validated module
receipts were reused, with no model/GPU rerun. Future model recipes must reuse
the recognized tracked-pidfd verified-exit receipt policy and continue refusing
live/unverified/unknown events. This is bounded own-input integration, with no
independent native capture, whole-state export or full-model numerical/quality
qualification.

[PR109](https://github.com/Zutfen-LLC/Strata/pull/109), source
`9a6d41029797304fbe1de79d5fb2bc311bae6c97`, adds the default-disabled ordinary
layer2 FFN → layer3 QSA pending read. Its specification remains
`src/core/verify.cpp::Verifier::record_window()` → `pre` →
`gr_read_group(0, pending, inj2_, inj_)` and
`src/kernels/cuda/fused_gr.cu::fused_gr_read_multi()`: apply the prior FFN
FMA before native-method HC normalization, injection and mixing. Vulkan
buffer ownership/submission expresses that dependency; it changes no placement
policy or QSA projection/attention/indexer algorithm. Layer0 PLE materialization
and control/head exceptions remain unchanged.

The small Vulkan gate passes: actual GDN2 producer/QSA3 consumer, shared scratch,
four-position pooled-block completion, materialized same-backend residual and
KV/indexer equality, provider rollback/retry, reset and five effective HC module
checks. Nine focused CTests pass. The probe checks five producer outputs and
seven paid consumer calls, including work before intentional refusal. Its
rebound layer0 HC operands are synthetic development roles, not actual layer3
model provenance. Complete state equality is asserted by the probe; arrays are
not exported. Guard sampling is discrete and does not establish continuous
coverage. This is a functional synthetic gate, not full QSA/model parity.

The actual-model diagnostic also passes using the same source and fresh native
planning: 384 primary/1490 helper ownership freezes and each arm's actual
admission passes. Layer0 and layer1→layer2 are fixed in both arms; only on adds
layer2→layer3. Each reset-separated request reports 28 earlier producer/read
calls in both arms and 28 new producer/read calls in on versus zero in control.
Actual layer3 GGUF-role/owner pending-v1, normal-v1 and mix-v2 checks and all 21
HC module uses pass. Both arms repeat generated IDs `[18,24,16]`; within-arm
heads/routes/receipts repeat exactly. First cross-arm route-rank divergence is
position 1/layer 5, first selected-set divergence position 1/layer 11. Matching-history
maximum absolute logit differences at positions 24–27 are approximately 1.61825,
0.66505, 0.70438 and 1.44528. These are diagnostics, not acceptance tolerances.
All three planned model processes exit with verified identity and three baseline
cleanup observations; no retry, new native capture or rebuild was needed.

This establishes bounded actual-path own-input behavior, not independent native
end-to-end numerical equivalence. Whole-model QSA state is not exported and no
live model failure is injected. QSA rollback/reset remains covered explicitly
by the synthetic coordinator, with real model reset/replay as separate evidence.
Native QSA projection/attention/indexer arithmetic and captured-window scheduling
remain unported or unqualified. The new boundary changes HC method as well as
materialization, so downstream changes cannot be attributed to materialization
alone. See the [implemented scope](https://github.com/Zutfen-LLC/Strata/blob/9a6d41029797304fbe1de79d5fb2bc311bae6c97/docs/native-hc-pending-layer2-3-qsa.md).

[PR110](https://github.com/Zutfen-LLC/Strata/pull/110), source
`a7d87b40941a62bae74242a1791afc510ef4293e`, implements the reusable library
family cycle. Its bounded Vulkan coordinator gate passes ten focused CTests,
actual post-projection QSA attention→FFN and QSA FFN→GDN tuple/control checks,
three own-input pending reads/control write and 22 effective module uses.
Provider rollback/retry/reset and complete state/control equality are asserted
by probe source; state arrays are omitted and guards are discrete samples.
Rebound development roles are synthetic, not actual model role provenance.

[PR111](https://github.com/Zutfen-LLC/Strata/pull/111), source
`3333e00cdae8cb56f9733512f9e822ae536f7222`, wires one default-disabled bounded
serial-HC product capability through that same family implementation. Explicit
FFN2/3 residency can be matched independently in control/on, while unrelated
QSA weights retain their prior placement. The initial GDN2→QSA3→GDN4 scope uses
actual projected output and native boundary predicates, preserves PLE/head and
the unchanged remainder, and charges actual resident banks/buffers plus host
bounds before original native-policy planning. Local family/residency/carry,
allocation, parser and default-off model tests pass. The bounded model cycle
now also passes fresh native-policy planning, same frozen 384/1490 ownership,
independent arm admission and matched FFN2/3 residency. Each reset request in on
reports FFN calls `[28,28]`, 28 outgoing tuples and 28 next-attention reads;
control has no new cycle work. Actual-owner own-input contracts, 33 module uses
and within-arm reset/replay pass. Both arms repeat `[18,24,16]`, but first
cross-arm rank/set divergences occur at position 1/layer 9 and position 2/layer 12;
matching-history logits differ (maximum about 1.76163), which is diagnostic only.
All three model children exit/reap and restore three baseline observations with
empty owner lists. The initial stale-source harness STOP remains preserved;
its corrected continuation reused build/image/roles/sentinels rather than
rebuilding or spending additional model loads. No whole-state arrays or live
model failure injection are claimed. This does not port native QSA kernels,
accepted-prefix commits, captured scheduling, quality or performance parity.

Do not attribute later route/head differences to down/up alone when upstream
injection and internal state already differ, and do not invent a final-logit
tolerance from matching generated tokens.
See the [contract](https://github.com/Zutfen-LLC/Strata/blob/1c3e835f4ce832ba7017ffb549c25facf6f3b9f2/docs/native-hc-numerical-contract.md)
and [serial implementation scope](https://github.com/Zutfen-LLC/Strata/blob/3d7eb782241cd45e202c87c152c2554eeb55dc27/docs/native-hc-serial-t1.md).

## Contract and inspected versions

The target is Strata's existing model-specific native engine: preserve its
ownership, capacity/placement policy, numerical order, state transitions and
execution methodology. Vulkan allocation, feature chains, barriers and command
submission implement that contract. They are not permission to invent routing,
hotness, compression, rebalancing or a different model algorithm.

This is a **source inspection and migration plan**, not a hardware qualification
or a claim of native algorithm parity. Source pins inspected on 2026-10-30:

- Native reference: `92c86668619bb6cef60d96de0812ea48a819e9cb`.
- Portable Vulkan candidate: `847336f4e22066b48ac7665da1a74881de1df9a5`
  ([draft PR #93](https://github.com/Zutfen-LLC/Strata/pull/93)); its native
  generation/cache/kernel sources are unchanged from the native reference.
- Diagnostic-only follow-up: `97fc53b522683bb8d978ae99782dacb4f5012c1b` adds
  resident-admission operands and a post-dense free-memory observation. It does
  not implement automatic sizing or change the admission predicate.
- Explicit capacity-planning follow-up: `0216ec9b79f8109e160affa73bb5526b71466874`
  ([draft PR #95](https://github.com/Zutfen-LLC/Strata/pull/95)), stacked on PR93.
  Its post-dense sizing, immutable plan binding and aligned estimator are
  inspected separately below; other rows retain their pinned reference.

Links below pin those versions so that a flag or later refactor cannot silently
change the reference. The historical `3dc7360` roadmap and first GDN-slice status
remain in Git history; their old gap descriptions are not current status.
Start with qwen4exp Q8_0, one sequence and a primary plus one expert helper.
Other native modes remain part of product parity, not implied by this first target.

## End-to-end map

"Preserved" below describes source correspondence or an ownership contract.
It does not mean that a Vulkan kernel, driver or complete model has passed an
independent native comparison. Method, numerical, lifecycle and performance
acceptance remain separate.

| Boundary | Native Strata method | Current portable Vulkan counterpart and required work |
| --- | --- | --- |
| Discovery and capabilities | [`device_count`, `device_summary`, primary selection][n-device] and generation's architecture/runtime checks select an actual backend/device. Backend-specific reserve rules exist; they are not universal hardware constants. | [`portable_model_main`][v-main] selects ordered GGML Vulkan devices and validates totals/free/reserves. Native-stage dot/FMA logical enablement and [BDA/int64 checks][v-address] are additional Vulkan plumbing. Physical support is not proof of logical enablement. Keep explicit unsupported errors; do not bypass checks to make a run start. Product-level discovery/integration remains separate from this opt-in CLI. |
| Model load and immutable weights | [`WeightTable::load`, `NativeDense` binding in generation][n-load] and native head/embedding paths bind supported model roles once. Native role/format support is broader than this Q8 target. | [`PortableModelSession::Impl`][v-model] reads the six-shard GGUF geometry and trusted expert identity; portable dense/head/FFN owners retain their copies. [`dense_resident_weights`][v-options] is still an explicit trial, not enabled by topology. Preserve dtype/provenance, including BF16 scalar recovery and BF16 indexer widening. Do not call a CPU-resident weight with repeated backend transfers equivalent to native device residency. |
| State, scratch and schedulers | [`session_bytes`/`session_init`][n-session] carve shared GDN/QSA/MoE/block scratch and per-layer state. [`Verifier::init` and captured windows][n-verify] retain model-specific execution resources. Native packs bypass the older `SessionGraphs` route ([selection][n-window-select]). | The [model][v-model] shares GDN and QSA scratch owners per dense backend. [Device admission][v-memory] distinguishes persistent weights/state, retained schedulers/private workspace and peak temporary/conversion storage. These are GGML/backend costs, not native arena sizes. Resident/production state flags exist, but their presence does not mean a run selected them. Audit ownership and actual high-water requirements before changing a bound; an unspent conservative allowance is not measured VRAM. |
| Reserves and expert capacity | After weights/session allocations, [generation reads current free memory][n-auto]. Auto mode prices reserve and future prompt resources; its per-layer mode computes `q = floor(min(requested byte ceiling, free room) / sum(layer slot strides))` ([implementation][n-size]). [`RemoteExperts::open`][n-helper] independently accounts for helper weights/free reserve and optionally auto-sizes. | The current [portable budget invariant][v-budget-doc] combines an independent policy ceiling, backend availability, reserve and conservative ledger. Fixed-plan mode charges **configured expert tensor plus runtime budgets** and admits the frozen plan or refuses. The [diagnostic follow-up][v-expert-preflight] exposes those operands. In explicit [native-capacity mode][v-capacity], post-dense ledger and physical-free room bound the requested byte ceiling; the native per-layer formula and exact aligned cache estimate determine capacity before plans freeze. Runtime allowances, caps and reserves remain charged. This covers per-layer primary plus one helper, not native global fill, auto tail extension, multiple helpers or adaptive policies. |
| Primary/helper ownership and profile order | [`ExpertCache::admit`][n-admit] supports per-layer no-eviction quotas. [Profile fill][n-fill] continues past a full layer; the [helper][n-helper] selects first distinct unclaimed source-ranked pairs. Native also supports other modes, multiple helpers, auto tail extension and optional adaptation. | [`partition_native`][v-select] mirrors the fixed-slot, per-layer primary plus one-helper selection. It does not implement all those other modes. [`dense_device`][v-topology] explicitly puts dense/state/head on the primary; portable's default remains contiguous split. GPU-plan assets are sorted **after ownership selection**: physical slot numbers may differ from native fill order, so compare ownership, masks and restored route order rather than slot-number equality. No invented alternating/balanced policy belongs in the native mode. |
| Resident experts and CPU misses | Native [`ExpertCache`][n-cache] owns persistent expert blobs; [`RemoteExperts`][n-helper] owns helper cache/resources. Uncached experts remain CPU work through the existing `ExpertPool`, not implicit graph fallback. | The opt-in [`NativeQ8Cache`][v-cache] uses bounded GGML blocks and one fixed 80-entry scratch; its [address formation][v-address] checks owners/ranges. This is new Vulkan storage plumbing, not CUDA's allocator. The older portable per-expert graph path is a separate mode. [`LocalExperts`][v-local] dispatches plan-owned entries, restores original slots, and sends misses to [the original native CPU pool][v-cpu]. Keep weight bytes, scratch, runtime allowance and CPU work distinct. |
| Router, shared expert and combine | Native [`route_multi`][n-router], dense MMVQ/BF16 shared expert methods and [`combine_k10_vec4`][n-combine] define tie/reduction behavior and ordered first-product/FMA/shared-add arithmetic. | Vulkan [router][v-router] emulates the native logical lane order; [shared source correspondence][v-shared-doc] preserves distinct dense versus grouped reductions; [combine][v-combine] preserves K10 ordering. These are shader ports, not the native kernel binaries. Descriptor components use Logical addressing; GU/down use physical64 ([ABI distinction][v-abi]). Denorm/Exp and cross-vendor arithmetic qualifications remain explicit. Shader instruction/unit tests alone do not establish native algorithm or full-model parity. With local `--remote-expert-opt`, [native helpers weight/reduce their owned rows][n-helper-reduce] and the main device adds their partials after local combine. Portable currently restores complete unweighted rows and combines all K10 slots in original order. Matching ownership and native component arithmetic therefore do not establish the same full-FFN reduction/addition order; this remains an explicit source-level gap. The observer preserves both methods and marks unavailable complete native rows rather than stripping the helper option. |
| Scheduling and useful overlap | [`Verifier::record_window`][n-verify] publishes routes and coordinates CPU work, cached experts and shared work; shared-stream defaults differ across CUDA/HIP where source policy warrants it. Captured execution and retained arenas are part of the native method. | [`PortableModelSession`][v-model] remains serial across layer/provider boundaries, and [`LocalExperts`][v-local] submits devices synchronously. Native-method router/shared/combine does **not** imply native fork/join overlap, helper concurrency, ring scheduling or captured execution. Preserve row association and one weighting/merge while migrating scheduling in bounded slices; do not add a generic scheduler framework or claim overlap from kernel timing. |
| Prefill, decode and output head | Native [`Prefill`][n-prefill] owns model-specific chunks/resources; native decode uses Verifier windows, not a loop over independent diagnostic layer executables. Head/sampling behavior is part of generation. | [`generate`, `advance_chunk`, `advance` and `step`][v-generate] support bounded serial/chunked prompt processing and final-prompt head projection. Chunking, GDN batched projections/causal work and prompt-head elision are separate opt-ins. The current CLI is token-ID and greedy; serial eager native QSA admits context up to 4096 with host coverage, while other modes remain limited to 256. It is not native serving, general long context, sampling, request batching or MTP. Preserve requested diagnostic logits while omitting only unused heads. |
| Transfers and state authority | Native session/window state is device-authoritative in its active execution path; scratch is reused and commits follow the model's native chronology. Native CPU experts still require their intentional exchanges. | [GDN state/production][v-gdn-doc] and [QSA production ownership][v-qsa-doc] remove some mirrors when selected. Host PLE, inter-layer residual/provider transfers and explicit device commit copies remain in parts of the portable path. Diagnostic snapshots must be explicit; do not require full host mirrors in normal inference or return stale observer vectors. Production poison/reset/replay semantics differ deliberately from eager diagnostic rollback. |
| Lifetime, reset and cleanup | [`Verifier` destruction][n-verify-free], session release and cache/helper close synchronize and free their owned resources. Reuse and conversation/checkpoint behavior are broader than one process. | Layer/session RAII, [`LocalExperts` release][v-local-free] and executor release preserve borrowed-backend lifetimes; failures poison until reset/replay where specified. A zero owned-allocation counter is not proof that driver usage has returned to baseline. Fresh/resident, cancellation, partial setup and reset tests remain necessary. External stdio supervisors and post-exit telemetry polls are test harness policy, not new native model behavior or a serving implementation. |
| Performance and product completion | [Generation wall summaries][n-timing] distinguish prefill, decode and time-to-first-token; internal device phases have different boundaries. Native product support also includes conversation, sampling/stops, families/formats, platforms and serving. | Portable [phase receipts][v-timing-doc] are inclusive/coarse host scopes: do not sum nested phases or equate them to GPU-active time. Compare matched source/model/quantization, placement, residency, context, prompt, CPU workers, lifecycle, arithmetic and timing boundaries. A short GEN3 or synthetic addressing case is not a throughput, quality, power or product-parity result. Broader support remains tracked by #70. |

## Capacity policy is not selection policy

The existing fixed-slot exporter demonstrates a **selection rule**, not a fit
calculation. A request for 20 experts per layer (960 across 48 layers) plus, for
example, 1490 helper slots is an explicit configuration. Neither count is a
native default, a measured fit, or an instruction to fill all VRAM. The same
applies to older example helper counts in [PORTABLE_MODEL.md](PORTABLE_MODEL.md).
The native per-layer flag is itself optional; the native engine also supports
global profile fill and automatic capacity paths.

The explicit native-sizing mode in PR95 implements this ordering for its bounded
primary-plus-one-helper target; retain it when broadening support:

1. Bind the selected native model/method and price or allocate dense weights,
   session state, reusable scratch and future prompt requirements on their owners.
2. Read current per-device free memory. Keep the configured reserve and policy
   ceilings. Reconcile each conservative Vulkan ledger with its real resource
   lifetimes; do not remove costs just because telemetry is smaller.
3. Derive the expert byte ceiling from **both** remaining ledger capacity and
   current free room. Price aligned blocks, fixed request scratch, component
   owners and backend runtime allowance before deciding how many whole blobs fit.
   Reserve is subtracted once in each independent admission calculation; there
   is no credit from the other device and no double subtraction in one ledger.
4. Apply the native per-layer stride/quota formula and original source-rank fill;
   then select the helper's first unclaimed distinct pairs with its own capacity.
   Only then freeze/bind the plan and allocate it. Retain native no-eviction and
   first-occurrence/row-order behavior.

This is the required sequence, **not an implemented automatic Vulkan sizer**.
The existing immutable-plan API should continue refusing a plan that does not
fit; it must not silently resize/rebalance it. An explicit native-sizing entry
point must resolve capacity before plan freeze. Do not manually try smaller
quotas until one passes, turn the full configured tensor ceiling into claimed
actual bytes, copy CUDA allocation sizes onto GGML, or weaken reserve/ABI guards.

Before changing sizing, distinguish three possibilities using recorded operands:
insufficient real free memory, conservative resources genuinely retained by this
backend, or a demonstrated accounting/lifetime mismatch. Only the last permits
an accounting correction, with proof that the resource is shared, temporary or
otherwise already charged. Allocator/BDA support removes addressing limits; it
does not settle how much memory the model should retain.

## Ordered implementation slices and acceptance

Each slice uses the existing model/components and harnesses. Its change should
name a row above, the native reference function, the intended semantic result
and its decisive test. Test constants and private operator scripts do not become
production defaults. Keep failed runs and unsupported cases visible.

Post-dense accounting and native quota/rank/helper-exclusion planning are
implemented and have bounded admission evidence. Preserve the conservative
ledger, measured allocations, byte ceilings/reserve and fixed-plan refusal
semantics. This is a shared prerequisite, not a new sizing policy to redesign
for each capability below.

1. **Generalize the tested serial HC policy to ordinary boundaries once.**
   The reusable GDN/QSA cycle is implemented and has bounded real-model evidence
   in PR110/111. [PR112](https://github.com/Zutfen-LLC/Strata/pull/112), source
   `de94a24edf0c9ebcc7933fd080251479e8274d4a`, adds one default-disabled ordinary
   serial scope using native `pre`/`post` predicates, actual post-projection
   tuples, existing components and state transactions. It preserves layer1 PLE,
   materializes the nonfused final FFN write and refuses unported split/fused-head
   modes. All 48 product predicates, shared-scratch family composition and the
   final-layer write/rollback/reset are CPU-tested; HC device arithmetic is mocked.
   Broad-scope admission succeeded, while the on request stopped at a normal-only
   sigmoid guard. [PR113](https://github.com/Zutfen-LLC/Strata/pull/113) retained
   the actual rejected operands. [PR114](https://github.com/Zutfen-LLC/Strata/pull/114)
   preserves native exp/add/divide and admits only checked finite-input tail
   identities. Its tiny same-input native/Vulkan gate passed each backend's
   original expression controls and exact tail identities; normal cross-backend
   bit differences remain. The finite-exp/subnormal-reciprocal transition is still
   unsupported. The rebuilt corrected ordinary on arm completed one original
   bounded request with admitted frozen placement and finite generated output.
   All ordinary paid-work counters, own-input numerical checks, finite head/route
   receipts, process exit and baseline cleanup passed. All 283 actual HC modules
   validated; a post-run checker classification error was corrected offline while
   preserving its original STOP, without a model rerun. This establishes functional
   ordinary serial coverage for that request and removes the diagnosed layer38
   position2 refusal. It is not a corrected-source reset pair, full native
   numerical/state equivalence, native PLE/head/QSA numerical or window-scheduling
   parity. Defaults remain disabled and the finite-exp/subnormal-reciprocal
   transition remains unsupported.

   The combined validation first prices selected banks and existing buffer/host
   envelopes, then uses original post-dense planning before any model requests.
   Keep the cycle profile as a matched control. For this 48-layer geometry,
   attention layers 2–47 and FFN layers 2–47 add, relative to the current on profile,
   32 GDN attention conversions, 11 resident QSA HC banks and 44 resident generic
   native FFN banks. Before alignment, these add 197,836,800 + 145,530,880
   + 812,810,240 = 1,156,177,920 persistent bank bytes. Shared expert banks are already
   resident and are not added again. Current conservative additional host bounds
   are about 781,582,336 bytes, including deferred tuples and a materialized final
   write. Raw additional component tensor payload is about 23.6 MB; alignment,
   pipeline metadata/private workspaces and changes in replaced GGML schedulers
   still require the real ledger. These are estimates, not VRAM observations,
   fit guarantees or permission to lower bounds. The native planner must derive
   expert capacity after the actual new dense ledger; do not preserve a quota by
   inventing streaming, compression or rebalancing. If dense admission itself
   cannot fit, stop scope expansion and work on numerical-family ports within
   the already admitted cycle instead.

   Exceptions remain precise: first-layer attention has no prior write; layer 1
   PLE materializes and applies its operation before HC (`verify.cpp:1230–1310`),
   and the current GDN native-HC implementation still refuses a PLE graph.
   Preserve that separate branch rather than enabling HC before PLE. Prior
   control-vector coverage clears pending and current coverage materializes as
   part of `cvec_apply` (`post`, 2223–2231); unsupported control modes must refuse,
   not be approximated by the predicate alone. At layer 47, the native non-fused
   variant uses `gr_write` before global-head HC; `STRATA_FUSE_HEAD_GR=1` selects
   a different pending-to-head path (`verify.cpp:2075–2081,2223–2287`). Preserve the
   selected variant and refuse unsupported fusion; the existing write component
   can express the non-fused endpoint but does not port head arithmetic. Split
   handoffs must follow native transport boundaries; none is inferred from the
   current primary-dense layout. PLE, control, global head/fusion and split work
   are distinct source-defined capabilities, not ordinary-layer exceptions to
   silently bypass.

   Acceptance is a parameterized full-boundary/geometry/admission test, existing
   family cycle controls, and one combined fresh-planning/same-image diagnostic
   with actual role/tuple provenance and reset. Reuse numerical contracts and
   bounded relevant receipts, not a new capture or per-layer audit campaign.
   This closes structural ordinary serial-HC coverage; it does not establish
   native GDN/QSA arithmetic, full model quality or window/state parity. Native
   numerical-family work follows; window scheduling should not capture an
   unqualified numerical method merely to produce a speed result.
2. **Port the remaining native numerical primitives by algorithm family.**
   HC method selection does not port dense projection, GDN recurrent math,
   QSA norm/RoPE/indexer/pooling/selection/attention/output, head math or every
   expert format. Use the selected native `record_window` calls and their
   existing kernel launchers as the specification, identical inputs and existing
   deliberate intrinsic contracts. Preserve dtype conversion, masks/ties,
   accumulation/reduction association and local-helper partial-reduction order;
   portable original-row combine is not automatically identical to native
   helper reduction. Group shared primitives and shape families, not layer
   numbers. Component controls must reject incorrect math; model quality and
   first-divergence diagnostics remain separate from own-input correctness.

   [PR115](https://github.com/Zutfen-LLC/Strata/pull/115) implements the ordinary
   T1 GDN recurrence/output norm/gate component. The actual unchanged native
   launcher and Vulkan pass zero/sparse/dense own-primitive state/output controls,
   reset/repeat and module protections; the selected native delta contraction
   matched. Dense cross-backend bits differ and remain diagnostic. An operand-free
   NoContraction parser correction reused the original outputs offline; it changed
   no shader, compiler flag or numerical policy.

   [PR116](https://github.com/Zutfen-LLC/Strata/pull/116) wires that same method into
   the model's existing shared serial schedulers and eager candidate/commit path,
   default-disabled, for ordinary GDN layers with layer1 PLE explicitly unchanged.
   Resident state/weights, aligned banks and an additional explicit host allowance
   remain subject to original admission/planning. Source tests cover shared warming,
   actual graph state progression, failure noncommit and reset; recurrence device
   arithmetic is mocked there. The initial model diagnostic admitted its native
   placement but stopped at a normal-only decay-log guard. The correction at
   `9da19578577db7d7b72a1d63a25b93837b993bc8` preserves native exp/add/divide/FMA
   order, admitting legitimate exp underflow and the established sigmoid tails;
   subnormal recurrence and unqualified transition classes still refuse. Reported
   native/Vulkan tiny checks pass the original failure input and tail boundaries.

   The corrected on-model continuation passes fresh admission and exact retained
   ownership, one load and two reset-separated 25-token/GEN3 requests. Seven tests,
   first-call GDN source-order/candidate-commit checks, all 35 GDN module uses and
   exact within-arm reset replay pass; the process is reaped and three baseline
   cleanup observations pass. The prior control is original-image evidence only,
   not a corrected-image same-image comparison. This removes the decay refusal for
   these bounded requests; `qualified=false`, full numerical/state/window parity
   and quality/performance remain unclaimed. Original stopped results are retained.

   [PR117](https://github.com/Zutfen-LLC/Strata/pull/117) ports standalone native
   T1 convolution/SiLU/q-k L2 and verify/commit history. Reported native/Vulkan
   zero/sparse/dense source-order/primitive/reset checks pass; history is exact
   across backends, and dense h differences remain diagnostic. Vulkan module,
   alias/noncommit/retry/canary and both lifecycle/cleanup checks pass. The original
   output-name runner STOP is preserved and corrected offline/additively; native
   zero/build were reused. No model or a/b qualification follows; `qualified=false`.

   The native specification for this port is the ordinary T1 recurrence and output norm/gate
   selected by `Verifier::record_window()` (`verify.cpp:1510–1531`) through
   `gdn_step_norm_multi()` and `gdn_step_norm_multi_kernel()`
   (`src/kernels/cuda/verify_kernels.cu:286–383,1111–1165`). The default portable counterpart
   still composes `ggml_gated_delta_net`, RMS norm and sigmoid in
   `PortableGdnLayer::Impl::build_tail()` (`portable_gdn.cpp:195–218`); HC coverage
   does not qualify that method. Port the native state-row layout, FMA chains,
   four row-group reductions, output scale/norm/gate order and selected T1 state
   update, adapting only Vulkan storage/submission and transaction ownership.
   Start with identical synthetic post-convolution q/k/v, gate/beta, z, gamma
   and state, compare output and next state against the existing native launcher,
   and reuse deliberate intrinsic bounds. No fresh model capture is needed for
   that development. PR117 now has bounded standalone convolution/L2 evidence;
   model integration remains separate. The next unresolved numerical dependency
   is `gdn_ab_multi()` softplus/log1pf selected by `Verifier::record_window()`.
   PR118 tests the existing Metal compensated positive expression against actual
   CUDA log1pf before any a/b integration; its tighter native accuracy requirement
   remains a gate even if the weaker Vulkan log primitive guarantee passes.
   The PR118 hardware probe failed its frozen one-adjacent-step target: CUDA
   log1pf passed all cases, while Vulkan's existing Metal compensated expression
   was two steps low for a near-exp(20) boundary input. Its emitted module passed
   validation; this is a numerical dependency STOP, not model qualification.
   Native uses a strict v>20 bypass, so equality still consumes exp/log1pf.
   Retained intermediates attribute the extra step to division: the Vulkan log
   is one representable step low, multiplication is RNE, and division is one
   further step below its RNE quotient. Both primitive bounds pass while the
   composed native-accuracy target fails. Existing native_f32_div_rn.glsl would
   remove that division step for this element, but is not log1pf or evidence of
   general compensated accuracy. Complete retained counterfactuals now show
   RNE division alone still fails boundary/ordinary inputs; ideal-log/RNE-division
   controls pass, and CUDA's compensated diagnostic also retains a failure.
   No qualified log1p implementation exists in current Vulkan dependencies.
   LLVM libc's [existing correctly rounded log1pf](https://github.com/llvm/llvm-project/blob/main/libc/src/__support/math/log1pf.h)
   is an external candidate with Apache-2.0/LLVM-exception licensing and FP64
   arithmetic/support requirements. ROCm OCML's [FP32 expansion implementation](https://github.com/ROCm/ROCm-Device-Libs/blob/amd-stg-open/ocml/src/log1pF.cl)
   uses NCSA-licensed AMD builtin/expansion helpers; the required Vulkan accuracy
   is not established. Selecting either is a separate backend primitive port,
   preserving native's outer softplus method, not an already-qualified shader.
   PR119 now implements a separate standalone adapter of pinned LLVM libc's
   accurate non-FMA/RNE configuration, preserving original coefficients/tables
   and in-domain hard cases. It requires actual logical FP64 and 64-bit RTE/signed
   controls. Its bounded four-fixture native/Vulkan hardware comparison now
   passes: Vulkan matches the original LLVM oracle/FP64 controls and RNE exactly;
   native CUDA log1pf stays within the unchanged gate. Cleanup passes; sampling
   is discrete with no Vulkan in-loop sample. This is primitive evidence only.
   It does not change the frozen gate, native outer softplus method or
   PR118's failed result. No a/b integration or default enablement follows.
   PR120 now ports ordinary T1 gdn_ab_multi as a standalone component: native
   BF16 rows, continuous per-lane FMA across chunks, XOR reduction, strict v>20
   bypass and gate/beta order. Local source-order/numerical/module tests pass;
   Hermes reports all four native/Vulkan cases and cleanup passed. Native had
   four in-loop guard samples; Vulkan had none. This is bounded component
   evidence, not model parity. PR121 (`afe94b9af3dda841b9c9cb308576a77d60bd2191`)
   now adds the explicit default-disabled ordinary T1 model seam: matched
   compat-BF16 alpha/beta replacement, completed QKV/mixed borrowing, reused
   recurrence candidate leaves, aligned reservation and existing commit/rollback.
   Fourteen local tests and the isolated generated workflow pass. Hermes now
   reports a successful bounded same-image model comparison with fresh native
   planning and two reset-separated requests per arm. The on arm runs actual
   convolution, a/b and recurrence on all 35 ordinary layers; PLE/QSA remain
   excluded. First-layer source-order/intrinsic controls, exact H/gate/beta
   producer links, accepted history/state commit, protected modules and within-arm
   head/route/state-digest reset replay pass. The initial harness STOP rejected a
   valid LLVM capability receipt before READY; an exact-schema classifier repair
   reused the successful build/control/plans and ran only the new on arm. The
   original STOP is preserved. Cross-arm generated IDs match, but routes and
   matching-history logits differ; these are diagnostics, not numerical parity.
   Owned exit/reap and three baseline cleanup observations pass. This establishes
   bounded ordinary T1 integration only: qualified=false, full native numerical,
   accepted-prefix/window, quality and performance parity remain unproven.
   Layer1 PLE stays unchanged.
   The default model prefix still uses GGML convolution/L2 and a/b preparation. Dense quantized
   projections also remain separate upstream work; do not attribute unequal model
   outputs to recurrence without equal inputs. Accepted-prefix commit and window
   scheduling follow numerical-family coverage, rather than capturing the
   current GGML graph and calling it the native method.
   [PR126](https://github.com/Zutfen-LLC/Strata/pull/126), tested at
   `eab5f489c7bf8f72ae95899c547190f9aabe6e64`, adds default-disabled native
   QSA weighted normalization/RoPE after completed Q/K/indexer-query projections.
   The bounded same-image model comparison passes fresh native planning and
   matched ownership, two reset-separated requests per arm, initial layer3
   own-input controls and accepted FP16 KV-key linkage. On executes 84 calls per
   ordinary QSA layer (1,008 per request); control executes none. All 24 new
   module uses and existing HC/GDN checks pass. Within-arm state/head/route reset
   is exact. Generated IDs match; routes differ from position1/layer41 and maximum
   retained matching-history absolute logit difference is 1.3833283185958862,
   diagnostic only. Owned exit/reap and three baseline cleanup observations pass;
   resource samples are discrete. This does not establish native projections,
   indexer append/pooling, scores/selection/attention, accepted-prefix/window,
   quality or performance parity. Defaults remain unchanged and qualified=false.
   [PR127](https://github.com/Zutfen-LLC/Strata/pull/127) now ports native
   `native_qsa_indexer_append` as a standalone candidate component: FP16-round raw
   keys, retain the three-row tail and cell-zero spare, then ordered pool/norm/RoPE.
   Hermes reports all five native/Vulkan cases pass own-input state/order,
   intrinsic, reset/repeat and Vulkan canary/noncommit/retry/module controls at
   `ba070a1072f4d9adc4172f2d0480a14bff75fc88`. Cleanup passes; native has five
   in-loop samples and Vulkan none. The default portable F32/four-row tail remains
   a distinct method. [PR129](https://github.com/Zutfen-LLC/Strata/pull/129) now
   passes bounded model validation at `f68c187ed5e3d991f14cc491f90705b4e61421b4`:
   23 tests, fresh native admission/matched ownership and two reset-separated
   requests per arm. On runs 28 indexer calls per ordinary QSA layer/request;
   control none. Actual layer3 positions0..7 own-input state/math, candidate/
   FFN-commit/next-input linkage, 24 new module uses and within-arm head/route/state
   reset pass. Cross-arm retained routes and matching-history logits match in
   this run, diagnostic only. All three processes exit/reap and restore baseline
   across three observations; samples are discrete. qualified=false. Scores,
   selection, attention and native accepted-prefix/window authority remain gaps.
   [PR130](https://github.com/Zutfen-LLC/Strata/pull/130) standalone native
   multi-score/register-radix selection passes the bounded native/Vulkan component
   comparison at `a9051d67`: eight cases each, with all sixteen score/selection
   output SHA/size pairs identical. Native capacity is 2051; competitive selection
   above that width is covered synthetically. See [component scope](native-qsa-score-select-t1.md).
   [PR131](https://github.com/Zutfen-LLC/Strata/pull/131) consumes the current
   indexer candidate/projected query and feeds native IDs to existing attention.
   Hermes reports its bounded model workflow passes at `64a60447`: build and 27
   CTests, fresh planning READY/QUIT/no GEN, two reset-separated requests per
   same-image arm, exited/reaped children/controller and no sealed STOP.
   Context 256 exercises identity selection only; full native attention/window
   parity remains unproven. See [model scope](native-qsa-score-select-model-t1.md).
   The next standalone [FP16 T1 attention component](native-qsa-attention-t1.md)
   follows the existing native chunk/merge and separate gate launchers. At
   `56b01f6e`, native two CTests/eight GPU cases and Vulkan four CTests/eight cases
   pass source-order, frozen exp/sigmoid, module protections, canaries and cleanup
   (Hermes 1793571148.601549). Samples are discrete, not continuous coverage.
   The default-disabled [model adapter](native-qsa-attention-model-t1.md) consumes
   actual query/selected-ID/current FP16 KV candidates and feeds gated output to
   WO under existing FFN-success state authority. Initial model control passes;
   on initially stopped before READY on an omitted output-leaf admission estimate.
   At corrected source `0079b358`, Hermes reports the bounded on-only recovery
   passes fresh admission against retained native-selected ownership and two
   reset-separated requests, both generating `[18,24,16]`. Each ordinary QSA
   layer runs 28 chunk/merge/gate calls per request. Layer3 positions 0..7
   own-input contracts and exact query/ID/gate-to-WO links, 72 attention plus
   566 preceding modules and within-arm state/route reset pass. The control is
   reused historical source-compatible evidence, not a new same-image pair.
   Owned exit/reap, empty owners and three baseline cleanup observations pass;
   922 resource samples are discrete, not continuous coverage. Offline compact
   packaging also passes under the unchanged 128 MiB cap, retaining authenticated
   historical references; original allocation, runner and packaging STOPs remain.
   This remains bounded experimental model evidence with qualified=false.
   [PR134](https://github.com/Zutfen-LLC/Strata/pull/134) tests the existing RN
   projection-input Q8_1 candidate against the unchanged native fast-math
   quantizer. Hermes reports native five-fixture controls pass; Vulkan producers
   and cleanup pass, but the exact packed-byte gate stops on six dense int8 code
   bytes (of 6,912), with half headers unchanged. Zero/sparse are exact;
   order/adversarial cross-backend checks were not reached. Native compiled
   scale uses reciprocal-constant multiply; ratio uses reciprocal then multiply.
   Offline exact-rounding calculations on the deterministic fixture reproduce
   all six offsets/signs: RN division rounds magnitude to 63.5, then code 64;
   reciprocal/multiply rounds just below, then code 63. This supports a numerical
   mismatch in the RN candidate, not a demonstrated adapter defect or FTZ issue.
   It is not an independent capture of the native reciprocal intermediate.
   The RN STOP and frozen exact gate remain. A separate reciprocal-then-multiply
   observation at `32fa5015` subsequently passes the bounded Vulkan-only comparison
   against retained native `ef63ef8f`, as reported by Hermes: all five frozen
   fixtures match all 6,912 packed bytes, including dense. Seven build tests,
   independent literal controls, repeat/canary/fixture/module checks and three
   baseline cleanup observations pass; no native rerun. This covers only the
   selected normal-domain fixtures and exact compiler/backend bindings. It does
   not establish general MUFU.RCP equivalence or model integration. Vulkan had
   no in-loop guard sample; qualified=false remains. Terra owns this separate
   candidate; the prior RN implementation and failed result are unchanged.
   The subsequent same-input replay at `e9ed479c` also passes, as reported by
   Hermes: eight retained layer3 gated inputs, positions 0..7, match all 55,296
   packed output bytes across native CUDA and Vulkan. Exact input echoes,
   canaries, repeats, modules, process exit/reap and three baseline cleanup
   observations pass; native has eight in-loop samples and Vulkan none.
   Inputs are from the portable model's actual gated candidate, supplied equally
   to each quantizer; this does not establish upstream native attention equality.
   Literal controls remain separate; no model rerun or integration was performed.
   [PR135](https://github.com/Zutfen-LLC/Strata/pull/135) adds the standalone
   ordinary unpacked Q8_0 T1 WO consumer at `ba1a8a8d`. The selected CUDA SASS
   matches the candidate scale-multiply/FMA and ordered four-warp/XOR reduction
   association in its supported finite fixture domain. The bounded physical
   comparison passes: all 2,560 rows / 10,240 bytes are exact, with literal,
   repeat/canary, actual module and lifecycle checks passing. Both hosts restore
   baseline across three observations and have no selected owners; each has one
   in-loop sample plus baseline, not continuous coverage. Collected output bytes
   and retained receipt manifests were independently checked. This original
   component excluded half-subnormal scales and did not compose the producer.
   [PR136](https://github.com/Zutfen-LLC/Strata/pull/136) subsequently connects
   the actual gated tensor through one retained packed buffer to native WO and
   the existing FFN commit boundary. Nine component cases match both packed and
   projected outputs exactly. Real-role preflight first refused 193 finite
   half-subnormal weight scales; [PR137](https://github.com/Zutfen-LLC/Strata/pull/137)
   faithfully admits them while retaining nonfinite refusal and the Q8_1 input
   restrictions. Native/Vulkan qualification matches all 2,560 literal output
   words, including all 2,046 signed nonzero half-subnormal encodings and the
   boundary, multiplication and cancellation controls. Kernels and shaders are
   unchanged; all twelve real WO role identities and scale counts are preserved.
   At source `51c4e65f`, the same-image model comparison passes 36 CPU tests,
   fresh native planning/frozen ownership/admission, and two reset-separated
   25-prompt-token GEN3 requests per arm. Both arms generate `[18,24,16]`.
   Twelve WO owners, 24 modules, eight layer3 own-input native packed buffers
   and mapped-role WO outputs, and within-arm state/route/reset replay pass.
   Owned exit/reap, restored baseline/clocks and empty owner scans pass. The
   sealed manifest and 23 retained-file references were independently checked;
   referenced original outputs must remain available. Earlier preflight/build
   stops remain preserved. Cross-arm logits differ (maximum retained matching-
   history absolute difference 1.3866307735); the first route-order difference
   is position 0 / layer 34, swapping IDs 9 and 163 within the same selected set.
   The arms deliberately compare GGML WO with the native method. Exact equality
   is not their acceptance criterion; these differences alone establish no new
   defect or full-model parity.
   Its precise downstream attribution has not been established. Sampling is
   discrete, not continuous; qualified=false and full_model_parity=false remain.
   [PR138](https://github.com/Zutfen-LLC/Strata/pull/138) adds default-disabled
   native T1 Q/K/V quantization and Q8_0 projections before norm/RoPE. One
   actual HC-mixed input produces the shared packed buffer for all three
   projections; Q/K producer-owned Vulkan banks feed normalization and the
   interleaved Q/gate bank feeds attention. The original component comparison
   passes exact packed and projected bytes. Its model attempt first exposed
   scheduler-owned input leaves, then a conservative Q8_1 half-scale refusal;
   both STOPs remain preserved. [PR140](https://github.com/Zutfen-LLC/Strata/pull/140)
   supports finite stored half-subnormal scales only for the 80-block QKV path,
   preserving full-F32 restrictions and the separate WO producer/reference
   domain. All 16 native-first constructed cases pass exact packed/projected
   comparisons and direct-consumer controls, covering all 1,023 positive
   half-subnormal scale encodings plus boundaries. This does not establish
   general reciprocal equivalence or replay the unretained rejected input.
   At corrected source `02c6d2b5`, fresh planning admits 96 primary / 1,490 helper
   experts; both same-image arms complete two reset-separated requests. Final
   checks pass all eight layer3 QKV and eight WO positions: same-input native
   packed bytes, mapped-role projections, actual consumer links, work counters,
   state/reset replay, modules, admission and owned exit/reap/cleanup. The QKV
   check covers twelve owners and 24 modules; WO checks cover 24 owners and
   48 modules. An owner-aware checker-path repair completed finalization offline,
   resolving the authenticated runtime owner rather than hard-coding its suffix.
   Ten resolver regressions pass; the 192-byte regenerated GDN derivative
   exactly matches the retained one. No model/GPU/replay/build rerun was needed
   for that repair, and prior failures remain preserved. The final result has
   6,096 manifest members and 25 host-retained references, totaling 133,578,533
   copied bytes including the manifest, below the unchanged 128 MiB cap.
   Cross-arm differences remain: first route-order difference position 0 /
   layer 8; largest retained matching-history absolute logit difference
   1.538769781589508 at position 24. These are diagnostics without a final-logit
   tolerance or quality/performance claim. Full state arrays are not exported;
   qualified=false, full_model_parity=false and native_window_parity=false remain.
   [PR141](https://github.com/Zutfen-LLC/Strata/pull/141) adds the ordinary T1
   BF16 indexer key/query projections from one actual HC-mixed input, with packed
   BF16 banks and producer-owned buffers feeding key append and query norm/RoPE.
   The native-first constructed component compares 5,120 output bytes exactly.
   At corrected model source `f61573b7`, two reset-separated 25-token / GEN3
   requests pass with QKV/WO experiments off. This uses a historical compatible
   control at `6c593d49`, not a same-image control. Fresh admission selects
   96 primary / 1,490 helper experts under the original bounds and frozen plans.
   Each request records 672 projection completions across twelve QSA layers.
   Checks pass twelve BF16 modules in addition to the existing 566 modules,
   eight actual producer/consumer positions, accepted state links and reset replay.
   All eight executed-head hashes match the retained control; matching-history
   logits at positions 24–27 are exact, with no route difference observed.
   These bounded observations do not establish general numerical or model parity.
   Offline import/receipt repairs completed checking and sealing without repeating
   model, component or GPU execution; original STOPs and 6,644 raw files remain
   preserved. Owned children exited/reaped, owner scans were clear, three baseline
   observations and automatic clocks were restored. The final seal contains 6,611
   members and 35 retained references, totaling 134,091,830 copied bytes including
   the manifest, below the unchanged 128 MiB cap. Normal-or-zero input/intermediate
   restrictions remain explicit; stored BF16 role scans alone do not qualify all
   inputs. The path remains diagnostic and default-disabled: qualified=false and
   full_model_parity=false, with no quality or performance claim.
   [PR142](https://github.com/Zutfen-LLC/Strata/pull/142) adds ordinary T1 GDN
   QKV and gate-input projections from one shared packed HC-mixed input. Actual
   producer banks feed convolution and recurrence; a/b retains the mixed F32
   input. Existing resident Q8_0 weights are reused, with added aligned banks and
   host controls charged to admission. The native-first component passes both
   frozen cases exactly: 2,880 packed bytes, 40,960 QKV bytes and 24,576 z bytes
   per case. Compiled native instruction review confirms the selected quantizer,
   DP4A/FMA association, reduction order and FTZ scope; it does not establish
   general reciprocal equivalence. All seventy real Q8_0 roles pass the finite
   stored-scale scan, including 825 half-subnormal scales; activation/intermediate
   domain checks remain separate.
   At `f5ec799c`, fresh same-image control/on planning selects 96 primary /
   1,490 helper experts under unchanged policy. Each arm completes two
   reset-separated 25-token / GEN3 requests, with QSA QKV, WO and BF16-indexer
   projection experiments off in both arms. The dedicated checker passes thirty-five
   GDN input owners / seventy modules, eight layer0 same-input native packed replays,
   mapped-role outputs and actual convolution/recurrence/a-b links. Existing checks
   also pass 566 HC module entries and the separate numerical-family receipts,
   within-arm state/reset/routes, admission, owned exit/reap and three baseline
   cleanup observations. Offline finalization passes without a GPU rerun; the
   retained final manifest has 6,732 entries and 37 large-artifact references.
   Whole-model divergence remains: first route-order difference position 0 /
   layer 1 swaps the order within the same expert set; largest retained
   matching-history absolute logit difference is 1.5297669172286987 at position 27.
   These diagnostics are not a final-logit tolerance or proof of defect attribution.
   The path stays default-disabled, qualified=false, full_model_parity=false and
   native_window_parity=false; no quality or performance claim is made. Full state
   arrays are omitted. Current resident snapshots preserve long-lived non-kworker
   identities; seven model-host kworkers ended. Historical CPU snapshot gaps,
   inaccessible root-cgroup fields and discrete monitoring remain explicit limits.
   [PR144](https://github.com/Zutfen-LLC/Strata/pull/144) completes the default-disabled
   ordinary T1 GDN output projection trial: actual gated recurrence feeds the shared
   Q8_1 producer, and the persistent native-order Q8_0 output bank feeds FFN HC.
   GDN-only finite stored half-scale admission is separately qualified; legacy WO
   and 80-block QKV domains remain distinct. Seven exhaustive/boundary cases plus
   the retained stopped input match all 6,912 packed and 10,240 projected bytes in
   native-first/Vulkan comparison. At source `6bd7335d`, the corrected model candidate
   completes two reset-separated requests using fresh admission and retained 96/1490
   plans; the control is historical-source-compatible at `0c71c648`, not same-image.
   Eight new candidate inputs pass exact native packed replay, actual mapped-role
   outputs and recurrence/FFN links; 35 output owners / 70 modules, reset, counters
   and cleanup pass. Final offline sealing preserves the original failures and all
   7,210 prior files; 19 sealer tests pass with byte-identical numerical revalidation.
   The final result has 7,180 manifest entries and 44 retained references.
   First route-order divergence is position 0 / layer 0 (experts 414 and 2 exchange
   order); largest matching-history absolute logit difference is 1.3079073429107666
   at position 27. Qualified, full-model and native-window parity remain false;
   no numerical tolerance, quality or performance claim is introduced. Details and
   limits are in [the output trial](native-gdn-output-t1.md).
   Final-head HC mixing/output projection, quantized KV, accepted-prefix and native
   window scheduling remain separate gaps.



3. **Replicate native state authority and commit/restore.** Native
   `Verifier::capture_commit()`/`Verifier::commit()` and restore paths handle GDN history/state and
   QSA KV/indexer tails for the accepted prefix. Reuse the existing session
   owners; port those transitions and remove host mirrors only when ownership
   and lifetime are proved. Test accepted-prefix/tail, reset A/B/A, failures,
   partial setup and bounded retained memory. Eager rollback and production
   poison/reset are different contracts; neither is evidence of speculative
   native-window commit parity. Keep unsupported combinations refusing.
4. **Replicate the native window's execution dependencies.** Port the selected
   `record_window()` group order, route publication, CPU/helper work, shared
   fork/join and reusable execution resources to Vulkan submissions/barriers.
   Device commands differ; the dependency graph and arithmetic policy do not.
   Then cover native T1/T8/prefill/decode windows and accepted-prefix commit,
   followed by the selected speculative/batching behavior. Do not introduce a
   generic replacement scheduler or infer overlap from individual kernel time.
   Check row association, one weighting/combine, segmentation/tails, final-prompt
   heads and state; measure actual transfers/submissions and bounded allocations.
5. **Handle native boundary exceptions and product coverage explicitly.**
   Layer0 starts without a prior pending tuple; layer1 PLE must materialize and
   apply its history-dependent operation first (`pre`, lines1230–1310).
   Control-vector coverage materializes before its operation, final-layer/head
   paths may use `gr_write` or fused head-GR, and pipeline handoff must express
   native transport boundaries (`post`/head, lines2223–2285). These are distinct
   source-defined behaviors, not reasons for inventing per-layer policies.
   Keep currently unsupported control/split/head-fusion modes explicit until
   ported. Broaden context/KV formats, quant roles, sampler/stops/checkpoints,
   serving/cancellation and model families using their existing native methods.
   Run matched native/Vulkan end-to-end quality and performance only for an
   explicitly supported configuration; token matches alone are not quality,
   native arithmetic, full-HC or speed qualification.

Preserve existing correctness accommodations while migrating: the
[pinned-backend per-head QSA graph][v-qsa-graph], exact BF16 indexer widening and
stable tie behavior must not be removed as incidental optimization. Replace a
workaround only after its affected regression and full relevant shape/cache tests
pass. Worker qualification, transport/security and distributed latency remain in
[#23](https://github.com/Zutfen-LLC/Strata/issues/23),
[#26](https://github.com/Zutfen-LLC/Strata/issues/26) and
[#11](https://github.com/Zutfen-LLC/Strata/issues/11); this map does not introduce
another worker protocol.

## Decisions to make explicitly

No user decision is needed to preserve current guards or document an existing
gap. Before broadening a slice, agree which native configuration is the target:
per-layer or global fill, fixed or auto capacity, helper count, prompt mode and
supported model/quant roles. These are native choices, not interchangeable flags.
Choose cross-backend acceptance criteria before declaring parity; strict
per-boundary identity, allowed numerical differences and task quality must not
be silently substituted for one another. User-set reserves/caps remain binding;
a platform-specific change needs separate evidence and a visible decision.

Keep private hardware results, operator paths and artifact links out of this
public map. Refer to source/test contracts here; retain execution evidence in
its appropriate private location. This document changes no runtime default and
does not authorize a new physical run.

[n-device]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/device.cu#L78
[n-load]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/program/generate.cpp#L4012
[n-session]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/session.cpp#L56
[n-verify]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/verify.cpp#L659
[n-window-select]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/program/generate.cpp#L7219
[n-auto]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/program/generate.cpp#L6087
[n-size]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/program/generate.cpp#L6234
[n-helper]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/remote_experts.cpp#L135
[n-admit]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/expert_cache.cpp#L643
[n-fill]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/program/generate.cpp#L6445
[n-cache]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/expert_cache.cpp
[n-router]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/kernels/cuda/native_router.cu#L100
[n-combine]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/kernels/cuda/native_moe.cu#L66
[n-prefill]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/include/strata/prefill/prefill.hpp
[n-verify-free]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/verify.cpp#L586
[n-timing]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/program/generate.cpp#L16981
[v-main]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_model_main.cpp#L304
[v-address]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/strata_vk_q8_gu.inc#L10
[v-model]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_model.cpp
[v-options]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/include/strata/core/portable_model.hpp#L51
[v-memory]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_memory.hpp#L40
[v-budget-doc]: PORTABLE_DEVICE_BUDGETS.md
[v-expert-preflight]: https://github.com/Zutfen-LLC/Strata/blob/97fc53b522683bb8d978ae99782dacb4f5012c1b/src/core/portable_memory.hpp#L37
[v-select]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/tools/export_local_ranked_assignments.py#L89
[v-topology]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_topology.hpp#L6
[v-cache]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/worker/vulkan_native_q8_cache.hpp#L40
[v-local]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_local_experts.cpp#L130
[v-cpu]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_cpu_experts.hpp#L270
[v-router]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/kernels/vulkan/native_router.comp#L1
[v-shared-doc]: native-shared-vulkan.md#native-binding-and-arithmetic
[v-combine]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/kernels/vulkan/native_moe_combine.comp#L15
[v-abi]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/native_q8_gu_spirv.hpp#L21
[v-generate]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_model.cpp#L721
[v-gdn-doc]: PORTABLE_MODEL.md#gdn-execution-policy
[v-qsa-doc]: PORTABLE_MODEL.md#qsa-device-state-policy
[v-local-free]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_local_experts.cpp#L115
[v-timing-doc]: PORTABLE_MODEL.md#coarse-model-phase-totals

[v-qsa-graph]: https://github.com/Zutfen-LLC/Strata/blob/847336f4e22066b48ac7665da1a74881de1df9a5/src/core/portable_qsa_graph.hpp#L6

[v-capacity]: https://github.com/Zutfen-LLC/Strata/blob/0216ec9b79f8109e160affa73bb5526b71466874/src/core/portable_model.cpp#L403

[n-helper-reduce]: https://github.com/Zutfen-LLC/Strata/blob/92c86668619bb6cef60d96de0812ea48a819e9cb/src/core/remote_expert_opt.cu#L23
