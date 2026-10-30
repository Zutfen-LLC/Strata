# Strata method: CUDA/HIP to Vulkan migration map

Tracking: [umbrella #65](https://github.com/Zutfen-LLC/Strata/issues/65), with
[GDN #66](https://github.com/Zutfen-LLC/Strata/issues/66),
[QSA #67](https://github.com/Zutfen-LLC/Strata/issues/67),
[scheduling #68](https://github.com/Zutfen-LLC/Strata/issues/68),
[prefill #69](https://github.com/Zutfen-LLC/Strata/issues/69) and
[product breadth #70](https://github.com/Zutfen-LLC/Strata/issues/70).
This updates the existing roadmap rather than introducing a parallel plan.

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
| Router, shared expert and combine | Native [`route_multi`][n-router], dense MMVQ/BF16 shared expert methods and [`combine_k10_vec4`][n-combine] define tie/reduction behavior and ordered first-product/FMA/shared-add arithmetic. | Vulkan [router][v-router] emulates the native logical lane order; [shared source correspondence][v-shared-doc] preserves distinct dense versus grouped reductions; [combine][v-combine] preserves K10 ordering. These are shader ports, not the native kernel binaries. Descriptor components use Logical addressing; GU/down use physical64 ([ABI distinction][v-abi]). Denorm/Exp and cross-vendor arithmetic qualifications remain explicit. Shader instruction/unit tests alone do not establish native algorithm or full-model parity. |
| Scheduling and useful overlap | [`Verifier::record_window`][n-verify] publishes routes and coordinates CPU work, cached experts and shared work; shared-stream defaults differ across CUDA/HIP where source policy warrants it. Captured execution and retained arenas are part of the native method. | [`PortableModelSession`][v-model] remains serial across layer/provider boundaries, and [`LocalExperts`][v-local] submits devices synchronously. Native-method router/shared/combine does **not** imply native fork/join overlap, helper concurrency, ring scheduling or captured execution. Preserve row association and one weighting/merge while migrating scheduling in bounded slices; do not add a generic scheduler framework or claim overlap from kernel timing. |
| Prefill, decode and output head | Native [`Prefill`][n-prefill] owns model-specific chunks/resources; native decode uses Verifier windows, not a loop over independent diagnostic layer executables. Head/sampling behavior is part of generation. | [`generate`, `advance_chunk`, `advance` and `step`][v-generate] support bounded serial/chunked prompt processing and final-prompt head projection. Chunking, GDN batched projections/causal work and prompt-head elision are separate opt-ins. The current CLI is token-ID, greedy and context <=256; it is not native serving, long context, sampling, request batching or MTP. Preserve requested diagnostic logits while omitting only unused heads. |
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

1. **Reconcile post-dense capacity before another sizing change.** The source
   audit is complete for this target: no duplicate retained owner was established,
   so the conservative ledger is retained. Preserve the
   diagnostic-only PR93 follow-up. Host tests must cover both devices, exact fit,
   one-byte refusal, overflow, no ledger mutation and no free-memory observation
   granting ledger credit. Once this map is adopted, the smallest decisive
   physical check is one guarded process with the *unchanged* plan/budgets and
   source pin, recording failing die, every dense operand, current free/total,
   effective cap and shortfall before refusal. No load-only-plus-GEN duplicate,
   guessed smaller quotas, changed reserves or admission bypass. Use the data to
   choose between a resource-lifetime correction and the native sizing slice.
2. **Implement native capacity planning at the existing load/plan seam.**
   Implemented for the explicit bounded mode in PR95, with host golden,
   immutable-plan and shared allocator-estimator tests. Preserve
   explicit fixed-plan behavior and only the supported native sizing mode,
   applying the byte-ceiling/quota/rank sequence above. Golden host cases compare
   with the native formula: equal/unequal layer sizes, exact/short room, alignment,
   primary/helper exclusion, duplicate/short profiles and overflow. Refuse missing
   memory/feature evidence and unsupported modes. A bounded real-model admission
   must reconcile complete dense+expert ledgers and actual allocations, retain
   the reserve, and execute the same request in that one process. No fit claim
   precedes this gate.
3. **Match native ownership/state and qualify arithmetic separately.** Select
   dense/state/production policies explicitly; do not infer them from topology or
   native FFN flags. Remove only proven redundant mirrors/owners. Test reset A/B/A,
   tails, partial setup, nonfinite/provider failure, poisoning and bounded memory.
   Compare router/shared/grouped/combine boundaries against existing native
   launchers, using identical real operands, original masks/ties/order and explicit
   numerical criteria. CPU-only/shader tests, same-backend replays and generated
   token matches remain different evidence. Preserve independent quality checks.
4. **Migrate scheduling and prefill without changing the model.** Follow native
   route-ready/shared fork/join, useful CPU/helper overlap and reusable execution
   resources. Make one seam change at a time; captured-window/command work is a
   later focused slice, not permission for an immediate graph rewrite. Compare
   T1 versus bounded chunks, arbitrary segmentation and tail states, final-prompt
   logits, original routes and one combine. Measure transfer/readback/submission
   counts and exposed wall time; verify no per-step allocation growth. Fixing
   correctness alone is not a speed result.
5. **Broaden product behavior, then measure matched end-to-end performance.**
   Track context/KV/RoPE, quant roles, samplers/stops, reuse/checkpoints, serving,
   cancellation, batching/MTP, families and platforms under #70 in bounded slices.
   Unsupported configurations must fail clearly. Benchmark only after the chosen
   arithmetic, residency, placement and lifecycle gates pass, using matched cold
   and resident cases and honest prefill/decode/TTFT boundaries. Do not select a
   different policy merely to improve a benchmark.

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
