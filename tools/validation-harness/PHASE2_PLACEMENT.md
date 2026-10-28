# Phase 2: one fixed multi-layer placement

Prospective parent-only admission; no hardware run or credential work is performed
by this change. Preserve the original twelve-case result and PR35 phase-1 artifacts.
Reuse the identical 100-case inventory, golds, scoring, numeric limits, prompt IDs,
32-new-token cap, accepted inference binaries and CPU resident on 02.

## Fixed placement policy before outputs

Prefer **32 CPU-backed Q8 experts across layers 0, 12, 24 and 47**, eight per layer,
from the existing trusted CPU assignment and retained bounded routing inventory.
Keep layer0/expert302 if it is still eligible; fill its layer's remaining slots
with the lowest expert IDs in that eligible inventory. On the other layers use
the eight lowest eligible IDs. Eligibility requires the current source identity,
CPU ownership, no local/other GPU assignment overlap, trusted source digest and
an available blob or existing selective source reader. No candidate-quality result
enters selection. If any layer has fewer than eight eligible entries but all have
at least four, freeze **16 total**, four per layer by the same policy. Otherwise
return the exact inventory shortage; do not launch a new capture or search campaign.

The production worker accepts multiple layers and experts, keeps one graph per
asset and looks up slots by `(layer, expert)`. Its bound is the configured tensor
budget and validated geometry, not two experts. The offline real-row probe's
one/two-expert and first-three-record limits do not apply to model inference.
Do not enlarge or require that probe for this full-inference screen.

## Budget and headroom

Each H2560/FF640 Q8_0 expert has 5,222,400 weight bytes:

| Experts | Weight bytes | Weight MiB | Proposed owned-tensor cap | Dedicated disk cache cap |
| --- | ---: | ---: | ---: | ---: |
| 32 | 167,116,800 | 159.375 | 256 MiB | 256 MiB |
| 16 | 83,558,400 | 79.6875 | 256 MiB | 256 MiB |

Read the accepted same-die single-expert `gpu_allocated` value A from its retained
receipt. Require `N*A <= 268435456` before admission; the same graph geometry and
backend determine per-graph padded tensor size. Worker preparation independently
enforces the actual summed allocations. Each graph must also be strictly below
16 MiB in the existing implementation; thus 32 graphs are strictly below 512 MiB
of owned tensor storage, even before using A. No engine change or permanent budget
increase is required. CLI supports tensor budgets up to 1,048,576 MiB and cache
budgets up to 1,073,741,824 MiB; those parser ceilings are not safe operating sizes.

The owned cap covers weights, tensor intermediates, padding and alignment. It does
not cover internal ggml compute buffers, pipelines, driver allocations, host memory
or all VRAM. Vulkan requires runtime reserve zero, so this is not an allocator-
enforced VRAM reservation. For the 8 GiB die require at least 6 GiB free before
setup and retain at least 4 GiB free after preparation/during the bounded screen,
using existing device-specific telemetry/guarding. Stop before GEN if preparation
fails or headroom is insufficient; do not silently raise limits or reduce/change
assignments after outputs. Do not report unavailable telemetry as a reserve pass.
The 256 MiB cache cap counts retained/unrelated files and pending blobs. Use a
fresh owned cache or verify its complete existing inventory; no automatic eviction
or deletion is requested. The RAM upgrade does not change this GPU accounting.

## Smallest read-only inventory request to operator, for parent dispatch

Return only these existing private receipt/metadata facts:

1. Current trusted source manifest and CPU-backed assignment paths/hashes/model-
   layout identities; local and other GPU placement identities for overlap checks.
2. Retained `--inventory` ASSET/ENTRY metadata for layers 0,12,24,47: eligible
   `(layer,expert)`, source digest, blob basename, bytes and availability; counts
   per layer, plus the already accepted `(0,302)` identity. Prefer the saved
   inventory. If absent, run the existing CPU-capable `gpu_capture_replay_check
   --inventory CAPTURE BLOB_DIR B_PLAN...` once on the retained capture and at
   most four trusted disjoint plans: at most 16 sampled records/225 entries, no
   compute replay or source-weight scan. Confirm its identity against current
   metadata rather than assuming historical assignments still apply.
3. Selected die's PCI identity, total/used/free VRAM from existing read-only
   device telemetry, the single-expert allocation A and accepted launch/binary
   receipts; existing free disk space for the dedicated cache and run artifacts.

Full GGUF shard presence alone does not supply trusted assignment/blob identities.
If blobs are missing, the existing GpuAssetSource reader can later extract only
selected expert ranges from the trusted source; no full model move is needed.
No keys, package installs, model loads, new captures or profiling are in this
inventory request. Do not dispatch it from this development task.

After parent admits the concrete inventory, use the existing metadata-only
`strata-gpu-manifest plan LOCAL_MANIFEST ASSIGNMENTS NEW_PLAN` command, freeze its
hash, and stage only selected real blobs. Reuse the 100 frozen GEN lines byte for
byte; create a new source/plan manifest before both arms because this scorer's
scope-label correction changes its source hash. Keep prior manifests immutable.
Run one CPU and one candidate suite: 200 requests, at most 6,400 generated tokens.
Same one endpoint/die, full CPU backing on 02, mTLS/NODELAY, owned PIDs before waits,
finite limits and resident-preserving cleanup. Parent handles dispatch and any
separate temporary credential approval.

Reports now identify the actual plan SHA-256 instead of asserting expert302.
Per-case participation remains endpoint-wide. Disclose that it does not establish
that every assigned expert/layer executed. Retained routing proves selection
provenance, not execution in these 100 new cases; do not overstate coverage or add
a profiling framework to resolve it. No statistical equivalence or speedup claim.

Local validation: seven `test_quality_v2.py` tests passed, including generic scope
and exact plan-hash assertions in the synthetic full-report test; Python compilation
and `git diff --check` passed. Inventory/golds, scoring functions, numeric limits
and all inference sources are unchanged. No real placement or hardware result is
claimed by these host-only checks.
