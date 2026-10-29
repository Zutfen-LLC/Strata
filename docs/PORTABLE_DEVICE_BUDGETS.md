# Portable device budgets

The portable CLI and the shared validation selector accept a scalar
`--device-cap-mib 7680` or ordered caps such as `--device-cap-mib 11776,7680`.
The two values follow **the order in `--vulkan`** (harness: `--devices`), not
NVIDIA indices or sorted PCI addresses. Expert plan slots, tensor/workspace
budgets and reserves use that same order. Scalar shorthand and the harness's
AMD identities, split and 7680 MiB default remain unchanged.

## Admission invariant

For each physical device independently:

- Policy ceiling P must be positive, overflow-safe and no larger than either
  verified physical total or backend device-local total.
- Available memory A is the minimum of physical free telemetry and backend
  effective free/budget evidence. Missing, invalid or mismatched identities and
  memory observations stop admission.
- Reserve R is at least 512 MiB, or the larger configured expert free reserve.
- The conservative ledger L must fit **L <= min(P, A - R)**, with A > R.

The ledger retains dense weights, persistent state, graph/scheduler allocations,
backend private workspace, staging/conversion and token temporary bounds, plus
configured expert tensor and workspace budgets. It uses the existing aligned
allocation accounting and conservative overlap bounds. Sampled allocation peaks
are receipts, not permission to omit costs or increase a limit. Memory on the
other device cannot cover a shortage. Expert tensor policy is charged even if
its assignment happens to allocate less.

P is a ceiling, not a request to allocate all P bytes. For example, an 8192 MiB
card with 9 MiB baseline usage has A=8183 MiB. With P=7680 MiB and R=512 MiB,
the effective admission limit is 7671 MiB. A conservative ledger of 7550 MiB fits;
requiring P+R <= A would wrongly reject it. Reserve is outside L and subtracted
once. The expert loader still checks current free against tensor + workspace +
reserve after dense allocation; that is the remaining allocation, not a second
reserve charge in the ledger.

The harness records policy, baseline usage, physical/backend totals, A, R and the
effective limit. Its `--device-free-bytes A,B` observation can only tighten the
engine's live backend admission. The engine records its possibly tighter limit,
P, R and the final admitted ledger in stderr. The existing `cap_bytes` field now
names the effective ledger limit; `policy_cap_bytes` names the configured ceiling.
Runtime guards retain physical identity/total checks, per-device aggregate growth
limits and free >= R. Engine backend checks also enforce the reserve at model
load and token/chunk checkpoints. Sampling does not prove every transient peak.

Fresh and resident references must bind ordered physical PCI identities, logical
indices, split, configured caps, expert plans and execution policy. Free-memory
observations may differ between launches and only tighten admission; they are not
numerical policy. A larger expert set is a changed placement, even with the same
quantization and CPU settings. It needs new matching fresh references.

## Bounded next candidate, after hardware preflight

This is a recipe, not a fit claim or authorization to run a model. Do not create
new fleet placements as part of this change.

1. Pin source/binary and the retained STRP profile, trusted manifest and previous
   assignments by SHA-256. Use the same quantization, CPU workers/threads, context,
   prompt, cache policy, execution flags and deadlines. If using batched F32 GDN
   projections, integrate the separate PR #82 correction and pin that combined
   source; this budget change does not include it.
2. Read `portable_model --list-vulkan` and physical vendor telemetry. Verify PCI
   identities and selected order; for a heterogeneous pair, put the 12 GiB device
   in slot 0 and the 8 GiB device in slot 1. Record unrelated usage and both free
   observations. Stop on missing or contradictory evidence. Do not unload other
   work to make a plan fit.
3. Use `tools/export_local_ranked_assignments.py` with its existing pinned profile
   and `--per-layer-per-device 24` in a **new** metadata directory. Its retained
   rank receipt can describe a conservative 24/16 per-layer candidate (1152/768):
   keep all slot-0 selections and only the first 16 slot-1 selections in source
   rank order within each layer. Freeze every exact `(layer, expert)` ID and the
   discarded tail; do not rerank, replace IDs or use outputs to choose experts.
   Verify any previously retained prefix explicitly. Use the existing
   `strata-gpu-manifest plan LOCAL_MANIFEST ASSIGNMENTS OUTPUT` for each frozen
   assignment. This needs no new planner or incremental 48-expert sweep.
4. Predeclare asymmetric policy ceilings from observed totals, separate tensor
   caps covering aligned plan allocations, workspace budgets and reserve. Weight
   bytes alone do not establish fit. Reversed backends may have different buffer
   sizes; previous-order ledger estimates are hypotheses. Require the recomputed
   full ledger to fit each effective limit before accepting any run.
5. The minimum next model check is one separately authorized, guarded load and
   bounded request with retained placement/memory receipts, followed by matching
   fresh/reference checks before performance comparison. A failure stops the
   attempt. Do not silently change caps, expert IDs, CPU settings or numerics.
