# Per-layer ranked placement: metadata preparation only

This optional candidate policy stacks on PR46. The default exporter remains the
first 960 distinct global source-ranked pairs, with its four output files
byte-identical to PR46 for the same input. All current hardware jobs stay pinned
to their existing binaries, plans and profiles. **Stop after metadata checks;
no physical run is released by this recipe.**

Parent's actual metadata check found that the global ranked960 candidate covers
all 48 layers, but combined counts range from 1 to 40 per layer. Native uses
per-layer placement. This motivates a separate uniform-quota candidate; it does
not demonstrate workload heat, better quality or performance.

`--per-layer-per-device N` requires an integer from 1 through 256. Within each
of the 48 layers, take the first 2N distinct pairs in the original profile order,
assigning alternating pairs to die0 and die1, starting with die0 for each layer.
Only then sort each device's assignment file. Reject any layer lacking 2N
distinct pairs. The existing full-profile SHA, STRP v1 geometry/bounds/table
checks and new-directory requirement remain in force. No router, weight,
frequency, CUDA plan or runtime behavior changes. `selected-ranks.tsv` retains
the original indices; the opt-in receipt adds N and per-device layer counts.

Initial candidate N=23 selects 23 per layer per die: 1104 per die, 2208 combined,
with no duplicate ownership. Use the original 196632-byte profile with SHA256
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
Its original generation trace is unknown; its order is a placement hypothesis.
Retain the origin evidence in the [PR46 handoff](portable-local-ranked-placement-v1.md#source-contract-and-open-evidence).
Do not regenerate or reorder the profile to fill a quota.

## Fresh source and metadata outputs

Parent exports the entire tracked source tree at this new draft's exact head,
using the full-source/archive-hash procedure in the PR46 handoff. Retain the
independently supplied commit and archive SHA. Use a fresh secondary-disk artifact
root and a complete copy of the original small profile. Reuse the existing
manifest and manifest-tool executable with their independently retained hashes;
do not invent a tool identity, rebuild C++ or copy model/cache payloads.

The following is for later parent/Hermes metadata preparation, not execution by
this coding task. ROOT is a fresh directory below the established artifact root;
SOURCE is the verified complete source export. Run outside a hardware clock
window. All outputs are fresh and previous plans/profiles/results remain intact.

```sh
set -eu
: "${ROOT:?fresh metadata artifact directory}"
: "${SOURCE:?verified complete source export of this new draft head}"
: "${PYTHON05:?existing Python}"
: "${PROFILE:?complete original profile copy}"
: "${MANIFEST:?existing local model manifest}"
: "${MANIFEST_SHA:?independently retained manifest SHA256}"
: "${PLAN_TOOL:?existing strata-gpu-manifest executable}"
: "${PLAN_TOOL_SHA:?independently retained tool SHA256}"
PROFILE_SHA=8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf
printf '%s  %s\n' "$PROFILE_SHA" "$PROFILE" "$MANIFEST_SHA" "$MANIFEST" \
  "$PLAN_TOOL_SHA" "$PLAN_TOOL" | sha256sum -c -
"$PYTHON05" -B -m unittest discover -s "$SOURCE/tools" \
  -p 'test_export_local_ranked_assignments.py' -v
"$PYTHON05" -B "$SOURCE/tools/export_local_ranked_assignments.py" \
  --profile "$PROFILE" --profile-sha256 "$PROFILE_SHA" \
  --per-layer-per-device 23 --out "$ROOT/assignments-per-layer23"
mkdir "$ROOT/plans-per-layer23"
for DIE in 0 1; do
  "$PLAN_TOOL" plan "$MANIFEST" \
    "$ROOT/assignments-per-layer23/assignments-$DIE.txt" \
    "$ROOT/plans-per-layer23/die$DIE.plan" \
    > "$ROOT/plans-per-layer23/die$DIE.log" 2>&1
done
sha256sum "$PROFILE" "$MANIFEST" "$PLAN_TOOL" \
  "$ROOT/assignments-per-layer23/"* "$ROOT/plans-per-layer23/"* \
  > "$ROOT/per-layer23.sha256"
```

Require receipt `experts_per_die=[1104,1104]`, `per_layer_per_device=23`, and
`per_device_layer_counts` equal to 23 in every one of 48 entries on both devices.
Each plan log must report `assigned=1104 assigned_bytes=5765529600`. The existing
manifest tool binds model/layout/selected digests using metadata only. Freeze
the actual output/plan hashes before any later experiment; none are predicted
here without the actual input file and manifest. Stop on any refusal.

## Admission remains a separate prerequisite

Parent's independent projection uses the guarded head admission in
[PR50](https://github.com/Zutfen-LLC/Strata/pull/50), source
`a4e1d9f5ad2b807ef304d2c84607978bd41f3877`. With retained padded tensor bytes
5250560 per expert, 1104 experts need 5796618240 bytes, rounded up to a
5529 MiB tensor cap per die. Projected full admitted totals are
7929068128 / 7721612384 bytes, below the separate 8053063680-byte (7680 MiB)
device caps. Runtime allowance 128 MiB per die and free reserve 512 MiB per die
remain unchanged. These are conditional admission projections, not observed
allocations or evidence of guard correctness. 1152 per die still fails die0.

The PR46 binary and its conservative head bound do not qualify this larger
candidate. PR50's physical guard validation is pending. Do not launch these
plans, relax caps/reserves, alter current jobs, reuse a prior pass as admission,
or add a discovery/native/quality run. Parent must first supply a reviewed
PR50-derived build/binary identity and successful bounded guard/admission
evidence, then explicitly release any physical checkpoint with its existing
resource/clock/cleanup rules. No physical command is included here.
