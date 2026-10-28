# Two local Vulkan plans from a pinned source ranking

Parent review and Hermes dispatch only. This metadata utility stacks on PR45
`3d7e20e1f9c7e173b468191251826a5e49ab4a47`. No fleet execution occurred here.
Do not touch the running 12-case quality suite or its plans. Wait for its final
result and owned-process/clock cleanup before this candidate's physical steps.

## Source contract and open evidence

`tools/export_local_ranked_assignments.py` reads the native STRP version 1 writer
format: 24-byte header, little-endian uint16 layer/expert pairs, then the complete
48 x 512 int32 **rank-index** table. It verifies exact length, version, geometry,
count <= slots, all pair bounds and table agreement. The table is not frequency
data. Both `tools/make_profile.py::write_profile` and
`src/core/expert_cache.cpp::write_expert_profile` produce this layout. Duplicate
pairs use the first occurrence for selection; the table must match the writer's
last-occurrence index. Invalid pairs anywhere cause refusal, without output.

The exporter takes the first 960 distinct source-ranked pairs. Ranks 0,2,4,...
among those distinct pairs belong to die0, ranks 1,3,5,... to die1. Only then are
each die's 480 assignments sorted for plan serialization. `selected-ranks.tsv`
preserves original source indices and owners. `assignments.json` pins input and
output hashes, geometry, counts and duplicate count. It never opens weights,
starts inference, reconstructs CUDA/helper ownership or changes routing.

Known candidate source on02:
`/home/hermes/strata-local-q8-20261023T152939Z/expert-profile.bin`.
Its actual SHA256, size, header and origin are **pending Hermes's read-only
report**, not supplied or inferred here. Obtain the complete original file,
not a prefix, text conversion, regenerated profile or CUDA/helper complement.
Pin its independently reported SHA before export; record its original path,
producer/source revision, generation command or retained origin receipt and
whether it was shipped, trace-ranked or learned. A complete unique 48x512 writer
profile is 196632 bytes; partial rankings have size 98328 + 4 * n_ranked. These
are format expectations, not observations of the candidate.

The format supports ordered placement but does not establish current workload
heat. `rank_learned_profile` may put current residents first, then heat/prior;
`make_profile` may preserve a complete base and append no trace evidence. If the
reported origin cannot establish workload relevance, label this strictly a
source-rank hypothesis. If the file does not match the supported format or has
fewer than 960 distinct pairs, retain the refusal and report that concrete gap;
do not synthesize another ranking. The suspected old-plan complement bias is
unproven. No counts/frequencies are invented.

## Export and plan preparation, outside any clock window

Use the exact full committed head supplied with this PR's publication receipt.
Parent exports the **whole tracked source tree**, including `tools/`, the entire
`tools/validation-harness/` and its imports, `data/`, and `agenthandoffs/`. Do not
send just changed files. For example, on the source machine:

```sh
set -eu
: "${EXPECTED_HEAD:?full reviewed candidate commit supplied independently}"
: "${EXPORT_DIR:?fresh source export directory}"
mkdir "$EXPORT_DIR"
git archive --format=tar "$EXPECTED_HEAD" > "$EXPORT_DIR/source.tar"
printf '%s\n' "$EXPECTED_HEAD" > "$EXPORT_DIR/source.commit"
sha256sum "$EXPORT_DIR/source.tar" > "$EXPORT_DIR/source.sha256"
```

Transfer only that source export and the small original profile/origin receipt.
On05 use a fresh root below `/srv/models/inferswarm-artifacts`, for example
`portable-local-ranked-v1-<commit-prefix>`. Verify the archive hash against the
parent's independent receipt before extracting to `$ROOT/source`. Keep the source
commit and archive hash beside the extracted tree. No new dependency download,
C++ rebuild, model copy or expert disk cache is needed. Reuse the already verified
binary and an existing `strata-gpu-manifest` executable; Hermes supplies that
tool's exact retained path and hash from its build receipt. Do not derive expected
hashes from candidate files being checked. Keep all old source/build/results.

All shell blocks use `set -eu` and stop on first failure. On05:

```sh
set -eu
: "${ROOT:?fresh secondary-disk artifact root}"
: "${PYTHON05:?existing Python}"
: "${PROFILE:?complete exported original profile under ROOT}"
: "${PROFILE_SHA:?Hermes independent original SHA256}"
: "${PLAN_TOOL:?existing strata-gpu-manifest absolute path}"
: "${PLAN_TOOL_SHA:?independent retained executable SHA256}"
SOURCE="$ROOT/source"
MANIFEST=/home/hermes/strata-pr10-9ed478e-20261025-05-peer/assets/manifest
BINARY=/srv/models/inferswarm-artifacts/portable-local-residency-large-v1/build/portable_model
BINARY_SHA=560507fa0b87a31206e03f172796fb3e60aae47ff5916aa45bd0de534bddae1b
printf '%s  %s\n' "$PROFILE_SHA" "$PROFILE" "$PLAN_TOOL_SHA" "$PLAN_TOOL" \
  "$BINARY_SHA" "$BINARY" \
  8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318 "$MANIFEST" \
  | sha256sum -c -
"$PYTHON05" -B -m unittest discover -s "$SOURCE/tools" \
  -p 'test_export_local_ranked_assignments.py' -v
"$PYTHON05" -B "$SOURCE/tools/export_local_ranked_assignments.py" \
  --profile "$PROFILE" --profile-sha256 "$PROFILE_SHA" --out "$ROOT/assignments"
mkdir "$ROOT/plans"
for DIE in 0 1; do
  "$PLAN_TOOL" plan "$MANIFEST" "$ROOT/assignments/assignments-$DIE.txt" \
    "$ROOT/plans/die$DIE.plan" > "$ROOT/plans/die$DIE.log" 2>&1
done
PLAN0="$ROOT/plans/die0.plan"
PLAN1="$ROOT/plans/die1.plan"
sha256sum "$PROFILE" "$MANIFEST" "$PLAN_TOOL" "$BINARY" \
  "$ROOT/assignments/"* "$PLAN0" "$PLAN1" > "$ROOT/inputs-outputs.sha256"
```

Require both plan logs to report `assigned=480 assigned_bytes=2506752000`.
The known pinned manifest is the existing 48-layer/512-expert Q8_0 model metadata;
the profile independently requires that geometry. The existing manifest tool binds
each plan to model/layout/selected digests; the local loader verifies those against
the mapped model on open. No full weight scan or cache regeneration. Export twice
to another fresh metadata directory and compare all four files if a manual
determinism receipt is desired; expected actual plan/output hashes become known
only from the pinned original profile and manifest, and must be frozen before
inference. Existing plans at `/home/hermes/amd-quality-pr37-split/plan-die*.plan`
are never overwritten. This creates no native/CUDA plan or ownership change.

## Bounded physical checkpoint after the current suite completes

Keep every existing host/cgroup/swap/resource/temperature/sensor guard and owned
cleanup/auto restoration rule. Dense split 32/16, Vulkan0/1 on PCI07:00.0 and
0b:00.0, four threads and twelve CPU expert workers remain fixed. Tensor caps
2404 MiB/die, runtime allowance 128 MiB/die, free reserve 512 MiB/die, total cap
7680 MiB/die and host anonymous cap 32768 MiB are unchanged. Admission is still
source/lifetime based. The retained conservative head bound still blocks 1152
per die. New ownership may fail actual hits or admission; preserve that STOP.

After verifying the current quality run has finished and restored its controls,
Hermes uses the existing finite temperature/clock guard, maximum 30 minutes for
this two-command checkpoint. Each command has a 325 s outer bound, with up to
15 s owned cleanup; total at most 680 s. Metadata work stays outside the window.
No discovery/native rerun, retry, altered gold, new tolerance or larger suite.

```sh
set -eu
sha256sum -c "$ROOT/inputs-outputs.sha256"
timeout --signal=TERM --kill-after=15s 325s \
  "$PYTHON05" -B "$SOURCE/tools/validation-harness/portable_local_expert_smoke_v1.py" \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA" --phase smoke \
  --plan "$PLAN0" --plan "$PLAN1" --tensor-mib 2404,2404 \
  --workspace-mib 128,128 --free-reserve-mib 512,512 --out "$ROOT/token"
```

Before the answer stage, require complete token summary, assigned480 per die,
actual entries >0 on **both** dies, row_checks=entries, finite reported row errors,
strict dense placements, unchanged budgets, passed growth/free admission and
owned cleanup. Zero hits, identity/admission/runtime guard failure or timeout is
STOP, with no replacement profile to force a pass. The wrapper checks hits/rows;
Hermes also inspects assigned counts and the retained resource receipts.

```sh
GGUF05=/srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
sha256sum -c "$ROOT/inputs-outputs.sha256"
timeout --signal=TERM --kill-after=15s 325s \
  "$PYTHON05" -B "$SOURCE/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF05" portable --single-case --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --suite /srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v2/suite \
  --expert-plan "0:$PLAN0" --expert-plan "1:$PLAN1" --expert-tensor-mib 2404,2404 \
  --expert-workspace-mib 128,128 --expert-free-reserve-mib 512,512 --out "$ROOT/case1"
```

Run the unchanged cardinality/hits/gold check in
[the previous recipe](portable-local-residency-large-v1.md#exactly-one-frozen-answerperformance-request)
with `COUNT=480`, `SOURCE`, `GGUF05` and `ROOT` set as above. It writes
`case1/case-score.json` using the original case1 gold and requires both dies'
actual hits and assigned counts. Exactly one request, no warmup, no row-reference
overhead in this timed case. Wrong gold or any STOP ends this checkpoint.

Retain original profile/origin receipt, all metadata hashes, selected ranks,
plan logs, exact argv, token/case stderr, row errors, actual GPU/cold CPU counts,
individual setup/full-consumption/selection/terminal-feed timings, resource and
clock cleanup receipts, source identity and `du -sb "$ROOT"`. Keep small evidence;
preserve every model and prior result, and create no duplicate weight cache.

Compare only against the retained same portable binary/workload/caps/dense split
with old 480-per-die ownership. Record all differences in operating profile,
cache state and clock coverage. Earlier case1 had 405 GPU /13920 total expert
entries (2.91%); that parent-supplied observation is context, not a predicted new
hit rate. One unmatched sample, fresh engines and unflushed OS/file caches do
not establish steady-state TPS or speedup. Prior full-model numerical parity
gates remain failed. This is neither broad quality equivalence nor an AMD versus
CUDA verdict. Parent handles independent review and release; no merge here.
