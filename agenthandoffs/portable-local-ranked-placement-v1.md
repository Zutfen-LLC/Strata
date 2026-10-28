> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Two local Vulkan plans from a pinned source ranking

**The ranked token and single frozen-case checkpoint completed.** See the
[physical result](#physical-ranked-placement-result) below. Preparation and
execution commands are retained for provenance, not a request to repeat them.
The ranked placement has not run the full 12-case suite. No further fleet test
is dispatched by this document.

Parent review and operator dispatch only. This metadata utility stacks on PR45
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
`${PRIVATE_ARTIFACT_A777A264}`.
operator's [read-only evidence](PRIVATE_RECEIPT),
relayed by parent, reports 196632 bytes, SHA256
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`,
STRP version 1, 48 layers, 512 experts, 24576 unique in-range ranked pairs and
a consistent inverse-rank table. Adjacent receipts show the file was copied
from `validation host:${PRIVATE_ARTIFACT_596BE285}`
and later used on02 local Q8 with per-layer primary placement and six coding
requests. **The original generation trace is unknown.** This establishes a
valid candidate order, not measured workload heat or how the order was built.
This coding task has not fetched the raw profile or independently inspected it.

Obtain the complete original file on02/05, not a prefix, text conversion,
regenerated profile or CUDA/helper complement. Preserve the original path and
origin receipt beside it. The runtime exporter validates the independently
supplied hash and full format before writing assignments; no raw profile is
needed on validation host. Partial writer rankings have size 98328 + 4 * n_ranked;
this candidate's reported 196632-byte size matches its complete ranking.

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
On05 use a fresh root below `${PRIVATE_ARTIFACT_50028F15}`, for example
`portable-local-ranked-v1-<commit-prefix>`. Verify the archive hash against the
parent's independent receipt before extracting to `$ROOT/source`. Keep the source
commit and archive hash beside the extracted tree. No new dependency download,
C++ rebuild, model copy or expert disk cache is needed. Reuse the already verified
binary and an existing `strata-gpu-manifest` executable; operator supplies that
tool's exact retained path and hash from its build receipt. Do not derive expected
hashes from candidate files being checked. Keep all old source/build/results.

All shell blocks use `set -eu` and stop on first failure. On05:

```sh
set -eu
: "${ROOT:?fresh secondary-disk artifact root}"
: "${PYTHON05:?existing Python}"
: "${PROFILE:?complete exported original profile under ROOT}"
PROFILE_SHA=8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf
: "${PLAN_TOOL:?existing strata-gpu-manifest absolute path}"
: "${PLAN_TOOL_SHA:?independent retained executable SHA256}"
SOURCE="$ROOT/source"
MANIFEST=${PRIVATE_ARTIFACT_32CC92CE}
BINARY=${PRIVATE_ARTIFACT_79D942D4}
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
inference. Existing plans at `${PRIVATE_ARTIFACT_1AC007FA}`
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
operator uses the existing finite temperature/clock guard, maximum 30 minutes for
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
operator also inspects assigned counts and the retained resource receipts.

```sh
GGUF05=${PRIVATE_ARTIFACT_ED0D8ADE}
sha256sum -c "$ROOT/inputs-outputs.sha256"
timeout --signal=TERM --kill-after=15s 325s \
  "$PYTHON05" -B "$SOURCE/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF05" portable --single-case --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --suite ${PRIVATE_ARTIFACT_021ADBD1} \
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

## Physical ranked-placement result

operator's [checkpoint report](PRIVATE_RECEIPT),
relayed by parent, records one resident-token run and one frozen case1 run,
each exactly once, using the source export at `4c5afefb49a57eea69d034ea0a8f4a1980957515`.
This coding task recorded the supplied report; it did not run fleet inference.
Evidence remains on05 under
`${PRIVATE_ARTIFACT_91120C95}`, including
`checkpoint-control/closeout.json`, `token/` and `case1/`. Preserve those receipts
and all old-plan evidence. The original profile generation trace remains unknown.

Case1 passed unchanged gold and generated four tokens identical to the retained
old-placement case. Both plans assigned 480 experts; all caps/reserves stayed
unchanged. Actual GPU expert entries increased on both dies, while the total
number of expert entries stayed 13920:

| Case1 expert entries | Old portable placement | Ranked portable placement |
| --- | ---: | ---: |
| Die0 GPU | 170 | 468 |
| Die1 GPU | 235 | 441 |
| Total GPU | 405 (2.91%) | 909 (6.53%) |
| Cold CPU | 13515 | 13011 |

This establishes more actual resident execution for this case. It does not
establish that source rank measures workload heat or transfers to other prompts.

| Case1 timing, seconds | Old portable placement | Ranked portable placement |
| --- | ---: | ---: |
| Wall | 104.007 | 104.259 |
| Setup | 18.086 | 18.124 |
| Full-consumption prompt | 74.720 | 74.696 |
| Full-consumption decode | 10.223 | 10.439 |

There is **no demonstrated speedup** from this single unmatched sample. The
comparison is within the same portable placement experiment; it is not a
steady-state result or a Vulkan-versus-CUDA verdict. The four-token answer check
does not waive the previously failed full-model numerical parity gates.

The resident-token diagnostic recorded 2 actual entries / 2 row checks on die0
and 6 / 6 on die1. Reported row maximum absolute errors were 5.96e-8 and 3.73e-8,
respectively. These are observed diagnostics, not a newly accepted tolerance or
whole-model parity result.

Sampled peak VRAM was 3794284544 / 3839131648 bytes; maximum temperature was
58 C and swap remained zero. Both dies had nonzero-busy 1500/945 MHz samples.
Both controls returned to auto, no owned process remained, and per-die VRAM
returned to the 8339456-byte baseline. The closeout receipt is the retained
cleanup evidence. No cap expansion, retries or additional fleet tests follow
from this checkpoint; 1152 per die remains blocked by the conservative bound.

The earlier **12-case 10/12 result compares native with the old local plans**.
It is not a ranked-plan quality result. **Ranked full quality was not run.**
This checkpoint supports only the token execution and single frozen-case facts
above, without broader qualification.

For future diagnosis, the current portable fresh process resets its lazy
expert-verification cache for every request; native performs the full CPU worker
LOAD before timing. First-use verification/loading work must be measured before
attributing a remaining gap to GPU or CPU computation. This is a diagnostic
priority, not a measured explanation of these timings, and requests no code
change or further run here.
