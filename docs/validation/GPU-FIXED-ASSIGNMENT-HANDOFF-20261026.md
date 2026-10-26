# Fixed fanout metadata and retained runner

Follow-on tooling for issue #18. Product engine/worker code remains at
`fd72e7a6f1fdf5ac85724b0af5605cea22bd5b4c`. The parent supplied a CUDA/CLI build and
all eight focused fixtures passing there (29.12 s), including actual mixed CUDA
rows, synthetic full CPU resident reuse, keeper survival, ordered mapped combine,
GPU-failure draining and unchanged backing counters. That is supplied evidence,
not a local run/private-log audit; actual resident reuse remains pending.

The existing metadata planner needs explicit assignments. The bounded
`tools/export_fixed_gpu_assignments.py` now derives them without loading a model,
using the existing `make_profile.read_profile` reader. It accepts only a complete
unique version-1 48x512 ranking, a pinned profile SHA256, fixed primary960/helper1378,
and no multiple-helper/peer/layer-split selection. Without per-layer mode this is
the engine's fixed global profile prefix. With the retained
`--expert-cache-per-layer` flag, it scans the entire ranking and admits the first
20 pairs in each of 48 layers, matching `ExpertCache::layer_slot_range` and
`admit`. The single helper then scans the original ranking excluding those actual
primary entries; it does not skip a global 960-pair prefix. It exports the exact complement bitmap and disjoint CPU-rank slices.
The complete resident's setup identity remains the final authoritative reuse gate.

```sh
python3 -B "$SRC/tools/export_fixed_gpu_assignments.py" --launch "$RETAINED_LAUNCH" --profile "$PROFILE" --profile-sha256 "$PROFILE_SHA256" --out "$NEW_ASSIGNMENTS"
cmake --build "$CUDA_BUILD" --target strata-gpu-manifest -j 4
for peer in 01 03a 03b 04; do
  timeout -k 5s 30s "$CUDA_BUILD/strata-gpu-manifest" plan "$MANIFEST" "$NEW_ASSIGNMENTS/$peer.txt" "$NEW_ASSIGNMENTS/$peer.plan"
done
```

The first three plans contain1850 experts /9661440000 bytes each, below9GiB.
The fourth contains3700 /19322880000 bytes, below18GiB. Total9250 experts /
48307200000 bytes. CPU backing remains22238 /116135731200 bytes; no arena shrink.
Check planner output against these byte counts; plans bind the manifest's actual
supported Q8_0 geometry and digests. Missing/full-model copies are unnecessary:
the coordinator sends only missing assigned blobs, using its trusted local source.

For H2560, FF640, T8/K10 the existing executor workspace is1598720 bytes, including
672000 bytes of grouped scratch. With1152MiB explicit reserve, minimum requested
free bytes are10870998272 for each small plan and20532438272 for the large one.
Allocator/driver effects can require more; actual READY is authoritative. Use
separate cache roots and verify disk headroom. Disk cache budget is separate from
VRAM reserve.

The retained harness previously refused every network GPU flag. It now admits
only an explicit `external_gpu_rows:true` configuration with one retained CPU
group, resident+entry-row flags, distinct existing loopback forwards and no legacy
GPU shorthand, CPU fallback or diagnostic mode. This manages only the existing
engine/CPU attach lifecycle. GPU workers and forwards remain externally owned;
the harness does not create, expose or supervise them. Defaults remain refused.

Prepare separate attached A0, B and A1 configs from the retained restore config:
all arms `restart:false`, `requests:[0]`, one fixed engine, original masks/settings,
`final_worker:"retain"`, `failure_worker:"retain"`, current `attach_root` and verified
idle identity. A0/A1 have no network GPU flags; B has the four explicit plan pairs
and `external_gpu_rows:true`. Preserve all existing pre-GEN placement/cache-hit/
zero-load markers. Stage and run one arm at a time using existing entry points:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$ARM_CONFIG"
timeout -k 30s "$OUTER_DEADLINE" python3 -B "$ARM_RUN_ROOT/run.py" --run-root "$ARM_RUN_ROOT"
```

B setup may include many GiB of serial acquisition on gigabit links: bound its
startup separately from the unchanged per-request deadline and exclude setup from
TPS. No extra requests/warmups. Compare B with A0 before A1; stop on failed gates.
Keep original token/first-logit policy. B's CPU-only entry count changes by design;
compare total CPU+GPU coverage, not the old CPU-only entry equality. Do not use a
cross-placement reference field that imposes that equality. A0/A1 exact checks
remain. Use the existing logit comparator read/compare functions once per pair;
its CLI expects three pairs, so do not duplicate one pair three times.

Transport/ordered arithmetic and CPU/CUDA kernel parity are separate. Report raw
logit byte equality, token equality and frozen numerical screen results separately;
a threshold pass cannot silently waive a required bitwise gate. Do not claim an
exact-output speedup on a failed acceptance gate.

Local validation: metadata exporter synthetic profile/coverage/hash/selection
refusal checks passed; existing harness suite45/45 passed in23.514s, including the
narrow opt-in regression. An initial sandbox run lacked socket permission; an
intermediate unconditional import broke two standalone owner fixtures, corrected
to import only in network mode before the final passing suite. No live process,
model, resident, network or deployment operation was performed here.

## Per-layer exporter correction

The original exporter rejected the actual retained `--expert-cache-per-layer`
flag before producing assignments. That was an exporter defect; no placement
flag should be removed to work around it. The correction supports this exact
mode and emits `primary.txt`, `helper.txt` and per-layer counts for inspection.
The regression uses a deliberately layer-skewed complete ranking: global-prefix
and per-layer ownership differ, while primary960/helper1378 and the full22238 /
116135731200-byte backing are retained. It verifies every bitmap bit and that all
9250 remote assignments are disjoint subsets of the corrected complement.
Actual retained-profile execution remains a metadata-only operator check.
Engine/worker product code remains at fd72e7a6.
