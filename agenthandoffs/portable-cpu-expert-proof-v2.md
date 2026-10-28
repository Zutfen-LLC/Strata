# Matched portable CPU expert proof — v2

Supersedes v1's publication checkpoint. Provider code is unchanged from
`fb9b4904e9d7b7eb10f38e0c0f8220ed87ef7125`, whose parent is exactly PR42
`59ce0bc5aae81b8cd4f3f17924e8b1cbb16a243d`. The additive supervisor flag
now forwards `--expert-dispatch-oracle`; all existing runtime guards remain.
Clean local build/test receipts are in `portable-cpu-expert-results-v2.md`.
Parent supplies the reviewed full remote SHA containing this recipe.

The comparison is original PORTABLE arithmetic versus optimized PORTABLE
arithmetic in the same binary. The previously failed CUDA parity screen is
preserved and does not gate or become accepted by this comparison. No new
floating-point policy, threshold, model conversion or runtime retry is allowed.

## Build and synthetic gate

Use a fresh source checkout at the parent-supplied SHA and a fresh build path.
Preserve old builds/results, resident models and the retained dependency.
Use the same existing CMake/compiler/Vulkan/SPIR-V settings as the successful
PR42 build. No install or dependency download. Example for05:

```sh
set -euo pipefail
: "${EXPECTED_SHA:?reviewed full remote SHA supplied by parent}"
: "${SOURCE:?isolated checkout at EXPECTED_SHA}"
: "${BUILD:?new build directory}"
: "${CHECKS:?new evidence directory}"
: "${CMAKE:?existing CMake executable version3.24 or newer}"
: "${LLAMA_SRC:?existing pinned llama.cpp checkout}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$SOURCE" status --porcelain --untracked-files=no)"
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test ! -e "$BUILD"
mkdir "$CHECKS"
"$CMAKE" -S "$SOURCE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON -DSTRATA_PORTABLE_QSA=ON \
  -DSTRATA_BUILD_TESTS=ON -DGGML_VULKAN=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_HIP_GFX906=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DCMAKE_PREFIX_PATH="${MODEL_CMAKE_PREFIX05:-}" \
  -DCMAKE_CXX_FLAGS="${MODEL_CXX_FLAGS05:-}" > "$CHECKS/configure.log" 2>&1
"$CMAKE" --build "$BUILD" --parallel 2 --target portable_model \
  portable_cpu_experts_test portable_model_test portable_head_test \
  portable_device_admission_test portable_eos_metadata_test \
  portable_ffn_test portable_gdn_test portable_ple_test portable_qsa_test \
  portable_qsa_gguf_test portable_qsa_bf16_test \
  > "$CHECKS/build.log" 2>&1
"$(dirname "$CMAKE")/ctest" --test-dir "$BUILD" --output-on-failure \
  -R '^portable_(cpu_experts|model|head|device_admission|eos_metadata|ffn|gdn|ple|qsa|qsa_gguf|qsa_bf16)_test$' \
  > "$CHECKS/ctest.log" 2>&1
sha256sum "$BUILD/portable_model" > "$CHECKS/binary.sha256"
```

Stop if any command fails; no rerun-until-pass. Actual before/after inference
requires the next explicit parent dispatch after review, not this document.

## Exactly one bounded pair, parent/Hermes only

Preserve `/home/hermes/portable-model-pr42-ae64892-real` and prior failures.
Reuse v4's retained manifest SHA256, six-shard sizes and device-identity
preflight. Neither arm edits source files, model files or metadata. Model
files must stay immutable throughout both sessions and every dispatch.
Size/mtime checks detect ordinary changes; they cannot establish adversarial
immutability, restored timestamps or concurrent-write safety. First-use
corruption rejection remains. Native execution reads the mappings rather
than the verified staging snapshot: stability must cover the first-use copy,
digest and native call as well as every subsequent hit. Parent ensures no
model writer is active during the pair. No locks/access settings are introduced.

Each arm uses the same candidate binary and supervisor, 38 retained prompt
IDs, context64, max-new3, threads4, workers12, split32, per-die7680MiB,
32GiB anonymous cap, 8GiB host reserve, 0.2s samples and inherited cgroup/swap
checks. Each owned process has its own timeout <=1800s; no automatic retry.
Parent may choose <=1200s per arm for <=2400s total. If the first arm fails,
stop before the second. Preserve its evidence and report the failure.

```sh
set -euo pipefail
: "${PAIR:?new private pair directory whose parent already exists}"
: "${DIE0:?verified first V340 Vulkan index}"
: "${DIE1:?verified second V340 Vulkan index}"
: "${SECONDS_PER_ARM:?parent-selected integer in1..1200}"
test "$SECONDS_PER_ARM" -ge 1
test "$SECONDS_PER_ARM" -le 1200
mkdir "$PAIR"
printf 'order=oracle,optimized; no page-cache flush or artificial warming\n' > "$PAIR/order.txt"
sha256sum "$BUILD/portable_model" > "$PAIR/binary-before.sha256"
/usr/bin/time -v -o "$PAIR/oracle-time.txt" \
  python3 "$SOURCE/tools/validation-harness/portable_model_smoke_v1.py" \
  --binary "$BUILD/portable_model" --evidence "$PAIR/oracle" \
  --devices "$DIE0,$DIE1" --split 32 --device-cap-mib 7680 \
  --timeout-seconds "$SECONDS_PER_ARM" --expert-dispatch-oracle
/usr/bin/time -v -o "$PAIR/optimized-time.txt" \
  python3 "$SOURCE/tools/validation-harness/portable_model_smoke_v1.py" \
  --binary "$BUILD/portable_model" --evidence "$PAIR/optimized" \
  --devices "$DIE0,$DIE1" --split 32 --device-cap-mib 7680 \
  --timeout-seconds "$SECONDS_PER_ARM"
sha256sum "$BUILD/portable_model" > "$PAIR/binary-after.sha256"
cmp "$PAIR/binary-before.sha256" "$PAIR/binary-after.sha256"
python3 - "$PAIR" > "$PAIR/compare.json" <<'PY'
import json, math, pathlib, sys
pair = pathlib.Path(sys.argv[1])
roots = [pair / 'oracle', pair / 'optimized']
records = []
for root in roots:
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['launched'] and summary['exit_status'] == 0, root
    assert summary['status'] == 'mechanism_checks_passed; numerical/quality verdict pending', root
    assert summary['prefix_equal'] and summary['generated_ids'] == [71093, 12305, 198], root
    assert math.isfinite(summary['seconds']) and summary['seconds'] > 0, root
    records.append(summary)
assert records[0]['model_file_working_set'] == records[1]['model_file_working_set']
# Only mode and new output destinations may differ in child argv.
commands = [json.loads((root / 'argv.json').read_text()) for root in roots]
assert commands[0].pop() == '--expert-dispatch-oracle'
for command in commands:
    for flag in ('--dump-logits', '--placements'):
        command[command.index(flag) + 1] = 'OUTPUT'
assert commands[0] == commands[1]
assert (roots[0] / 'stdout.jsonl').read_bytes() == (roots[1] / 'stdout.jsonl').read_bytes()
sets = [sorted(p.name for p in (root / 'logits').glob('pos-*.f32')) for root in roots]
assert sets[0] == sets[1] and len(sets[0]) == 41, 'expected38 prompt+3 generated rows'
for name in sets[0]:
    x, y = [(root / 'logits' / name).read_bytes() for root in roots]
    assert len(x) == 993280 and x == y, name
work = []
for arm, root in enumerate(roots):
    lines = [line for line in (root / 'stderr.log').read_text().splitlines()
             if line.startswith('expert_work ')]
    assert len(lines) == 41, root
    previous = dict(hits=0, verifications=0)
    for pos, line in enumerate(lines):
        row = {k: int(v) for k, v in (field.split('=') for field in line.split()[1:])}
        assert row['position'] == pos and row['native_batches'] == 48 * (pos + 1), row
        assert row['source_invalidations'] == 0, row
        assert row['hits'] + row['verifications'] == 480 * (pos + 1), row
        assert row['copied_bytes'] == row['digest_bytes'] == row['verifications'] * 5222400, row
        assert all(row[key] >= previous[key] for key in previous), row
        if arm == 0:
            assert row['hits'] == 0, row
        previous = {key: row[key] for key in previous}
    work.append(row)
assert work[1]['hits'] > 0, 'no repeated expert reuse observed'
assert work[0]['native_batches'] == work[1]['native_batches']
removed = work[0]['copied_bytes'] - work[1]['copied_bytes']
assert removed == work[1]['hits'] * 5222400
print(json.dumps(dict(byte_identical_positions=41, generated_ids=[71093,12305,198],
    oracle=work[0], optimized=work[1], removed_copy_bytes=removed,
    removed_fnv_logical_byte_accesses=2*removed,
    supervised_seconds=[s['seconds'] for s in records]), indent=2))
PY
```

Retain this comparison output in a new file plus both complete evidence trees.
The output acceptance criterion is byte identity of **every portable logit
row**, token/stop records and the expected portable IDs. Counters establish
removed copy/digest work while the number of native batches stays constant.
They are not timings. Any mismatch, metadata drift, guard or timeout failure
stops the proof; preserve evidence and do not retry or loosen thresholds.

The oracle retains the former copy/digest/native work, with the same new
metadata checks, counters and logging as the optimized arm. It is an arithmetic
and work comparison, not the exact uninstrumented historical executable.

Historical05 timing is not a matched baseline: prior builds, startup caches,
OS page residency and other workloads can differ. The fixed oracle-first
pair also gives the second arm a potentially warmer page cache. Do not flush
system caches, prewarm models, evict resident services or infer a general
speedup from this one pair. Record `/usr/bin/time` (supervisor plus child),
`summary.seconds` (loading/inference sampling interval), counters, sampled
RSS/VRAM/swap and the fixed order separately. Neither elapsed measurement
isolates FNV, copies or native pool duration. Numerical diagnosis can proceed
independently; no result here changes the failed native-CUDA parity verdict.
