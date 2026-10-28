# Larger local Vulkan residency: conditional handoff

**The480-per-die token and single-case stages completed successfully.** See
[their physical receipt](portable-local-residency-480-results-v1.md). Historical
commands below are not a request to repeat them. Parent separately released the
frozen 12-case screen with an explicit 65-minute guard; it completed with native
10/12 and portable 10/12, identical tokens on all 12 cases. The scorer returned
`inconclusive-baseline`; numeric parity remains FAILED. Full scoring/cleanup
receipts are in the linked result.

Prepared for parent review/dispatch after the16-per-die physical smoke passed.
The original source05601aba ledger rejected both retained counts. A source-grounded
[accounting correction](portable-memory-accounting-v1.md) now separates serial
QSA temporary capacity and verified backend-context conversion workspace from
retained bytes. **Build the latest C++ source in a fresh directory and record its
new binary SHA. The old05601aba binary must not be used for this larger run.**
Caps/reserves are unchanged. The480-per-die physical result is linked above.

The smoke wrapper already supports larger supplied plans and caps. Its flags are
`--plan` twice, `--tensor-mib`, `--workspace-mib`, `--free-reserve-mib`; it has no
expert-count flag. Only discovery chooses 16 experts per die. Smoke accepts the
supplied plan's count and forwards its caps. A focused host test verifies this.

Existing retained plan pairs may be reused from the completed campaigns rooted
at `/home/hermes/amd-1152-20261027` (1152 per die) or
`/home/hermes/amd-quality-pr37-split` (480 per die). Hermes supplies exact plan
filenames and hashes; none are invented here. Deleted cache payloads need not be
restored. The local loader reads only the plans and original mapped GGUF ranges.

## Readiness after the accounting correction

The old smoke ledger refused both counts; it must not be used as the admission
ledger for the corrected binary. The source/lifetime proof and conservative
480-per-die upper bounds are recorded in
[portable-memory-accounting-v1.md](portable-memory-accounting-v1.md).

The successful16-expert smoke measured84008960 padded tensor bytes per die.
Fixed expert geometry/allocation layout gives2520268800 bytes for480 experts,
rounded to a2404 MiB tensor cap. Host metadata/boundary/loading admission adds
991 MiB per die; host/cgroup and the32 GiB anonymous cap remain enforced.

Preserve caps7680 MiB/die, runtime allowance128 MiB/die and free reserve512 MiB/die.
The loader must pass its current live free-memory check after dense loading and
its growth/reserve checks during loading/inference. An observed1.2 GiB dense
footprint is not a substitute for the corrected admission ledger.

## Concrete supported pair and corrected build

The corrected conservative bound admits the retained480-per-die pair with
`TENSOR_MIB=2404,2404`;1152 remains blocked by the preserved head conversion
allowance on die1. This is based on the supplied padded allocation/admission
receipts, not an assumed observed peak. Verify exact retained inputs:

```sh
PLAN0=/home/hermes/amd-quality-pr37-split/plan-die0.plan
PLAN1=/home/hermes/amd-quality-pr37-split/plan-die1.plan
TENSOR_MIB=2404,2404
COUNT=480
printf '%s  %s\n' \
  74514c6c5329d2cd518ee8b13015621bc3ba67d3815e3a15753e823dffa8257f "$PLAN0" \
  064002c422b53f3508b9a771cafe8b8604ca5786865542c61db3a15a094fa061 "$PLAN1" \
  | sha256sum -c -
```

Retained1152 plans are `/home/hermes/amd-1152-20261027/plan-die0.plan` with SHA
21f25e49aa488a3dc1c215ff1fee5ffa0581345964db5cca198d346a49906434 and plan-die1.plan
in that same directory with SHA35598853921b0d41aca724e3a744ef433339369e1be380aa2b37512088d697a5.
Do not launch that pair under the current conservative head bound.

Use the original complete-source CMake recipe with a fresh SOURCE/BUILD/CHECKS
below the secondary disk, adding the new portable_memory_admission_test and
portable_qsa_memory_test targets. Build/test outside the finite clock window:

```sh
"$CMAKE" --build "$BUILD" --parallel 2 --target portable_model \
  portable_memory_admission_test portable_qsa_memory_test portable_timing_cli_test \
  portable_model_test portable_qsa_test > "$CHECKS/build.log" 2>&1
"$(dirname "$CMAKE")/ctest" --test-dir "$BUILD" --output-on-failure \
  -R '^portable_(memory_admission|qsa_memory|timing_cli|model|qsa)_test$' \
  > "$CHECKS/ctest.log" 2>&1
sha256sum "$BUILD/portable_model" > "$CHECKS/binary.sha256"
BINARY="$BUILD/portable_model"
BINARY_SHA=$(sha256sum "$BINARY" | cut -d' ' -f1)
```

After the first corrected-build token, capture the new private/conversion/temporary
fields and verify their sum plus scheduler/expert budgets equals admitted_bytes
on each die. Identity mismatch, cap refusal or zero actual hits is a STOP; do not
regenerate caches or alter plans/caps to get a pass.

## One larger resident token

Fresh output directories under the secondary disk only. Set ROOT to a new private
parent below `/srv/models/inferswarm-artifacts/portable-local-residency-large-v1`.
SOURCE is the complete latest source export; BINARY is the freshly rebuilt
portable_model with the corrected ledger. Record BINARY_SHA from that build and
verify it before launch. No cache/archive recreation.

```sh
: "${PLAN0:?Hermes verified exact filename}" "${PLAN1:?matching plan filename}"
: "${TENSOR_MIB:?derived separate caps}" "${ROOT:?fresh evidence parent}"
: "${BINARY_SHA:?new corrected-build binary SHA}"
sha256sum "$BINARY" "$PLAN0" "$PLAN1"
timeout --signal=TERM --kill-after=15s 325s \
  "$PYTHON05" -B "$SOURCE/tools/validation-harness/portable_local_expert_smoke_v1.py" \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA" --phase smoke \
  --plan "$PLAN0" --plan "$PLAN1" --tensor-mib "$TENSOR_MIB" \
  --workspace-mib 128,128 --free-reserve-mib 512,512 --out "$ROOT/token"
```

No discovery rerun: the retained plans supply ownership. Require selected assigned
counts, actual Vulkan hits on both dies, finite row checks, strict dense placement,
resource admission and clean owned-child cleanup. Retain any failure without
retry. Plan identity failure, zero hits, allocation/runtime-budget refusal or
300 s case timeout stops before the next stage.

## Exactly one frozen answer/performance request

The portable adapter now has `--single-case`: frozen case1 exactly once, timed,
no warmup or scoring. Existing `--benchmark` is four requests and must not be
substituted here. The new option is portable-only and does not change existing
default or benchmark semantics. It needs the latest Python export only.

```sh
GGUF05=/srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
timeout --signal=TERM --kill-after=15s 325s \
  "$PYTHON05" -B "$SOURCE/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF05" portable --single-case --binary "$BINARY" --binary-sha256 "$BINARY_SHA" \
  --suite /srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v2/suite \
  --expert-plan "0:$PLAN0" --expert-plan "1:$PLAN1" --expert-tensor-mib "$TENSOR_MIB" \
  --expert-workspace-mib 128,128 --expert-free-reserve-mib 512,512 --out "$ROOT/case1"
```

No row-reference overhead in this timed request. Inspect case-01 stderr expert
receipts for actual hits on both dies, assigned counts and unchanged plan/cap
identity. benchmark.json must be complete with exactly one request, excluded_warmup
false and measured_trials1. Retain generated IDs/count, setup, full-consumption and
selection timings and terminal feed. Check the decoded answer against unchanged
case1 gold/scorer, not native tokens as truth. No steady-state TPS or backend
verdict follows from one sample. Failure or wrong answer is a parent checkpoint;
do not automatically launch the suite.

After the timed command, this metadata/scoring check confirms exact cardinality,
actual hits on both dies, selected count and unchanged gold without printing the
private decoded answer:

```sh
: "${COUNT:?selected 1152 or 480}"
"$PYTHON05" -B - "$SOURCE" "$GGUF05" "$ROOT/case1" "$COUNT" <<'PY'
import json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(sys.argv[1])/'tools/validation-harness'))
import portable_quality_v1 as P
import portable_local_expert_smoke_v1 as S
root=pathlib.Path(sys.argv[3]); count=int(sys.argv[4])
result=json.loads((root/'benchmark.json').read_text())
assert result['status']=='complete' and len(result['requests'])==1
assert result['suite']['measured_trials']==1 and result['suite']['excluded_warmup']==0
request=result['requests'][0]; assert request['excluded_warmup'] is False
reports=S.devices((root/'case-01/stderr.log').read_text(),'token-complete')
assert set(reports)=={0,1}
assert all(int(r['assigned'])==count and int(r['entries'])>0 for r in reports.values())
tok=P.Q.tokenizer(sys.argv[2]); case=P.Q.CASES[0]
correct=P.Q.correct(tok.decode(request['tokens'],errors='strict'),case[2],case[3])
P.Q.write(root/'case-score.json',dict(case=1,correct=correct,generated_count=len(request['tokens'])))
print('case1_correct=',correct,'generated_count=',len(request['tokens']))
raise SystemExit(0 if correct else 1)
PY
```

Hermes owns any verified high profile, sensors, active-clock samples and auto
restoration. The two commands have external bounds 650 s plus 30 s cleanup total,
inside the existing 30-minute guard; build and metadata checks occur outside it.
Do not claim active-clock coverage on a die without nonzero-busy samples.

## Frozen suite only after successful admission/hits/case review

Use the same portable command without `--single-case`, with a fresh out directory,
for exactly the existing 12 frozen cases. Do not regenerate prompts, rerun native,
change gold or loosen gates. Existing 600 s/case and 3600 s suite limits remain.
A temporary high window still has its 30-minute deadline: use an explicit outer
1250 s bound and retain partial STOP on timeout, or let parent choose another
already-supported guarded operating profile. Never silently extend the clock
window. Prior native 10/12 makes the unchanged scorer's inconclusive-baseline
exit 1 distinct from runtime failure. Parent owns release of this later stage.

Retain selected plan hashes/counts, rejected-cap calculation, binary/Python source
identities, all individual outputs/resource summaries/row metrics and restoration
receipts. Preserve unrelated services/models and previous results. No duplicate
disk caches, cleanup, dependency changes, fleet SSH or merge by this task.
