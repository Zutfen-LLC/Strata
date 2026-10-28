# Larger local Vulkan residency: conditional handoff

Prepared for parent review/dispatch after the16-per-die physical smoke passed.
No new C++ implementation or fleet run here. Keep the existing source 05601aba
binary SHA 7d8718ac7dfa9897b248fb208f72117f3c731c765ed10697e0c70e2bc0d59ae4;
use this PR's latest complete Python source export. No rebuild is needed.

The smoke wrapper already supports larger supplied plans and caps. Its flags are
`--plan` twice, `--tensor-mib`, `--workspace-mib`, `--free-reserve-mib`; it has no
expert-count flag. Only discovery chooses 16 experts per die. Smoke accepts the
supplied plan's count and forwards its caps. A focused host test verifies this.

Existing retained plan pairs may be reused from the completed campaigns rooted
at `/home/hermes/amd-1152-20261027` (1152 per die) or
`/home/hermes/amd-quality-pr37-split` (480 per die). Hermes supplies exact plan
filenames and hashes; none are invented here. Deleted cache payloads need not be
restored. The local loader reads only the plans and original mapped GGUF ranges.

## Readiness and count selection; metadata only

First obtain the exact smoke summary.json and stderr.log paths plus exact retained
plan filenames/hashes. Require trusted manifest/model identity, disjoint ownership
and actual counts 1152/1152 or 480/480 from retained receipts. The local adapter
reconstructs and compares each full plan against the source manifest before expert
allocation; mismatched identities are a STOP, not permission to regenerate caches.

Per die, 1152 weights occupy 6016204800 bytes (5737.5 MiB); 480 occupy 2506752000
bytes (2390.625 MiB). Host GGML metadata alone is 2.25 GiB/die or 0.9375 GiB/die.
The implementation's additional host bound including boundary buffers and slack
is 2356 MiB/die or 991 MiB/die, respectively. Host/cgroup and 32 GiB anonymous-cap
checks still apply. These values do not include dense/runtime allocations.

Use actual padded tensor bytes from each 16-expert smoke receipt, rather than
weight bytes alone. Expert geometry and per-expert allocation layout are fixed
for this binary/backend; round the larger aggregate up to whole MiB. Recover the
dense admission ledger by subtracting the smoke tensor cap and workspace allowance
from the printed admitted_bytes. Do not replace it with observed 1.2 GiB usage.

This read-only calculation selects 1152 only if both per-device ledgers and the
host cap admit it, otherwise 480 with an explicit failed inequality. It does not
launch a model or replace live free-memory checks:

```sh
# Hermes resolves these actual retained files; no guessed subpaths.
: "${SMOKE_SUMMARY:?actual smoke summary.json}" "${SMOKE_STDERR:?actual smoke stderr.log}"
"$PYTHON05" -B - "$SMOKE_SUMMARY" "$SMOKE_STDERR" <<'PY'
import json, sys
M=1<<20
summary=json.load(open(sys.argv[1])); text=open(sys.argv[2]).read()
assert summary['status']=='complete'
reports=summary['expert_devices']
assert set(reports)=={'0','1'}
ledgers={}
for line in text.splitlines():
    if line.startswith('die='):
        fields=dict(part.split('=',1) for part in line.split())
        ledgers[int(fields['die'])]=fields
hostline=next(line for line in text.splitlines() if line.startswith('mapped_bytes='))
host=dict(part.split('=',1) for part in hostline.split())
base_host=sum(int(host[k]) for k in ('host_weight_bytes','host_scheduler','host_state_bound','expert_staging'))
per_host=(2<<20)+(32<<10)
base_host-=sum(int(r['assigned'])*per_host+(16<<20) for r in reports.values())
for count in (1152,480):
    tensor_caps=[]; reasons=[]
    for slot in (0,1):
        r=reports[str(slot)]; ledger=ledgers[slot]
        assigned=int(r['assigned']); padded=int(r['tensor_bytes'])
        assert assigned==16 and padded>0 and int(r['entries'])>0
        tensor=(count*padded+assigned*M-1)//(assigned*M)
        dense=int(ledger['admitted_bytes'])-int(r['tensor_cap'])-int(r['workspace_cap'])
        total=dense+tensor*M+128*M
        cap=int(ledger['cap_bytes'])
        tensor_caps.append(tensor)
        if total>cap: reasons.append(f'die{slot}: dense+tensor+workspace={total} > cap={cap}')
    new_host=base_host+2*(count*per_host+(16<<20))
    if new_host>32768*M: reasons.append(f'host bound {new_host} > 32GiB')
    if reasons:
        print(f'COUNT {count} REJECTED: '+ '; '.join(reasons)); continue
    print(f'COUNT={count} TENSOR_MIB={tensor_caps[0]},{tensor_caps[1]} HOST_BOUND={new_host}')
    break
else:
    raise SystemExit('Neither retained placement admits; STOP without a tiny replacement or cap relaxation')
PY
```

Record its output/rejection reason. Set PLAN0/PLAN1 to the matching retained pair
for that selected count, and TENSOR_MIB to the printed pair. Do not mix campaigns
without independently confirming disjoint ownership. Keep workspace 128 MiB/die,
free reserve 512 MiB/die and total cap 7680 MiB/die. Before launch, current idle/free
observations must still support the plan. During preparation the loader requires
live free memory after dense loading >= tensor cap + 128 MiB + 512 MiB reserve,
then enforces observed growth/free reserve around asset loading and inference.
Driver transient peaks remain subject to the existing outer sampled guards.

## One larger resident token

Fresh output directories under the secondary disk only. Set ROOT to a new private
parent below `/srv/models/inferswarm-artifacts/portable-local-residency-large-v1`.
SOURCE is the complete latest Python export; BINARY is the retained source 05601aba
portable_model, verified against the SHA above. No cache/archive recreation.

```sh
: "${PLAN0:?Hermes verified exact filename}" "${PLAN1:?matching plan filename}"
: "${TENSOR_MIB:?derived separate caps}" "${ROOT:?fresh evidence parent}"
BINARY_SHA=7d8718ac7dfa9897b248fb208f72117f3c731c765ed10697e0c70e2bc0d59ae4
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
