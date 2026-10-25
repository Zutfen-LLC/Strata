# Packed native activation equivalence: bounded offline proof

Tracks [#14](https://github.com/Zutfen-LLC/Strata/issues/14). The small Linux target
`cpu_packed_activation_proof` moves no production boundary. No wire protocol, worker lifecycle,
socket default, model run or dependency changes are included. Desktop executable checks are **NOT RUN**.
Independent focused static review found an incorrect codec call signature; it was corrected and
rereviewed with no remaining blocking finding. Static review is not compiler or runtime evidence.

The parent reports the role05 same-build screen **PASS** at
`06bbd80930f6b1a3b19656e924d1e5c76cc348f1`: CTest 1/1 (0.30 s), 16 selected records,
48 full/peer row comparisons and retained full partials bit-exact. Pack took 0.13 s;
rows took 9.11 s with about 77 MB peak memory. Edge case6's unsupported native scale is the
expected refusal case. SPA1 is 92884 bytes, SHA256
`531d113d5d110fa4a53fa464ca8e504f913945d8af233113040f2d0dd6af2e0a`;
SRA1 is 8110340 bytes, SHA256
`0aaeb0762418bc150d7c4ca2e6c168e38b5069fb2df825cb94dd75312a6d727d`.
Reported proof ELF SHA256 is
`352872b6f623383b69ba6d2116a36ddbbc2b891d96745cc60d4f4d2b69ca3e33`.
These are supplied Hermes results, not desktop execution or private receipt inspection.
No worker contact or GEN was reported. **Cross-build comparison remains pending.**

The current request carries 10240 F32 bytes per 2560-value token. Native Q8_0 quantization already
produces 2720 bytes on workers. This proof serializes that existing result; it neither recovers the
original F32 bits nor introduces a different quantizer. Requests accounted for 18.67% of the supplied
traffic. The optimistic combined-byte saving ceiling is under 13.71% before unchanged metadata.
This is not reply compression, measured latency, or a near-local TPS prediction.

## Gates and limits

`--check` uses existing synthetic Q8 weights and the direct native reference, without creating a
service. It checks active/inactive tokens, zero work, original accumulation order, signed zero fields,
finite constants/subnormals/varying signs/half-step quantizer inputs, nonfinite F32 rejection,
truncated/excess blocks, nonfinite FP16 scales, explicit little endian fields, wrong geometry/rotation,
reference differences and selected weight corruption. The native quantizer sees extreme finite F32
inputs unchanged. Its raw output and a scale-support flag enter the cross-build artifact. If it emits
an infinite/NaN FP16 scale, restoration refuses it and logs `native_scale_supported=0`; the proof
does not silently broaden or change production acceptance. In particular, all finite F32 values are
not claimed transferable by this diagnostic. Native zero quantization need not preserve F32 negative
zero; the serializer preserves every resulting scale and quantized-byte bit, including negative zero.

`--pack CAPTURE OUT` selects the SAME first/last available records at layers 0/12/24/47, T1/T2,
K10 as the earlier compression diagnostic: at most 16 records and 32 real token rows (87040 useful
bytes). It quantizes only active tokens once per packing operation, with existing 4096-byte stride
scratch zeroed. Output SPA1 includes edge results, selected record indices, a digest of each original
production request and full-mask active token fields. Digests bind original IDs, masks, weights,
sequence and F32 inputs. It reads no weights. Fields are FP16 scale bits in little endian followed by
32 signed-byte bit patterns, 80 blocks per active row; C structs and stride padding are never saved.
The declared bundled Q8_0 field layout, ggml public sizes and runtime NativeFmt geometry are checked.

`--rows CAPTURE MANIFEST MODEL_ROOT PACKED_REFERENCE ROW_OUT` first regenerates SPA1 and requires
exact equality with the supplied reference, before opening model assets. After that byte gate,
restoring regenerated fields is equivalent to consuming the reference fields. Each selected record
is checked with the full original mask and both peer masks from the original full ownership bitmap.
Each path compares original native scratch against restored scratch, all unweighted native rows,
and the original ordered weighted partial. Full partials must also bitmatch the retained capture.
Peer sums are checked against same-build original native peer sums, not the differently ordered full
sum. Output SRA1 contains all full/peer rows and ordered partials as explicit little endian F32 bits.
No peer/helper/GPU merge order changes. No timed measurements or performance verdict are emitted.

One digest-verified existing expert blob (at most 5222400 bytes) is read at a time. There is no arena.
Capture storage is bounded by its existing reader; reference/output files are bounded to 16 MiB.
Inactive token scratch, unused stride bytes and unowned rows remain zero. Input paths and artifacts
are private. Only Q8_0 gate/up/down, 2560/640 geometry without rotation and retained Linux x86-64
native configurations are supported. Wrong formats fail explicitly. Do not run an ELF on a different
CPU: build/run each role locally with its verified compatible retained configuration. Unsupported
CPU/build, scale failure on a real row, mismatch, resource refusal or timeout is a stop, not a pass.

Every process has a 180-second internal deadline, no owned swap, <=2 GiB RSS, >=8 GiB MemAvailable,
and unchanged host OOM count. This applies intentionally to CTest too; low-memory hosts fail closed.
External address space is limited to 2 GiB, with a 190-second timeout. Admission checks require 10 GiB
available and cgroup headroom. No tiny inherited file-size limit: synthetic checks write Q8 fixtures.

## Evidence levels

One build passing its own pack/row gates establishes **same-build equivalence on these fixtures**.
It does not establish cross-host, compiler or ISA equivalence. For the intended deployment:

1. Build each role's proof using its actual retained native CPU compiler/flags/ggml configuration.
   Record effective compile/link commands, CPU features, compiler version and ELF hashes. The 05
   cache alone cannot stand in for the sender or 02 worker configuration.
2. Run `--check` and `--pack` on each role against the identical SHA-verified capture. Compare the
   two SPA1 files with `cmp`. Edge output is part of this comparison. Stop if they differ.
3. Make each receiving arithmetic build run `--rows` with the OTHER build's identical SPA1 artifact.
   Keep its local pack as a separately retained reference. Compare SRA1 outputs with `cmp` across
   the relevant native builds. A local `PACKED_ROWS_DONE` deliberately reports
   `cross_build_rows=REQUIRES_CMP` until that external comparison succeeds.

Retain exact files/logs and label same-build versus cross-build checks separately. Equal SHA-256
values support provenance; `cmp` is the exact-byte gate. These finite samples do not prove equality
for every activation, architecture or future compiler, and cover first-logit/prefill, not long decode.
A format/capability contract, raw fallback, placement cost and live latency remain outside this task.
No protocol adoption or follow-on model run is automatically authorized by a pass.

## Parent bindings and bounded Hermes commands

This desktop authors and publishes only. The parent dispatches. No worker contact, GEN, resident
reload, firewall change, package installation, compression sweep or shared runtime mutation.
Preserve all old source/builds/ELFs/receipts and resident worker 809314 on02.

For the first role on05, use the parent-supplied retained isolated compression source at
`e5685da112f9e75262176635743b0790586d88c3` and its matching CPU build. Keep machine paths in
private bindings. The shared runtime at e145df6e is not an input to mutate. Existing capture, manifest and model-root bindings were supplied privately to the parent.
Capture SHA256 is `218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208` (37544692 bytes);
manifest SHA256 is `8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318`.

Export `BASE_SRC`, `BASE_PIN`, `CPU_BUILD`, exact published `PIN`, `CAPTURE`, and fresh absolute
`NEW_SRC`, `NEW_BUILD`, `RUN_ROOT` (all absent, existing parents). Export `ROLE=05` for provenance.
Role02 uses its real retained Release CUDA cache, with GPU backends deliberately disabled only
for the offline proof. See [the separate role02 relay/build handoff](CPU-PACKED-ACTIVATION-ROLE02-20261025.md).
Do not substitute role05 native settings, guess dependency paths or use the resident's port. For arithmetic it also needs verified
`MANIFEST` and `LOCAL_ROOT` bindings. Missing other-role bindings block cross-build evidence only.

Fetch the exact PIN into an existing isolated source through GitHub, without checking it out there.
Extract this marked block from PIN to a fresh private `BLOCK`; refuse empty extraction, export
bindings, then `bash "$BLOCK"`. For example:

```sh
set -euo pipefail
umask 077
test ! -e "$BLOCK"
git -C "$BASE_SRC" fetch origin "$PIN"
git -C "$BASE_SRC" show "$PIN:docs/validation/CPU-PACKED-ACTIVATION-PROOF-20261025.md" |
  sed -n '/^# BEGIN PACKED_BUILD_PACK$/,/^# END PACKED_BUILD_PACK$/p' > "$BLOCK"
test -s "$BLOCK"
bash "$BLOCK"
```

The following is a fixed build/pack command set, not a request to execute it from this desktop.
It uses a new source archive rather than changing any retained checkout. Build only the proof and
its libraries; no worker ELF or engine target is rebuilt. Compiler selection and flags come from
the role's retained cache; GPU backends are off for the offline target. Native ISA flags, IPO settings and linker flags remain. Toolchain/sysroot/cross-target settings
are refused explicitly. Saved effective commands must be checked against the deployment receipt
before calling this an actual-role comparison.

```sh
# BEGIN PACKED_BUILD_PACK
set -euo pipefail
umask 077
: "${ROLE:?}" "${BASE_SRC:?}" "${BASE_PIN:?}" "${CPU_BUILD:?}" "${PIN:?}" "${CAPTURE:?}" "${NEW_SRC:?}" "${NEW_BUILD:?}" "${RUN_ROOT:?}"
export ROLE BASE_SRC BASE_PIN CPU_BUILD PIN CAPTURE NEW_SRC NEW_BUILD RUN_ROOT
export GIT_OPTIONAL_LOCKS=0
PROJECT_OBJECTS=${CANDIDATE_GIT:-$BASE_SRC}
test "$(git -C "$BASE_SRC" rev-parse HEAD)" = "$BASE_PIN"
test -z "$(git -C "$BASE_SRC" status --porcelain)"
test ! -e "$RUN_ROOT"; test ! -e "$NEW_SRC"; test ! -e "$NEW_BUILD"
test -x /usr/bin/time
test "$(wc -c < "$CAPTURE")" = 37544692
mkdir -m 700 "$RUN_ROOT" "$NEW_SRC"
printf '%s  %s\n' 218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208 "$CAPTURE" | sha256sum -c - > "$RUN_ROOT/input-hashes.log"
test "$(git -C "$PROJECT_OBJECTS" rev-parse "$PIN^{commit}")" = "$PIN"
git -C "$PROJECT_OBJECTS" archive "$PIN" | tar -x -C "$NEW_SRC"
python3 -B - <<'PY'
import os,pathlib,re,subprocess,json
e=os.environ; cache={}
for k in ('BASE_SRC','CPU_BUILD','CAPTURE','NEW_SRC','NEW_BUILD','RUN_ROOT'): assert pathlib.Path(e[k]).is_absolute()
for line in (pathlib.Path(e['CPU_BUILD'])/'CMakeCache.txt').read_text().splitlines():
    m=re.match(r'([^:#]+):([^=]+)=(.*)',line)
    if m: cache[m[1]]=(m[2],m[3])
assert cache['CMAKE_HOME_DIRECTORY'][1]==e['BASE_SRC']
assert e['ROLE'] in ('05','02')
assert cache['CMAKE_BUILD_TYPE'][1]=='Release'
assert cache['STRATA_ENABLE_CUDA'][1]==('ON' if e['ROLE']=='02' else 'OFF')
assert cache['STRATA_NATIVE_EXPERTS'][1]=='ON' and cache['STRATA_BUILD_TESTS'][1]=='ON'
assert cache.get('STRATA_HIP_GFX906',('BOOL','OFF'))[1]=='OFF'
retained_ggml=cache.get('STRATA_GGML_DIR',('PATH',''))[1] or str(pathlib.Path(e['CPU_BUILD'])/'_deps/strata_llamacpp-src')
ggml=e['GGML_RELAY_SRC'] if e['ROLE']=='02' else retained_ggml
assert pathlib.Path(ggml).is_absolute()
assert (pathlib.Path(ggml)/'ggml/include/ggml.h').is_file()
assert subprocess.check_output(['git','-C',ggml,'rev-parse','HEAD'],text=True).strip()=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert not subprocess.check_output(['git','-C',ggml,'status','--porcelain'],text=True).strip()
# Native retained builds only. A toolchain/sysroot/cross-target needs a separately reviewed recipe.
for k,(t,v) in cache.items():
    if t not in ('INTERNAL','STATIC') and re.fullmatch(
        r'CMAKE_(TOOLCHAIN_FILE|SYSROOT.*|SYSTEM_.*|OSX_.*|STAGING_PREFIX|LINKER_TYPE)',k):
        assert not v, f'unsupported retained setting: {k}'
keys=[k for k,(t,v) in cache.items() if t not in ('INTERNAL','STATIC') and
      (k.startswith('STRATA_') or k.startswith('GGML_') or
       re.fullmatch(r'CMAKE_(C|CXX)_(COMPILER.*|FLAGS.*|STANDARD.*|EXTENSIONS|LINKER_LAUNCHER|LINK_WHAT_YOU_USE|VISIBILITY_PRESET)',k) or
       re.fullmatch(r'CMAKE_(INTERPROCEDURAL_OPTIMIZATION.*|(EXE|SHARED|MODULE|STATIC)_LINKER_FLAGS.*|AR|RANLIB|LINKER|NM|OBJDUMP|STRIP|POSITION_INDEPENDENT_CODE|BUILD_TYPE|MAKE_PROGRAM|VISIBILITY_INLINES_HIDDEN|LINK_WHAT_YOU_USE|BUILD_RPATH|INSTALL_RPATH|SKIP_RPATH|SKIP_BUILD_RPATH|BUILD_WITH_INSTALL_RPATH|INSTALL_RPATH_USE_LINK_PATH|LINK_SEARCH_START_STATIC|LINK_SEARCH_END_STATIC)',k))]
values={k:cache[k] for k in keys}
intentional={k:dict(retained=values.get(k),proof=v) for k,v in dict(
    STRATA_GGML_DIR=('PATH',ggml),STRATA_ENABLE_CUDA=('BOOL','OFF'),
    STRATA_ENABLE_HIP=('BOOL','OFF'),STRATA_ENABLE_SYCL=('BOOL','OFF')).items()}
values.update(STRATA_GGML_DIR=('PATH',ggml),FETCHCONTENT_FULLY_DISCONNECTED=('BOOL','ON'),
    STRATA_ENABLE_CUDA=('BOOL','OFF'),STRATA_ENABLE_HIP=('BOOL','OFF'),STRATA_ENABLE_SYCL=('BOOL','OFF'),
    CMAKE_EXPORT_COMPILE_COMMANDS=('BOOL','ON'),
    CMAKE_RUNTIME_OUTPUT_DIRECTORY=('PATH',str(pathlib.Path(e['RUN_ROOT'])/'bin')),
    CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=('PATH',str(pathlib.Path(e['RUN_ROOT'])/'bin')))
r=pathlib.Path(e['RUN_ROOT'])
with (r/'initial.cmake').open('x') as f:
    for key,(kind,value) in sorted(values.items()):
        assert ']===]' not in value
        if kind not in ('BOOL','STRING','PATH','FILEPATH'): kind='STRING'
        f.write(f'set({key} [===[{value}]===] CACHE {kind} "" FORCE)\n')
(r/'generator.txt').write_text(cache['CMAKE_GENERATOR'][1])
with (r/'build-provenance.json').open('x') as f: json.dump(dict(role=e['ROLE'],pin=e['PIN'],base=e['BASE_PIN'],project_objects=e.get('CANDIDATE_GIT',e['BASE_SRC']),retained_ggml=retained_ggml,ggml=ggml,intentional=intentional,values=values),f,indent=2)
for lang in ('C','CXX'):
    compiler=cache[f'CMAKE_{lang}_COMPILER'][1]
    (r/f'{lang}-compiler.txt').write_bytes(subprocess.check_output([compiler,'--version']))
PY
uname -a > "$RUN_ROOT/uname.txt"
lscpu > "$RUN_ROOT/lscpu.txt"
cp "$CPU_BUILD/CMakeCache.txt" "$RUN_ROOT/retained-CMakeCache.txt"
timeout --signal=TERM --kill-after=5s 90s cmake -S "$NEW_SRC" -B "$NEW_BUILD" -G "$(cat "$RUN_ROOT/generator.txt")" \
  -C "$RUN_ROOT/initial.cmake" > "$RUN_ROOT/configure.log" 2>&1
timeout --signal=TERM --kill-after=5s 600s cmake --build "$NEW_BUILD" --target cpu_packed_activation_proof \
  --parallel 4 --verbose > "$RUN_ROOT/build.log" 2>&1
cp "$NEW_BUILD/compile_commands.json" "$RUN_ROOT/compile_commands.json"
cp "$NEW_BUILD/CMakeCache.txt" "$RUN_ROOT/proof-CMakeCache.txt"
python3 -B - <<'PY'
import os,pathlib,re,json
r=pathlib.Path(os.environ['RUN_ROOT']); actual={}
for line in (r/'proof-CMakeCache.txt').read_text().splitlines():
    m=re.match(r'([^:#]+):([^=]+)=(.*)',line)
    if m: actual[m[1]]=m[3]
for key,(kind,value) in json.loads((r/'build-provenance.json').read_text())['values'].items():
    assert actual.get(key)==value, f'configuration changed: {key}'
PY
python3 -B - <<'PY'
import os,pathlib,re,json
if os.environ['ROLE']!='02': raise SystemExit(0)
e=os.environ; r=pathlib.Path(e['RUN_ROOT']); provenance=json.loads((r/'build-provenance.json').read_text())
old=pathlib.Path(e['CPU_BUILD']); new=pathlib.Path(e['NEW_BUILD'])
replacements={e['BASE_SRC']:'@SRC',e['NEW_SRC']:'@SRC',str(old):'@BUILD',str(new):'@BUILD',
              provenance['retained_ggml']:'@GGML',provenance['ggml']:'@GGML'}
def normalize(s):
    for k in sorted(replacements,key=len,reverse=True): s=s.replace(k,replacements[k])
    return s
report={}
for target in ('strata_kernels_cpu','strata_expert_worker','strata_worker_tier','ggml-cpu','ggml-base'):
    pairs=[]
    for root in (old,new):
        found=list(root.rglob(f'CMakeFiles/{target}.dir/flags.make'))
        assert len(found)==1, f'missing/ambiguous retained native flags: {target}'
        text=found[0].read_text()
        # Retain ordinary assignments AND CMake's commented source-specific options/defines.
        pairs.append([normalize(line.strip()) for line in text.splitlines() if '=' in line])
    report[target]=dict(retained=pairs[0],proof=pairs[1],equal=pairs[0]==pairs[1])
with (r/'native-flags-comparison.json').open('x') as f: json.dump(report,f,indent=2)
assert all(v['equal'] for v in report.values()), 'effective native compile flags differ'
for language in ('C','CXX'):
    versions=[]
    for root in (old,new):
        found=list((root/'CMakeFiles').glob(f'*/CMake{language}Compiler.cmake'))
        assert len(found)==1, 'missing/ambiguous compiler identity'
        text=found[0].read_text()
        fields={k:re.search(r'^set\('+k+r'\s+(.*?)\)$',text,re.M).group(1)
                for k in (f'CMAKE_{language}_COMPILER',f'CMAKE_{language}_COMPILER_ID',f'CMAKE_{language}_COMPILER_VERSION')}
        versions.append(fields)
    assert versions[0]==versions[1], 'retained compiler executable/identity/version changed'
    with (r/f'{language}-retained-proof-identity.json').open('x') as f: json.dump(versions,f,indent=2)
links=list(new.rglob('CMakeFiles/cpu_packed_activation_proof.dir/link.txt'))
assert len(links)==1, 'missing/ambiguous proof link command'
link=links[0].read_text(); (r/'proof-link-command.txt').write_text(link)
assert not re.search(r'(?:lib)?(?:cuda\w*|cublas\w*|amdhip\w*|sycl\w*|nvrtc\w*|strata_engine)(?:[.\s/-]|$)|libstrata_kernels\.a',link,re.I), 'GPU/engine link dependency'
print('ROLE02_NATIVE_FLAGS ok=1 targets=5 compiler_identity_bits=1 gpu_backend_delta=explicit')
PY
sha256sum "$RUN_ROOT/bin/cpu_packed_activation_proof" > "$RUN_ROOT/elf.sha256"
# Runtime overrides are not guessed. Refuse inherited arithmetic/ISA overrides for this fixed run.
cat > "$RUN_ROOT/admit.py" <<'PY'
import os,pathlib,re,json,sys
keys=('STRATA_FORCE_ISA','STRATA_FORCE_AVX2','STRATA_NO_AVXVNNI','STRATA_CPU_YMM','STRATA_KQ256',
      'STRATA_NO_IQ128','STRATA_NO_IQ256','STRATA_NO_IQ512','STRATA_CPU_IQ_ALL_EXACT1',
      'STRATA_IQ_MT_MIN','STRATA_IQ3S_MT1','STRATA_NO_IQ4NL','STRATA_IQ_GATHER','STRATA_IQ256_GATHER',
      'STRATA_IQ_PREFETCH','STRATA_NO_Q8K_AVX1','STRATA_NO_Q8K_AVX2','STRATA_NO_TQ_KERNELS',
      'STRATA_NO_NVFP4_512','STRATA_NO_NVFP4_256')
assert not any(k in os.environ for k in keys), 'runtime override requires verified deployment binding'
need=10<<30; r=pathlib.Path(os.environ['RUN_ROOT'])
available=int(re.search(r'^MemAvailable:\s+(\d+)',pathlib.Path('/proc/meminfo').read_text(),re.M)[1])*1024
assert available>=need
def unescape(s): return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),s)
mounts=[]
for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
    left,right=line.split(' - ',1); fields=left.split()
    if right.split()[0]=='cgroup2' and unescape(fields[3])=='/': mounts.append(pathlib.Path(unescape(fields[4])))
assert len(mounts)==1, 'full cgroup2 ancestry must be visible'
member=[s.split(':',2)[2] for s in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')]
assert len(member)==1 and member[0].startswith('/') and '..' not in pathlib.PurePosixPath(member[0]).parts
mount=mounts[0]; p=mount.joinpath(*pathlib.PurePosixPath(member[0]).parts[1:]); rows=[]
while p!=mount:
    current=int((p/'memory.current').read_text()); row=dict(path=str(p),current=current)
    for key in ('memory.max','memory.high'):
        value=(p/key).read_text().strip(); assert value=='max' or int(value)-current>=need
        row[key]=value
    rows.append(row); p=p.parent
with (r/sys.argv[1]).open('x') as f: json.dump(dict(available=available,membership=member[0],ancestors=rows,runtime_overrides='absent'),f,indent=2)
PY
python3 -B "$RUN_ROOT/admit.py" check-admission.json
( ulimit -v 2097152
  /usr/bin/time -v -o "$RUN_ROOT/check-resources.log" timeout --signal=TERM --kill-after=5s 190s \
    ctest --test-dir "$NEW_BUILD" --output-on-failure --no-tests=error -R '^cpu_packed_activation_check$' > "$RUN_ROOT/check.log" 2>&1 )
grep -Eq 'cpu_packed_activation_check.*Passed' "$RUN_ROOT/check.log"
! grep -Eq 'Skipped|Not Run|FAILED' "$RUN_ROOT/check.log"
python3 -B "$RUN_ROOT/admit.py" pack-admission.json
( ulimit -v 2097152
  /usr/bin/time -v -o "$RUN_ROOT/pack-resources.log" timeout --signal=TERM --kill-after=5s 190s \
    "$RUN_ROOT/bin/cpu_packed_activation_proof" --pack "$CAPTURE" "$RUN_ROOT/activations.spa" > "$RUN_ROOT/pack.log" 2>&1 )
grep -Eq '^PACKED_DONE ok=1 records=([1-9]|1[0-6]) bytes=[0-9]+ weights_read=0$' "$RUN_ROOT/pack.log"
sha256sum "$RUN_ROOT/activations.spa" > "$RUN_ROOT/pack.sha256"
test "$(git -C "$BASE_SRC" rev-parse HEAD)" = "$BASE_PIN"
test -z "$(git -C "$BASE_SRC" status --porcelain)"
# END PACKED_BUILD_PACK
```

After preserving the role's SPA1 and exchanging only those small private artifacts through the
parent's established artifact channel, export `PACKED_REFERENCE` to the other's absolute local
artifact path. Keep `RUN_ROOT` exported for the admission check. The fixed run refuses inherited
arithmetic/ISA overrides and records their absence; the parent must verify that absence matches the
retained deployment. A deployment override needs an explicit revised binding, not an automatic unset.
For a clearly labeled initial SAME-BUILD screen only, it may instead point at the
role's own SPA1; that screen does not satisfy cross-build acceptance. Set `ROW_LABEL=same-build`
or `ROW_LABEL=cross-build` accordingly. This second fixed block runs each role's arithmetic once.
Cross-build mode requires two independent packs and exact `cmp` first; the parent retains source
host/role/hash receipts for transferred artifacts. No binary or model files are transferred.

```sh
# BEGIN PACKED_ROWS
set -euo pipefail
umask 077
: "${RUN_ROOT:?}" "${CAPTURE:?}" "${MANIFEST:?}" "${LOCAL_ROOT:?}" "${PACKED_REFERENCE:?}" "${ROW_LABEL:?}"
case "$ROW_LABEL" in same-build|cross-build) ;; *) exit 2;; esac
test ! -e "$RUN_ROOT/rows-$ROW_LABEL.sra"
printf '%s  %s\n' 218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208 "$CAPTURE" \
  8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318 "$MANIFEST" | sha256sum -c - > "$RUN_ROOT/rows-$ROW_LABEL-inputs.log"
cmp "$RUN_ROOT/activations.spa" "$PACKED_REFERENCE"
python3 -B "$RUN_ROOT/admit.py" "rows-$ROW_LABEL-admission.json"
( ulimit -v 2097152
  /usr/bin/time -v -o "$RUN_ROOT/rows-$ROW_LABEL-resources.log" timeout --signal=TERM --kill-after=5s 190s \
    "$RUN_ROOT/bin/cpu_packed_activation_proof" --rows "$CAPTURE" "$MANIFEST" "$LOCAL_ROOT" "$PACKED_REFERENCE" \
    "$RUN_ROOT/rows-$ROW_LABEL.sra" > "$RUN_ROOT/rows-$ROW_LABEL.log" 2>&1 )
grep -Eq '^PACKED_ROWS_DONE ok=1 records=([1-9]|1[0-6]) packed_reference_bits=1 same_build_rows_bits=1 retained_capture_bits=1 cross_build_rows=REQUIRES_CMP sockets=0 worker_loads=0 model_GEN=0$' "$RUN_ROOT/rows-$ROW_LABEL.log"
python3 -B - "$RUN_ROOT/rows-$ROW_LABEL.log" <<'PY'
import sys
lines=open(sys.argv[1]).read().splitlines(); rows=[]; done=[]
for line in lines:
    fields=dict(p.split('=',1) for p in line.split()[1:] if '=' in p)
    if line.startswith('PACKED_ROWS '): rows.append(fields)
    if line.startswith('PACKED_ROWS_DONE '): done.append(fields)
assert len(done)==1 and len(rows)==int(done[0]['records'])*3
assert len({(v['record'],v['peer']) for v in rows})==len(rows)
assert all(v['row_bits']=='1' and v['partial_bits']=='1' and v['capture_bits']==str(int(v['peer']=='-1')) for v in rows)
PY
sha256sum "$RUN_ROOT/rows-$ROW_LABEL.sra" > "$RUN_ROOT/rows-$ROW_LABEL.sha256"
# END PACKED_ROWS
```

For the final cross-role gate, bring both SRA1 files to the existing private evidence location and run
`cmp "$ROWS_ROLE_A" "$ROWS_ROLE_B"`, then retain SHA-256 receipts and exit status. Do the same for
SPA1. Until both cross-role files have actually been compared, cross-build rows remain **NOT RUN**.
A differing `cmp` is **FAILED**, even if both local runs passed.
Stop on any command failure, timeout, missing/duplicate metric, source/digest/bit mismatch, unsupported
format/CPU, memory admission failure or new OOM. Preserve partial logs; no automatic rerun, sample
expansion or tuning. Total model work is at most 16 records × three masks × two scratch paths,
sequentially, under the per-process bounds above.
