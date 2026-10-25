# Packed activation proof: second-role relay and native settings

Use this with the [bounded proof handoff](CPU-PACKED-ACTIVATION-PROOF-20261025.md). It does not
change that diagnostic, its inputs, numerical gates or sample bounds. This recipe is authored and its host operations are **NOT RUN** by the desktop thread.
Shell/embedded Python syntax checks and a tiny synthetic Git-only relay check passed locally;
that check preserved an existing commit/tree/blob identity through a one-commit shallow restore
and passed `git fsck --strict`. Independent focused static review found no remaining blocker. It is not a dependency transfer or numerical proof result. The parent dispatches exact commands.
The first-role run proceeds separately; a missing second-role binding does not block it.
For the already compiled02 attempt, use **Continue the compiled02 attempt** below; do not repeat
the historical relay/build steps.

Role02's retained source is clean at `e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb`. Its real cache is
Release, CUDA ON, Unix Makefiles, GCC through `/usr/bin/cc` and `/usr/bin/c++`, `-O3 -DNDEBUG`.
The parent reports an E5-2683 v3 class CPU with AVX2/F16C/FMA and no runtime arithmetic/ISA overrides.
These are supplied inventory facts, not execution by this thread. Retained output hashes were
reported to match deployed engine/worker artifacts; this recipe does not rebuild those executables.
The parent keeps their full receipts privately.

The initial inventory incorrectly reported no pinned dependency source on02. A later verified
retained prefix is available; use the continuation below for the already compiled attempt. There
is no intact CPU-only cache on02. Use the real CUDA
cache as `CPU_BUILD`, not the role05 cache. Disabling CUDA/HIP/SYCL for this offline proof is an
intentional recorded configuration difference. Compiler, native ISA, C/C++ flags, IPO and linker
settings remain inherited from02. The shared build block now accepts ROLE02's CUDA-ON input and an
explicit fresh dependency relay path. It compares retained versus new effective native flags for
five linked CPU libraries, including source-specific flags, and compiler executable/ID/version.
A mismatch stops before CTest or pack. Saved link commands and cache deltas remain part of evidence.
The proof remains CPU-only; no worker/engine executable target or service is started.

## Minimal private bindings

The parent has supplied the exact verified `GGML05_SRC` and full `CPU_BUILD` path on02.
No existing-host binding remains missing; only fresh output/relay paths and the final published PIN
are chosen at dispatch. Other known
inputs are the retained source, original capture, same-hash manifest and model root already in the
parent's inventory. Do not publish private paths or guess an existing dependency directory.

Choose fresh absolute `EXPORT05` on05; fresh `IMPORT02`, `GGML_RELAY_SRC`, `NEW_SRC`, `NEW_BUILD`,
`RUN_ROOT`, `BLOCK`, `CANDIDATE_GIT` on02; and a fresh controller transfer directory. Their parents must exist.
`EXPORT05`, `IMPORT02` and source/build/run paths must be disk-backed and outside retained sources,
builds, model assets and evidence directories. `GGML_RELAY_SRC` is a new path, never the absent old
cache path. Retain transfer hashes at all three hops. No worker port, resident PID, package install,
network dependency fetch or shared-runtime mutation is involved.

The dependency transfer uses an exact object snapshot rather than all ancestor history. It contains
the pinned commit and its entire tree, with a shallow boundary at that commit on import. This keeps
its original commit/tree identity, modes and blobs; parent commits are intentionally absent. Git's
object verification and `fsck --strict` check the result. Only objects already present in the clean
verified05 dependency are used. No source edits, synthetic commit, clone from the network or build
in the dependency checkout occurs. Submodules, oversized trees or missing objects fail explicitly.
The object snapshot is capped at 512 MiB of source objects and a 64 MiB largest blob; the pack is
also capped at 512 MiB. Packing/unpacking each has a 120-second external bound and 2 GiB address
space, one pack thread and no delta search. The 37,544,692-byte capture is the existing retained file.
Keep files private and sequentially stage this small transfer under the parent's existing resource
admission policy; let the in-flight05 proof finish before running the05 export. Preparing this
recipe does not delay that proof and the transfer is not an additional model workload.

## Export existing objects and capture on05

Export `GGML05_SRC`, `CAPTURE05`, `EXPORT05`. The fixed pinned dependency is
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Do not run this on the desktop or against a guessed source.

```sh
# BEGIN PACKED_RELAY_EXPORT05
set -euo pipefail
umask 077
: "${GGML05_SRC:?}" "${CAPTURE05:?}" "${EXPORT05:?}"
export GGML05_SRC CAPTURE05 EXPORT05
export GIT_OPTIONAL_LOCKS=0 GIT_NO_LAZY_FETCH=1
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
test ! -e "$EXPORT05"
mkdir -m 700 "$EXPORT05"
python3 -B - <<'PY'
import os,pathlib,subprocess,json,hashlib
r=pathlib.Path(os.environ['EXPORT05']); src=pathlib.Path(os.environ['GGML05_SRC'])
pin='3cf03257f219afbe7334045ff7c6a06ac68c627d'
def git(*args): return subprocess.check_output(['git','-C',str(src),*args])
assert src.is_absolute() and r.is_absolute()
assert git('rev-parse','HEAD').decode().strip()==pin
assert not git('status','--porcelain').strip()
partial=subprocess.run(['git','-C',str(src),'config','--get-regexp',r'^(extensions\.partialclone|remote\..*\.promisor)$'],capture_output=True)
assert partial.returncode==1, 'partial/promisor sources unsupported: no lazy dependency fetch'
tree=git('rev-parse',pin+'^{tree}').decode().strip()
entries=git('ls-tree','-r','-t',pin).decode().splitlines()
assert all(not line.startswith('160000 ') for line in entries), 'submodules unsupported'
objects=sorted({pin,tree,*[line.split()[2] for line in entries]})
object_input=('\n'.join(objects)+'\n').encode()
sizes=subprocess.check_output(['git','-C',str(src),'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=object_input).decode().splitlines()
assert len(sizes)==len(objects) and all(len(line.split())==3 for line in sizes)
lengths=[int(line.split()[2]) for line in sizes]
assert sum(lengths)<=512<<20 and max(lengths)<=64<<20
(r/'objects.txt').write_bytes(object_input)
with (r/'source.json').open('x') as f: json.dump(dict(commit=pin,tree=tree,objects=len(objects),source_bytes=sum(lengths)),f,indent=2)
cap=pathlib.Path(os.environ['CAPTURE05']); assert cap.is_absolute() and cap.stat().st_size==37544692
with cap.open('rb') as f: assert hashlib.sha256(f.read()).hexdigest()=='218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208'
PY
( ulimit -v 2097152
  timeout --signal=TERM --kill-after=5s 120s git -C "$GGML05_SRC" -c pack.threads=1 \
    -c pack.windowMemory=32m pack-objects --window=0 --compression=1 --stdout \
    < "$EXPORT05/objects.txt" > "$EXPORT05/ggml-native.pack" )
test "$(wc -c < "$EXPORT05/ggml-native.pack")" -le 536870912
cp --reflink=auto -- "$CAPTURE05" "$EXPORT05/cpu-capture.bin"
( cd "$EXPORT05"; sha256sum ggml-native.pack source.json objects.txt cpu-capture.bin > transfer.sha256 )
test -z "$(git -C "$GGML05_SRC" status --porcelain)"
# END PACKED_RELAY_EXPORT05
```

The controller relays only `ggml-native.pack`, `source.json`, `objects.txt`, `cpu-capture.bin` and
`transfer.sha256` using its established private transfer channel: on05 → fresh controller directory →
fresh `IMPORT02`. Run `(cd "$TRANSFER_DIR" && sha256sum -c transfer.sha256)` at each hop and retain
stdout/exit status with the source host/path receipt. Do not transfer model files, ELFs, cache settings
from05, Git credentials or the entire dependency history. Parent-supplied hashes bind the transfer;
Git object IDs below independently bind the original pinned source.

## Import into a new private02 dependency path

Export `IMPORT02` and `GGML_RELAY_SRC`. Imported source contains a shallow Git history of exactly
one existing commit so the shared proof guard can verify the real pinned commit, clean status and
available headers. It does not pretend to be a full historical clone.

```sh
# BEGIN PACKED_RELAY_IMPORT02
set -euo pipefail
umask 077
: "${IMPORT02:?}" "${GGML_RELAY_SRC:?}"
export IMPORT02 GGML_RELAY_SRC
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
test ! -e "$GGML_RELAY_SRC"
( cd "$IMPORT02"; sha256sum -c transfer.sha256 )
python3 -B - <<'PY'
import os,pathlib,json,hashlib
r=pathlib.Path(os.environ['IMPORT02']); dst=pathlib.Path(os.environ['GGML_RELAY_SRC'])
assert r.is_absolute() and dst.is_absolute()
m=json.loads((r/'source.json').read_text()); assert m['commit']=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert 0<m['source_bytes']<=512<<20 and (r/'ggml-native.pack').stat().st_size<=512<<20
assert (r/'cpu-capture.bin').stat().st_size==37544692
with (r/'cpu-capture.bin').open('rb') as f: assert hashlib.sha256(f.read()).hexdigest()=='218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208'
PY
git init -q "$GGML_RELAY_SRC"
( ulimit -v 2097152
  timeout --signal=TERM --kill-after=5s 120s git -C "$GGML_RELAY_SRC" unpack-objects \
    < "$IMPORT02/ggml-native.pack" > "$IMPORT02/unpack.log" 2>&1 )
printf '%s\n' 3cf03257f219afbe7334045ff7c6a06ac68c627d > "$GGML_RELAY_SRC/.git/shallow"
git -C "$GGML_RELAY_SRC" -c core.autocrlf=false checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d > "$IMPORT02/checkout.log" 2>&1
git -C "$GGML_RELAY_SRC" fsck --strict --no-reflogs > "$IMPORT02/fsck.log" 2>&1
python3 -B - <<'PY'
import os,pathlib,json,subprocess
r=pathlib.Path(os.environ['IMPORT02']); src=pathlib.Path(os.environ['GGML_RELAY_SRC']); m=json.loads((r/'source.json').read_text())
def git(*args): return subprocess.check_output(['git','-C',str(src),*args]).decode().strip()
assert git('rev-parse','HEAD')==m['commit'] and git('rev-parse','HEAD^{tree}')==m['tree']
assert git('rev-list','--count','HEAD')=='1' and not git('status','--porcelain')
assert (src/'ggml/include/ggml.h').is_file()
with (r/'import-identity.json').open('x') as f: json.dump(dict(commit=m['commit'],tree=m['tree'],shallow=True,fsck='passed',status='clean'),f,indent=2)
PY
# END PACKED_RELAY_IMPORT02
```

## Build role02 from its own retained CUDA cache

Keep `GGML_RELAY_SRC` exported. Set `ROLE=02`, `BASE_SRC` to the verified retained02 source,
`BASE_PIN=e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb`, `CPU_BUILD` to its actual CUDA cache,
`CAPTURE="$IMPORT02/cpu-capture.bin"`, and fresh `NEW_SRC`, `NEW_BUILD`, `RUN_ROOT`, `BLOCK`,
`CANDIDATE_GIT` (a bare repository path).
`PIN` is the exact published commit containing this handoff and the compiler correction. Do not use
consumed05 paths. The source fetch below is for this Strata candidate only; dependency network
fetches stay disabled and the restored ggml source is the sole dependency binding.

Use this fail-closed wrapper in place of the role05 extraction example. Only candidate project
source is fetched from the existing project's origin, into a fresh private object store. This is
not a dependency fetch. Retained source/cache/Git metadata stay read-only.

```sh
# BEGIN PACKED_CANDIDATE02
set -euo pipefail
umask 077
: "${BASE_SRC:?}" "${CANDIDATE_GIT:?}" "${PIN:?}" "${BLOCK:?}" "${GGML_RELAY_SRC:?}"
export CANDIDATE_GIT GGML_RELAY_SRC GIT_OPTIONAL_LOCKS=0
test ! -e "$CANDIDATE_GIT"; test ! -e "$BLOCK"
git init --bare -q "$CANDIDATE_GIT"
timeout --signal=TERM --kill-after=5s 120s git -C "$CANDIDATE_GIT" fetch --depth=1 \
  "$(git -C "$BASE_SRC" remote get-url origin)" "$PIN"
test "$(git -C "$CANDIDATE_GIT" rev-parse "$PIN^{commit}")" = "$PIN"
git -C "$CANDIDATE_GIT" show "$PIN:docs/validation/CPU-PACKED-ACTIVATION-PROOF-20261025.md" | \
  sed -n '/^# BEGIN PACKED_BUILD_PACK$/,/^# END PACKED_BUILD_PACK$/p' > "$BLOCK"
test -s "$BLOCK"
bash "$BLOCK"
# END PACKED_CANDIDATE02
```

The shared block deliberately admits CUDA ON for ROLE02 and CUDA OFF for ROLE05, then forces only
GPU backends off for the proof. Source is archived from `CANDIDATE_GIT`. The retained source HEAD,
content/Git metadata, cache, ELFs, manifest, assets and resident stay untouched. No checkout of PIN
in the retained source.

Require `ROLE02_NATIVE_FLAGS ok=1 targets=5 compiler_identity_bits=1 gpu_backend_delta=explicit`
in the wrapper's retained stdout before any execution verdict is accepted. This is an exact
comparison of normalized effective native assignments, including CMake's per-source options/defines;
only source/build/dependency directory prefixes are normalized. A missing retained Makefile/compiler
receipt or an actual flag/compiler difference is a blocker, not permission to copy05 settings.
The cache provenance records original dependency and GPU settings plus intentional proof overrides.
The build log and a separate receipt preserve the proof's actual link command, with an automatic guard against GPU/engine link dependencies. Cache
comparison also retains native linker flags and RPATH/search settings.

Same-build screen: use the shared `PACKED_ROWS` block with `ROW_LABEL=same-build` and the local
SPA1 as `PACKED_REFERENCE`. The parent-supplied02 manifest and model root are read-only bindings.
Hash-check the manifest against `8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318`.
This is independent of the ongoing05 result and establishes no cross-build claim.

For cross-role gates, relay SPA1 artifacts privately, require exact `cmp`, then run the same bounded
rows block using the other role's SPA1 and `ROW_LABEL=cross-build`. Compare the two SRA1 files
exactly with `cmp` at the controller, retaining both hashes, producer role/PIN/ELF/compiler receipts
and exit status. Same-build restoration, cross-build packed bytes, retained capture partials and
cross-build native rows/ordered partials are separate outcomes. No one outcome substitutes for another.

Stop on any failure, missing binding/receipt, unsupported case, timeout, new OOM or source/flag/bit
mismatch. Preserve partial outputs and the failed05 build evidence. No automatic rerun, extra sample,
model GEN, service query, reload, firewall change, new package or deployment follows from this recipe.

## Continue the compiled02 attempt after the prefix/PIC gate

The parent reports that02 compiled/linked at `0dd834ba`, then stopped at the native-flags gate.
CTest, pack and rows did **not** run. Retain that failure and all existing outputs. The corrected
inventory found the original pinned dependency under another verified prefix on02; no relay or
source copy needs repeating. The parent also reports GNU time1.9-0.2 was installed by Hermes after
Eric explicitly authorized it. This continuation requests no installation.

The PIC difference has a specific cause: `CMakeLists.txt:2108` sets
`POSITION_INDEPENDENT_CODE ON` on `ggml-base` inside the CUDA shared-library branch. Turning CUDA
off removes that target property. Restore that property on **ggml-base only**, using a deferred
CMake include in the existing isolated build. Do not set global PIC or change native flags. The
incremental build may recompile ggml-base and relink the proof; it must not recompile other targets.
Preserve the original ELF path and hash; link the corrected proof into CONTINUE_ROOT/bin. The only waived flags are the exact reported
`ggml-base` warning differences: C `-Wdouble-promotion`, C++ `-Wextra-semi`. They are recorded as
warning-only deltas, not numerical flags. Every other assignment, including PIC/ISA/optimization,
definitions and source-specific options, must compare exactly after verified path normalization.

Parent exports the existing `RUN_ROOT`, retained `CPU_BUILD`, and
`VERIFIED_RETAINED_GGML02` (the newly verified original dependency prefix). Choose fresh
`CONTINUE_ROOT` under the existing private evidence directory. Source/build/dependency paths are
read from the existing proof cache/provenance; no new source/archive/relay campaign or candidate
checkout. The original SPA1/SRA1 outputs are absent because execution never started. This block
creates its new check/pack outputs under CONTINUE_ROOT. `PIN` in the original build evidence stays
`0dd834bad7fabd6266ff3bd95147c9f0f4ca4937`; the include hook and new ELF hash identify the explicit
PIC correction separately. The continuation itself is **NOT RUN** by this desktop thread.

```sh
# BEGIN PACKED_CONTINUE02
set -euo pipefail
umask 077
: "${RUN_ROOT:?}" "${CPU_BUILD:?}" "${VERIFIED_RETAINED_GGML02:?}" "${CONTINUE_ROOT:?}" "${CAPTURE:?}"
export RUN_ROOT CPU_BUILD VERIFIED_RETAINED_GGML02 CONTINUE_ROOT CAPTURE GIT_OPTIONAL_LOCKS=0
test ! -e "$CONTINUE_ROOT"
test -f "$RUN_ROOT/native-flags-comparison.json"
test ! -e "$RUN_ROOT/check.log"
test ! -e "$RUN_ROOT/activations.spa"
mkdir -m 700 "$CONTINUE_ROOT"
python3 -B - <<'PY'
import os,pathlib,re,json,subprocess,hashlib,shlex
r=pathlib.Path(os.environ['RUN_ROOT']); c=pathlib.Path(os.environ['CONTINUE_ROOT']); old=pathlib.Path(os.environ['CPU_BUILD'])
p=json.loads((r/'build-provenance.json').read_text()); assert p['role']=='02' and p['pin']=='0dd834bad7fabd6266ff3bd95147c9f0f4ca4937'
cache={}
for line in (r/'proof-CMakeCache.txt').read_text().splitlines():
    m=re.match(r'([^:#]+):([^=]+)=(.*)',line)
    if m: cache[m[1]]=m[3]
source=pathlib.Path(cache['CMAKE_HOME_DIRECTORY']); build=pathlib.Path(cache['CMAKE_CACHEFILE_DIR'])
assert all(v.is_absolute() for v in (r,c,old,source,build)) and c.parent==r
assert not cache.get('CMAKE_PROJECT_INCLUDE'), 'existing project include needs explicit review'
verified=pathlib.Path(os.environ['VERIFIED_RETAINED_GGML02']); used=pathlib.Path(p['ggml'])
for dep in (verified,used):
    assert dep.is_absolute()
    assert subprocess.check_output(['git','-C',str(dep),'rev-parse','HEAD'],text=True).strip()=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
    assert not subprocess.check_output(['git','-C',str(dep),'status','--porcelain'],text=True).strip()
assert (verified/'ggml/include/ggml.h').read_bytes()==(used/'ggml/include/ggml.h').read_bytes()
doc=(source/'docs/validation/CPU-PACKED-ACTIVATION-PROOF-20261025.md').read_bytes()
assert hashlib.sha256(doc).hexdigest()=='6b344f36203578632a39bfe71ed54aad20a621fd3d07f05438d41c78ccdf1ac4'
block=doc.decode().split('# BEGIN PACKED_BUILD_PACK\n',1)[1].split('# END PACKED_BUILD_PACK',1)[0]
start='sha256sum "$RUN_ROOT/bin/cpu_packed_activation_proof" > "$RUN_ROOT/elf.sha256"'
assert block.count(start)==1
# Reuse the published admission/check/pack tail, changing only its ELF location and output root.
tail=block[block.index(start):].replace('"$RUN_ROOT/bin/cpu_packed_activation_proof"','"$PROOF_ELF"')
(c/'check-pack.sh').write_text('set -euo pipefail\numask 077\n'+tail)
(c/'restore-ggml-base-pic.cmake').write_text('if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)\n  cmake_language(DEFER CALL set_property TARGET ggml-base PROPERTY POSITION_INDEPENDENT_CODE ON)\nendif()\n')
values=dict(NEW_SRC=str(source),NEW_BUILD=str(build),ORIGINAL_PROOF_ELF=str(r/'bin/cpu_packed_activation_proof'),PROOF_ELF=str(c/'bin/cpu_packed_activation_proof'))
# BASE_SRC comes from the retained cache; do not infer it from directory naming.
for line in (old/'CMakeCache.txt').read_text().splitlines():
    if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL='): values['BASE_SRC']=line.split('=',1)[1]
assert 'BASE_SRC' in values
values.update(BASE_PIN=p['base'],CAPTURE=os.environ['CAPTURE'])
with (c/'bindings.sh').open('x') as f:
    for key,value in values.items(): f.write('export '+key+'='+shlex.quote(value)+'\n')
with (c/'correction.json').open('x') as f: json.dump(dict(original_pin=p['pin'],verified_retained_ggml=str(verified),used_ggml=str(used),pic_target='ggml-base',pic=True,warning_only=dict(C_FLAGS='-Wdouble-promotion',CXX_FLAGS='-Wextra-semi')),f,indent=2)
PY
source "$CONTINUE_ROOT/bindings.sh"
printf '%s  %s\n' 218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208 "$CAPTURE" | sha256sum -c - > "$CONTINUE_ROOT/capture-input-hashes.log"
sha256sum "$ORIGINAL_PROOF_ELF" > "$CONTINUE_ROOT/proof-before-pic.sha256"
cp "$NEW_BUILD/CMakeCache.txt" "$CONTINUE_ROOT/cache-before-pic.txt"
timeout --signal=TERM --kill-after=5s 90s cmake -S "$NEW_SRC" -B "$NEW_BUILD" \
  -DCMAKE_PROJECT_INCLUDE:FILEPATH="$CONTINUE_ROOT/restore-ggml-base-pic.cmake" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH="$CONTINUE_ROOT/bin" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE:PATH="$CONTINUE_ROOT/bin" > "$CONTINUE_ROOT/reconfigure.log" 2>&1
python3 -B - <<'PY'
import os,pathlib,re,json,shlex
r=pathlib.Path(os.environ['RUN_ROOT']); c=pathlib.Path(os.environ['CONTINUE_ROOT'])
p=json.loads((r/'build-provenance.json').read_text()); old=pathlib.Path(os.environ['CPU_BUILD']); new=pathlib.Path(os.environ['NEW_BUILD'])
replacements={os.environ['BASE_SRC']:'@SRC',os.environ['NEW_SRC']:'@SRC',str(old):'@BUILD',str(new):'@BUILD',os.environ['VERIFIED_RETAINED_GGML02']:'@GGML',p['ggml']:'@GGML'}
def normalize(s):
    for key in sorted(replacements,key=len,reverse=True): s=s.replace(key,replacements[key])
    return s
report={}; warnings=[]
for target in ('strata_kernels_cpu','strata_expert_worker','strata_worker_tier','ggml-cpu','ggml-base'):
    original=[]
    for root in (old,new):
        found=list(root.rglob(f'CMakeFiles/{target}.dir/flags.make')); assert len(found)==1
        original.append([normalize(line.strip()) for line in found[0].read_text().splitlines() if '=' in line])
    adjusted=[list(v) for v in original]
    if target=='ggml-base':
        for key,flag in [('C_FLAGS','-Wdouble-promotion'),('CXX_FLAGS','-Wextra-semi')]:
            positions=[[i for i,line in enumerate(v) if line.startswith(key+' =')] for v in adjusted]
            assert all(len(v)==1 for v in positions)
            tokens=[shlex.split(adjusted[j][positions[j][0]].split('=',1)[1]) for j in (0,1)]
            assert '-fPIC' in tokens[0] and '-fPIC' in tokens[1], 'PIC not restored exactly'
            if tokens[0]==tokens[1]: continue  # Exact equality is preferred; no exception is needed.
            assert tokens[0].count(flag)==1 and tokens[1].count(flag)==0, 'unexpected warning delta'
            for j in (0,1): adjusted[j][positions[j][0]]=key+' = '+shlex.join([t for t in tokens[j] if t!=flag])
            warnings.append(dict(target=target,assignment=key,retained_only=flag,classification='warning-only'))
    report[target]=dict(retained=original[0],proof=original[1],equal_except_listed_warnings=adjusted[0]==adjusted[1])
with (c/'native-flags-comparison.json').open('x') as f: json.dump(dict(targets=report,warning_only=warnings),f,indent=2)
assert all(v['equal_except_listed_warnings'] for v in report.values()), 'meaningful native assignment differs'
for language in ('C','CXX'):
    identities=[]
    for root in (old,new):
        files=list((root/'CMakeFiles').glob(f'*/CMake{language}Compiler.cmake')); assert len(files)==1
        text=files[0].read_text()
        identities.append({k:re.search(r'^set\('+k+r'\s+(.*?)\)$',text,re.M).group(1) for k in (f'CMAKE_{language}_COMPILER',f'CMAKE_{language}_COMPILER_ID',f'CMAKE_{language}_COMPILER_VERSION')})
    assert identities[0]==identities[1], 'compiler identity changed'
with (c/'compiler-identity.json').open('x') as f: json.dump(dict(C='equal',CXX='equal'),f)
print(f'ROLE02_CORRECTED_FLAGS ok=1 pic=ggml-base warning_only={len(warnings)} meaningful_native_assignments=equal')
PY
timeout --signal=TERM --kill-after=5s 600s cmake --build "$NEW_BUILD" --target cpu_packed_activation_proof \
  --parallel 4 --verbose > "$CONTINUE_ROOT/incremental-build.log" 2>&1
python3 -B - <<'PY'
import os,pathlib,re
c=pathlib.Path(os.environ['CONTINUE_ROOT']); b=pathlib.Path(os.environ['NEW_BUILD'])
for line in (c/'incremental-build.log').read_text().splitlines():
    if re.search(r'Building (?:C|CXX) object ',line): assert 'ggml-base.dir/' in line, 'unexpected target recompiled'
links=list(b.rglob('CMakeFiles/cpu_packed_activation_proof.dir/link.txt')); assert len(links)==1
link=links[0].read_text(); (c/'proof-link-command.txt').write_text(link)
assert not re.search(r'(?:lib)?(?:cuda\w*|cublas\w*|amdhip\w*|sycl\w*|nvrtc\w*|strata_engine)(?:[.\s/-]|$)|libstrata_kernels\.a',link,re.I)
PY
cp "$NEW_BUILD/CMakeCache.txt" "$CONTINUE_ROOT/cache-after-pic.txt"
cp "$NEW_BUILD/compile_commands.json" "$CONTINUE_ROOT/compile_commands-after-pic.json"
sha256sum "$PROOF_ELF" > "$CONTINUE_ROOT/proof-after-pic.sha256"
sha256sum -c "$CONTINUE_ROOT/proof-before-pic.sha256"
# The original failure, cache/build logs and original ELF path/bytes remain preserved.
RUN_ROOT="$CONTINUE_ROOT" bash "$CONTINUE_ROOT/check-pack.sh" > "$CONTINUE_ROOT/check-pack-driver.log" 2>&1
# END PACKED_CONTINUE02
```

The required `CAPTURE` binding is the existing verified relayed file. This uses the
already relayed capture and dependency. Require corrected flags PASS, focused CTest Passed and
PACKED_DONE before continuing. Do not infer numerical success from compilation. Then use the
published `PACKED_ROWS` block with `RUN_ROOT="$CONTINUE_ROOT"`, preserving its new check/pack files,
`MANIFEST` and `LOCAL_ROOT` read-only. The corrected proof links to the new CONTINUE_ROOT/bin path, leaving the original ELF unchanged.
Use separate `same-build` and `cross-build` labels. Cross-role SPA1/SRA1 `cmp` gates remain unchanged.
Stop on any other flag difference, unexpected target compilation, resource refusal or bit mismatch.


## Resume after the warning-equality guard

The parent reports that the PIC reconfigure succeeded and the native flags now match exactly,
including both warning flags. The earlier guard wrongly required a warning difference. No build,
CTest or pack followed that reconfigure. The corrected gate above accepts exact equality first;
otherwise only the two documented retained-only warnings may be removed for comparison. All
PIC/numerical/compiler checks remain. This resume reuses the configured build and existing
`bindings.sh`/`check-pack.sh`; it does not reconfigure, stage sources or relay dependencies.

Export the same `RUN_ROOT`, `CPU_BUILD`, `VERIFIED_RETAINED_GGML02`, existing `CONTINUE_ROOT`, and
`RESUME_DOC` pointing at a freshly retrieved copy of THIS published document. Run only this block.
It extracts the corrected flag gate and the existing incremental-build/check-pack suffix. The old
failure and reconfigure evidence stay intact. Actual resume execution is **NOT RUN** here.

```sh
# BEGIN PACKED_RESUME02
set -euo pipefail
umask 077
: "${RUN_ROOT:?}" "${CPU_BUILD:?}" "${VERIFIED_RETAINED_GGML02:?}" "${CONTINUE_ROOT:?}" "${RESUME_DOC:?}"
export RUN_ROOT CPU_BUILD VERIFIED_RETAINED_GGML02 CONTINUE_ROOT RESUME_DOC GIT_OPTIONAL_LOCKS=0
test -f "$CONTINUE_ROOT/reconfigure.log"
test -f "$CONTINUE_ROOT/restore-ggml-base-pic.cmake"
test -f "$CONTINUE_ROOT/check-pack.sh"
test ! -e "$CONTINUE_ROOT/incremental-build.log"
test ! -e "$CONTINUE_ROOT/resume-warning-guard.sh"
source "$CONTINUE_ROOT/bindings.sh"
python3 -B - <<'PY'
import os,pathlib
c=pathlib.Path(os.environ['CONTINUE_ROOT'])
s=pathlib.Path(os.environ['RESUME_DOC']).read_text()
block=s.split('# BEGIN PACKED_CONTINUE02\n',1)[1].split('# END PACKED_CONTINUE02',1)[0]
start="python3 -B - <<'PY'\nimport os,pathlib,re,json,shlex\nr="
assert block.count(start)==1
suffix=block[block.index(start):]
assert "if tokens[0]==tokens[1]: continue" in suffix
assert 'cmake -S ' not in suffix and 'cmake --build ' in suffix
with (c/'resume-warning-guard.sh').open('x') as f: f.write('set -euo pipefail\numask 077\n'+suffix)
PY
bash "$CONTINUE_ROOT/resume-warning-guard.sh" > "$CONTINUE_ROOT/resume-warning-guard.log" 2>&1
# END PACKED_RESUME02
```

Require the corrected flags gate, incremental build, CTest and pack to pass before the unchanged
rows/artifact-comparison steps. An exact comparison reports `warning_only=0`. Stop on any failure;
do not retry completed stages or relax any other flags.
