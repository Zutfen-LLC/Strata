# Packed activation proof: second-role relay and native settings

Use this with the [bounded proof handoff](CPU-PACKED-ACTIVATION-PROOF-20261025.md). It does not
change that diagnostic, its inputs, numerical gates or sample bounds. This recipe is authored and its host operations are **NOT RUN** by the desktop thread.
Shell/embedded Python syntax checks and a tiny synthetic Git-only relay check passed locally;
that check preserved an existing commit/tree/blob identity through a one-commit shallow restore
and passed `git fsck --strict`. Independent focused static review found no remaining blocker. It is not a dependency transfer or numerical proof result. The parent dispatches exact commands.
The first-role run proceeds separately; a missing second-role binding does not block it.

Role02's retained source is clean at `e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb`. Its real cache is
Release, CUDA ON, Unix Makefiles, GCC through `/usr/bin/cc` and `/usr/bin/c++`, `-O3 -DNDEBUG`.
The parent reports an E5-2683 v3 class CPU with AVX2/F16C/FMA and no runtime arithmetic/ISA overrides.
These are supplied inventory facts, not execution by this thread. Retained output hashes were
reported to match deployed engine/worker artifacts; this recipe does not rebuild those executables.
The parent keeps their full receipts privately.

There is no intact CPU-only cache or existing pinned dependency directory on02. Use the real CUDA
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
