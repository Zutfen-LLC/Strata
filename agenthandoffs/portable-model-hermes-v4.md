> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Parent-dispatched local portable model retry — v4

Supersedes the held v3 recipe after exact BF16 indexer binding and the complete
saved-header audit. Both independent EOS semantics and BF16 representation
are corrected. No further unsupported actual tensor role/type/shape or active
metadata was found in the saved six-shard inventory. This is readiness for
parent review and one newly dispatched bounded smoke, not physical or quality
acceptance. No retry is dispatched by this document.

Preserve `${PRIVATE_ARTIFACT_F5058E02}` and all prior source,
build and evidence directories. Select the new reviewed full PR head and use
fresh paths below. No model changes, lossy conversions, installs, merges,
performance sweep or automatic retry. The 38-token prompt, context 64,
max-new 3, reference row at position 37, comparison limits and existing
host/swap/per-die guards are unchanged.

The actual BF16 indexer Q/K weights are widened exactly to F32 with a 10 KiB
row buffer per layer, then projected with F32 accumulation on the requested
Vulkan die. Load conversion is explicit CPU work, not a graph-operation
fallback. The real run must report all 24 `BF16_TO_F32_EXACT` role conversions
and strict Vulkan graph assignments for every layer and the final head.

Audit: `portable-model-actual-header-audit-v1.md`.
Local verification: `portable-model-bf16-results-v1.md`.

This is the Strata product's full local CPU/Vulkan model path, stacked on
corrected PR41 `1d8eae1c3906962932b77e366f324c51aa89819d`. Parent/operator
executes the stages separately. This task does not access the fleet.
Use the exact full source SHA reported with the draft PR as
`EXPECTED_PORTABLE_MODEL_HEAD`; do not derive that expected value from the
candidate checkout itself. Do not merge, install, fetch new dependencies,
change services, run an optimization sweep or retry after a failed stage.

## Stage 1: isolated source/build and synthetic checks

For this retry, set the full `EXPECTED_PORTABLE_MODEL_HEAD` from the new PR
publication checkpoint (never from the candidate itself). Parent prepares
that exact source at a fresh path; suggested deterministic path names are:

```sh
: "${EXPECTED_PORTABLE_MODEL_HEAD:?new full reviewed PR42 head}"
MODEL_TAG05=$(printf '%.7s' "$EXPECTED_PORTABLE_MODEL_HEAD")
MODEL_SRC05=${PRIVATE_ARTIFACT_34F3F6F4}
MODEL_BUILD05=${PRIVATE_ARTIFACT_BD75CC36}
MODEL_CHECKS05=${PRIVATE_ARTIFACT_4094BD40}
MODEL_RUN05=${PRIVATE_ARTIFACT_6D714E01}
```

The failed old source/build/results are never overwritten. The parent must
choose the new head and dispatch this single retry explicitly.


Use fresh source/build/evidence paths; preserve earlier work and resident
models. The retained ggml dependency on05 is
`${PRIVATE_ARTIFACT_9CA76AB8}`, commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Reuse existing compilers, Vulkan
headers and the same previously working SPIR-V include configuration.

```sh
set -eu
: "${MODEL_SRC05:?fresh exact candidate source checkout}"
: "${MODEL_BUILD05:?fresh isolated build path}"
: "${MODEL_CHECKS05:?fresh synthetic evidence directory}"
: "${EXPECTED_PORTABLE_MODEL_HEAD:?full draft PR head supplied by parent}"
: "${MODEL_CMAKE05:?absolute existing cmake executable, version3.24 or newer}"
GGML_SRC05=${PRIVATE_ARTIFACT_9CA76AB8}
test "$(git -C "$MODEL_SRC05" rev-parse HEAD)" = "$EXPECTED_PORTABLE_MODEL_HEAD"
test -z "$(git -C "$MODEL_SRC05" status --porcelain --untracked-files=no)"
test "$(git -C "$GGML_SRC05" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test ! -e "$MODEL_BUILD05"
test ! -e "$MODEL_CHECKS05"
mkdir "$MODEL_CHECKS05"
# Optional values below are only existing host-specific settings from the
# accepted PR41 build, not a request to install or change any toolchain.
"$MODEL_CMAKE05" -S "$MODEL_SRC05" -B "$MODEL_BUILD05" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON \
  -DSTRATA_PORTABLE_QSA=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DGGML_VULKAN=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DSTRATA_GGML_DIR="$GGML_SRC05" \
  -DCMAKE_PREFIX_PATH="${MODEL_CMAKE_PREFIX05:-}" \
  -DCMAKE_CXX_FLAGS="${MODEL_CXX_FLAGS05:-}" \
  > "$MODEL_CHECKS05/configure.log" 2>&1
"$MODEL_CMAKE05" --build "$MODEL_BUILD05" --parallel 2 \
  --target portable_model portable_head_test portable_model_test \
  portable_device_admission_test portable_eos_metadata_test portable_qsa_bf16_test \
  portable_ffn_test portable_gdn_test portable_ple_test \
  portable_qsa_test portable_qsa_gguf_test \
  portable_ffn_vulkan_test portable_gdn_vulkan_test \
  portable_ple_vulkan_test portable_qsa_vulkan_test \
  > "$MODEL_CHECKS05/build.log" 2>&1
"$(dirname "$MODEL_CMAKE05")/ctest" --test-dir "$MODEL_BUILD05" \
  -R '^portable_(head|model|device_admission|eos_metadata|ffn|gdn|ple|qsa|qsa_gguf|qsa_bf16)_test$' \
  --output-on-failure > "$MODEL_CHECKS05/cpu.log" 2>&1
```

The header-only EOS regression uses the artifact's distinct EOS metadata,
BOS/padding IDs and248320 vocabulary, with no weight payload reads. It checks
that valid independent EOS fields reach manifest admission and that missing,
non-integer or out-of-vocabulary values are still rejected for each role.

The synthetic full-model test uses sparse files at actual H2560/48-layer
geometry, full vocabulary 248320, actual BF16 indexer / Q8 dense / F32 norm
role encodings, real in-process CPU experts,
configured host PLE, fresh-token HC, a nonzero expert/reference check,
late-layer failure poison/reset, PLE boundary versus tokenizer generation EOS,
prompt separators, generated EOS/reset, and bounded generation. It reads no
real weights. It is several GiB of host memory, not a full-model payload scan.
The real model's vocabulary248320 is read from its embedding tensor.

## Stage 2: strict device fixtures

Preserve the accepted Vulkan-index-to-PCI mapping: device0 is
`0000:07:00.0`, device1 is `0000:0b:00.0`, each a V340L die with8GiB.
If the established visibility mapping is different, pass its verified indexes
as DIE0/DIE1 and record them. Do not infer a physical identity from the backend
name alone. Run sequentially; any failure/timeout terminates the stage.

```sh
set -eu
: "${DIE0:?verified ggml index for PCI0000:07:00.0}"
: "${DIE1:?verified ggml index for PCI0000:0b:00.0}"
test "$DIE0" != "$DIE1"
timeout 90s "$MODEL_BUILD05/portable_device_admission_test" --vulkan "$DIE0,$DIE1" \
  > "$MODEL_CHECKS05/device-admission.log" 2>&1
for DIE in "$DIE0" "$DIE1"; do
  timeout 120s "$MODEL_BUILD05/portable_qsa_bf16_test" --vulkan "$DIE" \
    > "$MODEL_CHECKS05/bf16-die${DIE}.log" 2>&1
  timeout 180s "$MODEL_BUILD05/portable_head_test" --vulkan "$DIE" \
    > "$MODEL_CHECKS05/head-die${DIE}.log" 2>&1
  timeout 180s "$MODEL_BUILD05/portable_ffn_vulkan_test" --device "$DIE" --expect-device V340 \
    > "$MODEL_CHECKS05/ffn-die${DIE}.log" 2>&1
  timeout 180s "$MODEL_BUILD05/portable_gdn_vulkan_test" --device "$DIE" --expect-device V340 \
    > "$MODEL_CHECKS05/gdn-die${DIE}.log" 2>&1
  timeout 180s "$MODEL_BUILD05/portable_ple_vulkan_test" --device "$DIE" --expect-device V340 \
    > "$MODEL_CHECKS05/ple-die${DIE}.log" 2>&1
  timeout 180s "$MODEL_BUILD05/portable_qsa_vulkan_test" --device "$DIE" --expect-device V340 \
    > "$MODEL_CHECKS05/qsa-die${DIE}.log" 2>&1
done
```

The device-admission fixture creates two distinct handles to the first die
and requires session admission to reject their shared physical device
identity. It also checks that the two actual die identities reach the next
admission guard. Equal descriptions are not used as identity. The CLI rejects
repeated Vulkan indices before initializing either backend.

Require the BF16 fixture's exact F32 parity, role/lifetime/budget checks and
strict device success, the head's `PASS strict Vulkan portable head` line,
and all prior
fixture success lines, including QSA's high-score first-key spare regression.
The head fixture uses actual H2560/HC4/rank320 with a bounded64-row output
matrix; the later real model stage exercises the entire vocabulary head.
Scheduled tensor computation must use the requested die with no fallback.
Host routing, unweighted CPU expert rows, weighted FFN merge, PLE row gathers
and recurrent-state staging are intentional CPU work and must be reported.

## Stage 3: one real-model request, both dies

Run only when parent dispatches this stage. No new full-file hashes are
required. Check retained manifest identity (a1.5MB read) and existing shard
sizes/header admission. This manifest covers experts in shards4–6; it is
not a six-shard checksum manifest.

```sh
set -eu
: "${MODEL_RUN05:?fresh real-model evidence directory}"
: "${MODEL_TIMEOUT05:?parent-selected single-run seconds, at most1800; suggested1200}"
MANIFEST05=${PRIVATE_ARTIFACT_32CC92CE}
printf '%s  %s\n' \
  8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318 \
  "$MANIFEST05" | sha256sum -c -
python3 - <<'PY'
from pathlib import Path
root=Path('${PRIVATE_ARTIFACT_9AA3D2B0}')
sizes=[10946624,682434912,54400261312,49446841216,49668930400,34015618784]
for n,size in enumerate(sizes,1):
    path=root/f'Qwen3.8-Flash-Next-Q8_0-0000{n}-of-00006.gguf'
    assert path.stat().st_size==size,(str(path),path.stat().st_size,size)
    print(path,size)
PY
python3 "$MODEL_SRC05/tools/validation-harness/portable_model_smoke_v1.py" \
  --binary "$MODEL_BUILD05/portable_model" --evidence "$MODEL_RUN05" \
  --devices "$DIE0,$DIE1" --split 32 --device-cap-mib 7680 \
  --timeout-seconds "$MODEL_TIMEOUT05"
```

The supervisor launches one process once: retained38-token prompt, context64,
max-new3, threads4, CPU expert workers12, layers0–31 on the first die,
layers32–47 and global head on the second. No GPU expert residency, listeners,
external engine, MTP or speculation. It does not retry. Time or memory-bound
failure terminates only its own child process group and retains evidence.

Before `Popen`, the existing supervisor stats the six unique shards (no
payload scan) and sums their sizes as a **worst-case file working-set bound**.
It records that term, the explicit32GiB anonymous admission cap passed to the
CLI, and an8GiB reserve. For this retained artifact the total is
231174706208B (about215.3GiB). Both host MemAvailable and any finite effective
memory-cgroup allowance must admit that total. The known05 host has about
251GiB RAM and recently about249GiB available, so an idle host should pass.
Mapped logical size is not a measurement or guarantee of actual residency;
the CLI's32GiB host cap does **not** bound total resident model memory.

The supervisor records MemAvailable, SwapTotal/SwapFree/SwapUsed, pswpin and
pswpout baselines and the child's initial VmSwap. Historical system swap need
not be zero. The existing0.2s loop starts during loading, before the first
completed token. Every sample records those host/cgroup values and child
VmSwap/RSS/high-water RSS. It terminates only its owned process group if
host/effective-cgroup available memory falls below8GiB, child VmSwap becomes
nonzero, or system pswpout grows from the pre-launch baseline. It records the
failing sample and preserves the stopped run without retrying.

The inherited cgroup-v2 membership and all visible ancestor memory.max and
memory.current values are read, never changed; the minimum finite remaining
allowance also constrains admission and the ongoing8GiB floor. Unreadable or
unsupported applicable memory accounting stops preflight rather than silently
ignoring a ceiling. This is a sampled guard with conservative initial
headroom, not an atomic RAM reservation or a promise that transient OOM is
impossible if unrelated workloads change between samples.

Before launch it also reads aggregate used/total VRAM on each named PCI device.
Each must have room for its separate7680MiB cap; the remaining *observed*
headroom is recorded. During the run it samples both devices every0.2seconds,
records process RSS/high-water RSS and stops if observed per-die growth
exceeds that cap. These are aggregate sampled values, not an exact transient
peak or unique attribution in the presence of other GPU users. CLI snapshots
also record Vulkan memory budget/availability before model loading and after
each token. Preserve baseline and peak values together with actual scheduler
allocations, admitted native scratch/state/workspace and head reservations.
No pooled16GiB claim is valid.

`placements.tsv` includes actual scheduler backend, operation, layer and any
fallback. The supervisor verifies non-metadata tensor work for all48 layers
and head against the exact contiguous placement; QSA attention and GDN ops
must be present. Host ROUTE_AND_MERGE and the 24 startup BF16_TO_F32_EXACT conversions are
identified separately. Only those two indexer roles may use the load-conversion
marker; it cannot mask a CPU matrix operation or fallback. CPU fallback
is never enabled by this recipe. Model state remains fail-stop on any partial
token failure. A new process/reset-and-replay is a new separately dispatched
run, not an automatic retry.

Return `argv.json`, `stdout.jsonl`, `stderr.log`, `placements.tsv`,
`memory.jsonl`, `summary.json` and exact source/build identities. Keep raw
logits host-local. The summary reports generated IDs, the retained prefix
`[71093,12305,198]`, first divergence (including early termination), exit
status and sampled memory. A prefix match is not a quality verdict.

## Stage 4: full-vocabulary numerical comparison

The first-generation logit row follows the last prompt token at position37:
`$MODEL_RUN05/logits/pos-37.f32`, expected993280B (248320F32). The retained02
reference is `${PRIVATE_ARTIFACT_91D2F803}`,
SHA256 `4143957a612227b67ed3c47ce930365e5e8249d03ef77ee1442ca4722c33fa96`.
Parent/operator compares where both rows are already available through approved
local access; this recipe does not transfer files or upload logits to Git.

The existing comparison utility expects three rows on its command line. For
this one retained row, call its same metric implementation without duplicating
one row into a fictitious three-position result:

```sh
# Run on the host where approved local reference and candidate rows exist.
# Set REFERENCE_F32 and CANDIDATE_F32 to those exact local paths.
python3 - "$MODEL_SRC05/tools/validation-harness/compare_logits.py" \
  "$REFERENCE_F32" "$CANDIDATE_F32" <<'PY'
import hashlib,importlib.util,json,pathlib,sys
source,reference,candidate=sys.argv[1:]
assert hashlib.sha256(pathlib.Path(reference).read_bytes()).hexdigest()=='4143957a612227b67ed3c47ce930365e5e8249d03ef77ee1442ca4722c33fa96'
spec=importlib.util.spec_from_file_location('retained_compare_logits',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
a,b=module.read(reference),module.read(candidate)
assert len(a)==len(b)==248320
metrics=module.compare(a,b)
print(json.dumps({'position':37,'limits':module.LIMITS,'metrics':metrics},indent=2,allow_nan=False))
raise SystemExit(0 if metrics['numeric_pass'] else 1)
PY
```

Retain all full numerical metrics and failures. Do not tune existing limits
against the candidate. No cross-backend bit identity is promised. Additional
teacher-forced positions require their corresponding reference rows; the
single retained row cannot establish them. Parent owns numerical and quality
interpretation. The retained02 engine is our existing product baseline, not
an external engine. Its actual local argv is
`${PRIVATE_ARTIFACT_8D7D2A05}`; NDon input
configuration is not evidence of that local process's actual arguments.
