# Fixed two-host CPU group: Hermes handoff

Base source: `c9cef19a35369d5a3721d11515d524ebd6476da1`. Candidate is the full published PR head
containing this file; bind `CANDIDATE` to that exact SHA, never a moving branch. Source, fixtures and
commands are authored. **Builds, executable tests and inference: NOT RUN on Geekom/Valinor.**
No two-host numerical acceptance or speedup is established. No fleet work was dispatched by this coding task.

The change adds only the opt-in fixed CPU endpoint group, necessary per-host use of the existing
guardian/config driver, focused fixtures and bounded host capture/replay diagnostics. The latest opt-in
entry-row extension preserves legacy wire bytes and the default one-endpoint route. It does not add
discovery, rebalancing or a network GPU mode.

## Prerequisites and focused checks

The original full-model comparison at 9ed478e5543885e019205512f2450ab4336d393e has completed.
The parent reports that the split arm failed every frozen first-logit screen and was slower; the
local baseline/restore first logits were bit-identical. This is supplied evidence, not execution
or raw-receipt inspection by this coding thread. **Do not rerun that campaign or either duplicate-arena
diagnostic.** The codec benchmark has completed. The current authorized next step is the
[fixed row-mode distributed handoff](CPU-ROWS-SPLIT-20261025.md), runtime source e145df6e. All executable
commands in this historical file are no longer authorization to repeat prior work.

### Completed reply codec checks (historical commands)

Supplied Hermes update at `e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb`: all3 focused CTests
passed in20.83s and benchmark-source comparison passed. The verbose healthy test log was23710870B;
the handoff's erroneous post-hoc1MiB limit aborted before the benchmark, which had NOT RUN then.
The parent authorized only the unchanged benchmark suffix at that SAME e145df6e source/binary pin.
Do not rebuild or repeat the passed tests. This documentation correction changes future test logging
only; the pending benchmark remains pinned to e145df6e. Preserve the original verbose log.

Supplied Hermes model-free benchmark at `b9ab00c64dc283b57c62da047ccb355100d6762a`: focused
check and all20 metrics/guards passed; no resident contact. Row means in nanoseconds:

| Shape | Encode | Frame copy | Decode | Central prepare/reduce | Block copy |
| --- | --- | --- | --- | --- | --- |
| T1/K10/H2560 | 209560.433 | 4405.718 | 74335.216 | 17199.223 | 4333.788 |
| T2/K10/H2560 | 419738.368 | 13175.709 | 146577.792 | 34366.481 | 11265.723 |

Weighted encode/decode means were27713.943/3454.104ns at T1 and41409.729/6986.844ns at T2.
These parent-supplied measurements target scalar codecs; they do not isolate the live critical path.
The following narrow change copies contiguous float payload bytes only on4-byte IEEE754 little-endian
hosts. Other hosts retain the previous scalar codec. Headers/indices, schema/API, finite/exact-set/size
checks, buffer caps, native arithmetic and framing remain unchanged. No raw structs or unaligned float
casts are used. New fixtures cover scalar-reference bytes, signed zero, subnormals/finite boundaries,
unaligned input, legacy permitted nonfinite payloads and row nonfinite refusal. Portable fallback is
statically inspected; execution here and cross-endian execution remain NOT RUN.

The benchmark source MUST byte-match b9ab00c6 before measurement; counts/workload/timing are unchanged.
Preserve its retained baseline binary/logs. Only focused CPU fixtures and the same benchmark are
authorized here: no new production worker build, reload, engine build or model test. The complete
current candidate resident stays untouched; never request shutdown timing.

Supplied Hermes results at `e56a837bc75ff1076e8d4bff6a01c306d4b302b2`: after correcting an old-worker
binding (which correctly refused row capability), LEGACY2 and ROWS2 both matched A0 FIRST logits
(993280B bitwise),213 tokens and102724 entries. Each made11136 calls with0 failures. One local loopback
sample measured prompt1.899/2.486s, decode14.663/18.393s and reported14.5/11.5TPS, weighted/rows.
Request bytes were124299072 each; replies123236352/1052500104B; native pool service9.148/9.239s;
row scatter/reduction213.079ms. Complete candidate residency remained allocation1, swap/OOM0.
These are parent-supplied results, not local execution or private-receipt inspection. They establish
prompt0 single-resident correctness only; the original split failure remains uncleared.

Retained client phase totals, weighted/rows: prepare.018/.021s, queue wake.218/.241s,
encode/send.545/.562s, receive/decode10.340/14.205s, finish wait10.961/14.930s. Receive/decode
includes waiting for worker compute, encoding and transmission. This supports reply-side critical-path
growth, not an isolated client-decoder bottleneck. These scopes overlap; never sum them as elapsed time.
Power-of-two percentile buckets are coarse bounds. No worker shutdown is needed to obtain more timing.

The benchmark calls existing reply encode, encode_frame and decode functions independently; central
work mirrors indexed preparation/scatter and calls the shared production reducer. Weighted central
work copies/checks the already reduced reply. Central scratch is retained, so the tiny production
coverage-vector allocation, locks, queueing, expert computation and I/O are excluded. The same-size
block copy copies the complete frame. bytes_per_iteration describes the input/output dataset, not
total memory traffic; central rows performs several passes. No phase total predicts end-to-end time.

Two dense fixed shapes: T1/K10/H2560 and T2/K10/H2560. Weighted frames are10272/20512B; row frames
102456/204876B. Independent exact little-endian payload/frame and signed-zero checks run outside
timing, before/after each mode. Every phase uses64 warmups and4096 measured iterations, checks retained
capacities and reports aggregate nanoseconds only. Explicit vector scratch capacities must stay below
16MiB (not an RSS/AS claim). A29s cooperative check and30s process alarm bound execution. Repeated
identical buffers are cache-hot synthetic data, not the actual mask distribution or storage/LAN bandwidth.

Execute ONLY on the retained CPU-fixture role. The existing focused fixtures use small synthetic native
services on owned loopback sockets. The benchmark uses no sockets or expert kernels. No real model,
resident endpoint/control, production worker/CUDA/native-OFF build, wrapper or driver is needed.
Bind exactly these six variables using the verified bindings supplied privately with this handoff:

| Binding | Required source |
| --- | --- |
| CANDIDATE | Exact published commit containing this section, supplied with the handoff; never branch tip. |
| SRC | Existing CPU-fixture source checkout recorded in the e56a837 CPU build receipt. |
| CPU_BUILD | Its existing compatible CPU build cache from that SAME receipt; no fresh configure/cache. |
| BENCH_OUTPUT | That cache's recorded runtime-output directory plus /cpu_row_codec_bench. |
| PRIOR_ROOT | The retained b9ab00c6 benchmark artifact root, with its binary and benchmark.log; read-only. |
| RUN_ROOT | New absolute disk-backed artifact directory for this benchmark; MUST NOT exist. |

Missing binding/cache, dirty source, different CMake source binding or incompatible options means STOP
and report the exact mismatch. Do not search for alternatives, install anything, alter build flags,
borrow another cache or author a sequencing wrapper. The following is the complete command block;
run once in bash after filling those bindings. It updates only the existing fixture checkout/cache and
builds only three fixture/benchmark targets. Existing production ELFs and resident processes are not
targets. Fetch CANDIDATE once, then mechanically extract the marked block from that exact Git object
using the accompanying command; do not transcribe/edit it or create another driver.

```sh
# BEGIN CPU_REPLY_CODEC_CHECK
set -euo pipefail
umask 077
: "${CANDIDATE:?exact published SHA}"
: "${SRC:?source path from retained CPU receipt}"
: "${CPU_BUILD:?cache path from same receipt}"
: "${BENCH_OUTPUT:?recorded runtime-output directory plus /cpu_row_codec_bench}"
: "${PRIOR_ROOT:?retained b9ab00c6 benchmark artifact root}"
: "${RUN_ROOT:?fresh absolute disk-backed output root}"
for p in "$SRC" "$CPU_BUILD" "$BENCH_OUTPUT" "$PRIOR_ROOT" "$RUN_ROOT"; do test "${p#/}" != "$p"; done
test "${#CANDIDATE}" = 40
test ! -e "$RUN_ROOT"
test -f "$CPU_BUILD/CMakeCache.txt"
test -z "$(git -C "$SRC" status --porcelain)"
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CPU_BUILD/CMakeCache.txt")" = "$SRC"
grep -Eq '^STRATA_BUILD_TESTS:BOOL=ON$' "$CPU_BUILD/CMakeCache.txt"
grep -Eq '^STRATA_ENABLE_CUDA:BOOL=OFF$' "$CPU_BUILD/CMakeCache.txt"
test -x "$PRIOR_ROOT/cpu_row_codec_bench"
grep -Eq '^CPU_ROW_CODEC_DONE ok=1 check_only=0 ' "$PRIOR_ROOT/benchmark.log"
git -C "$SRC" cat-file -e "$CANDIDATE^{commit}"
git -C "$SRC" checkout --detach "$CANDIDATE"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
mkdir -m 700 "$RUN_ROOT"
git -C "$SRC" rev-parse HEAD > "$RUN_ROOT/source-sha.txt"
cp -- "$PRIOR_ROOT/cpu_row_codec_bench" "$RUN_ROOT/prior-cpu_row_codec_bench"
cp -- "$PRIOR_ROOT/benchmark.log" "$RUN_ROOT/prior-benchmark.log"
git -C "$SRC" show b9ab00c64dc283b57c62da047ccb355100d6762a:tests/worker/cpu_row_codec_bench.cpp > "$RUN_ROOT/benchmark-source-baseline.cpp"
cmp -- "$RUN_ROOT/benchmark-source-baseline.cpp" "$SRC/tests/worker/cpu_row_codec_bench.cpp"
timeout --signal=TERM --kill-after=5s 60s cmake -S "$SRC" -B "$CPU_BUILD" -DFETCHCONTENT_FULLY_DISCONNECTED=ON > "$RUN_ROOT/configure.log" 2>&1
timeout --signal=TERM --kill-after=5s 180s cmake --build "$CPU_BUILD" --target expert_worker_test cpu_worker_group_test cpu_row_codec_bench --parallel 4 > "$RUN_ROOT/build.log" 2>&1
test -x "$BENCH_OUTPUT"
cp -- "$BENCH_OUTPUT" "$RUN_ROOT/cpu_row_codec_bench"
sha256sum "$RUN_ROOT/prior-cpu_row_codec_bench" "$BENCH_OUTPUT" "$RUN_ROOT/cpu_row_codec_bench" > "$RUN_ROOT/binary-sha256.txt"
timeout --signal=TERM --kill-after=5s 240s ctest --test-dir "$CPU_BUILD" --no-tests=error -R '^(expert_worker_test|cpu_worker_group_test|cpu_row_codec_bench_check)$' > "$RUN_ROOT/check.log" 2>&1
grep -Eq '100% tests passed, 0 tests failed out of 3' "$RUN_ROOT/check.log"
for name in expert_worker_test cpu_worker_group_test cpu_row_codec_bench_check; do
  grep -Eq "Test[[:space:]]+#[[:space:]]*[0-9]+: $name[[:space:]].*Passed" "$RUN_ROOT/check.log"
done
timeout --signal=TERM --kill-after=5s 32s "$RUN_ROOT/cpu_row_codec_bench" > "$RUN_ROOT/benchmark.log" 2>&1
test "$(grep -Ec '^CPU_ROW_CODEC T=' "$RUN_ROOT/benchmark.log")" = 20
for t in 1 2; do
  for mode in weighted rows; do
    for phase in encode frame_copy decode central_prepare_reduce block_copy; do
      test "$(grep -Ec "^CPU_ROW_CODEC T=$t K=10 H=2560 entries=$((t*10)) mode=$mode phase=$phase warmup=64 measured=4096 " "$RUN_ROOT/benchmark.log")" = 1
    done
  done
done
grep -Eq '^CPU_ROW_CODEC_DONE ok=1 check_only=0 scratch_limit_bytes=16777216 deadline_seconds=30 model=0 sockets=0 expert_kernels=0 phases_independent=1$' "$RUN_ROOT/benchmark.log"
test "$(wc -c < "$RUN_ROOT/benchmark.log")" -le 65536
cat "$RUN_ROOT/binary-sha256.txt" "$RUN_ROOT/benchmark.log"
# END CPU_REPLY_CODEC_CHECK
```

With the six bindings exported, execute that exact block mechanically (bash, no manual transcription):

```sh
set -euo pipefail
test ! -e "$RUN_ROOT"
git -C "$SRC" fetch origin "$CANDIDATE"
git -C "$SRC" show "$CANDIDATE:docs/validation/CPU-FIXED-GROUP-20261025.md" |
  sed -n '/^# BEGIN CPU_REPLY_CODEC_CHECK$/,/^# END CPU_REPLY_CODEC_CHECK$/p' | bash
test -f "$RUN_ROOT/benchmark.log"
grep -Eq '^CPU_ROW_CODEC_DONE ok=1 check_only=0 ' "$RUN_ROOT/benchmark.log"
```

Any nonzero command, alarm/timeout, CPU_ROW_CODEC_FAIL, missing/duplicate metric or changed capacity is
a failed measurement: preserve the fresh output, report it, and stop without automatic retry. Do not
lengthen counts/deadlines, alter affinities or investigate the resident. Future full checks use quiet
CTest output: require its3/3 success summary and all three named Passed lines, not hidden per-test stdout.
Detailed CTest logs may be large; their size is not a test failure. Expected benchmark: exactly20 metric lines and DONE
check_only=0, no failures. Return those small logs/pins and per-phase weighted/rows means. Compare codec
and frame-copy costs with the block-copy reference, without claiming that their sum equals the live
decode delta. No latency pass threshold is introduced. Authored checks: NOT RUN locally.

### Completed batched entry rows, one candidate full resident (historical commands)

Supplied Hermes evidence at `abdf92e3270c38e327f8c2ce265a0a36ba21bfaf`: EO0 matched A0 first
logits/213 tokens/102724 entries,28s overall. EO1 ordered matched A0 FIRST logits (993280B, all
metrics0),213 tokens and102724 entries;11136 shadow windows passed,108577 wire frames,
max11 exchanges/window, peak row storage204800B. EO1 overall49s, prompt4.73s/decode33.06s.
No load/allocation change, swap/OOM0, resident intact. These are parent-supplied results, not local
execution or private-receipt inspection. This proved arithmetic recovery for prompt0, not optimized
throughput, final-token logits, physical two-host correctness or full-model acceptance. Do not repeat it.

The new optional `--remote-cpu-entry-rows` uses one batched request/reply per active group peer/window,
no shadow or per-row RPC, and no new threads. LOCAL_SETUP flag4 requests row schema1/max80;
only that opt-in ACK appends8B. Old workers refuse clearly without fallback. Message20 has the
existing INFER payload; message21 contains session/sequence/schema/count and ascending indexed
unweighted native rows. Exact requested-set/identity/finite checks precede publication. The engine
reconstructs CPU entry order and calls the shared reducer once, preserving the existing helper/GPU
merge boundary. Legacy weighted requests/replies/defaults stay unchanged. Mode is session-scoped,
not a weight-cache key; existing capable residency can switch reply mode without another load.

Caps:80 indexed rows,819200B float row data,819380B payload/819396B frame. Worker native row scratch,
packed decoder rows and central rows each bound at819200B; row encode/receive payloads bound at819380B
and send frames at819396B, per peer. Mask/index/weight tables cap at80. These are separate retained
buffers, not a claim that aggregate scratch is819200B. Zero-work peers send nothing. Any failure drains
all pending peer exchanges and exposes no group partial. Existing per-peer I/O overlaps GPU work;
central reduction stays in finish(), with a separate row_reduce_ms scope that overlaps neither request
service nor network timing by definition. It is already included in group wall/wait time. Do not sum
overlapping scopes as elapsed time.

#### Build/fixture gates before touching the current resident

Use exact candidate SHA, retained caches/native/compiler/ccache settings, and fresh disk-backed
artifact roots/logs. The cached target outputs MUST differ from the live resident executable path/inode.
Preserve existing ELF outputs BEFORE builds replace them; never overwrite retained binaries/logs.
No package, access, CPU-policy or monitor changes. The old full resident remains untouched until
ALL these gates pass. CPU fixture role only (new root must not exist):

```sh
set -e
umask 077
mkdir -m 700 "$CPU_ARTIFACT_ROOT"
cp -- "$CPU_WORKER_OUTPUT" "$CPU_ARTIFACT_ROOT/prior-worker"
cmake --build "$CPU_BUILD" --target expert_worker_test cpu_worker_group_test worker_local_test cpu_capture_group_test strata-expert-worker --parallel 4 > "$CPU_ARTIFACT_ROOT/build.log" 2>&1
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(expert_worker_test|cpu_worker_group_test|worker_local_test|worker_local_binary_test|cpu_capture_group_test)$' > "$CPU_ARTIFACT_ROOT/tests.log" 2>&1
```

The new cases are in the existing cpu_worker_group_test: exact legacy/row codec bytes, mask
normalization/signed zero, malformed/missing/duplicate/foreign/nonfinite/oversize rows, max retained
capacities, old-worker refusal/no fallback, shared sequence ordering, reconnect capability reset,
one/two real peers bitwise unsplit, alternating T8/T1/T2/T1 and masks/K, zero work, copied input
lifetime, reversed completion, failure draining and central overflow. Other listed tests retain the
normal worker/real executable/local-cache/ordered-proof compatibility gates. Keep fixture source
creation clear of tiny inherited RLIMIT_FSIZE. These checks are authored, NOT RUN locally.

After CPU gates, bulk CUDA role, outside inference (worker target remains CPU-only):

```sh
set -e
umask 077
mkdir -m 700 "$CUDA_ARTIFACT_ROOT"
cp -- "$CUDA_ENGINE_OUTPUT" "$CUDA_ARTIFACT_ROOT/prior-engine"
cp -- "$CUDA_WORKER_OUTPUT" "$CUDA_ARTIFACT_ROOT/prior-worker"
cmake --build "$CUDA_BUILD" --target strata strata-expert-worker --parallel 8 > "$CUDA_ARTIFACT_ROOT/build.log" 2>&1
cp -- "$CUDA_ENGINE_OUTPUT" "$CUDA_ARTIFACT_ROOT/engine-rows"
cp -- "$CUDA_WORKER_OUTPUT" "$CUDA_ARTIFACT_ROOT/worker-rows"
sha256sum "$CUDA_ARTIFACT_ROOT/engine-rows" "$CUDA_ARTIFACT_ROOT/worker-rows"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$CUDA_ARTIFACT_ROOT/engine-rows" --native-experts on
```

Small incremental native-OFF role, with its existing local output binding:

```sh
set -e
umask 077
mkdir -m 700 "$OFF_ARTIFACT_ROOT"
cp -- "$OFF_ENGINE" "$OFF_ARTIFACT_ROOT/prior-engine"
cmake --build "$OFF_BUILD" --target strata --parallel 2 > "$OFF_ARTIFACT_ROOT/build.log" 2>&1
python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_ENGINE"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$OFF_ENGINE" --native-experts off
```

Stop/report any failure, keeping bounded log tails and pins. Other backend links/runtime remain
NOT RUN. Coding task performs no builds, fleet operations or replacement; parent dispatches Hermes.

#### One managed replacement, then weighted/rows on the SAME candidate resident

Use three private configs with unchanged static bundle/run helpers: LOAD, LEGACY, ROWS. No new
sequencing script. Stage/verify LOAD before old service release so every path/import/asset is concrete.
Keep exact model/manifest/native/profile/request assets, GPU960/1378 and full22238 CPU experts /
116135731200B, workers12/loaders4, original turbo/affinities/budgets/deadlines and original monitor
policy. No split, remote peer config or duplicate arena. Candidate engine and worker bindings are the
preserved new ELFs above. The configured4-thread setup affinity must remain broad until READY.

After all gates, parent-authorized release uses the OLD service's retained owner controller and
identity receipts, with no engine active. Its existing `C.stop_owned_worker()` records STOP intent,
waits/reaps the actual guardian and verifies hierarchy absence before checking the port. Do not call
it from a fresh import or forge a Popen handle for an attached resident: the helper deliberately
refuses that. If the retained owning controller/handle is unavailable, report that precise runtime
blocker before stopping anything. An uncertain STOP must be reconciled through its original receipt/
owner, never resent. Only a completed owned stop/reap permits LOAD; no simultaneous old/new arena.

| Config | Existing-driver fields |
| --- | --- |
| LOAD | New root, no attach_root; one arm LOAD, restart:true, requests:[]; final_worker:retain, failure_worker:stop. Newly owned partial-load failure uses existing stop cleanup; no automatic restore/retry. |
| LEGACY | Separate new root; attach_root=completed LOAD root; attached_worker_idle:true; one arm LEGACY, restart:false, requests:[0], reference:reference; final/failure worker both retain. |
| ROWS | Same as LEGACY with name ROWS/new root and only added engine option --remote-cpu-entry-rows. Both use SAME candidate engine/worker bytes/full resident. |

All three use cpu_route:group and exactly one existing numeric loopback --remote-cpu-worker. Remove
--remote-cpu-regroup-diagnostic entirely, including its value, from BOTH warm configs. Production
row mode has no shadow/reuse-only diagnostic behavior. Use --remote-cpu-resident, trusted manifest,
no upload fallback. Before either warm attach verify idle complete forever residency, original owned
guardian/worker/monitor/listener, VmSwap0, healthy original guard and unchanged OOM counters. Require
observed exact cache-hit reuse/zero local/network load READY markers before GEN. The row config also
requires `CPU entry rows ready schema=1 ordered_reduce=1`. Missing markers refuse before requests.

G1 pre-stage normalization is mandatory: argv[0]=@engine, worker_argv[0]=@worker, engine
--remote-cpu-local-manifest AND worker --local-manifest values=@manifest. Assets bind existing absolute
SOURCE files, not old argv literals/future bundle paths. New run roots must NOT be precreated. Both
warm arms use original prompt0 GEN256, capture_first_logits:true, no global dump prefix/capture env,
and the retained A0 reference enforcing exact213 tokens/102724 entries. Enable engine RPC timing
identically in both (`--remote-cpu-rpc-timing`); worker timing, if enabled, stays fixed across them.
Use request_seconds600 for both as a bounded diagnostic ceiling, with per-frame/startup/cleanup and
monitor limits unchanged. Cold LOAD startup/read/verification is separate from warm request timing.

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$LOAD_PRIVATE_CONFIG"
# ALL gates and completed owned old-worker stop/reap must precede the next command.
python3 -B "$LOAD_ROOT/run.py" --run-root "$LOAD_ROOT"
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$LEGACY_PRIVATE_CONFIG"
python3 -B "$LEGACY_ROOT/run.py" --run-root "$LEGACY_ROOT"
```

STOP until LEGACY token/entry gate passes and its993280B finite FIRST logits bitmatch retained A0,
using the exact unchanged raw comparison below with LEGACY/ROWS paths. A failed baseline leaves the
new complete resident retained and stops the comparison. Only after success and fresh idle/cache checks:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$ROWS_PRIVATE_CONFIG"
python3 -B "$ROWS_ROOT/run.py" --run-root "$ROWS_ROOT"
```

Run this unchanged comparator once after LEGACY (RESULT_FIRST_RAW_F32=LEGACY-first-logits.0), then
after ROWS (ROWS-first-logits.0), each in its corresponding root. It must succeed before continuing:

```sh
PYTHONPATH="$SRC/tools/validation-harness" python3 -B -c '
import json,pathlib,sys
import compare_logits as c
a,b=map(pathlib.Path,sys.argv[1:]);m=c.compare(c.read(a),c.read(b))
m["bitwise_equal"]=a.read_bytes()==b.read_bytes()
print(json.dumps(m,indent=2,allow_nan=False))
raise SystemExit(0 if m["bitwise_equal"] and m["numeric_pass"] else 1)
' "$A0_PROMPT0_FIRST_RAW_F32" "$RESULT_FIRST_RAW_F32"
```

Require ROWS exact A0 first logits/tokens/entries plus reply_mode=entry_rows/schema1. One active peer
sends one request/window; no ordered shadow counters or per-rank frames. Compare decode/prompt
seconds, same213-token decode TPS, worker-before/after LEGACY fields infer_in_bytes/infer_out_bytes,
requests and cpu_service_ns deltas, client exchange/finish wait, and group row_reduce_ms. All fields
already exist in the driver receipts/reports. Scopes overlap; do not sum them or use overall startup
as decode time. Report the single sample per mode and actual bytes, without a speedup generalization.
Ten CPU rows at T1 cost102456B in one reply versus10272B weighted; actual mask occupancy determines
the measured bytes. Do not compare against the deliberately chatty ordered proof as production gain.

Never issue STATUS/STOP/UNLOAD while an engine owns the session. After owner reaping, verify unchanged
new resident bytes/allocation/load/upload/unloads/expirations and swap/OOM0; retain that new resident,
including on a numerical failure. Preserve exact binaries/configs/outputs/errors; no automatic retry
or original-worker restoration. Return one flat receipt with gates, hashes, cold setup separately,
both warm numerical/performance results and final ownership/cache cleanup. A fixed two-host campaign
is later and is NOT included or authorized by this first full-resident validation.

### Completed ordered-row proof (historical commands)

The parent supplies decisive results at `6c4b4cbf8f1592739f003f65b163479b23e3a715`: focused CPU,
CUDA/native-OFF links and CLI gates passed. RG0 off bitmatched A0 first logits,213 tokens and102724
entries. RG1 on bitmatched B first logits exactly (993280B); versus A0 maxabs0.7277, relativeL2.0852,
probability max.00249 and KL.00039 failed the frozen screen, with equal top1. RG1's213 tokens matched
both A0/B. DONE, final tier report and arm summary all agree RG0 entries102724/RG1 entries102720;
original B also102720. Reported routed columns6263/6259 and virtual frames11081/11088 are aggregate
evidence, not a retained per-entry routing explanation. No reload, allocation1, swap/OOM0, engines
reaped. These are supplied Hermes results, not local execution/private-receipt inspection.
Regrouping is sufficient to reproduce prompt0 divergence; this does not clear other inputs, physical
peer correctness or performance. No further off/on regrouping run is requested.

The existing option now accepts `--remote-cpu-regroup-diagnostic ordered`. It retains the same
one-endpoint/trusted-manifest/text/reuse-only/complete-idle-forever guards and the existing full
resident. For every nonempty live window it first obtains the normal full-mask shadow reply.
For each active rank it sends one unit-weight entry per token using the unchanged protocol, skips
empty ranks, and places returned rows in token/rank order. The existing exported
`ExpertWorkerService::accumulate_partial` then uses the ORIGINAL mask/weights/frame order. Only
finite output bit-identical to the shadow is published. Signed-zero and one-ULP differences refuse;
there is no numerical tolerance or shadow fallback. A failed exchange publishes nothing.

Bounds: at most11 sequential exchanges/window,819200B row storage (T8/K10/H2560), plus bounded
request/reply/accumulator scratch. No extra weight arena, worker binary/protocol change, model
reload, GPU graph instrumentation or sequencer. Unit-row transport can expose rounding/batch-shape
differences; the gate stops rather than assuming equivalence. This deliberately chatty proof is
separate from any future batched per-entry reply implementation. Its timing is NOT optimized
production throughput; no protocol optimization belongs in this task.

#### Fresh outputs before builds, then focused gates — authored, NOT RUN locally

Pin this candidate SHA in the existing source/build trees. All variables below are private absolute
bindings. Reuse retained CMake/native/compiler/ccache settings. Before ANY build, choose new
disk-backed artifact roots (parents exist, roots MUST NOT exist), preserve cached engine ELFs,
and use new log names. Do not overwrite old build logs, retained ELFs or prior result roots:

CPU fixture role only:

```sh
set -e
umask 077
mkdir -m 700 "$CPU_ARTIFACT_ROOT"
cmake --build "$CPU_BUILD" --target cpu_capture_group_test --parallel 4 > "$CPU_ARTIFACT_ROOT/cpu-build.log" 2>&1
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^cpu_capture_group_test$' > "$CPU_ARTIFACT_ROOT/cpu-test.log" 2>&1
```

After that passes, bulk CUDA role only, outside inference. `CUDA_ENGINE_OUTPUT` is the existing
cached target ELF path; preserve it before the incremental build replaces that cached output:

```sh
set -e
umask 077
mkdir -m 700 "$CUDA_ARTIFACT_ROOT"
cp -- "$CUDA_ENGINE_OUTPUT" "$CUDA_ARTIFACT_ROOT/prior-engine"
sha256sum "$CUDA_ARTIFACT_ROOT/prior-engine"
cmake --build "$CUDA_BUILD" --target strata --parallel 8 > "$CUDA_ARTIFACT_ROOT/cuda-build.log" 2>&1
cp -- "$CUDA_ENGINE_OUTPUT" "$CUDA_ARTIFACT_ROOT/engine-ordered"
sha256sum "$CUDA_ARTIFACT_ROOT/engine-ordered"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$CUDA_ARTIFACT_ROOT/engine-ordered" --native-experts on
```

Small incremental native-OFF role only, using its local retained output binding `OFF_ENGINE`:

```sh
set -e
umask 077
mkdir -m 700 "$OFF_ARTIFACT_ROOT"
cp -- "$OFF_ENGINE" "$OFF_ARTIFACT_ROOT/prior-off-engine"
sha256sum "$OFF_ARTIFACT_ROOT/prior-off-engine"
cmake --build "$OFF_BUILD" --target strata --parallel 2 > "$OFF_ARTIFACT_ROOT/off-build.log" 2>&1
python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_ENGINE"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$OFF_ENGINE" --native-experts off
```

Run only the lines belonging to each build role, stop on any failure and return bounded log tails.
Keep fixture source creation clear of a tiny inherited RLIMIT_FSIZE. The existing focused CTest
now covers real resident shadow/row bit identity over multi-token/K10/masks/zero work, increasing
sequences and unchanged cache accounting. Scripted checks cover original input/rank masks/unit
weights, skipped ranks, row failure after partial progress, one-ULP/signed-zero refusal, oversize
refusal before exchange, and the exact11-frame/819200B maxima. Prior regrouping cases remain.
All executable checks here remain NOT RUN by the coding thread; other backends remain unvalidated.

#### Same candidate off control, then one ordered prompt0

Copy the proven G1 private config into two separate new output roots/one-arm configs `EO0` and
`EO1`; do not precreate the bundle roots. Use the SAME preserved `engine-ordered` bytes in both.
Keep `cpu_route:"group"`, one existing numeric loopback `--remote-cpu-worker`, `restart:false`,
`requests:[0]` (original GEN256), current full resident attach receipts/idle check, workers12/loaders4,
GPU960/1378/CPU22238 placement,116135731200B arena, affinities/turbo/native/model/profile assets.
No warmups/worker startup; success/failure both retain. No peer configs or `STRATA_CPU_CAPTURE`.
Use `capture_first_logits:true` with no global dump prefix. Keep per-frame RPC deadlines unchanged;
set harness `request_seconds:600` for BOTH requests. Up to11 exchanges, repeated activation quantization
and smaller work batches justify this bounded10-minute diagnostic ceiling, rather than interpreting
the old decode duration as a reliable timeout. Startup/cleanup/monitor budgets stay unchanged.

Pre-stage `assets.engine` binds the new immutable candidate file; `assets.worker`/`assets.manifest`
bind existing retained SOURCE files. Normalize engine argv[0] to `@engine`, worker argv[0] to
`@worker`, engine `--remote-cpu-local-manifest` AND worker `--local-manifest` values to `@manifest`.
Never leave old absolute argv literals or future bundle paths in assets. Mechanical staging then
resolves exactly these four tokens. Add `--remote-cpu-regroup-diagnostic off` to EO0 and `ordered`
to EO1; that value is the only arithmetic change. Both arms keep `reference:"reference"` bound to
the retained A0 summary, enforcing213 exact tokens/102724 entries. Require existing peers1/cache-hit/
exact reuse/zero setup load READY markers plus `CPU regroup diagnostic=off reuse_only=1` or
`CPU regroup diagnostic=ordered reuse_only=1` respectively.

Stage/run EO0 using the existing bundle/run entry points below. Compare `EO0-first-logits.0`
to retained A0 using the unchanged functions AND raw-byte equality. Require finite993280B, all metrics0,213 exact
tokens and102724 entries. A failed control stops here; no EO1 staging/launch. Fresh live preflight
must confirm original owned listener/guardian/worker/monitor, complete idle forever cache, VmSwap0,
healthy original guard and unchanged OOM. Only then use the same entry points for EO1:

```sh
set -e
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$EO0_PRIVATE_CONFIG"
python3 -B "$EO0_ROOT/run.py" --run-root "$EO0_ROOT"
PYTHONPATH="$SRC/tools/validation-harness" python3 -B -c '
import json,pathlib,sys
import compare_logits as c
a,b=map(pathlib.Path,sys.argv[1:]);m=c.compare(c.read(a),c.read(b))
m["bitwise_equal"]=a.read_bytes()==b.read_bytes()
print(json.dumps(m,indent=2,allow_nan=False))
raise SystemExit(0 if m["bitwise_equal"] and m["numeric_pass"] else 1)
' "$A0_PROMPT0_FIRST_RAW_F32" "$EO0_ROOT/EO0-first-logits.0"
```

STOP until EO0 token/entry/bitwise-logit gates and idle/cache checks pass. Then:

```sh
set -e
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$EO1_PRIVATE_CONFIG"
python3 -B "$EO1_ROOT/run.py" --run-root "$EO1_ROOT"
PYTHONPATH="$SRC/tools/validation-harness" python3 -B -c '
import json,pathlib,sys
import compare_logits as c
a,b=map(pathlib.Path,sys.argv[1:]);m=c.compare(c.read(a),c.read(b))
m["bitwise_equal"]=a.read_bytes()==b.read_bytes()
print(json.dumps(m,indent=2,allow_nan=False))
raise SystemExit(0 if m["bitwise_equal"] and m["numeric_pass"] else 1)
' "$A0_PROMPT0_FIRST_RAW_F32" "$EO1_ROOT/EO1-first-logits.0"
```

Frozen screens stay unchanged.
An internal shadow bit mismatch is a refusal, not a successful diagnostic. Return the first error,
no retries. Require final report `diagnostic_regroup=ordered`, checked_windows matching nonempty
logical requests, max_exchanges<=11, peak_row_bytes<=819200 and positive wire_frames. No worker
STATUS/STOP/UNLOAD during engine ownership; only post-reap checks. Preserve exact outputs, all
hashes, sent line, internal gate counters, token/logit/entry results and unchanged resident bytes,
allocation/unloads/expirations/load/upload/resource receipts in one flat result. No full campaign,
extra capture or additional arena. A passing ordered result supports arithmetic recovery for this
prompt; it does not establish efficient two-host routing or production throughput.

### Completed live off/on regrouping (historical commands)

The parent supplies a record0 microfixture PASS at `bc8e57541ad9b934774c1b184bf31b757ef3d8cc`:
independent-full/capture bits matched before children, both actual peer/independent bits and group/manual
bits matched. Aggregate selected payload104448000B across20 experts, reported owned RSS bound238391296B,
swap/OOM0, children reaped, original full resident untouched. Regroup/unsplit maxabs1.49e-8, relativeL2
6.4e-8. Per-peer counts/bytes await a supplied receipt correction and are not transcribed here. This is
supplied execution, not a local run or raw-receipt inspection. Do not repeat the microfixture.

`--remote-cpu-regroup-diagnostic off|on` is optional; omission leaves normal behavior unchanged. Both
explicit values require one `--remote-cpu-worker`, trusted manifest, resident reuse, text, no upload
fallback, no network GPU and no `STRATA_CPU_CAPTURE`. Setup requires an idle complete forever cache
and an exact cache hit; it cannot load/upload. `off` uses the normal exchange. `on` keeps the actual
one-member group/socket owner, but sends original global-ordinal peer0 then peer1 masks sequentially
against the full arena. Inputs and signed weights stay identical within a window; no renormalization.
Empty peers send nothing. Copy the first active reply then add peer1 in F32, exactly like the group.
Wire sequences increase; a refused/failed exchange publishes no partial. No worker/protocol/arena or
CUDA graph-node/callback/synchronization change. This isolates live feedback from physical two-peer state.

#### Focused gates — authored, NOT RUN locally

Reuse retained build directories, native settings, compilers and ccache. On the CPU fixture role:

```sh
cmake --build "$CPU_BUILD" --target cpu_capture_group_test --parallel 4
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^cpu_capture_group_test$'
```

This extends the existing one-small-arena test: unchanged normal group, ordered masks/input bytes,
both active and each one-empty peer, signed zero, multiple windows/increasing sequences, full-bitmap
ownership across layers, cache reuse, and a real peer0 success followed by peer1 refusal with no result.
Existing positive-owned child cleanup is reused. No full model/external endpoint. After that gate,
link the candidate engine outside inference using the bulk CUDA role (parallel8) and the retained
native-OFF directory on the small incremental role. Do not change CMake options or resident worker:

```sh
cmake --build "$CUDA_BUILD" --target strata --parallel 8
cmake --build "$OFF_BUILD" --target strata --parallel 2
python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_ENGINE"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$OFF_ENGINE" --native-experts off
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$CANDIDATE_ENGINE" --native-experts on
```

Native-OFF preflight and Qwen3.6 explicitly refuse the new option. Other backend links/runtime remain
NOT RUN unless separately supplied. Keep draft and preserve all baseline/candidate ELF hashes.

#### Two warm requests; on is forbidden until the control passes

Copy the proven G1 private config twice. Each has one arm, `RG0` or `RG1`, `restart:false`,
`requests:[0]`, its own new disk-backed `run_root`, `cpu_route:"group"`, and the SAME candidate engine.
Keep original prompt0 GEN256, GPU placement960/1378, full22238 CPU partition116135731200B, workers12,
loaders4, affinities, CPU policy, model/profile/native assets, deadlines and environment unchanged.
No warmups/extra requests. Current attached ownership receipts and `attached_worker_idle:true` are
required. Both `final_worker`/`failure_worker` stay `"retain"`; omit peer configs. No worker startup.

| Pre-stage field | Value |
| --- | --- |
| `assets.engine` | Existing absolute candidate ELF, identical for both runs |
| `assets.worker`, `assets.manifest`, `assets.reference` | Existing absolute retained worker/manifest/A0 summary source files |
| engine `argv[0]`, worker `worker_argv[0]` | `@engine`, `@worker` |
| engine `--remote-cpu-local-manifest`, worker `--local-manifest` | Both values `@manifest` |
| endpoint | One existing numeric loopback `--remote-cpu-worker`; no legacy endpoint flag |
| added option | RG0: `--remote-cpu-regroup-diagnostic off`; RG1: same option with `on` |
| `capture_first_logits` | `true`; remove global first-logit prefix and `STRATA_CPU_CAPTURE` |
| arm `reference` | RG0: `"reference"`; RG1: omit, retaining diagnostic differences as results |
| READY markers | Existing peers1/exact cache-hit reuse/zero setup load markers plus `CPU regroup diagnostic=off reuse_only=1` or `on reuse_only=1` |

The four `@asset` tokens above are mandatory source-config normalization: do not leave old argv
absolute literals or put future staged paths in assets. One existing mechanical staging operation
rewrites these tokens. Before each run verify original owned guardian/worker/monitor/listener,
complete idle forever arena, VmSwap0, original healthy guard and unchanged OOM. No worker
STATUS/STOP/UNLOAD during engine ownership. After complete engine-owner reaping, verify unchanged
arena bytes/allocation/unloads/expirations, zero new load/upload, idle forever and resource accounting.

Run RG0 using the existing driver, then gate the raw993280B first-logit row:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$RG0_PRIVATE_CONFIG"
python3 -B "$RG0_ROOT/run.py" --run-root "$RG0_ROOT"
PYTHONPATH="$SRC/tools/validation-harness" python3 -B -c '
import json,pathlib,sys
import compare_logits as c
a,b=map(pathlib.Path,sys.argv[1:]);m=c.compare(c.read(a),c.read(b))
m["bitwise_equal"]=a.read_bytes()==b.read_bytes()
print(json.dumps(m,indent=2,allow_nan=False))
raise SystemExit(0 if m["bitwise_equal"] and m["numeric_pass"] else 1)
' "$A0_PROMPT0_FIRST_RAW_F32" "$RG0_ROOT/RG0-first-logits.0"
```

Require all commands succeed, exact A0 213-token array/102724 entries from the reference gate,
and bit-identical first logits. **Failure stops here; do not stage/run RG1.** After a passing control
and fresh idle/cache verification, stage/run RG1 with the same two entry points and its config/root.
Compare using unchanged functions/screens, not the three-request CLI:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$RG1_PRIVATE_CONFIG"
python3 -B "$RG1_ROOT/run.py" --run-root "$RG1_ROOT"
PYTHONPATH="$SRC/tools/validation-harness" python3 -B -c '
import json,sys
import compare_logits as c
a,b,on=map(c.read,sys.argv[1:])
print(json.dumps(dict(limits=c.LIMITS,a0_vs_on=c.compare(a,on),b_vs_on=c.compare(b,on)),indent=2,allow_nan=False))
' "$A0_PROMPT0_FIRST_RAW_F32" "$B_PROMPT0_FIRST_RAW_F32" "$RG1_ROOT/RG1-first-logits.0"
```

The last command reports failed screens as data; its exit0 is not numerical acceptance. Compare
complete RG1 tokens in `arm-RG1-summary.json` against retained A0/B request0: equality, length and
first differing token separately. Return one flat receipt with source/ELF/config hashes, exact sent
line, off gates, on metrics/tokens, READY/cache and final resource/cleanup accounting. The final
peer report's `virtual_peer_frames=P0/P1` counts actual frames; tier requests/timings count logical
windows with sequential exchanges. No performance claim, automatic retry or complete campaign.

If off matches A0 and on reproduces B-scale first-logit divergence, regrouping can amplify in the live
loop; physical peer correctness remains unproved. If on stays within the frozen screen, investigate
actual two-peer/full-size slot/remote integration for this prompt. Intermediate outcomes are
inconclusive. No threshold tuning, original split-failure clearance or extra capture/microfixtures.

### Current status: resident-monitor incident; duplicate-arena diagnostics withdrawn

The supplied sequential attempt at `dc772896dcd78ca8e3dbed331e821ee8f24c4f69` stopped during the
first temporary subset load at157s. No actual-subset numerical result was produced. Full replay
lowered the original worker's VmSwap from422148KiB to276464KiB; it later reached1060228KiB.
The external guard terminated replay, and the temporary children were reaped.

Subsequent supplied triage confirms that the **original resident's own monitor** recorded sustained
owned VmSwap above1GiB for30s, then caused the guardian to stop/unload the116135731200B arena and
exit0. Observed swap1060228KiB exceeded its1048576KiB threshold. Recorded minimum MemAvailable
was38157612KiB (about36.4GiB). No kernel/cgroup OOM event was found. The external guard targeted
the replay timeout process only; it was not the1800s deadline. All original and temporary processes
were confirmed absent after the incident, with the service port free. This is a **resource stop, not a numerical
failure**. It does not establish subset mapping or group correctness.

The source agrees with the confirmed signal path: the retained monitor sends through its worker
pidfd, and the guardian owns worker cleanup. The diagnostic controls only its forked temporary
children and never sends STOP/UNLOAD to the full endpoint. No production, monitor, protocol or
diagnostic code change is needed for this incident correction.

The prior handoff incorrectly instructed the external guard to tighten its ceiling after a swap
decrease. That was an added interpretation, not the parent's fixed-baseline policy. **The baseline
is the original worker's freshly measured, stable prelaunch VmSwap and stays fixed for the run.
Decreases never lower that ceiling; an increase above the initial ceiling requires stopping.**
Record initial, maximum and final values. The independent resident monitor retains its own original
thresholds and may stop the worker regardless of the diagnostic guard. At1060228KiB, both the
initial422148KiB diagnostic ceiling and the monitor's1GiB threshold were exceeded; ratcheting is
not required to explain either stop. Neither guard policy is changed here.

**Stop duplicate-arena experiments, including the sequential one-half mode.** The previous
`--local-subsets` procedure and config template are withdrawn from runtime use; their source remains
for history, not authorization to retry. Do not attempt recovery sweeps, swapoff, mlock, settings
changes or another diagnostic load. Parent owns restoring only the normal full resident with four
loaders/twelve inference workers and the unchanged original guard policy. This coding task launches
nothing and makes no zero-swap or timing-qualification claim.

### Completed record0 real-weight microfixture, no full resident connection (historical commands)

The parent supplies a completed G1 PASS using the retained engine/worker and `ddb0d652` helpers:
one original prompt0 GEN256 completed213 tokens and102724 entries, exactly A0. Its993280B raw-F32
first-logit file was bit-identical to A0; all reported error metrics were0. READY reported one group
member, cache hit/reused116135731200B and zero setup local/network bytes. The original full arena
remained allocation1 with zero unload/expiry/swap, and engine owners were reaped. Staging was fixed
through four existing `@asset` tokens, without source change. A post-summary wrapper KeyError was
corrected read-only, without rerun. These are supplied Hermes results, not execution or private
receipt inspection by this coding thread. They do not clear the original failed split comparison.

The remaining unconfirmed seam is actual subset loading/lookup or two-peer integration. Source
uses global `(layer, expert)` keys for loading, digest verification and inference lookup; no static
mapping defect was found. The new mode of the existing `cpu_group_numeric_diagnostic` fixes the next
input to captured record0, layer0/T2/K10. It derives peer ownership from the COMPLETE original CPU
bitmap before filtering to that record's referenced experts. IDs are never renumbered.

The independent reference reads one real Q8 blob at a time, verifies its trusted global-ID digest,
evaluates native gate/up/down rows without the worker pool, and reduces signed weighted rows in
frame-entry order. Its full result must bitmatch the captured reply before any child is created.
Only then do two owned loopback workers load their disjoint selected IDs through the original
manifest/local source path. Each actual peer must bitmatch its independent reference; the real
`ExpertWorkerGroup` must bitmatch the manual peer sum. The first READY peer uses the existing setup
keepalive while the second loads; both exchanges drain before comparisons. Group setup reuses the
small subsets, with no reload. No worker protocol, production arithmetic, capture format, engine,
GEN, remote endpoint, full resident connection or new sequencing framework is added.

Bounds are enforced in code: at most20 distinct selected experts, each5222400B, and104448000B
(99.61MiB) TOTAL arena payload across BOTH children. Each child budget is its exact assigned payload
rounded up only to MiB. Reference payload staging is one5222400B blob, released before children;
row scratch is204800B plus bounded native packing/hidden buffers and capture/manifest storage.
The existing guards each cap parent+one-child RSS at1GiB, so combined owned RSS is conservatively
bounded by2GiB. They require MemAvailable>=8GiB, owned VmSwap0, finite-cgroup headroom and unchanged
host/cgroup OOM counters, with a180s work deadline. Each child log is exclusive0600 and capped1MiB;
the output directory is new/private. Cleanup signals/reaps only positive retained child PIDs, including
failure during second-child startup. No full or half-complement arena is loaded or replaced.

On the CPU fixture host, retain the existing cache/compiler/native options. Build only the new
focused test and existing diagnostic target; no CUDA/native-OFF/product rebuild is requested:

```sh
cmake -S "$SRC" -B "$CPU_BUILD"
cmake --build "$CPU_BUILD" --target cpu_capture_micro_test cpu_group_numeric_diagnostic --parallel 4
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^cpu_capture_micro_test$'
```

The small Q8 regression checks original ownership despite skipped GPU/unselected IDs, shape/ownership
and single-active-peer refusals, digest/full-reference failure before children, positive-owned cleanup
when the second executable exits before listening, and successful peer/group bits. Its success case
delays second setup11s, beyond the first peer's10s idle deadline. Fixture payload files are larger
than tiny stdout limits: bound test output through the retained drain/timeout guard rather than a
small inherited file-size limit. All executable checks are **NOT RUN on Geekom/Valinor**.

After that test passes, parent selects runtime on the CPU fixture host, preserving the separate
full resident host. Bind these existing inputs; do not copy a full model:

| Binding | Required input |
| --- | --- |
| `DIAGNOSTIC` | New CPU `cpu_group_numeric_diagnostic`, exact candidate SHA and mechanically verified executable hash |
| `WORKER_A`, `WORKER_B` | Retained compatible original peer0/peer1 worker ELFs, already used by the earlier same-host diagnostic; verify their complete hashes (reported prefixes `e9f25fa5` / `3b41fa8b`), not newly built fixture workers |
| `CAPTURE` | Existing37544692B capture, SHA256 `218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208`; transfer only this file if absent |
| `MANIFEST` | Original trusted local manifest, original full hash, at most8MiB; identity/index must match the capture |
| `LOCAL_ROOT` | Existing valid local Q8 assets on the fixture host; source files are read-only and not staged/copied |
| `NEW_MICRO_OUTPUT` | New absolute directory beneath the existing disk-backed private run root, not tmpfs; parent directory already exists |

The two ELF sizes and actual manifest size come from their retained file receipts, not guesses.
Before weight reads, bounded `MICRO_RANGE` lines list every selected global ID, role, source basename,
exact offset/length and expected logical source-file size (at most60 ranges, total<=104448000B).
If an existing source file is absent/invalid, the mode refuses before workers. Return those precise
missing bindings and ranges; do NOT transfer entire shards or synthesize altered manifests. The
current `LocalSource` requires the original manifest's filenames/sizes; sparse-range staging is not
implemented by this change. Reuse the valid assets from the earlier subset run where available.

Run once under the retained external owned-process/output guard, after live fixture-host headroom,
cgroup, swap and OOM checks. Do not infer safety from MemAvailable alone. No port/PID of the original
full resident is accepted by this CLI:

```sh
timeout --signal=TERM --kill-after=10s 240s "$DIAGNOSTIC" \
  --worker-a "$WORKER_A" --worker-b "$WORKER_B" \
  --capture-record0 "$CAPTURE" "$MANIFEST" "$LOCAL_ROOT" "$NEW_MICRO_OUTPUT"
```

Require the independent-full/capture bit gate with `workers_started=0`, then both actual-peer bit
gates and actual-group/manual bit gate. Final `CPU_CAPTURE_MICRO` must report ok/reference/peer0/
peer1/group/owned_reaped all1, payload<=104448000, GEN0 and full_endpoint_connections0. Each child
must have one arena allocation, exactly its selected loaded/verified bytes, no upload/failure/unload/
expiry and two INFER requests (manual then group). Return source/binary/manifest/capture pins,
selected IDs/ranges and original/micro slot receipts, all gates, descriptive regrouping error,
resource/OOM and owned-child cleanup receipts. `peak_owned_rss_bound` sums two independently sampled
parent+child maxima; it is a conservative bound, not a simultaneous RSS measurement. Preserve every
failure and do not repeat automatically. No inference performance claim is made.

Interpret failures in order: reference mismatch stops before workers and questions input/weight/kernel
alignment; a peer mismatch witnesses selected-subset loading/execution; matching peers with a group
mismatch witnesses the group path. An all-pass result covers only these selected real inputs/weights
and local two-peer integration. `MICRO_SLOT` exposes the original split rank and the smaller fixture
rank: full-size compacted slot ranks, the remaining layers, remote state/network behavior, live
two-peer engine integration and closed-loop amplification remain untested. Frozen screens and the
original failed split verdict stay unchanged. The withdrawn full/half duplicate-arena tests stay withdrawn.

### Completed G1 live-engine request — supplied PASS; historical commands

The parent supplies a preliminary PASS for the `b224ef87f73d5da9724cb1f8f2a87859fc4adfbd`
single-member saved replay in5s; its full receipt is pending. Do not repeat it. The next discriminator
uses the retained original engine, one full resident worker and the original prompt0 GEN256 line.
This exercises live engine/group interaction without two-peer state or another arena. The original
split-model numerical/performance verdict remains failed.

This additive change touches only the existing Python harness, focused regressions and docs.
Top-level `cpu_route` defaults to `"legacy"`; explicit `"group"` requires exactly one
`--remote-cpu-worker` in base argv, bound to the same worker/control endpoint. The opposite flag,
duplicates, invalid route values or endpoint/manifest mismatches are refused before staging or
owner startup. The selected route survives `arm_config`, including existing single-peer rewriting.
The omitted-field legacy argv and existing two-peer sequencing remain unchanged. No new sequencer,
worker guard, engine feature or C++ build is introduced.

Hermes runs these focused Python tests once on its controller, with exact candidate source and
the retained Python. No model, GPU, native binary or package installation is needed:

```sh
cd "$SRC/tools/validation-harness/tests"
timeout --signal=TERM --kill-after=5s 120s python3 -B -m unittest -v \
  test_harness.HarnessTests.test_cpu_route_legacy_default_preserves_argv \
  test_harness.HarnessTests.test_cpu_route_single_group_survives_staging_and_arm_selection \
  test_harness.HarnessTests.test_cpu_route_invalid_conflicting_and_duplicate_flags_refused \
  test_harness.HarnessTests.test_cpu_route_exact_endpoint_and_manifest_bindings \
  test_peers.PeerTests.test_explicit_group_route_survives_single_peer_argv_rewrite \
  test_peers.PeerTests.test_membership_config_keeps_exact_total_and_legacy_primary_path \
  test_harness.HarnessTests.test_worker_controls_never_queue_behind_engine_session
```

These tests are authored, **NOT RUN on Geekom/Valinor**. After they pass, copy the existing private
warm-attach config and change only the following fields. Use the new source pin for harness helpers,
but keep the retained `9ed478e` engine and full worker binaries; verify their complete recorded
hashes rather than relying on the supplied prefixes `66750645` and `e9f25fa5`.

| Config field | One-request value/delta |
| --- | --- |
| `run_root` | New private output root; never overwrite an earlier bundle |
| `cpu_route` | `"group"` |
| engine `argv` | Replace the one `--remote-cpu-experts` token with `--remote-cpu-worker`; keep its existing numeric loopback endpoint and all other arguments |
| `attach_root`, `attached_worker_idle` | Current recovery receipt root, whose `WORKER/identities.json` must match the live guardian/worker/monitor; `true` after exclusive idle verification |
| `final_worker`, `failure_worker` | Both `"retain"` |
| `worker_bytes` | `116135731200` |
| `assets.engine`, `assets.worker`, manifest/model/profile/request assets | Retained original bindings; no binary replacement or asset regeneration |
| `assets.reference` | Existing retained A0 summary JSON with the original request1 token array and worker-entry count |
| `arms` | Exactly one copied warm arm, named `G1`, `restart:false`, `requests:[0]`, `reference:"reference"` |
| arm workers/loaders/masks and engine/worker policy | Keep12/4, caller0/pool1..12 and all retained engine/GPU/environment settings |
| arm `placement_markers` | Keep960 primary/1378 helper/22238 CPU markers; replace any legacy CPU-ready marker with `fixed CPU group ready peers=1`; require `cache_hit=1 reused=116135731200 B` and `setup network=0 B local=0 B hits=0 misses=0` |
| `capture_first_logits` | `true`; remove global `STRATA_DUMP_FIRST_LOGITS` so the driver supplies a unique prefix |
| `env.STRATA_CPU_CAPTURE` | Remove; no new host capture |
| `worker_peers`, arm `cpu_peers` / `peer_bytes` | Omit; this is one attached full resident, no remote controls |

Keep `--remote-cpu-local-manifest`, `--remote-cpu-resident`, native assets, primary/helper GPU placement,
prompt cache, greedy/speculative settings, budgets and all deadlines unchanged. Do not retain both
CPU endpoint flags. `requests:[0]` sends the exact first GEN256 line from the retained request asset,
then waits for its complete DONE and QUIT/reaping. Its A0 output is213 tokens;256 is a cap, not an
expected token count. The existing reference gate compares the complete token array and entries.

Before launch, Hermes must verify the current original guardian/worker/monitor identities, owned
listener, complete idle forever-resident116135731200B arena, VmSwap0 and healthy unchanged resource
guard. Preserve the fixed prelaunch swap ceiling, MemAvailable>=8GiB and unchanged OOM counters.
The retained engine has normal resident negotiation, **not a reuse-only switch**: an empty arena
could load. Do not launch if the live resident preflight fails. A mismatched complete cache is
refused by the worker without replacing it. Same placement/manifest/geometry and one group member
produce the same full partition (`ordinal % 1`), so require an observed cache hit/reused116135731200B
and zero new local-load/upload bytes at READY before GEN. Do not issue worker STATUS/STOP/UNLOAD
while the engine owner is alive; inspect the existing READY log and retained session receipts.
The configured placement markers enforce those reuse/zero-load observations before the existing
driver sends GEN; a missing marker fails the arm without changing its sequence.

Use the unchanged staging/run entry points after the focused tests and parent dispatch:

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$PRIVATE_CONFIG"
python3 -B "$RUN_ROOT/run.py" --run-root "$RUN_ROOT"
```

The first operation mechanically stages/hashes the new helpers and unchanged assets once; it starts
no owner. After complete engine-owner reaping, require the original resident still complete, idle and
forever, with resident/loaded/verified/peak bytes and allocation/unload/expiry counters unchanged,
zero new local-load/upload bytes, no failures and VmSwap0. Preserve stdout, first-logit file and
failure receipts even if the token reference gate fails. No automatic repeat or restore load.

Compare the new `G1-first-logits.0` to the retained A0 prompt0 raw-F32 file with the existing bounded
`read()`/`compare()` functions (the comparator CLI expects three pairs). This invokes one pair without
adding a script or changing any numerical threshold. The retained row is993280B of raw F32, with no
count header:

```sh
PYTHONPATH="$SRC/tools/validation-harness" python3 -B -c '
import json, pathlib, sys
import compare_logits as c
a, b = map(pathlib.Path, sys.argv[1:])
metrics = c.compare(c.read(a), c.read(b))
metrics["bitwise_equal"] = a.read_bytes() == b.read_bytes()
print(json.dumps(metrics, indent=2, allow_nan=False))
raise SystemExit(0 if metrics["bitwise_equal"] and metrics["numeric_pass"] else 1)
' "$A0_PROMPT0_FIRST_RAW_F32" "$RUN_ROOT/G1-first-logits.0"
```

Return one flat receipt: source/helper/engine/worker/config pins, exact sent line, DONE213 and complete
token equality, raw-F32 length/finite/bit comparison plus unchanged error metrics, placement/affinity,
observed setup cache hit, zero new load/upload bytes, final owned cache/resource accounting and
cleanup. This is one diagnostic request, not a performance campaign. A failure is preserved; a pass
does not clear two-peer/subset/remote behavior or the original failed split comparison.

### Completed single-member saved replay — preliminary supplied PASS; historical commands

The parent reports normal full residency restored with the retained worker binary, four loaders and
twelve inference workers. That is a supplied snapshot, not a live check by this coding thread.
Hermes must reverify the owned receipt, listener identity, complete forever-resident116135731200B
arena, exclusive idle session, current swap and original monitor health immediately before replay.
No worker replacement, second worker, additional arena, engine, GEN or settings change is requested.
Duplicate-arena procedures remain withdrawn.

The new `--single-group` option first runs the existing capture/request/logit validation and direct
full/masked replay unchanged. All full replies must bitmatch before anything masked; the existing
detail gates must pass. It then closes that session and verifies cache accounting before opening an
actual **one-member `ExpertWorkerGroup`** with the identical full partition and trusted manifest.
`resident_reuse_only=true` refuses empty/mismatched residency; fallback/upload is disabled. Every
record, including zero-work records, goes through actual group post/finish with the original
activations, IDs, weights, masks, T/K and layer. Fresh session/sequence identity is supplied by the
existing tier. Every result must be finite and bitwise identical to the saved full reply. No numerical
tolerance or production arithmetic changes are included.

The group drains/joins its existing tier, disconnects without STOP/UNLOAD and verifies complete idle
forever residency again. Resident/loaded/verified/peak bytes, allocation, unload and expiry counters
must equal the original snapshot. Exactly one additional cache hit and one arena's reused bytes are
expected for this group session; local reads/uploads must stay zero. Scratch is one bounded request,
reply, ID conversion and group row at a time, reusing the existing capture in memory. No masked
reference arrays or child processes are added by this mode. The direct phase and group phase each
have a180s deadline, with5s cache-control and10s group-frame limits.

On the CPU fixture host, use the existing build cache/options at the exact new source pin. Reconfigure
only to discover the new test target; do not change compiler or native options:

```sh
cmake -S "$SRC" -B "$CPU_BUILD"
cmake --build "$CPU_BUILD" --target cpu_capture_test cpu_capture_group_test cpu_capture_replay --parallel 4
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(cpu_capture_test|cpu_capture_group_test)$'
```

`cpu_capture_group_test` owns one small real Q8 fixture worker, with one arena allocation. It checks
T2/T1 and K10, mixed ownership/weights, zero work followed by active work, full-validation failure
preventing group setup, empty/stale-cache refusal, group bit-mismatch cleanup preserving the primary
error, valid reconnect after failure and final owned-child reaping. It reuses existing fixture/lifecycle
helpers and needs no model or external endpoint. The unchanged `cpu_capture_test` covers corrupt,
truncated, oversize and malformed capture refusals. Avoid a tiny inherited file-size cap that would
kill fixture creation; bound stdout/stderr using the retained validation policy. **These checks are
authored, NOT RUN on Geekom/Valinor.** No CUDA/engine rebuild is needed for this diagnostic-only change.

After those focused gates pass, mechanically stage/hash only the replay executable using the retained
transfer procedure. Preserve all worker/engine binaries and private assets. The six positional input
bindings stay exactly those of the completed replay: absolute saved capture, trusted manifest, request
asset, retained A0 raw-F32 first-logit file, numeric loopback address and current owned full-worker port.
Verify the37544692B capture SHA256
`218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208`.
No config template, weight staging or subset flag is needed. Parent dispatches one invocation after
live checks under the existing external guard and unchanged resident monitor:

```sh
timeout --signal=TERM --kill-after=10s 420s "$REPLAY" \
  "$CAPTURE" "$MANIFEST" "$REQUESTS" "$RETAINED_A0_FIRST_RAW_F32" 127.0.0.1 "$FULL_WORKER_PORT" \
  --single-group
```

The external guard targets replay only. Measure original/replay VmSwap before launch; both must be
zero for this restored-resident run and remain zero. Keep the original prelaunch ceiling fixed, never
ratchet after decreases. Require MemAvailable>=8GiB, unchanged host/cgroup OOM counters and bounded
replay RSS/output using the retained guard. On failure preserve receipts, reconcile the replay PID,
verify the original worker/guardian/monitor and cache, and do not retry or send STOP/UNLOAD.

Return exit status, source/replay/worker/config/capture hashes, the two summary lines and final owned
residency/resource receipt. Require `CPU_CAPTURE_REPLAY ok=1 resident_unchanged=1 full=960` with its
existing gates true, then `CPU_SINGLE_GROUP_REPLAY ok=1 resident_unchanged=1 records=960 cache_hit=1`;
active+zero must equal960. The first line's `grouped` counter is the older masked arithmetic, while
the second line is actual one-member group execution. If the first phase fails, no group is opened.

A pass isolates the real-input group post/finish wrapper from direct replay. It does not test two-peer
fan-out, actual subset slot maps, remote state, the live engine/helper-GPU merge or closed-loop error
amplification. It cannot clear the original failed screens or rule out accumulated rounding across
layers. No second-endpoint alias to the single-session resident is proposed.

### Prior duplicate-arena attempt — historical evidence, do not repeat

The supplied run at `6351cde18db499b0c6a28fcad01e6228098914aa` ended with a **resource stop, not
a numerical failure**. Its CPU build and2/2 CTests passed after an external RLIMIT_FSIZE wrapper
was corrected to permit synthetic fixture payloads. Full replay again passed960 replies. The first
temporary half completed58067865600B/11119blobs in66.05s; the second was incomplete when the external
guard observed422148KiB of swap in the original full worker and terminated replay at176s. Both
temporary children were reaped. No actual-subset/group numerical outcome was produced. The original
cache stayed logically complete with unchanged identity; this does not mean physical VmSwap0.

The parent supplied a completed PASS at `7c0dfce4f88856856c6d33a281824b1aae7dff1b`, whose direct
parent is `80119730422bfe7071c97420af8f5866344162c9`. CPU checks2/2 and CUDA/native-OFF links passed.
One original prompt produced one token/DONE, warm reuse of116135731200B and first logits bit-identical
to retained A0. The saved capture is37544692B, SHA256
`218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208`.
Replay returned0 in3s: windows20, records/full/grouped960, maxabs9.53674316e-7, maxrel0.333,
relativeL2 6.112e-8, entry_bits1, peer_bits1, detail_envelope1. First regrouping bit difference was
record0, prefill position0/T2/slice0/layer0/K10; the largest was record47/layer47. The complete full
arena remained unchanged: one allocation, zero failures/unloads/expirations. These are supplied
Hermes results; neither execution nor private receipt inspection occurred in this coding thread.

Those results hold baseline activations/IDs/weights fixed. They do not establish closed-loop error
propagation, subset slot-map correctness, real concurrent group behavior or remote peer state.
The failed original model screen remains failed. **Do not recapture, launch an engine or rerun GEN.**

The withdrawn sequential implementation kept one temporary half at a time, with58067865600B peak
temporary arena and60GiB additional owned-RSS budget. Even that reduced footprint did not preserve
the original worker's swap policy. Its focused fixture was authored; no desktop tests were run.
Neither duplicate-arena attempt produced an actual-subset numerical verdict. The capture format,
protocol, arithmetic and original acceptance screens remain unchanged.

### Completed real-input capture and resident replay (historical commands; do not repeat)

The parent supplies the following completed evidence, not execution by this coding thread. At
8011973 the seven-case K10/T1 diagnostic passed all 140 metric lines, including both retained
worker binaries, full/subset masks, native row/batch references and actual group/manual merging.
Only regrouping versus unsplit differed: maximum absolute errors were 1.53e-5 (dense/shifted),
3.81e-6 (sparse), and 4.58e-5 (cancellation), within the unchanged published envelope. This does
**not** clear the failed model comparison. Its synthetic activations/weights and T1 shape miss real
values and multi-token prompt batches.

Retained alignment also found matching three GEN lines, prompt tokens/hashes, GEN256 lengths,
engine binary, model/manifest/profile assets, no resume/reuse, and correct first-logit ordinals.
The first-logit files are **raw F32 with no count header**. No unintended configuration difference
was found. No intermediate CPU inputs or partials were retained. The next discriminator captures
those actual host inputs, without CUDA graph nodes, callbacks, events, copies or synchronization
changes. Production inference arithmetic and SEW1 remain unchanged.

Bind `CANDIDATE` to the exact commit containing this section. Use retained build caches, unchanged
CMake options, and no model operation until the focused gates pass. Build-target versus CTest names:

```sh
# CPU fixture host: reconfigure only to discover the new targets, preserving all cached options.
cmake -S "$SRC" -B "$CPU_BUILD"
cmake --build "$CPU_BUILD" --target cpu_capture_test cpu_capture_replay expert_tier_test --parallel 4
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(cpu_capture_test|expert_tier_test)$'

# Engine host: outside inference, retained CUDA/native-ON cache, existing ccache and parallelism.
cmake -S "$SRC" -B "$CUDA_BUILD"
cmake --build "$CUDA_BUILD" --target strata --parallel 8

# Incremental native-OFF link check on its existing validation host/cache.
cmake --build "$NATIVE_OFF_BUILD" --target strata --parallel 2
```

`cpu_capture_test` has no socket or model. It checks multi-token format round trips, fixed capacity,
all-full-before-masked ordering, early mismatch refusal, zero-work, complete layer/token coverage,
exclusive publication, missing slices/layers, corrupt/truncated/oversize input, non-finite inputs,
wrong ownership and lifecycle/bounds refusals. These new checks are **authored, NOT RUN locally**.
Do not rebuild or replace the retained worker binary for this diagnostic. Mechanically hash/stage
the replay executable to the engine host through the existing approved transfer operation; preserve
the old binaries. The tool links only CPU libraries and does not need model files copied.

Use the existing merged config driver and its retained local warm configuration. Change only:

| Private config binding | Required value |
| --- | --- |
| engine asset | newly built candidate engine, mechanically staged/hash verified |
| worker asset/argv, manifest and model assets | exact existing full resident worker and validated assets |
| `attach_root`, `attached_worker_idle` | existing owned receipt root; true only after exclusive idle use established |
| arms | one local warm entry, `restart:false`, `requests:[0]`, no `reference` token-array gate |
| workers / loader_threads / masks / engine policy | retained 12 / 4 and unchanged measured policies |
| CPU route | original single `--remote-cpu-experts` endpoint; no group/remote GPU endpoint |
| peer configuration | remove top-level `worker_peers` and arm `cpu_peers` / `peer_bytes`; no remote control bindings |
| `capture_first_logits` | true (driver supplies its unique first-logit prefix) |
| `env.STRATA_CPU_CAPTURE` | new absolute private filename, absent along with its `.partial` sibling |
| final_worker / failure_worker | retain / retain |

Keep the original retained request asset: its first **plain GEN256** line is sent unchanged, with
no keyed GEN options, resume, session restore or warmup. Capture includes every prompt window and
the first decode window. After that first logit row, diagnostic mode emits only its first sampled
token and normal DONE; it then accepts QUIT. This is deliberately not a complete token-array or
performance run. The original requested length is retained in the capture metadata. A second GEN
is refused. No harness code or per-experiment sequencing script is needed.

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$CAPTURE_CONFIG"
python3 -B "$CAPTURE_RUN_ROOT/run.py" --run-root "$CAPTURE_RUN_ROOT"
```

Before starting, establish the existing resident is exclusively idle and complete with **forever**
residency. Both capture and replay refuse a finite TTL (avoids expiry between status and setup),
an empty cache or a cache miss. The server's existing retained-key mismatch refusal preserves the
old arena. No LOAD/UPLOAD fallback is available in these diagnostic paths. Capture allocates one
96MiB buffer plus bounded codec scratch before callbacks, writes no files in callbacks, and
publishes a private mode0600 file exclusively after the existing first-logit host copy. Its integrity
digest detects accidental corruption, not malicious edits. Never publish raw activation files.

Wait for the engine broker to finish QUIT, reap the engine/monitor, and check the existing owned
idle receipt **before** any worker control/replay connection. Never probe STATUS during an engine
session. Keep worker/pool affinity, binary and arena unchanged. Under the retained bounded output
guard, invoke exactly once (outer timeout200s, tool budget180s plus bounded frame deadlines):

```sh
timeout --signal=TERM --kill-after=10s 200s "$REPLAY" \
  "$CAPTURE" "$MANIFEST" "$REQUESTS" "$RETAINED_A0_FIRST_RAW_F32" "$WORKER_NUMERIC_HOST" "$WORKER_PORT"
```

All file bindings must be absolute. The tool validates the entire bounded format, exact original
GEN line/token binding, trusted manifest/index identity, and captured first-logit digest against
the retained A0 raw file **before contacting the worker**. A logit mismatch stops interpretation.
It then attaches using a fresh resident cache-hit session, rewrites only session/sequence IDs,
and first replays **every full-mask reply bitwise**. Any mismatch stops before masked work; report
its record index, do not treat subsequent grouping as valid. Only after that full pass does it replay
the two deterministic alternating masks on this same full arena and add active peers in configuration
order. Zero-work peers send nothing. This isolates batching/masking/reduction on actual values,
including T>1, without different slot maps, network peers or a second copy of the model.

Output is bounded summary metrics: maximum absolute/relative error (relative denominator floor1e-12),
relative L2 (norm floor1e-30), first bit-differing record and its phase/position/T/slice/layer/K.
If a numerical discrepancy exists, one record with the largest absolute error is decomposed into
at most T*K<=80 individual contributions, with two masked reconfirmations. Report entry-order and
per-peer bit identity and the unchanged `1e-4 + 4e-6*sum_abs_contributions` envelope for that record;
this is not an envelope gate over un-decomposed records. No threshold is tuned after results.
Non-finite/shape errors, full replay mismatch, decomposition mismatch or failed resident checks
fail the tool. Ordinary regrouping bit differences remain visible; a passing tool never clears
the original failed model screen or proves network/subset slot identity.

The tool disconnects normally and reconnects only after that session closes, checking the arena
remains complete/idle with unchanged bytes, loaded/verified counts, allocations, expiry and unloads.
It never sends STOP, UNLOAD, UPLOAD or LOCAL_LOAD. Preserve any failure and receipts; do not retry
uncertain operations or reload the model. Return the source/binary pins, focused test result,
capture size/hash, one first-logit comparison, replay summary/first differing record, and final
resident receipt. No complete A/B/A rerun, worker replacement or model reload is requested.

### Completed synthetic discriminator (historical commands; do not repeat)

The parent reports identical ggml source bytes at the pinned revision, matching effective compiler
flags, and a fixed retained K2/T8 fixture passing both retained worker ELFs bitwise (20,480 floats).
Those results narrow the search but do not establish equality at every shape or explain the model
failure. This follow-up changes only tests, the fixture generator and the manual build target.
Production routing, arithmetic, protocol and all numerical thresholds are unchanged.

Use the existing CPU/native-ON build on the designated fixture host. Bind the full new diagnostic
source SHA as CANDIDATE, and verify clean source as below. WORKER_A and WORKER_B are existing,
compatible, preserved worker ELFs from the failed comparison. Keep their exact hashes and the same
fixed reference build/runtime environment in the private receipt. Do not rebuild or replace those
workers. No GPU or model assets are needed.

~~~sh
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
cmake --build "$CPU_BUILD" --parallel 2 --target cpu_group_numeric_diagnostic worker_local_test
~~~

There is deliberately **no CTest entry** for the manual diagnostic: its two ELF inputs must be
explicit. Under the existing owned process-tree/RAM/output guard, run these once, in order, with a
180-second wall cap and 2 GiB RAM cap per invocation (no automatic repeat):

~~~sh
OMP_NUM_THREADS=2 "$CPU_BUILD/worker_local_test" --binary "$WORKER_A" --load-threads 4
OMP_NUM_THREADS=2 "$CPU_BUILD/cpu_group_numeric_diagnostic" --worker-a "$WORKER_A" --worker-b "$WORKER_B"
~~~

The first invocation checks the shared generator's unchanged default two-expert fixture after its
bounded extension; it does not repeat both earlier cross-binary checks. The new diagnostic constructs
32 Q8_0 experts per layer at H2560/FF640 across two layers, with only20 CPU experts per layer. Every
T1 request has10 **distinct** IDs. Full synthetic CPU arenas are208,896,000 B per worker; each split
subset is104,448,000 B. The generated fixture payload is334,233,600 B, not a model-sized asset.
Each child has a256MiB arena budget,12 inference workers and4 loader threads; only fresh loopback
port0 children can be controlled. Output is a fixed number of metric lines; child logs cap at1MiB
each. Successful completion prints child reaping and zero real-model loads. Failure cleanup signals
only the children it forked, with bounded TERM then KILL/reap. Leave the real resident worker alone.
Do not set STRATA_WORKER_FIXTURE_KEEP; normal exit removes the synthetic files.

Seven fixed cases cover dense/shifted routing, sparse CPU/GPU ownership, work on each peer alone,
zero CPU work, and arithmetic cancellation. IDs0–9 have distinct seeded weights; IDs10–31 contain
adjacent identical expert pairs so the signed cancellation case has a known small residual.
Signed weights are a deliberate numerical stress case, not a router-policy change or normalization.
The diagnostic also checks copied queued inputs, inactive-peer INFER counts, cancellation before
connection, and valid reuse afterward. It does not claim to test in-flight native-kernel cancellation.

The comparisons run in this order:

| Stage label | What a failure distinguishes |
| --- | --- |
| common-pool-vs-independent-rows | Native twelve-worker row execution differs from independent single-entry native calls |
| peerN-batch-vs-common-rows | Changing the native batch membership changes a routed row, before weighting/reduction |
| binaryA/B-full, binaryA/B-maskedN, binaryA-vs-binaryB-full | Real retained worker arithmetic/masking differs at K10/T1 or from the common reference |
| subsetA/B-vs-common-batch-mask | Actual subset ownership/setup/slot mapping differs from a full-arena masked batch |
| group-vs-manual-peers | The production group differs from independent concurrent tier posts and configuration-order addition |
| group-vs-grouped-common-reference | The combined real group differs from the same grouping over independent native rows |
| reference-grouping-vs-unsplit, group-vs-unsplit, group-vs-actual-fullA | Quantifies reference/actual regrouping against both independent and actual full-worker baselines |

All row, masked-reply and same-grouped-reference gates are **bitwise**. Only grouping versus unsplit
uses the already-published 1e-4 + 4e-6*sum_abs_contributions envelope. Metrics report maximum
absolute/relative error, relative L2 (norm floor1e-30), RMSE, first differing index and hexadecimal
float values. Relative errors can be large for the deliberately near-zero cancellation result;
the fixed absolute/scale gate remains decisive. Non-finite results fail. A bits=0 result at an
unsplit comparison is reported even when its fixed envelope passes; do not hide it as bit identity.

Return the first differing stage and all fixed-case metrics, exit status, duration/peak memory,
exact diagnostic/worker hashes, and child/file cleanup. An all-pass result narrows the failure to
inputs or behavior absent from this fixture; it does **not** clear the model's failed logit screen.
Do not tune thresholds, rerun models, add seeds or start another campaign. Broader tracing needs a
separate decision after these results. **This diagnostic is authored and NOT RUN on the desktop.**

### Historical original validation procedure — do not repeat

Parent/Hermes owns execution and source/binary pins. Keep the healthy existing full worker on02
resident until all build/fixture gates pass. Do not probe, unload, stop or replace it for preflight.
Use retained caches/assets and existing build directories. Do not reconfigure policies or install
anything. On each selected checkout establish clean exact source first:

```sh
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
```

On the existing Hermes Python controller, run the synthetic suite once under its retained bounded
output guard. It needs Linux /proc and ordinary loopback sockets, no model, GPU or native binary:

```sh
cd "$SRC/tools/validation-harness/tests"
timeout --signal=TERM --kill-after=10s 240s python3 -B -m unittest -v test_harness test_peers test_numerics
```

The new peer tests exercise real tiny children: remote-role staging/pre-START binding, alternate
loopback readiness, fast STOP reconciliation, explicit abort/reaping, lost-channel uncertainty,
bounded transport stderr, and config-only local/split selection. Existing affinity/owner tests stay.
No local synthetic test demonstrates real network throughput or full-model correctness.

On05 use the existing CPU/native-ON test build; build-target names and CTest names are different:

```sh
cmake --build "$CPU_BUILD" --parallel 4 --target strata-expert-worker cpu_worker_group_test expert_tier_test expert_worker_test worker_local_test worker_service_residency_test
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(cpu_worker_group_test|expert_tier_test|expert_worker_test|worker_local_binary_test|worker_service_residency_test)$'
```

The group fixture covers malformed/overlap/missing ownership, one/two peers, T8/T1/T2/T1, changing
K/masks, zero-work, copied input lifetime, concurrent fan-out, reversed completion, failure draining,
non-finite/overflow refusal, setup keepalive, resident reconnect and explicit geometry/partition
refusal. It compares grouped bits exactly. Its independent unsplit gate is frozen before execution:
`abs(delta) <= 1e-4 + 4e-6 * sum(abs(weighted entry contributions))`, with max absolute/relative,
scale-relative and RMSE printed. This scale covers cancellation; the existing single-tier bitwise
gates are unchanged. Preserve any first failure; do not loosen gates to pass.

On02, outside inference, use its retained CUDA/native-ON cache, ccache and approved bulk parallel8:

```sh
cmake --build "$CUDA_BUILD" --parallel 8 --target strata strata-expert-worker worker_dispatch_test cpu_group_dispatch_test
ctest --test-dir "$CUDA_BUILD" --output-on-failure -R '^(worker_dispatch_test|worker_dispatch_host_zero_test|cpu_group_dispatch_test)$'
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$CUDA_BUILD/strata" --native-experts on
```

The new CUDA fixture uses real group/local services, the real dispatcher and mapped combine at the
production K10 contract (T1 on its two-expert synthetic pack). It asserts no local weight fallback,
zeroed worker rows, exact grouped reference and exactly one mapped contribution. Multi-token
grouping is covered by the CPU fixture, old cache/worker merge gates by the existing CUDA tests.

On03 do the small incremental native-OFF boundary check with its existing build (no model):

```sh
cmake --build "$OFF_BUILD" --parallel 2 --target strata worker_disabled_dispatch_test
ctest --test-dir "$OFF_BUILD" --output-on-failure -R '^worker_disabled_dispatch_test$'
python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_BUILD/strata"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$OFF_BUILD/strata" --native-experts off
```

These commands request no HIP/SYCL build. Their backend build/link validation remains NOT RUN until
the parent selects available hardware; do not claim it from CUDA or CPU results. If a named retained
build lacks the required configure options, report that exact prerequisite rather than silently
substituting a different backend. Stop after the first failed gate and keep receipts/cleanup.

## One local / split / local comparison, only after gates pass

Use candidate engine and matching candidate worker binaries for every arm: this compares ownership,
not different source releases. Retain the old baseline binaries separately. Bind one private config
to the [PEERS.md contract](../../tools/validation-harness/PEERS.md), stage mechanically once per host,
then run that static driver. No per-arm wrapper edits, automatic retries or generation warmups.
The six explicit load/warm entries yield exactly three retained GEN256 requests in each A0/B/A1.
Cold load/READY timing stays separate from inference. Loading cache conditions must be disclosed.

Engine plus RTX3060/3060Ti stay on02. Keep primary960/helper1378 placement, Q8 manifest/assets,
same retained three prompts, worker12 and loader4 on each participating host, turbo and identical
engine/worker masks/policy. Verify post-swap02 usable251.5GiB, 8x32GB,2133MT/s and05 usable94.2GiB
from parent runtime receipts. All22238 CPU experts occupy116135731200B local; B alternates them in
layer/expert order,11119 and58067865600B per peer.05 receives only that subset, never the116GB arena.
Budget each host from actual headroom; do not infer safety from installed RAM alone.

Use private actual numeric LAN bindings and the existing exact source allowlist. No new SSH/control
route, firewall/NIC/access permission or tunnel is assumed. If an authorized command channel from the
engine controller to the existing second-host guardian cannot be bound, stop before START and report
it. Do not invent addresses or change access to make the experiment run.

The current resident full bitmap cannot stand in for a split subset. After prerequisites pass,
parent names the owned worker replacement/handoff; the driver must own the retained guardian handle
before a restart. It refuses arbitrary PID adoption and occupied unowned ports. If current ownership
cannot be transferred, report the concrete handoff blocker before touching residency. Cold loads and
explicit replacement are necessary at A0/B/A1 partition changes; matching warm sessions reconnect
without reload. Final A1 stays resident;05 is stopped/reaped before A1. Keep useful weights until the
next named owned replacement, not an unrelated cleanup.

Report per-peer entries, requests, resident/loaded/verified bytes, allocation/unload/failure counters,
service and exchange timing, endpoint order, and group post-to-finish/caller-wait wall scopes. Peers
overlap, so never sum their elapsed scopes as wall time. Transport and slower peers can dominate;
equal expert counts do not establish balance. Report all nine request times and tokens, per-arm
aggregate and B versus both A0/A1 with drift disclosed. A flat or slower result closes this round;
do not add tuning arms or rerun until selecting a win.

## Numerical and correctness gates fixed before the run

Set `capture_first_logits: true` for every arm. It uses the existing
`STRATA_DUMP_FIRST_LOGITS` diagnostic: raw F32 first next-token row per matched prompt, `.0`/`.1`/`.2`.
This does **not** compare logits at every continuation position. Missing, malformed or non-finite
rows fail the comparison. Run the same source helper twice, A0 versus B and A0 versus A1:

```sh
python3 -B "$SRC/tools/validation-harness/compare_logits.py" \
  --baseline "$ENGINE_RUN_ROOT/A0-first-logits.0" "$ENGINE_RUN_ROOT/A0-first-logits.1" "$ENGINE_RUN_ROOT/A0-first-logits.2" \
  --candidate "$ENGINE_RUN_ROOT/B-first-logits.0" "$ENGINE_RUN_ROOT/B-first-logits.1" "$ENGINE_RUN_ROOT/B-first-logits.2"
python3 -B "$SRC/tools/validation-harness/compare_logits.py" \
  --baseline "$ENGINE_RUN_ROOT/A0-first-logits.0" "$ENGINE_RUN_ROOT/A0-first-logits.1" "$ENGINE_RUN_ROOT/A0-first-logits.2" \
  --candidate "$ENGINE_RUN_ROOT/A1-first-logits.0" "$ENGINE_RUN_ROOT/A1-first-logits.1" "$ENGINE_RUN_ROOT/A1-first-logits.2"
```

Frozen screening limits: max absolute logit error `<=1e-2 + 1e-4*max(abs(baseline logits))`, relative
L2 `<=1e-4`, max absolute softmax probability error `<=1e-3`, KL(A0||other) `<=1e-5`. Max relative
element error uses denominator floor1e-12 and is reported, not used as a cancellation-unstable gate.
These are prospective screening limits, not measured acceptance; no after-result threshold tuning.
Report top1 equality and full256 token arrays separately, including first divergence. Keep exact
baseline outputs. Bind the retained prompt task oracle before execution; record task correctness
separately from token equality and numeric pass. If those prompts have no task oracle, mark task
correctness UNASSESSED and leave full-model acceptance pending rather than inventing one.

Return one flat receipt with exact source/binary/config hashes, host policy/topology, commands,
pass/fail/skip/NOT RUN, first-failure output, cleanup/uncertainty, all nine measurements, numeric
metrics, token comparison and task verdict. No broad provenance framework, further campaign, merge
or deployment is authorized by this handoff.
