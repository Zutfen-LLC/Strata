# Hermes: bounded fixed network CUDA worker gate

Use the exact full candidate SHA supplied in the handoff. Base:
`34a59d83a51bdb3f0299d65306740bd357818715`. Branch:
`inferswarm/fixed-cuda-worker`. This prompt authorizes only bounded synthetic fixtures
on the stated hosts. Stop after reporting; do not merge or run a real-model trial.

**Geekom is editing/static-review only.** Do not configure, compile, run tests or
Python, execute test binaries/benchmarks, or load a model there, including through
subprocesses/reviewers. Do not alter drivers, installations, firewall, routing,
security settings, resource limits or existing services anywhere. Do not use 02.

## Reported first attempt and continuation

The user reports candidate `57384717c2ba73252436a9510c386bc7b94b1d6f` failed the
first 05 build after 168 seconds at `tests/worker/gpu_worker_test.cpp:105`:
three unqualified codec writes were hidden by the fixture's two-argument `put<T>`.
The adjacent nonce write also encoded four bytes instead of the required eight.
This is a candidate fixture source defect, not a host/toolchain/environment failure.
The additive correction qualifies the writes and asserts exact SGW1 field widths,
offsets and decode expectations. It has not been compiled or executed by Codex.

No tests ran in that attempt; 04, 03 and cross-host stages are **NOT RUN**. The user
reports 18 recipe targets existed, candidate/tree/main and the clean ggml pin were
verified, and 05 guards showed 123.8 GiB available RAM and 49 GiB disk. These are
historical prerequisites, not results for the corrected candidate or a substitute
for fresh resource guards. The shared 05 source remains clean at `466b01d`; the
dedicated failed build is preserved. No owned processes/services were contacted.

Preserve these user-reported evidence locations unchanged:

* Controller: `/home/zutfen/strata-val-57384717-controller/REPORT.md` and
  `05-evidence.tar.gz`, reported SHA-256
  `e0e03d8b15e8a843f606c24e0d3e83ad7b77f37b3ddae2b5ca304c82770516e3`.
* 05: `/home/hermes/strata-val-57384717-05/REPORT.md` and its failed build/logs.

Codex could not access those evidence files locally and has not independently
inspected their contents. Hermes must inspect and cite the actual receipts before
reusing their build/source provenance. Write a new append-only retry evidence
directory. Keep the old failure and all earlier residency/native-OFF/ISA evidence.

Continue in strict order **05 -> 04 -> 03 -> cross-host**, stopping at the first
failure. Every previously unrun test gate still requires actual execution.

## 0. Preserve and pin

Roles: 05 is CPU worker compatibility and selective-assets/protocol fixtures;
04 is the single real CUDA network worker on its RTX 3090; 03 is the engine/router
fixture host with its two RTX 3060 GPUs. A fake executor fixture establishes protocol
behavior only; real CUDA and dispatcher checks must pass independently.

Preserve every previous CPU residency/native-OFF/ISA receipt and every running
service, source tree and successful build directory. Use new dedicated clean source
worktrees, separate ON/OFF build directories, and new evidence/cache directories.
The verified dedicated failed 05 build may be reused incrementally as described below.
Do not reset, rebase, clean, amend or force push. Never overwrite evidence.

On the controller resolve the canonical source from existing receipts. Accept only
`git@github.com:Zutfen-LLC/Strata.git` or its canonical HTTPS equivalent. Require
clean tracked/untracked state and record HEAD, branch, remote, and worktree list.
Read branch and main refs without changing them; verify branch equals CANDIDATE and
base is its ancestor. Main must still be the stated base; if it has drifted, stop and
report the commits for assessment. Fetch only the candidate branch into dedicated
sources and verify FETCH_HEAD exactly. Use new detached worktrees for fresh builds;
the verified failed 05 source may instead advance to that exact candidate after the
reuse checks below. Do not change the shared old residency checkout.

Prerequisites already installed: git, Bash, timeout, CMake/CTest, the retained C++
compiler, CUDA toolkit matching the existing validated sm_86 setup, `nvidia-smi`,
Python 3 for the existing CLI/ISA checks, nm/objdump, sha256sum, and retained ggml
source at `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Verify source/cache/compiler
provenance against retained receipts where they exist. 04 has no historical Strata
receipts: establish and record an honest initial source/toolchain/device/service
baseline there instead of inventing prior validation. Set `STRATA_GGML_DIR` to the
verified pinned checkout and `FETCHCONTENT_FULLY_DISCONNECTED=ON`.

The user already approved one preparation exception: Hermes may transfer the existing,
verified clean ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d` source from a validated
host to a new directory on 04 through existing trusted transport. The user reports
04 GPU/toolkit/transport checks were completed, but this transfer was **NOT PERFORMED**.
Transfer source only, with self-contained Git identity/object availability; include
no credentials, build outputs, model assets or external object-store dependency.
Inventory the export, record the exact commit/tree/clean state and transport archive
hash before transfer, and verify matching hash plus commit/tree/clean state after
transfer on 04 before configuring. Never copy credentials or the source host's Git
configuration into the export. Preserve the donor checkout and use a fresh destination.
Record export/checkout byte counts and require destination disk headroom for both
plus the build reserve. Cap each export/transfer/verification command at 600 seconds
inside the 04 stage's outer budget; preserve any failed transfer's receipt.
Start the 04 stage's 5400-second clock and its new RUN before source export/transfer;
initialize the bounded receipt helper below before this preparation, not afterward.
This exception permits no external downloads, installations or network/security changes.
Missing or unverifiable source/provenance remains a blocker.

For each host set absolute `SRC`, `RUN`, `BUILD`, `GGML_SRC`; require RUN absent and
create it separately. BUILD must be new unless it is the verified dedicated failed
05 build. On 03 also set fresh `OFFBUILD`. Use the operator's existing validation
root. New paths must be outside previous source/build/evidence paths; the verified
failed 05 source and build are the narrow reuse exceptions. Keep the new RUN separate.
Record `hostname`, UTC times, `git show -s --format=fuller HEAD`, clean status, full
origin, compiler/toolkit versions, GGML SHA, build commands/cache and binary SHA-256s
in `$RUN/identity.txt`. Use the same candidate on all hosts; label each receipt 03/04/05.

For incremental reuse on 05, resolve the failed source/build paths from the actual
receipts and `CMAKE_HOME_DIRECTORY`, not guessed paths. Require a clean dedicated
source at the failed candidate (or an already verified corrected candidate), canonical
origin, the recorded compiler/generator/options/ggml pin, and matching cache/input
provenance. Preserve cache and build-manifest copies in the new RUN before updating
the dedicated source to CANDIDATE without reset. Do not touch the shared `466b01d`
checkout. Save `git diff --name-status` and source hashes from the failed candidate
to CANDIDATE; the code correction is confined to the fixture source, with documentation
updates. CMake and production inputs must match unless a later reviewed correction
explicitly changes them.

Run a bounded serial incremental build of all required targets using that unchanged
configuration. Do not discard the build or require a clean rebuild merely because
checkout mtimes changed. Save each target's pre/post binary hash and compile/link
command/input provenance. Confirm dependency tracking recompiles the corrected
fixture for each target using it and the target build succeeds before execution.
Unchanged binaries may be reused only with proven matching complete input/config
provenance; never execute an old failed-candidate test image containing the defective
fixture or a binary from an incomplete/unverified rebuild. If tracking cannot prove
the inputs, stop and report the precise target; use a separately justified targeted
rebuild or fresh build, not a blanket clean. Archive new logs without overwriting old
ones. Reconfigure only if a verified configuration/input change requires it.

## 1. Bounds and command receipts

Run one host stage and one test process at a time. Set
`CMAKE_BUILD_PARALLEL_LEVEL=1`, `CTEST_PARALLEL_LEVEL=1`, `OMP_NUM_THREADS=2`.
Before each build require at least 8 GiB host MemAvailable and 8 GiB remaining under
every finite applicable cgroup memory limit, plus 5 GiB free disk in the new build
area. Before GPU fixtures require at least 2 GiB free VRAM on the selected GPU and
2 GiB available host/cgroup headroom. Read existing usage; do not free someone else's
resources or change cgroup limits. If availability is unknown or insufficient, stop.

Use a 5400-second outer wall budget per host (including builds); no command gets
more than 2700 seconds for configure/build, 180 seconds for a new fixture, 120 seconds
for CLI/native parity, or 600 seconds for the existing ISA checker. Per-frame worker
deadline 5000 ms, total worker setup 60000 ms for fixtures. Never run broad CTest or
benchmarks. RC 124, signal, skip, omitted test, nonzero or missing evidence is a failed
gate, not a pass. A missing CUDA device is a blocker, not an acceptable skipped test.

Use a fail-fast shell and save each exact command, stdout/stderr, return code and wall
seconds, including failed commands. This helper is to run on the designated remote
host, never on Geekom:

```sh
set -euo pipefail
export CMAKE_BUILD_PARALLEL_LEVEL=1 CTEST_PARALLEL_LEVEL=1 OMP_NUM_THREADS=2
DEADLINE=$((SECONDS + 5400))
run() {
  local name=$1 cap=$2 rc start remaining
  shift 2
  remaining=$((DEADLINE - SECONDS))
  (( remaining > 0 )) || return 124
  (( cap <= remaining )) || cap=$remaining
  test ! -e "$RUN/$name.log"
  printf '%q ' "$@" > "$RUN/$name.command"
  printf '\n' >> "$RUN/$name.command"
  start=$SECONDS
  set +e
  timeout --signal=TERM --kill-after=15s "$cap" "$@" > "$RUN/$name.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$RUN/$name.rc"
  printf '%s\n' "$((SECONDS-start))" > "$RUN/$name.seconds"
  (( rc == 0 )) || { cat "$RUN/$name.log"; return "$rc"; }
}
```

Record baseline GPU processes/memory, CPU worker PID/arena/partition/session and
source/services from receipts before any stage, then compare after cleanup. Never
send stop/unload to a pre-existing worker. Capture `nvidia-smi` memory/process samples
before, during and after each owned GPU fixture, and `/proc/PID/status` VmRSS/VmHWM
when the process lasts long enough. The worker's allocated-byte counters do not
replace external VRAM/RSS evidence. Fast sampling does not establish an exact RSS peak.

## 2. 05: CPU compatibility, asset/cache and protocol fixtures

Use the verified failed CPU-only native-ON build incrementally when its provenance
matches. In that case skip the configure command below, preserve its configuration
and run the build command with new logs. Otherwise use a new build with retained
toolchain options; do not guess missing provenance:

```sh
run configure 2700 cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR="$GGML_SRC" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
run build 2700 cmake --build "$BUILD" -j1 --target strata-expert-worker \
  strata-gpu-manifest gpu_assets_test gpu_worker_test worker_local_test \
  worker_residency_test worker_compat_test expert_tier_test
run assets 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^gpu_assets_test$'
run protocol 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^gpu_worker_test$'
run cpu-local 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^worker_local_(test|binary_test)$'
run cpu-residency 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^worker_residency_test$'
run cpu-compat 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^worker_compat_test$'
run cpu-tier 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^expert_tier_test$'
```

Reconcile the two compatibility baselines explicitly; they are distinct gates:

* The user reports retained historical base
  `5260f2b7c8166021d86f57e4eb80fc44b050e570` has matched provenance on 05. Preserve
  and use that valid evidence for the historical SEW1/local-v1 old/new tests that
  CMake originally describes. If the failed candidate build already points its
  `STRATA_WORKER_COMPAT_BASE` and `STRATA_WORKER_COMPAT_BIN` to this exact baseline,
  keep those settings. Build/run `worker_legacy_client_test` and
  `worker_current_client_old_worker_test` serially, 180 seconds each, and label both
  receipts **historical 5260f2b7 compatibility**. Neither test ran in the failed attempt.
* Merged main `34a59d83a51bdb3f0299d65306740bd357818715` remains the feature base
  and the separately requested immediate-pre-feature binary comparison. The retained
  `5260f2b7` binary is not this main binary. If an exact-main worker/client source and
  binary with verified provenance is unavailable, build the main CPU worker in a
  separate owned baseline directory under the same serial/resource/time limits.
  Use a separate candidate compatibility build with
  `STRATA_WORKER_COMPAT_BASE=$MAIN_BASESRC` and
  `STRATA_WORKER_COMPAT_BIN=$MAIN_BASEBIN`; build/run the same two compatibility
  targets with separate **merged-main compatibility** receipts, 180 seconds each.
  Do not toggle the historical failed build's configuration merely to replace its
  valid baseline. A new main baseline build is for missing comparison evidence,
  not an mtime-driven clean rebuild of the candidate.

For each pair require both CMake settings, exact source/binary/input identities and
successful target builds before running. Preserve prior historical receipts. Do not
silently substitute either baseline, call historical results merged-main results, or
claim either direction from candidate-to-candidate checks. Report any unavailable or
unexecuted comparison as **NOT RUN**; it prevents claiming the complete requested
compatibility gate passed. Stop at that missing prerequisite before later host stages.

Asset acceptance: only selected three-role ranges read; source bytes equal missing
blob payloads; no whole shard/layer reads/downloads. Cold transfer equals assigned
bytes; warm restart is zero source/network/persisted bytes and positive reuse.
Adding an assignment fetches exactly its missing addition; corrupt/truncated cache
is refetched and rejected until complete; partial upload never publishes. Metadata
digest/length/duplicate/out-of-range failures, no-follow path containment, exclusive
cache ownership, atomic publication, disk refusal preserving old assets, and explicit
cleanup all need successful assertions. Retained disk is independently bounded.
Protocol acceptance: explicit CUDA/numerics negotiation; READY-before-load refusal;
delayed READY succeeds within the total setup budget even beyond a short frame
budget, and total setup expiry refuses READY. Capacity refusal; stale plan/session/
sequence, malformed/nonfinite/duplicate replies through both codecs and the actual
client failure latch; simulated GPU/executor loss latches failure with no CPU retry. Label fake
executor results as such. Record assigned/cache/transferred/staging counters.

## 3. 04: actual RTX 3090 network CUDA fixtures

Verify the selected CUDA ordinal is the RTX 3090 on inferswarm04 and record UUID and
free VRAM. Use `CUDA_VISIBLE_DEVICES` only for owned test processes, never globally.
Configure a new native-ON CUDA sm_86 build, same ggml/toolchain/disconnected settings:

```sh
run configure 2700 cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR="$GGML_SRC" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
run build 2700 cmake --build "$BUILD" -j1 --target strata-cuda-expert-worker \
  gpu_worker_cuda_test native_expert_parity
run cuda-network 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^gpu_worker_cuda_test$'
run native-q8-reference 120 "$BUILD/native_expert_parity" --synthetic q8_0/q8_0
```

Require real client -> loopback socket -> service -> grouped CUDA kernels -> bound
reply, all assigned GPU copies completed before READY. Record current/peak weight,
scratch, total GPU allocation bytes/count, configured reserve and free VRAM at fit
check. Force capacity refusal using a fixture executor/free-memory boundary; if the
actual CUDA allocation-failure branch is not exercised, report that limitation.
The actual worker must produce finite outputs, with numerical metrics within the
predeclared bounds: network versus direct same-device grouped CUDA relative L1
`<=1e-6` **and** max absolute error `<=1e-5`; network versus GGML-dequantized float
reference relative L1 `<=3e-2` (the existing native-expert-parity bound). Mixed merge
versus the same per-owner results must meet `1e-6` relative L1 and `1e-5` max absolute.
Record comparison reference, seeds, dimensions, T/K, masks, max absolute error, RMSE
and relative L1. No CPU/GPU bitwise claim. Code inspection or fake execution does
not establish CUDA capacity/READY or numerical correctness.

## 4. 03: native-OFF and mixed engine routing

Use two new build directories; never toggle a successful ON cache OFF. Configure
OFF with CUDA sm_86 and `STRATA_NATIVE_EXPERTS=OFF`, tests ON and the same retained
toolchain/source settings. Build `strata worker_disabled_dispatch_test`, then:

```sh
run off-dispatch 120 ctest --test-dir "$OFFBUILD" -V -j1 --no-tests=error -R '^worker_disabled_dispatch_test$'
run off-cli 120 python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFFBUILD/strata"
run off-isa 600 python3 -B "$SRC/tools/check_isa_guard.py" "$OFFBUILD/libstrata_kernels_cpu.a"
```

All four new GPU options must refuse before resource/model access, including Qwen3.6
family dispatch, while option-shaped values and help/version retain their behavior.
Then configure ON separately as in stage 3 and build exactly
`strata gpu_worker_dispatch_test worker_dispatch_test`. The disabled dispatcher
fixture belongs only to the OFF build.

```sh
run route 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^gpu_worker_dispatch_test$'
run cpu-route 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^worker_dispatch_test$'
run cpu-host-zero 180 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^worker_dispatch_host_zero_test$'
run on-isa 600 ctest --test-dir "$BUILD" -V -j1 --no-tests=error -R '^isa_guard_check$'
run isa-fixtures 120 python3 -B -m unittest -v tools.test_check_isa_guard
```

Run the last command from SRC. Require exact test discovery and non-skip results.
The real dispatcher fixture must show local-GPU/remote-GPU/CPU entry masks disjoint
and exhaustive, CPU requests include only uncovered entries, unchanged full CPU
partition, no local source reads for remote-owned rows, and one merge per owner.
GPU disconnect/execution failure, stale session/plan and malformed/duplicate result
must abort without remote partial accumulation or CPU retry; a new CPU-only session
may recover. Report which cases are protocol-only, mock-executor, or real CUDA.

## 5. Synthetic cross-host transport and final gate

After stages 2–4 pass, use the candidate fixture client's plan-export/peer modes to
generate bounded synthetic metadata on 03, copy **only that small plan** to 04, and
launch a new owned `strata-cuda-expert-worker` on 04 with a fresh cache, 64 MiB disk
cap, explicit 256 MiB runtime reserve, 5000 ms frame and 60000 ms setup budgets.
Record its PID/log and loopback port. Use an existing trusted, owned forward route
from 03 to that port. If no approved route exists, stop and report this stage blocked;
do not change network/security settings. Never connect to existing 05/04 services.
Use `gpu_worker_test --emit-plan FILE` on 03, then
`gpu_worker_test --peer NUMERIC_HOST PORT FILE` against the owned 04 worker.
`--emit-plan FILE --addition` emits the three-expert variant; the default has two.
Build `gpu_worker_test` on 03 if not already present. Preserve exact invocations in
receipts. Do not substitute real model blobs for a missing fixture.

Run the 03 peer fixture cold then warm, each under 120 seconds, with identical plan,
deterministic data and a fresh session. Require cold transferred bytes equal assigned
payload, warm bytes zero with same complete cache, new session identities, successful
CUDA outputs and the declared numerical metrics. Change only the explicit assignment
in a new plan and restart the owned worker against the same retained cache; require
only additions transferred. Record network payload versus logical source read versus
cache persisted/retained bytes separately. Assignment changes happen between sessions.

Stop only the owned fixture worker/tunnels/samplers with their recorded PIDs; TERM,
15-second bounded wait, then KILL only if still that owned process. Preserve logs
and failed/incomplete cache for inspection; do not use pkill, broad rm, or unload
existing services. Confirm owned processes/ports exited, GPU allocations released,
and baseline services/PIDs/05 arena/partition unchanged. Explicit cleanup can remove
only this run's owned disposable fixture directory after archiving its evidence.

Produce `$RUN/REPORT.md` per host plus a controller report linking them, containing
exact candidate/base, tool/source/binary/cache/plan identities; all commands/RCs;
passed/failed/not-run separately; tests and reference metrics; transfer/read/disk/RSS/
GPU allocation/timing counters; negative-case failures; cleanup and service-preservation
proof. Hash the evidence manifest and retain previous evidence separately. Report
blockers with exact action, target, reason and denied/pending/uncertain status.

**STOP at remote validation/review.** Full-model 03+04+05 transfer/load/trial needs
a separate explicitly authorized window after these fixtures pass. Propose its
resource budget and exact fixed plan for approval. Eventual acceptance must show
04 really executes assigned experts and 05 only uncovered requests, with its roughly
103 GiB arena unchanged; transfer/memory/numerical/timing evidence is required.
