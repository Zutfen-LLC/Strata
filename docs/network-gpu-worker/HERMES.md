# Hermes: bounded fixed network CUDA worker gate

Use the exact full candidate SHA supplied in the handoff. Base:
`34a59d83a51bdb3f0299d65306740bd357818715`. Branch:
`inferswarm/fixed-cuda-worker`. This prompt authorizes only bounded synthetic fixtures
on the stated hosts. Stop after reporting; do not merge or run a real-model trial.

**Geekom is editing/static-review only.** Do not configure, compile, run tests or
Python, execute test binaries/benchmarks, or load a model there, including through
subprocesses/reviewers. Do not alter drivers, installations, firewall, routing,
security settings, resource limits or existing services anywhere. Do not use 02.

## 0. Preserve and pin

Roles: 05 is CPU worker compatibility and selective-assets/protocol fixtures;
04 is the single real CUDA network worker on its RTX 3090; 03 is the engine/router
fixture host with its two RTX 3060 GPUs. A fake executor fixture establishes protocol
behavior only; real CUDA and dispatcher checks must pass independently.

Preserve every previous CPU residency/native-OFF/ISA receipt and every running
service, source tree and successful build directory. Use new dedicated clean source
worktrees, separate ON/OFF build directories, and new evidence/cache directories.
Do not reset, rebase, clean, amend or force push. Never overwrite evidence.

On the controller resolve the canonical source from existing receipts. Accept only
`git@github.com:Zutfen-LLC/Strata.git` or its canonical HTTPS equivalent. Require
clean tracked/untracked state and record HEAD, branch, remote, and worktree list.
Read branch and main refs without changing them; verify branch equals CANDIDATE and
base is its ancestor. Main must still be the stated base; if it has drifted, stop and
report the commits for assessment. Fetch only the candidate branch into dedicated
sources, verify FETCH_HEAD exactly, and detach new worktrees at that SHA. Do not
change the old residency checkout.

Prerequisites already installed: git, Bash, timeout, CMake/CTest, the retained C++
compiler, CUDA toolkit matching the existing validated sm_86 setup, `nvidia-smi`,
Python 3 for the existing CLI/ISA checks, nm/objdump, sha256sum, and retained ggml
source at `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Verify source/cache/compiler
provenance against retained receipts. Set `STRATA_GGML_DIR` to that existing checkout
and `FETCHCONTENT_FULLY_DISCONNECTED=ON`. Missing dependencies/provenance are blockers;
do not install, download or silently select a different compiler/source.

For each host set absolute `SRC`, `RUN`, `BUILD`, `GGML_SRC`; require RUN and BUILD
absent and create them separately. On 03 also set fresh `OFFBUILD`. All must be under
the operator's existing validation root, outside previous source/build/evidence paths.
Record `hostname`, UTC times, `git show -s --format=fuller HEAD`, clean status, full
origin, compiler/toolkit versions, GGML SHA, build commands/cache and binary SHA-256s
in `$RUN/identity.txt`. Use the same candidate on all hosts; label each receipt 03/04/05.

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

Configure a new CPU-only native-ON build with retained toolchain options:

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

For actual old/new SEW1 compatibility, use a separately retained, verified exact-base
worker/client source and binary, or build the base in a new directory under this
same serial bound. Configure a separate candidate compatibility build with
`STRATA_WORKER_COMPAT_BASE=$BASESRC` and `STRATA_WORKER_COMPAT_BIN=$BASEBIN`, both
proven from merged main. The CMake checks require both. Build and run
`worker_legacy_client_test` and `worker_current_client_old_worker_test` serially,
180 seconds each; save both directions and exact source/binary identities. Do not
claim old/new compatibility from only candidate-to-candidate tests.

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
