# Hermes: fixture-first validation, then bounded CPU residency

**NOT RUNTIME VALIDATED.** The published source has not been built or executed. Earlier tests were
interrupted and one residency test failed. Static-only source corrections require verification on 05.
Do not run builds, tests, binaries or benchmarks on Geekom. Start with section 1 on inferswarm05.

These commands are for Hermes on its controller and existing dedicated inferswarm05/03 checkouts.
The coding task does not connect to either host. Use fail-fast Bash, preserve all logs and the first
failure, and stop dependent work on failure. Do not install packages, change host trust/drivers,
reset unrelated work, or launch broad CI. This recipe supersedes the session-scoped worker cleanup in
[the local-source recipe](../cpu-worker-local/HERMES.md). Normal engine completion leaves the owned
worker and forwards running for the bounded reconnect window. Errors trigger owned cleanup.

## 1. Verify the additive candidate and build incrementally

Set the full `CANDIDATE` SHA from the handoff and `BRANCH=inferswarm/persistent-cpu-weight-residency`.
From the controller verify the remote fork and branch:

```sh
: "${CANDIDATE:?Published full 40-character SHA}"
BRANCH=inferswarm/persistent-cpu-weight-residency
test "${#CANDIDATE}" -eq 40
test "$(git ls-remote git@github.com:Zutfen-LLC/Strata.git "refs/heads/$BRANCH" | cut -f1)" = "$CANDIDATE"
```

On 05 supply its existing absolute `SRC`, configured absolute `BUILD`, and that same SHA/branch.
Preserve the build generator/toolchains and pinned ggml cache from the successful prior receipt:

```sh
cd "$SRC"
test -z "$(git status --porcelain)"
test "$(git remote get-url origin)" = git@github.com:Zutfen-LLC/Strata.git
git fetch origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$CANDIDATE"
git merge-base --is-ancestor 5260f2b7c8166021d86f57e4eb80fc44b050e570 "$CANDIDATE"
git switch --detach "$CANDIDATE"
```

Use conservative serial execution. Require at least 6 GiB current MemAvailable AND cgroup headroom;
record current limits and reserve first. No model assets are needed. Use only the existing CPU dependency
cache; supply absolute `GGML_SRC`, `BASESRC`, `BASEBUILD`, `RUN05`, and `PYTHON05=$(command -v python3)`.
`BASESRC`/`BASEBUILD` are separate dedicated validation paths, never the candidate checkout/build.
Use the existing Makefiles generator and retained CPU/ggml options. No package installation or downloads.

Record every exit code and stop on the first failure:

```sh
set -euo pipefail
: "${SRC:?Existing absolute candidate checkout on 05}"
: "${BUILD:?Existing absolute candidate CPU build on 05}"
: "${GGML_SRC:?Retained pinned ggml source checkout on 05}"
: "${BASESRC:?Dedicated exact-base validation checkout path on 05}"
: "${BASEBUILD:?Separate absolute base CPU build directory on 05}"
: "${RUN05:?Owned existing absolute log directory outside source}"
PYTHON05=$(command -v python3)
export CMAKE_BUILD_PARALLEL_LEVEL=1 CTEST_PARALLEL_LEVEL=1 OMP_NUM_THREADS=2
run_recorded() {
  local label=$1 rc=0
  shift
  printf '%q ' "$@" > "$RUN05/$label.command"
  printf '\n' >> "$RUN05/$label.command"
  "$@" > "$RUN05/$label.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$RUN05/$label.rc"
  test "$rc" -eq 0
}
if test ! -e "$BASESRC"; then
  git -C "$SRC" worktree add --detach "$BASESRC" 5260f2b7c8166021d86f57e4eb80fc44b050e570
fi
test -z "$(git -C "$BASESRC" status --porcelain)"
test "$(git -C "$BASESRC" rev-parse HEAD)" = 5260f2b7c8166021d86f57e4eb80fc44b050e570
# Preserve any existing base generator/options; a new base build uses 05's Makefiles generator.
run_recorded base-configure cmake -S "$BASESRC" -B "$BASEBUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR="$GGML_SRC" \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF
run_recorded base-build timeout --signal=TERM --kill-after=10s 1800 \
  cmake --build "$BASEBUILD" --target strata-expert-worker -j1
run_recorded candidate-configure cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_GGML_DIR="$GGML_SRC" -DSTRATA_WORKER_COMPAT_BASE="$BASESRC" \
  -DSTRATA_WORKER_COMPAT_BIN="$BASEBUILD/strata-expert-worker"
run_recorded candidate-build timeout --signal=TERM --kill-after=10s 1800 \
  cmake --build "$BUILD" --target strata-expert-worker strata-worker-manifest \
  worker_residency_test worker_compat_test worker_legacy_client_test worker_local_test \
  expert_worker_test expert_worker_e2e_test expert_tier_test worker_socket_test -j1
sha256sum "$BUILD/strata-expert-worker" "$BASEBUILD/strata-expert-worker" > "$RUN05/worker-binaries.sha256"
# Rebuild first. Never run a retained pre-fix worker_residency_test binary.
run_recorded residency env STRATA_WORKER_FIXTURE_KEEP=1 \
  timeout --signal=TERM --kill-after=5s 120 bash -c \
  'ulimit -v 4194304; exec ctest --test-dir "$1" -V -j1 -R "^worker_residency_test$"' _ "$BUILD"
run_recorded focused env STRATA_WORKER_FIXTURE_KEEP=1 \
  timeout --signal=TERM --kill-after=5s 900 bash -c \
  'ulimit -v 4194304; exec ctest --test-dir "$1" -V -j1 -R "^(worker_local_test|worker_local_binary_test|expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_socket_test|worker_compat_test|worker_legacy_client_test|worker_current_client_old_worker_test)$"' _ "$BUILD"
run_recorded controls timeout --signal=TERM --kill-after=5s 60 \
  "$PYTHON05" "$SRC/tests/worker/worker_control_test.py" \
  --worker "$BUILD/strata-expert-worker" --control "$SRC/tools/worker_control.py"
```

The 4 GiB virtual-memory cap is a fixture backstop, not a full-model setting. It may itself expose a
platform limit; report that failure rather than raising it to hide a problem. CTest must list all ten
named suites across the two invocations; missing compatibility tests are a failed gate, not a skip.
The two cross-version tests compile the **actual base client source/header** against the candidate
worker, and run the current client against the **actual exact-base worker binary**. Both exercise upload,
local-v1, fresh sessions, native bitwise output and legacy STATUS; current-to-old also requires explicit
resident negotiation failure with zero upload. This is a wire check, not a C++ ABI promise.

The retained failure was `connect failed, cancelled or timed out` after expiry assertions. Static
inspection found repeated log filenames allowing stale-port discovery; the source now creates each log
uniquely before fork. It also found a cleanup path that could signal PID -1 after reaping an unsuccessful
child. All child signals are now guarded by positive unreaped ownership, including after a wait.
The first residency stage must pass `failed child is reaped and cannot be signalled again`, then all
later stages must complete with **zero failures**, a new child log per launch and clean owned process
exits. Preserve the printed synthetic fixture directories and logs on failure; do not rerun for a PASS.
The earlier negative-PID defect is a plausible desktop-disruption cause, not a confirmed diagnosis.

Require cold/warm evidence: complete assigned payload 20,889,600 B; exact reconnect after an 11-second
gap; fresh session/READY and fresh A/B/A native results; stale session/replay rejection; no loaded/hash/
network-weight delta on a hit; one allocation; mismatch/incomplete/corrupt refusal; expiry/unload/stop;
disabled TTL and TTL-without-client-opt-in behavior. The controls script checks real status/unload/stop
and invalid TTL values. New binary processes use two pool workers; default test service pools are also
capped at two. Fixture cleanup cannot signal other tasks by using a negative or group PID.

**Stop and return this fixture/build report now.** Final source remains NOT RUNTIME VALIDATED until
these commands actually pass on 05. Sections 2 onward are a later full-model validation window, gated on
this result and selection of that window. They must not be launched just to complete this code handoff.

### Later gated CUDA fixture check on 03

On 03 preserve its CUDA sm_86 Ninja configuration, toolchain and pinned dependencies:

```sh
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$BUILD" --target strata strata-expert-worker worker_residency_test worker_local_test worker_dispatch_test -j1
ctest --test-dir "$BUILD" --output-on-failure -j1 \
  -R '^(worker_residency_test|worker_local_test|worker_local_binary_test|worker_dispatch_test|worker_dispatch_host_zero_test)$'
STRATA_VERIFY_DEVICE_PLAN=1 timeout 120 "$BUILD/worker_dispatch_test"
```

These checks use synthetic fixtures. Stop if the engine does not compile. Geekom did not validate
CUDA compilation. Keep the existing trusted immutable shard-copy receipts, native index, pack and
manifest from [local-source preparation](../cpu-worker-local/HERMES.md#2-one-bounded-manifest-preparation-then-reuse).
Do not reread/hash 188 GB on each iteration. Verify the saved manifest SHA-256 on both hosts and the
saved index/profile/binary/source SHAs. Names and sizes alone are not identity verification.
Use absolute profile `PROFILE03="$SRC/data/expert-profile.bin"` on 03; record its SHA-256.

## 2. Later full-model window: reserve resources and launch an independently owned worker

Use the existing frozen cache occupancy. For `G` distinct GPU-owned slots the arena is
`(24576-G)*5222400` bytes. Require this plus **16 GiB reserve** within current 05 MemAvailable AND cgroup
headroom. The arena cap is not an RSS cap. Check both 3060 UUIDs/free VRAM on 03 and the existing
primary/dense/KV/context reserve. Record limits, memory/page-cache, storage and GPU baselines. No swap.
Keep every placement/window/speculation/KV flag identical across sessions; even `auto` GPU cache counts
can change under other VRAM use. The worker refuses a changed complete partition until unload/expiry.

Supply the verified absolute `ROOT05`, `MANIFEST05`, `BUILD`, `SRC`, `RUN05`, and `ARENA_MIB` on 05.
`RUN05` is an owned log directory outside the clean checkout. Confirm port 7431 is free. Launch through
the existing controller connection, with stdin and output detached so closing that SSH session cannot
stop the worker. The idle weight window is **30 minutes**, and the owned process has a **2-hour hard cap**:

```sh
: "${ARENA_MIB:?Arena limit with separate 16 GiB reserve}"
: "${ROOT05:?Verified immutable shard directory}"
: "${MANIFEST05:?Trusted absolute manifest path}"
: "${RUN05:?Owned existing absolute log directory}"
nohup timeout --signal=TERM --kill-after=15s 7200 \
  "$BUILD/strata-expert-worker" --port 7431 --workers 0 \
  --memory-budget-mib "$ARENA_MIB" --deadline-ms 10000 --local-setup-ms 600000 \
  --local-root "$ROOT05" --local-manifest "$MANIFEST05" --residency-ms 1800000 \
  < /dev/null > "$RUN05/worker.out" 2> "$RUN05/worker.log" &
printf '%s\n' "$!" > "$RUN05/worker-wrapper.pid"
```

Verify listening with a bounded status probe. Save the child worker PID, `/proc/PID/exe`, start time,
command line and wrapper identity after launch; stop/kill only these owned matching processes. Maintain
the existing controller-owned `-L 127.0.0.1:7431:127.0.0.1:7431 inferswarm05` and
`-R 127.0.0.1:7431:127.0.0.1:7431 inferswarm03` forwards, using existing trust and recorded PIDs.
Launch the forwards detached on the controller so a finished Hermes shell does not tear them down.
Supply its owned absolute `RUNCTL` log directory and confirm the ports are free first:

```sh
nohup timeout --signal=TERM --kill-after=15s 7200 ssh -N -o BatchMode=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:7431:127.0.0.1:7431 inferswarm05 \
  < /dev/null > "$RUNCTL/forward05.log" 2>&1 &
printf '%s\n' "$!" > "$RUNCTL/forward05-wrapper.pid"
nohup timeout --signal=TERM --kill-after=15s 7200 ssh -N -o BatchMode=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:7431:127.0.0.1:7431 inferswarm03 \
  < /dev/null > "$RUNCTL/forward03.log" 2>&1 &
printf '%s\n' "$!" > "$RUNCTL/forward03-wrapper.pid"
```

Record/verify each child SSH process identity as well as its wrapper. A forward failure blocks engine
launch. No direct 03-to-05 access is needed.
Do not attach worker cleanup to a successful engine QUIT. Retain controller ownership of the forwards
for reconnect; use an ERR/INT/TERM cleanup handler for failures and the hard wall as a backstop.
On 05 run `python3 "$SRC/tools/worker_control.py" status --port 7431`; expect empty session/zero bytes.
No `/usr/bin/time` dependency exists. Use shell `time` and `/proc`/cgroup samples if timing/RSS is needed.

## 3. Cold session, engine exit, idle gap, exact reconnect

On 03 supply the verified absolute `SHARD03`, `PACK03`, `MANIFEST03`, `PROFILE03`, `RUN03`, `SRC`, `BUILD`,
and positive budgeted `PRIMARY_SLOTS`. PACK03 is the existing native reference pack with no experts.bin.
Create fixed short A/B/A inputs with the built-in tokenizer. Each request starts fresh with
`--prompt-cache 0`; use identical deterministic sampling settings from the prior trial.

```sh
cd "$SRC"
PROFILE03="$SRC/data/expert-profile.bin"
export SHARD03 RUN03
python3 - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, 'tools')
from strata_tokenizer import Tokenizer
t = Tokenizer.from_gguf(os.environ['SHARD03'])
a, b = (t.encode(s) for s in ('Hello', 'Good morning'))
assert all(1 <= len(ids) <= 64 for ids in (a,b))
gen = lambda ids: 'GEN 128 ' + ','.join(map(str, ids)) + '\n'
r = Path(os.environ['RUN03'])
(r/'cold.in').write_text(gen(a) + 'QUIT\n')
(r/'warm.in').write_text(gen(a) + gen(b) + gen(a) + 'QUIT\n')
PY
run_engine() {
  local tag=$1
  STRATA_ROUTE_TAIL_SKIP=0 STRATA_LOOKAHEAD=0 STRATA_LOOKAHEAD_STATS=0 STRATA_FS_SLOTS=0 \
    timeout --signal=TERM --kill-after=15s 1200 "$BUILD/strata" --serve \
    --native "$SHARD03" --pack "$PACK03" --gpu 0,1 --mmap-experts \
    --remote-cpu-experts 127.0.0.1:7431 --remote-cpu-deadline-ms 10000 \
    --remote-cpu-local-manifest "$MANIFEST03" --remote-cpu-resident --remote-cpu-setup-ms 600000 \
    --expert-cache "$PRIMARY_SLOTS" --expert-cache-per-layer --expert-profile "$PROFILE03" \
    --expert-cache-device1 auto --remote-expert-opt \
    --adapt-every 0 --adapt-swaps 0 --pcie-frac 0 --short-read 64 --prefill 64 \
    --prompt-cache 0 --spec 2 --suffix-draft 0 --lookup-chain 0 --max-context 1024 \
    < "$RUN03/$tag.in" > "$RUN03/$tag.out" 2> "$RUN03/$tag.log"
}
run_engine cold
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv > "$RUN03/after-cold-gpu.csv"
python3 "$SRC/tools/worker_control.py" status --port 7431 > "$RUN03/after-cold-cache.json"
sleep 11
run_engine warm
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv > "$RUN03/after-warm-gpu.csv"
python3 "$SRC/tools/worker_control.py" status --port 7431 > "$RUN03/after-warm-cache.json"
```

The executable takes flags directly: **there is no `generate` subcommand** in this build. The engine
wrapper is allowed to finish; it is not the worker's wrapper. Omit both upload-fallback flags. A local
miss or incompatible cache fails the trial instead of uploading the full CPU partition.

After each QUIT require engine exit RC 0 and verify its owned PID has disappeared from both GPU devices
and VRAM returns to its recorded baseline within normal driver/accounting delay. Check GPU service
participation independently; a resident worker says nothing about GPU release. Between sessions require
`state=empty`, `session=0`, `complete=1`, positive resident bytes and positive TTL. After warm setup,
require `cache_hit=1`, positive reused bytes and a new nonzero session id in engine/worker logs.
The old 10000 ms I/O deadline remains in effect; the eleven-second gap is covered by the idle cache.

Compare the cold/warm JSONs: loaded_bytes and verified_bytes must not increase, network_weight_bytes
must stay zero, cache_hits/reused_bytes must increase, arena_allocations must stay one, peak/resident
bytes must equal the complete assigned payload. Require fresh worker requests and nonzero CPU service,
finite completed outputs and successful worker entries. Compare the generated A token streams from
cold and warm A/B/A, retaining raw output and finish reasons. Report any shorter EOS or difference;
do not claim full-model parity from those examples. Preserve the prior inference correctness checks.

If a partition mismatch occurs, retain the failure and its identity evidence. Unload explicitly before
a deliberately changed placement. Do not silently rebalance or relaunch with fallback enabled.

At this point **finish the inference task and return the resident handoff**: both engines have exited,
GPU resources have been released, and the CPU arena remains ready until its reported idle deadline.
Return the 05 worker PID/start identity, port/forward ownership, remaining TTL and hard process deadline,
plus the JSON evidence. Leave the worker and bounded forwards alive on successful completion. Do not
run unload/stop as a success trap. The synthetic suite is required to pass unload, expiry and clean exit before this window.
The following controls are for the later explicit release or an intentionally selected expiry trial.

## 4. Later release or a separate unload/expiry verification window

When the operator chooses to release the retained cache, use 05's local endpoint (or the still-owned forward):

```sh
python3 "$SRC/tools/worker_control.py" unload --port 7431 > "$RUN05/after-unload.json"
```

Require empty session, complete=0, resident_bytes=0, incremented unloads, and `reason=explicit` in the
worker log. Keep PID identity and sample RSS/headroom; heap byte accounting is exact while RSS may lag.
If a cold reload is in the approved window, run the same cold engine once more and require cache_hit=0
and loaded/verified bytes increasing by the complete payload, without overlapping arenas.
Then leave the engine stopped and wait until the **remaining idle TTL** expires. Status does not renew
it. Check at most once per minute within the owned two-hour wall; expect complete=0, resident_bytes=0,
session=0, incremented expirations and `reason=expiry`. Allow 100 ms sweep plus scheduling delay.
For a separately selected expiry-only shorter trial, choose `--residency-ms 30000` at the initial worker launch and keep
both engine sessions within that window; do not shorten the window by restarting a populated worker.

Finally stop the now-empty worker with `worker_control.py stop`, verify its owned worker PID and timeout
wrapper exit, then close only the owned forwards. Require both `arena released; joining CPU pool` and
`CPU pool joined; resident=0 B`, final summaries and actual process exit. If a wrapper hits its hard
wall, record RC 124 as a timeout and preserve which cleanup marker was last seen; do not attribute it
to arena release without evidence. Do not leave an unbounded worker or tunnel after controller loss.

Report cold/warm setup phases, bytes reused/loaded/verified/network, session ids, complete partition,
requests/CPU service, engine GPU release, exact source/binary/manifest/index/profile identities, reserves,
unload/expiry/exit evidence and all exit codes. Mark CUDA/full-size memory/full-model behavior untested
until this recipe runs. The coding task has not run it.
