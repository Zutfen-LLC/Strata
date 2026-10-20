# Hermes: checker-only continuation after 7992bb0d

Use the exact new `CANDIDATE` SHA from the coding handoff. The [7992bb0d receipt](evidence/hermes-7992bb0d-receipt.md)
is an attributed user report, not locally inspected evidence. It records complete OFF PASS and ON
configure/build plus four math/two dispatch PASS on 03, followed by `isa_guard_check` RC 8 for the
pre-existing unregistered TQ AVX512 object. Device-plan and all 05 gates did not run. **7992bb0d is not
an overall PASS.** The new checker head is unvalidated until the following gates pass.

The code diff only registers `tq_avx512.cpp` and `tq_rows_avx512` in the checker and adds deterministic
regression source. No kernel, CPUID, build definition or runtime input changed from BASE. Reuse verified
safe 03 outputs; no configure/build or repeated math/dispatch/50-CLI run is needed there. Run one quick
OFF archive scan with the new checker, because it shares the changed policy; no OFF engine rebuild.
On 05 use normal bounded CPU-only dependency checking/build if its safe cache predates BASE. Do not
force recompilation for mtimes. No full-model work, model hash sweeps, benchmark, broad CI, merge, PR,
service/tunnel operations, driver changes or network GPU implementation. No activity on Geekom beyond
edits/static inspection. Do not infer a host/application crash from an assistant transport disconnect.

**Sequential gates:** new checker regressions, quick OFF scan, corrected ON ISA check, then remaining
03 device-plan. Only after all 03 gates pass may 05 start. At the first failure, timeout, skipped required
test, missing required identity or drift, stop the entire handoff, preserve the first failure and report;
no retry, later gate or other-host continuation. Older 05 results remain separate, not a substitute.

## Identities and preservation

```sh
set -euo pipefail
: "${CANDIDATE:?New full correction SHA from handoff}"
: "${CONTROLLER_SRC:?Absolute canonical controller checkout}"
BASE=7992bb0d293126ed0738a9b73d6c3fec7b744e78
MAIN=e88ee39d1d6be6e3364183af3b39b62a7ed6f211
SAFE=16d065e23211f2d422c56a3497a8be200c1e1d4d
BRANCH=inferswarm/persistent-cpu-weight-residency
[[ "$CANDIDATE" =~ ^[0-9a-f]{40}$ ]] && test "$CANDIDATE" != "$BASE"
canonical_origin() {
  case "$(git -C "$1" remote get-url origin)" in
    git@github.com:Zutfen-LLC/Strata.git|ssh://git@github.com/Zutfen-LLC/Strata.git|https://github.com/Zutfen-LLC/Strata.git|https://github.com/Zutfen-LLC/Strata) ;;
    *) echo 'STOP: origin is not the canonical fork'; exit 1 ;;
  esac
}
# From a verified canonical controller checkout (SSH and canonical HTTPS are both valid):
canonical_origin "$CONTROLLER_SRC"
test "$(git -C "$CONTROLLER_SRC" ls-remote origin "refs/heads/$BRANCH" | cut -f1)" = "$CANDIDATE"
test "$(git -C "$CONTROLLER_SRC" ls-remote origin refs/heads/main | cut -f1)" = "$MAIN"
```

Preserve `/home/hermes/strata-val-7992bb0d-03-r2` and controller
`/home/zutfen/strata-val-7992bb0d-controller/REPORT.md` and associated receipts. Reported archive SHA-256:
`01094f70846ba3c829ff2925e20eddcadc0b46ec607f09cfe35f6f502d417e4a`.
Do not overwrite the failed ISA output or rehash this archive/model assets merely to repeat a known
receipt. Recover full binary/archive/cache/toolchain identities from existing small receipts; do not
guess them. Preserve the separate 79f4da0 link failure and 16d065e full-model PASS.

On 03 resolve `OFFBUILD03`, `ONBUILD03` and their source directories from the retained receipts and each
cache's `CMAKE_HOME_DIRECTORY`. Require known-safe BASE source/outputs, exact compiler paths/versions,
generator/options, native OFF/ON as appropriate, CUDA sm_86, and ON ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Compare these facts and saved binary/archive hashes against
the receipts before reuse; a hash computed now without source/build provenance is insufficient. Sources may
differ between builds. Save current cache copies in new receipt directories; never toggle known-good
ON/OFF options or clean caches. Stop rather than reconstruct missing source/build provenance.

For each dedicated source: require clean status, canonical SSH or HTTPS origin, exact HEAD from its
receipt (BASE on 03; known safe receipt revision on 05), and MAIN/SAFE ancestry. Fetch only BRANCH,
verify FETCH_HEAD equals CANDIDATE and BASE is an ancestor, then detach without reset. If two builds
share a source advanced by this same run, verify clean CANDIDATE instead of switching twice.
On every updated source prove no compiled input changed from BASE:

```sh
git -C "$SRC" diff --exit-code "$BASE" "$CANDIDATE" -- . \
  ':!docs/cpu-worker-residency/**' ':!tools/check_isa_guard.py' ':!tools/test_check_isa_guard.py'
```

Any unexpected diff is a blocker, not permission to rebuild or expand scope. Ensure CTest's ISA command
uses the new checker in that cache's source tree, not another checkout. Record its actual command and
the checker file SHA. Python unit tests below use synthetic nm/objdump text and do not launch tools.
Read 05 receipts or operate 05 only after the complete 03 success gate. Future integration includes all
worker prerequisites and additive corrections, not only the two original residency commits.

## Common controls for each authorized stage

Use fail-fast Bash, existing tools only, `-j1`, `CTEST_PARALLEL_LEVEL=1`, `OMP_NUM_THREADS=2`. Require
`git`, `timeout`, Python 3, CMake/CTest, `sha256sum`, `rg`, `objdump` and `nm` already available; no installs.
Supply a new
absolute `RUN` outside source/builds, require it absent and create with `mkdir "$RUN"`. Set
`BUDGET_SECONDS=2700` on 03 and `900` on 05. Timeouts have at most 15 seconds cleanup; RC 124 is failure.
Keep existing limits/services unchanged. Require at least 8 GiB host and finite cgroup headroom before
each execution group and at least 1 GiB free per assigned device before the small 03 device-plan test.
Unknown resource layouts are blockers. No assumption that `/usr/bin/time` exists. Use these helpers:

```sh
for tool in git timeout python3 cmake ctest sha256sum rg objdump nm; do
  command -v "$tool" >/dev/null
done
export CMAKE_BUILD_PARALLEL_LEVEL=1 CTEST_PARALLEL_LEVEL=1 OMP_NUM_THREADS=2
: "${BUDGET_SECONDS:?2700 on 03; 900 on 05}"
DEADLINE=$((SECONDS + BUDGET_SECONDS))
run() {
  local name=$1 cap=$2 rc remaining started
  shift 2
  remaining=$((DEADLINE - SECONDS))
  if (( remaining <= 0 )); then echo 'STOP: stage budget expired'; exit 124; fi
  if (( cap > remaining )); then cap=$remaining; fi
  printf '%q ' "$@" > "$RUN/$name.command"
  printf '\n' >> "$RUN/$name.command"
  started=$SECONDS
  set +e
  timeout --signal=TERM --kill-after=15s "$cap" "$@" > "$RUN/$name.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$RUN/$name.rc"
  printf '%s\n' "$((SECONDS - started))" > "$RUN/$name.seconds"
  if (( rc != 0 )); then echo "STOP: $name returned $rc; retain $RUN"; exit "$rc"; fi
}
cache_get() { sed -n "s/^$1:[^=]*=//p" "$2/CMakeCache.txt"; }
reserve() {
  # Start gate: 8 GiB available in host RAM AND every finite applicable cgroup-v2 cap.
  # Unknown cgroup layouts are blockers, not unlimited memory. Do not alter limits/services.
  run "$1" 10 python3 - <<'PY'
from pathlib import Path
import json
minimum = 8 * 1024**3
available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith('MemAvailable:'))) * 1024
records = {'MemAvailable': available, 'minimum_required': minimum, 'cgroups': []}
if available < minimum:
    raise SystemExit(json.dumps(records))
entry = next(s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
root = Path('/sys/fs/cgroup').resolve()
if not (root / 'cgroup.controllers').is_file():
    raise SystemExit('unknown cgroup-v2 mount')
node = (root / entry.split('::', 1)[1].lstrip('/')).resolve()
if node != root and root not in node.parents:
    raise SystemExit(f'unknown cgroup path: {node}')
while True:
    cap_file = node / 'memory.max'
    # The real hierarchy root has no local memory.max/current. A namespace mount root can
    # expose a cap, so check it when present instead of always treating the mount as unlimited.
    if node == root and not cap_file.exists():
        records['cgroups'].append([str(node), 'hierarchy root; no local memory.max', None])
        break
    cap = cap_file.read_text().strip()
    used = int((node / 'memory.current').read_text())
    records['cgroups'].append([str(node), cap, used])
    if cap != 'max':
        if int(cap) - used < minimum:
            raise SystemExit(json.dumps(records))
    if node == root:
        break
    node = node.parent
print(json.dumps(records, indent=2))
PY
}
reject_matches() {
  local log=$1 pattern=$2 rc
  set +e
  rg -q "$pattern" "$log"
  rc=$?
  set -e
  # 1 means no matches. Both a match (0) and any matcher/read failure (>1) stop the handoff.
  if (( rc != 1 )); then echo "STOP: rejected match or matcher failure in $log (rc=$rc)"; exit 1; fi
}
prove_file() {
  local name=$1 path=$2 expected=$3
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]]
  run "$name-sha256" 30 sha256sum "$path"
  test "$(cut -d ' ' -f1 "$RUN/$name-sha256.log")" = "$expected"
}
prove_archive() {
  local name=$1 build=$2 expected=$3 ninja
  shift 3
  if test -n "$expected"; then
    prove_file "$name" "$build/libstrata_kernels_cpu.a" "$expected"
    return
  fi
  # Only the archive digest is missing: source/cache/toolchain provenance and prior executable
  # digests must already be verified. Do not fall back for unknown source or output lineage.
  test "$(cache_get CMAKE_GENERATOR "$build")" = Ninja
  ninja=$(cache_get CMAKE_MAKE_PROGRAM "$build")
  test -x "$ninja"
  run "$name-no-work" 30 env LC_ALL=C NINJA_STATUS='[%f/%t] ' "$ninja" -n -j1 -C "$build" "$@"
  if ! rg -q -F 'ninja: no work to do.' "$RUN/$name-no-work.log"; then
    echo 'STOP: dependency check did not prove no work'; exit 1
  fi
  reject_matches "$RUN/$name-no-work.log" '^\[[0-9]+/[0-9]+\]'
  # Freshly captured archive identity, not a claimed historical digest or a recomputed model hash.
  run "$name-sha256" 30 sha256sum "$build/libstrata_kernels_cpu.a"
  printf '%s\n' 'new archive hash; retained source/cache/toolchain plus no-work proof' > "$RUN/$name-provenance.txt"
}
```

## 1. Resume the 03 gates without recompiling verified engine outputs

Set verified `OFFSRC03`, `ONSRC03`, build paths and the identity variables above. From the retained
BASE receipts supply full executable SHA-256 values `ON_STRATA_SHA` and `ON_DISPATCH_SHA`; verify
caches/toolchain/source provenance as above. Compare `OFF_ARCHIVE_SHA`/`ON_ARCHIVE_SHA` too if recorded.
The prior recipe did not require archive digests. If only those digests are absent, preserve the old
archives/logs, verify retained exact source/cache/toolchain and BASE-identical compiled inputs, then
require the bounded serial Ninja dry-run below to prove no work for the linked targets. Stop if any
work is scheduled; do not configure, recompile or relink. Record freshly captured archive hashes as new
provenance, not retrospective hash proof. Prior executable hashes stay mandatory for reused engine/
dispatch execution. Advance both sources, keep cache options intact, and copy caches/actual CTest ISA
command into the new RUN. No actual 03 configure/build.

```sh
test "$(cache_get CMAKE_HOME_DIRECTORY "$OFFBUILD03")" = "$OFFSRC03"
test "$(cache_get CMAKE_HOME_DIRECTORY "$ONBUILD03")" = "$ONSRC03"
test "$(cache_get STRATA_NATIVE_EXPERTS "$OFFBUILD03")" = OFF
test "$(cache_get STRATA_NATIVE_EXPERTS "$ONBUILD03")" = ON
cp "$OFFBUILD03/CMakeCache.txt" "$RUN/off-cache.txt"
cp "$ONBUILD03/CMakeCache.txt" "$RUN/on-cache.txt"
prove_file on-engine "$ONBUILD03/strata" "$ON_STRATA_SHA"
prove_file on-dispatch "$ONBUILD03/worker_dispatch_test" "$ON_DISPATCH_SHA"
prove_archive off-archive "$OFFBUILD03" "${OFF_ARCHIVE_SHA:-}" strata worker_disabled_dispatch_test
prove_archive on-archive "$ONBUILD03" "${ON_ARCHIVE_SHA:-}" strata worker_dispatch_test
sha256sum "$ONSRC03/tools/check_isa_guard.py" "$ONSRC03/tools/test_check_isa_guard.py" > "$RUN/checker.sha256"
reserve checker-resources
# unittest module resolution must use the new source checkout.
cd "$ONSRC03"
run checker-regressions 60 python3 -B -m unittest -v tools.test_check_isa_guard
run off-isa 120 python3 -B "$OFFSRC03/tools/check_isa_guard.py" "$OFFBUILD03/libstrata_kernels_cpu.a"
run on-isa 600 ctest --test-dir "$ONBUILD03" -V -j1 --no-tests=error -R '^isa_guard_check$'
# CTest's skip return code must not turn an unavailable check into PASS.
reject_matches "$RUN/on-isa.log" 'Skipped|Not Run|DISABLED'
reserve device-plan-resources
# Capture current free GPU memory and enforce the 1 GiB per assigned-device admission check here.
run on-device-plan 120 env STRATA_VERIFY_DEVICE_PLAN=1 "$ONBUILD03/worker_dispatch_test"
printf '%s\n' "$CANDIDATE" > "$RUN/03-passed.sha"
```

Require explicit zero ISA violations and actual test execution. The regression suite must accept only
the registered TQ entry/anonymous helper and still reject weak TQ symbols/clones, unrelated public code,
the entry in an unrelated object, and EVEX/VEX static initialization. It also exercises the existing
policy cases. Reused BASE math/dispatch/CLI PASS is not a newly executed CANDIDATE result.

## 2. Only after all 03 gates pass, finish 05

Verify the small `03-passed.sha` via the controller and all 03 RCs; then permit 05 access. Set verified
`SRC05`, `ONBUILD05`, new RUN and BUDGET_SECONDS=900, and `GATE03_RECEIPT` to the copied success receipt.
Resolve source/toolchain/options/ggml and complete binary hashes from existing safe 05 receipts. Record
its original source revision. Advance source to CANDIDATE using the identity gate above.

If its compiled outputs already correspond to BASE and match their receipt hashes, reuse them. If its
cache predates the common-kernel correction, normal bounded incremental CPU-only configure/build below
is necessary to reach those inputs; this is not a forced clean rebuild for checker changes. Do not
reuse older outputs as though they contained BASE. Preserve the existing generator/compiler/ggml cache,
no CUDA/HIP/SYCL enabling and no download/tool changes. Stop on missing provenance or an unexpected
expensive rebuild rather than broadening work. Keep exact hashes before and after dependency checking.

```sh
test "$(cat "$GATE03_RECEIPT")" = "$CANDIDATE"
test "$(cache_get CMAKE_HOME_DIRECTORY "$ONBUILD05")" = "$SRC05"
test "$(cache_get STRATA_NATIVE_EXPERTS "$ONBUILD05")" = ON
cp "$ONBUILD05/CMakeCache.txt" "$RUN/cache-before.txt"
reserve cpu-build-resources
# Needed only if inputs are not already BASE-equivalent; otherwise skip these two steps and record why.
run cpu-configure 120 cmake -S "$SRC05" -B "$ONBUILD05" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
run cpu-build 600 cmake --build "$ONBUILD05" --target \
  strata-expert-worker worker_local_test worker_residency_test router_dot_parity q2_bitplane_parity -j1
reserve cpu-test-resources
for name in router_dot_parity router_dot_parity_avx q2_bitplane_parity q2_bitplane_parity_avx \
            worker_local_binary_test worker_residency_test; do
  run "$name" 120 env STRATA_WORKER_FIXTURE_KEEP=1 ctest --test-dir "$ONBUILD05" \
    -V -j1 --no-tests=error -R "^$name$"
  reject_matches "$RUN/$name.log" 'Skipped|Not Run|DISABLED'
done
sha256sum "$ONBUILD05/strata-expert-worker" "$ONBUILD05/worker_residency_test" \
  "$ONBUILD05/router_dot_parity" "$ONBUILD05/q2_bitplane_parity" > "$RUN/binaries.sha256"
cp "$ONBUILD05/CMakeCache.txt" "$RUN/cache-after.txt"
printf '%s\n' "$CANDIDATE" > "$RUN/05-passed.sha"
```

Use only safe fixture lineage, never forbidden pre-fix binaries. Return one report with source/origin/main,
exact cache/toolchain and BASE-equivalence proof, full binary/archive/checker hashes, per-step commands,
RCs/timings, actual named outcomes and new receipt locations. Separate reused 7992bb0d evidence from
new checker/device-plan/05 execution, and passed/failed/blocked/not-run. Preserve historical failures.
HIP/SYCL/Windows runtime validation and full-model numerical parity remain unclaimed. Stop before merge.
