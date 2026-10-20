# Hermes: bounded common-kernel correction validation

Use only the exact new `CANDIDATE` SHA from the coding handoff. **79f4da0 native-OFF failed to link**;
OFF tests and native-ON checks on 03 did not run. Its independent 05 fixture passes and recovered
identities are in the [attributed failure receipt](evidence/hermes-79f4da0-failure.md).
The [16d065e full-model PASS](evidence/hermes-16d065e-receipt.md) remains separate. The subsequent
common-kernel correction has no runtime validation. This coding task ran no configure/build/tests
and accessed neither host.

Only dedicated checkouts/builds and small synthetic fixtures on 03/05 are authorized. No real-model
reload, benchmark, broad CI, service/tunnel operations, driver changes, merge, history rewrite or
network GPU implementation. Preserve old logs, first-failure evidence, known-good native-ON caches
and unrelated work. **Run the complete OFF and ON stage on 03 before starting 05. At the first failure,
timeout, missing required identity or drift, stop the whole handoff. Do not retry or continue on 05.**
The earlier independently executed 05 result is retained, not a precedent for bypassing this gate.

The [dependency audit](NATIVE_OFF_AUDIT.md) explains the correction: real Q2 AVX1 and BF16 router kernels
are common; native quant formats, ggml and worker libraries stay opt-in. BF16 AVX2 arithmetic/flags were
moved unchanged. CUDA and shared HIP use this CPU target. HIP/Windows are unvalidated; SYCL's separate
CPU source list is synchronized, without a claim of worker support or backend validation. Metal is
unchanged. This diff does not require another full-model load or establish full-model numerical parity.

## 0. Pin identities and preserve the failed run

On the controller:

```sh
set -euo pipefail
: "${CANDIDATE:?Full new correction SHA from coding handoff}"
BRANCH=inferswarm/persistent-cpu-weight-residency
BASE=79f4da0e09b4f7ff0e331809316cd2fc18684b51
MAIN=e88ee39d1d6be6e3364183af3b39b62a7ed6f211
SAFE=16d065e23211f2d422c56a3497a8be200c1e1d4d
[[ "$CANDIDATE" =~ ^[0-9a-f]{40}$ ]] && test "$CANDIDATE" != "$BASE"
test "$(git ls-remote git@github.com:Zutfen-LLC/Strata.git "refs/heads/$BRANCH" | cut -f1)" = "$CANDIDATE"
test "$(git ls-remote git@github.com:Zutfen-LLC/Strata.git refs/heads/main | cut -f1)" = "$MAIN"
```

Read 05 receipts only after the full 03 stage passes. Preserve receipts on 03 and 05 under `/home/hermes/strata-native-off-79f4da0` and controller
`/home/zutfen/strata-native-off-79f4da0-summary/REPORT.md`, failure log, identities and fixtures.
Do not overwrite them. Use new owned log directories outside all source/build directories.
Recovered manifest SHA-256 is `8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318`.
Hermes freshly rehashed and matched evidence archive SHA-256
`fd623801b5a99bc4ca1396f7a7b0271dec4ad493f2782c6e331851810c0e5189`.
Read remaining full binary/index/profile identities from retained small receipts. Do not rehash model
assets, prepare manifests again or invent a missing digest. Missing identities required to identify
these source/build/toolchain/binary inputs are blockers; report other historical gaps explicitly.

The failed native-OFF build `/home/hermes/strata-trial-17ecafc/build-native-off` may be used incrementally
only after comparing its source, old candidate, compiler paths/versions, CUDA host compiler/toolchain,
generator and options with the retained receipt. Save its pre-change `CMakeCache.txt` in the new log
directory and record the preserved first-failure log path before configuring. Do not delete/recreate
this cache. Resolve each source path from that build's `CMAKE_HOME_DIRECTORY`, not a guessed path.
Resolve known-good `ONBUILD03` and `ONBUILD05` from successful receipts, preserving each cache's own
source path/toolchain/options. Their source directories may differ from the OFF source.

Require each source clean, canonical origin `git@github.com:Zutfen-LLC/Strata.git`, exact old HEAD as
recorded in its receipt (79f4da0 or safe 16d065e), and MAIN/SAFE ancestry. Fetch only BRANCH, require
`FETCH_HEAD=$CANDIDATE` and BASE ancestry, then detach at CANDIDATE without reset. If two builds share
a source already advanced by this same run, verify its clean CANDIDATE HEAD instead of switching again.
Stop on unexpected state. Include all seven pre-OFF-correction commits (five worker prerequisites and
two residency commits) and all additive corrections in any future integration; do not select only the
two residency commits for main.

Native-ON ggml must remain `3cf03257f219afbe7334045ff7c6a06ac68c627d`, verified against the retained
cache/checkout receipt. No new downloads, tool installation or compiler changes. If failed-cache reuse
cannot be verified, stop and propose a distinct empty OFF directory with the exact retained toolchain
and options. Never toggle a successful native-ON cache OFF.

## Shared shell controls

On each host only when its stage is authorized, use Bash with `set -euo pipefail`. Supply the pinned
identity variables above and absolute build/source paths verified from receipts. Set a new `RUN`
(outside sources/builds); require it absent and create it with `mkdir "$RUN"`. Keep 03 within 2,700
seconds and the later 05 stage within 900 seconds, plus up to 15 seconds for timeout cleanup.
Use these helpers; every nonzero result stops the stage and the entire handoff:

```sh
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
```

Require existing `timeout`, Python 3, CMake, the recorded Ninja/Make generator, `sha256sum`, `objdump`
and `nm`. Use serial `-j1`; preserve current cgroup caps and at least the 8 GiB start reserve. Check
again before each build/test group. Report current GPU free memory and require at least 1 GiB free on
each device assigned to the small 03 fixtures. These are admission checks, not a claim memory cannot
change later. Stop for insufficient headroom; do not stop other services to create space. No assumption
that `/usr/bin/time` exists. RC 124 is a timeout/failure, not a successful shutdown diagnosis.

## 1. All native-OFF checks on 03, then focused native-ON checks

Set verified `OFFBUILD03`, its `OFFSRC03`, known-good `ONBUILD03` and its `ONSRC03`; they may have different
source directories. Advance each required source using the identity gate above. Set new `RUN` and
`BUDGET_SECONDS=2700`; use the helpers. Compare cache/toolchain identities before any reconfigure.
Require OFF cache: Ninja, Release, CUDA ON, sm_86, HIP/SYCL OFF, native OFF, tests ON, MMQ prefill/ORCA OFF.
Require ON cache: its original native-ON Ninja sm_86 configuration and pinned ggml. Keep all other
options, compiler and host-compiler/toolchain settings from the retained receipts.

```sh
test "$OFFBUILD03" != "$ONBUILD03"
test "$(cache_get CMAKE_HOME_DIRECTORY "$OFFBUILD03")" = "$OFFSRC03"
test "$(cache_get CMAKE_HOME_DIRECTORY "$ONBUILD03")" = "$ONSRC03"
test "$(cache_get STRATA_NATIVE_EXPERTS "$OFFBUILD03")" = OFF
test "$(cache_get STRATA_NATIVE_EXPERTS "$ONBUILD03")" = ON
cp "$OFFBUILD03/CMakeCache.txt" "$RUN/off-cache-before.txt"
cp "$ONBUILD03/CMakeCache.txt" "$RUN/on-cache-before.txt"
reserve off-cpu-resources
run off-configure 120 cmake -S "$OFFSRC03" -B "$OFFBUILD03" \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
# Fast common CPU link/numerical gate before the engine link; synthetic only.
run off-cpu-build 180 cmake --build "$OFFBUILD03" --target router_dot_parity q2_bitplane_parity -j1
for name in router_dot_parity router_dot_parity_avx q2_bitplane_parity q2_bitplane_parity_avx; do
  run "off-$name" 90 ctest --test-dir "$OFFBUILD03" -V -j1 --no-tests=error -R "^$name$"
done
reserve off-engine-resources
run off-engine-build 1800 cmake --build "$OFFBUILD03" --target strata worker_disabled_dispatch_test -j1
run off-dispatch 30 ctest --test-dir "$OFFBUILD03" -V -j1 --no-tests=error -R '^worker_disabled_dispatch_test$'
run off-cli 120 python3 "$OFFSRC03/tests/worker/worker_disabled_cli_test.py" --engine "$OFFBUILD03/strata"
run off-isa 120 python3 "$OFFSRC03/tools/check_isa_guard.py" "$OFFBUILD03/libstrata_kernels_cpu.a"
sha256sum "$OFFBUILD03/strata" "$OFFBUILD03/worker_disabled_dispatch_test" \
  "$OFFBUILD03/router_dot_parity" "$OFFBUILD03/q2_bitplane_parity" > "$RUN/off-binaries.sha256"
cp "$OFFBUILD03/CMakeCache.txt" "$RUN/off-cache-after.txt"
# Inspect/save actual engine+dispatcher link graph/commands: no worker libraries or ggml in OFF;
# common router/Q2 objects present once, with per-file ISA flags. Stop if this does not hold.
reserve on-build-resources
run on-configure 120 cmake -S "$ONSRC03" -B "$ONBUILD03" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
run on-build 1800 cmake --build "$ONBUILD03" --target strata worker_dispatch_test router_dot_parity q2_bitplane_parity -j1
reserve on-test-resources
for name in router_dot_parity router_dot_parity_avx q2_bitplane_parity q2_bitplane_parity_avx \
            worker_dispatch_test worker_dispatch_host_zero_test isa_guard_check; do
  run "on-$name" 120 ctest --test-dir "$ONBUILD03" -V -j1 --no-tests=error -R "^$name$"
done
run on-device-plan 120 env STRATA_VERIFY_DEVICE_PLAN=1 "$ONBUILD03/worker_dispatch_test"
sha256sum "$ONBUILD03/strata" "$ONBUILD03/worker_dispatch_test" \
  "$ONBUILD03/router_dot_parity" "$ONBUILD03/q2_bitplane_parity" > "$RUN/on-binaries.sha256"
cp "$ONBUILD03/CMakeCache.txt" "$RUN/on-cache-after.txt"
printf '%s\n' "$CANDIDATE" > "$RUN/03-passed.sha"
```

Record every named test actually executed and output, not just an aggregate count. CLI checks cover
six unavailable flags, option-shaped values and the Qwen3.6 parser with a tiny metadata-only GGUF. Its
family metadata probe may precede the second refusal; payload/model load, worker connections and GPU
initialization must not. Router/Q2 checks exercise real math, including forced AVX1. No model assets
are used. Save new source, cache and binary identities with each result.

## 2. Only after the complete 03 stage passes, start 05

Read/copy the small `03-passed.sha` via the controller and check the 03 logs/RCs have no failure. Do not
start 05 merely because OFF passed if a later 03 ON check failed. On 05 resolve its dedicated `SRC05`
from `ONBUILD05/CMakeCache.txt`; verify clean source and retained identities, then advance as in section
0. Preserve its native-ON CPU-only Makefiles/compiler/ggml cache. Set a new `RUN`, `BUDGET_SECONDS=900`
and `GATE03_RECEIPT` to the verified copied success receipt; use the same helpers.

```sh
test "$(cat "$GATE03_RECEIPT")" = "$CANDIDATE"
test "$(cache_get CMAKE_HOME_DIRECTORY "$ONBUILD05")" = "$SRC05"
test "$(cache_get STRATA_NATIVE_EXPERTS "$ONBUILD05")" = ON
cp "$ONBUILD05/CMakeCache.txt" "$RUN/on-cache-before.txt"
reserve cpu-build-resources
run cpu-configure 120 cmake -S "$SRC05" -B "$ONBUILD05" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
run cpu-build 600 cmake --build "$ONBUILD05" --target \
  strata-expert-worker worker_local_test worker_residency_test router_dot_parity q2_bitplane_parity -j1
reserve cpu-test-resources
for name in router_dot_parity router_dot_parity_avx q2_bitplane_parity q2_bitplane_parity_avx \
            worker_local_binary_test worker_residency_test; do
  run "$name" 120 env STRATA_WORKER_FIXTURE_KEEP=1 ctest --test-dir "$ONBUILD05" \
    -V -j1 --no-tests=error -R "^$name$"
done
sha256sum "$ONBUILD05/strata-expert-worker" "$ONBUILD05/worker_residency_test" \
  "$ONBUILD05/router_dot_parity" "$ONBUILD05/q2_bitplane_parity" > "$RUN/binaries.sha256"
printf '%s\n' "$CANDIDATE" > "$RUN/05-passed.sha"
```

Normal dependency-based build/relink is sufficient. Unchanged safe binaries identified by source/cache
lineage and matching receipts are valid; do not force recompilation merely for mtimes. Common CPU
library changes may naturally rebuild/relink affected targets. Never execute unsafe pre-fix fixtures.

Return one report with source/remote/main, cache/toolchain/ggml identities, exact commands, exit codes,
elapsed times, actual named outcomes, complete binary hashes and new receipt locations. Preserve and
attribute 79f4da0 OFF FAIL, independent 05 PASS and 16d065e full-model PASS separately. Keep passed,
failed, blocked and not run distinct. HIP/SYCL/Windows/older-CPU checks remain open; do not expand this
handoff to execute them. Stop before merge and return the first failure without a retry.
