#!/usr/bin/env bash
# pr7_build_gates_05.sh — PR7 candidate 723c5a45 serial CPU build + focused loader gates on 05.
# Detached driver with per-step rc/log receipts; stop on first failure.
set -euo pipefail
SRC=__HOST_PATH__
BUILD=__HOST_PATH__
RES=__HOST_PATH__
BIN=$RES/cpu-bin
CANDIDATE=723c5a45cd0fe289cb57c7c3d1e504c068bd0cf8
BASE=fbd3bfd1e710c277b818ff37417ba6ee313a5621
mkdir -m 700 -p "$RES"
cp __HOST_PATH__/timed_run.py "$RES/timed_run.py" 2>/dev/null || true
step() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RES/driver.log"; }

step "gate: candidate ref tip"
TIP=$(git -C "$SRC" ls-remote origin refs/heads/feature/cpu-local-load-threads | awk '{print $1}')
[ "$TIP" = "$CANDIDATE" ] || { echo "TIP MISMATCH: $TIP" | tee -a "$RES/driver.log"; exit 10; }

step "gate: clean tree + fetch + ancestry + detach"
[ -z "$(git -C "$SRC" status --porcelain)" ]
git -C "$SRC" fetch origin feature/cpu-local-load-threads 2>&1 | tee -a "$RES/fetch.log"
[ "$(git -C "$SRC" rev-parse FETCH_HEAD)" = "$CANDIDATE" ]
git -C "$SRC" cat-file -e "$CANDIDATE^{commit}"
git -C "$SRC" merge-base --is-ancestor "$BASE" "$CANDIDATE"
git -C "$SRC" switch --detach "$CANDIDATE" 2>&1 | tee -a "$RES/driver.log"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE" ]
step "gate: cache binding"
[ "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "$SRC" ]
test ! -e "$BIN"
mkdir -m 700 "$BIN"

step "configure (disconnected, tests on, runtime out=$BIN)"
python3 "$RES/timed_run.py" "$RES/configure" cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN" \
  || { rc=$?; echo "rc=$rc" > "$RES/configure.rc"; exit 20; }
echo 0 > "$RES/configure.rc"
grep -iE "error|warning:.*dependency|downloading" "$RES/configure.log" | head -5 > "$RES/configure.flags" || true

step "serial build: worker + five test targets"
START=$(date +%s)
python3 "$RES/timed_run.py" "$RES/build" cmake --build "$BUILD" --parallel 1 --target strata-expert-worker worker_loader_test \
  worker_local_test worker_residency_test worker_service_residency_test worker_compat_test \
  || { rc=$?; echo "rc=$rc" > "$RES/build.rc"; exit 21; }
echo 0 > "$RES/build.rc"; python3 -c "import json; print(json.load(open('''$RES/build.time.json'''))['elapsed_s'])" > "$RES/build.elapsed_s"

step "record candidate hashes"
sha256sum "$BIN/strata-expert-worker" "$BIN"/worker_*_test > "$RES/cpu-bin.sha256"
cat "$RES/cpu-bin.sha256" >> "$RES/driver.log"

step "focused ctest gates"
ctest --test-dir "$BUILD" -V -j1 -R \
  '^(worker_loader_(threads|binary)_(1|2|4)|worker_local_test|worker_local_binary_test|worker_residency_test|worker_service_residency_test|worker_compat_test)$' \
  > "$RES/ctest.log" 2>&1 || { rc=$?; echo "rc=$rc" > "$RES/ctest.rc"; exit 22; }
echo 0 > "$RES/ctest.rc"
grep -E "tests passed|Failed" "$RES/ctest.log" | tail -3 >> "$RES/driver.log"

step "worker_control_test"
python3 "$SRC/tests/worker/worker_control_test.py" --worker "$BIN/strata-expert-worker" --control "$SRC/tools/worker_control.py" \
  > "$RES/control-test.log" 2>&1 || { rc=$?; echo "rc=$rc" > "$RES/control-test.rc"; exit 23; }
echo 0 > "$RES/control-test.rc"; tail -2 "$RES/control-test.log" >> "$RES/driver.log"

step "teardown check: no owned fixture processes"
pgrep -af "worker_loader|worker_local_test|worker_residency|worker_service|worker_compat|strata-expert-worker" | grep -v grep >> "$RES/teardown.log" || echo "none" > "$RES/teardown.log"

step "ALL GATES PASSED"
echo done > "$RES/STATUS"
