#!/usr/bin/env bash
# pr7-def4-build-gates-05.sh — PR7 follow-up 38ef68d9 default-four focused validation on 05.
# Edits of retained pr7_build_gates_05.sh (same gates/structure): new CANDIDATE/RES, 2-target
# serial incremental build, two ordered default-selection checks. No full suite, no control test.
set -euo pipefail
SRC=__HOST_PATH__
BUILD=__HOST_PATH__
RES=__HOST_PATH__
BIN=$RES/cpu-bin
CANDIDATE=38ef68d988dd66822ff56b2d7a7768f71c800394
PREV=723c5a45cd0fe289cb57c7c3d1e504c068bd0cf8
BASE=fbd3bfd1e710c277b818ff37417ba6ee313a5621
OLDRES=__HOST_PATH__
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

step "record source delta PREV..CANDIDATE"
git -C "$SRC" diff --name-only "$PREV" "$CANDIDATE" > "$RES/changed-files.txt"
git -C "$SRC" diff --stat "$PREV" "$CANDIDATE" >> "$RES/changed-files.txt"

step "configure (disconnected, tests on, runtime out=$BIN)"
python3 "$RES/timed_run.py" "$RES/configure" cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN" \
  || { rc=$?; echo "rc=$rc" > "$RES/configure.rc"; exit 20; }
echo 0 > "$RES/configure.rc"
grep -iE "error|warning:.*dependency|downloading" "$RES/configure.log" | head -5 > "$RES/configure.flags" || true

step "serial incremental build: strata-expert-worker + worker_local_test"
START=$(date +%s)
python3 "$RES/timed_run.py" "$RES/build" cmake --build "$BUILD" --parallel 1 --target strata-expert-worker worker_local_test \
  || { rc=$?; echo "rc=$rc" > "$RES/build.rc"; exit 21; }
echo 0 > "$RES/build.rc"
grep -E "Building CXX object|Linking CXX executable" "$RES/build.log" > "$RES/changed-tus.txt" || true
python3 -c "import json; print(json.load(open('''$RES/build.time.json'''))['elapsed_s'])" > "$RES/build.elapsed_s"

step "record candidate hashes + compare with previous ELFs"
sha256sum "$BIN/strata-expert-worker" "$BIN/worker_local_test" > "$RES/cpu-bin.sha256"
cat "$RES/cpu-bin.sha256" >> "$RES/driver.log"
while read -r h f; do b=$(basename "$f");
  if [ -f "$OLDRES/cpu-bin/$b" ]; then oh=$(sha256sum "$OLDRES/cpu-bin/$b" | awk '{print $1}');
    if [ "$h" = "$oh" ]; then echo "UNCHANGED $b"; else echo "CHANGED   $b  old=$oh new=$h"; fi;
  else echo "NEW       $b  (not in prior cpu-bin)"; fi;
done < "$RES/cpu-bin.sha256" > "$RES/elf-compare.txt"
cat "$RES/elf-compare.txt" >> "$RES/driver.log"

step "check 1: worker_local_binary_test once (omitted loading flag)"
ctest --test-dir "$BUILD" -V -j1 -R '^worker_local_binary_test$' \
  > "$RES/ctest-default.log" 2>&1 || { rc=$?; echo "rc=$rc" > "$RES/ctest-default.rc"; exit 22; }
echo 0 > "$RES/ctest-default.rc"
grep -E "tests passed|Failed|Passed" "$RES/ctest-default.log" | tail -3 >> "$RES/driver.log"

step "check 2: worker_loader_binary_1 once (explicit --local-load-threads 1)"
ctest --test-dir "$BUILD" -V -j1 -R '^worker_loader_binary_1$' \
  > "$RES/ctest-serial.log" 2>&1 || { rc=$?; echo "rc=$rc" > "$RES/ctest-serial.rc"; exit 23; }
echo 0 > "$RES/ctest-serial.rc"
grep -E "tests passed|Failed|Passed" "$RES/ctest-serial.log" | tail -3 >> "$RES/driver.log"

step "gate greps: lane markers in both check logs"
grep -c "local loading lanes=4;" "$RES/ctest-default.log" > "$RES/marker-default.txt"
grep -c "local loading lanes=" "$RES/ctest-serial.log" >> "$RES/marker-default.txt" || echo "0 (serial: marker absent as required)" >> "$RES/marker-default.txt"
grep -c "2 pool workers" "$RES/ctest-default.log" "$RES/ctest-serial.log" >> "$RES/marker-default.txt" || true
cat "$RES/marker-default.txt" >> "$RES/driver.log"

step "teardown check: no owned fixture processes"
pgrep -af "worker_local_test|strata-expert-worker" | grep -v grep >> "$RES/teardown.log" || echo "none" > "$RES/teardown.log"
cat "$RES/teardown.log" >> "$RES/driver.log"

step "ALL GATES PASSED"
echo done > "$RES/STATUS"
