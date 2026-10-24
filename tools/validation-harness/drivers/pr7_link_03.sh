#!/usr/bin/env bash
# pr7_link_03.sh — PR7 candidate 723c5a45 incremental CUDA link of `strata` with the
# new shared worker library, using 03's retained bf51f09e native-OFF cache. No model launch.
set -euo pipefail
SRC=__HOST_PATH__
BUILD="${PR7_BUILD:-__HOST_PATH__}"
RES="${PR7_RES:-__HOST_PATH__}"
BIN=$RES/bin
CANDIDATE=723c5a45cd0fe289cb57c7c3d1e504c068bd0cf8
BASE=fbd3bfd1e710c277b818ff37417ba6ee313a5621
mkdir -m 700 -p "$RES"
step() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RES/driver.log"; }

step "gate: ref tip + fetch + ancestry + detach (same receipts as 05)"
TIP=$(git -C "$SRC" ls-remote origin refs/heads/feature/cpu-local-load-threads | awk '{print $1}')
[ "$TIP" = "$CANDIDATE" ]
git -C "$SRC" fetch origin feature/cpu-local-load-threads > "$RES/fetch.log" 2>&1
[ "$(git -C "$SRC" rev-parse FETCH_HEAD)" = "$CANDIDATE" ]
git -C "$SRC" cat-file -e "$CANDIDATE^{commit}"
git -C "$SRC" merge-base --is-ancestor "$BASE" "$CANDIDATE"
git -C "$SRC" switch --detach "$CANDIDATE" >> "$RES/fetch.log" 2>&1
[ "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE" ]

step "gate: cache binding + new BIN"
[ "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "$SRC" ]
test ! -e "$BIN"; mkdir -m 700 "$BIN"

step "configure into retained OFF cache with new runtime dir"
python3 "$RES/timed_run.py" "$RES/configure" cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN" \
  || { rc=$?; echo "rc=$rc" > "$RES/configure.rc"; exit 20; }
echo 0 > "$RES/configure.rc"

step "incremental link: strata only"
python3 "$RES/timed_run.py" "$RES/link" cmake --build "$BUILD" --parallel 1 --target strata \
  || { rc=$?; echo "rc=$rc" > "$RES/link.rc"; exit 21; }
echo 0 > "$RES/link.rc"
sha256sum "$BIN/strata" > "$RES/link.sha256"; cat "$RES/link.sha256" >> "$RES/driver.log"

step "link sanity: required DSOs resolve"
ldd "$BIN/strata" | grep -c "not found" > "$RES/ldd-notfound.count" && [ "$(cat "$RES/ldd-notfound.count")" = "0" ] || { echo "ldd not-found > 0" | tee -a "$RES/driver.log"; exit 22; }
ldd "$BIN/strata" > "$RES/ldd.txt"

step "ALL LINK GATES PASSED"
echo done > "$RES/STATUS"
