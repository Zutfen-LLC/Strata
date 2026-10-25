#!/usr/bin/env bash
# Role05 direct continuation; run under Hermes's existing external RSS/swap guard.
set -euo pipefail
umask 077
export RUN_ROOT=/home/hermes/strata-pr10-wire05-20261025-r1-out
SRC=/home/hermes/strata-pr10-wire05-20261025-r1-src
BUILD=/home/hermes/strata-pr10-wire05-20261025-r1-build
OBJECTS=/home/hermes/strata-pr10-compress-74ffe391-src
REF=/home/hermes/strata-pr10-packed-06bbd809-20261025-05
CAPTURE=/home/hermes/strata-pr10-micro-bc8e5754-20261025-05/cpu-capture.bin
MANIFEST=/home/hermes/strata-wl-metadata/local.manifest
MODEL=/srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66
export SRC BUILD OBJECTS
test "$(git -C /home/hermes/strata-cand-66235df1-build-cpu/_deps/strata_llamacpp-src rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test -z "$(git -C /home/hermes/strata-cand-66235df1-build-cpu/_deps/strata_llamacpp-src status --porcelain)"
for path in "$SRC" "$BUILD" "$RUN_ROOT"; do test -d "$path"; done
test -f "$REF/admit.py"
test -z "$(find "$RUN_ROOT" -maxdepth 1 -name 'direct1-*' -print -quit)"
ulimit -f unlimited
# Check compiled project units, headers and CMake inputs once; Git filters honor attributes.
python3 -B - <<'PY' > "$RUN_ROOT/direct1-source-check.log"
import json,os,pathlib,subprocess
P=pathlib.Path;src=P(os.environ['SRC']);build=P(os.environ['BUILD']);git=os.environ['OBJECTS']
pin='fa39c2f60d0e3b6c3154469b40b569dcd43f8479'
cmd=['git','-C',git,'--work-tree='+str(src)]
assert (src/'.gitattributes').read_bytes()==subprocess.check_output(cmd+['show',pin+':.gitattributes'])
paths={P(row['file']) for row in json.loads((build/'compile_commands.json').read_text()) if P(row['file']).is_relative_to(src)}
for folder in ('include','src/worker','src/kernels/cpu','third_party/ggml'):
    paths.update(p for p in (src/folder).rglob('*') if p.suffix in ('.h','.hh','.hpp','.hxx','.cuh','.inc','.def'))
paths.update(src.rglob('CMakeLists.txt'));paths.update((src/'cmake').rglob('*.cmake'))
for path in sorted(paths):
    rel=str(path.relative_to(src))
    expected=subprocess.check_output(cmd+['-c','core.autocrlf=false','cat-file','--filters','--path='+rel,pin+':'+rel])
    assert path.read_bytes()==expected, 'compiled/CMake input differs: '+rel
print('compiled/CMake inputs match',pin,'files='+str(len(paths)))
PY
OOM=$(awk '$1=="oom_kill" {print $2}' /proc/vmstat)
# Established admission + timeout/ulimit/time checks; no /proc process-tree monitor.
bounded() {
    local label=$1 seconds=$2 cap=$3; shift 3
    python3 -B "$REF/admit.py" "direct1-$label-admission.json"
    ( if test "$cap" = 1; then ulimit -v 2097152; fi
      /usr/bin/time -v -o "$RUN_ROOT/direct1-$label-resources.log" \
        timeout --signal=TERM --kill-after=5s "${seconds}s" "$@" \
        > "$RUN_ROOT/direct1-$label.log" 2>&1 )
    test "$(awk '$1=="oom_kill" {print $2}' /proc/vmstat)" = "$OOM"
    if test "$cap" = 1; then
        awk '/Maximum resident set size/ {seen=1; if ($NF>2097152) exit 1} END {if (!seen) exit 1}' "$RUN_ROOT/direct1-$label-resources.log"
        grep -Eq '^[[:space:]]*Swaps: 0$' "$RUN_ROOT/direct1-$label-resources.log"
    fi
}
bounded build 600 0 cmake --build "$BUILD" --parallel 2 --target \
    cpu_packed_wire_test cpu_worker_group_test cpu_packed_activation_proof expert_worker_test \
    expert_tier_test worker_residency_test worker_service_residency_test
for name in cpu_packed_wire_test cpu_worker_group_test cpu_worker_group_packed_test \
    cpu_packed_activation_check expert_worker_test expert_tier_test worker_residency_test worker_service_residency_test; do
    bounded "$name" 180 1 ctest --test-dir "$BUILD" -V --no-tests=error --timeout 180 -j1 -R "^${name}$"
    grep -Eq "$name.*Passed" "$RUN_ROOT/direct1-$name.log"
    ! grep -Eq 'Skipped|Not Run|FAILED' "$RUN_ROOT/direct1-$name.log"
done
PROOF="$RUN_ROOT/bin/cpu_packed_activation_proof"
test ! -e "$RUN_ROOT/activations.spa"; test ! -e "$RUN_ROOT/rows-same-build.sra"
bounded pack 190 1 "$PROOF" --pack "$CAPTURE" "$RUN_ROOT/activations.spa"
grep -Eq '^PACKED_DONE ok=1 records=16 bytes=92884 weights_read=0$' "$RUN_ROOT/direct1-pack.log"
cmp "$RUN_ROOT/activations.spa" "$REF/activations.spa"
bounded rows 190 1 "$PROOF" --rows "$CAPTURE" "$MANIFEST" "$MODEL" \
    "$REF/activations.spa" "$RUN_ROOT/rows-same-build.sra"
grep -Fxq 'PACKED_ROWS_DONE ok=1 records=16 packed_reference_bits=1 same_build_rows_bits=1 retained_capture_bits=1 cross_build_rows=REQUIRES_CMP sockets=0 worker_loads=0 model_GEN=0' "$RUN_ROOT/direct1-rows.log"
cmp "$RUN_ROOT/rows-same-build.sra" "$REF/rows-same-build.sra"
sha256sum "$RUN_ROOT/activations.spa" "$RUN_ROOT/rows-same-build.sra" > "$RUN_ROOT/direct1-artifacts.sha256"
echo 'ROLE05 direct continuation PASS; role02/backends/live A/B not run'
