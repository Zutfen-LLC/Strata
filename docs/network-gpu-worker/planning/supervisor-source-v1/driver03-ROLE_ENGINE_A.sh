#!/bin/bash
# --- sanitized historical source (supervisor-source-v1) ----------------------
# Strata bf51f09e full-model trial - ROLE_ENGINE_A driver (host identity removed). (v2)
# Sanitized excerpt of the private original driver03.sh: NOT a byte-identical
# original and NOT runnable deployment instructions. Lines tagged
# "# sanitization:" are the only intentional textual edits; all logic,
# thresholds, units, ordering, error paths and original defects are preserved.
# -----------------------------------------------------------------------------
# Four serial engine sessions: B0 B1 G0 G1, global monotonic wall from GLOBAL_T0.
# All stops are positive-PID TERM of recorded owned processes (timeout wrappers).
set -uo pipefail
RUN=${RUN_A:?}                # sanitization: private evidence dir (was hardcoded path)
BIN=${BIN_ENGINE_A:?}         # sanitization: private engine executable path
SHARD=${NATIVE_SHARD_A:?}     # sanitization: private native reference input path
PACK=${NATIVE_PACK_A:?}       # sanitization: private native pack path
MANIFEST=${MANIFEST_A:?}      # sanitization: private SWLOCAL1 manifest path
PROFILE=${PROFILE_A:?}        # sanitization: private trusted profile path
GPU_PLAN=$RUN/plan_B.bin
REQ=$RUN/requests.in
GLOBAL_T0=${GLOBAL_T0:?}
STOP_AT=$(( GLOBAL_T0 + 5400 - 45 ))
mkdir -p "$RUN"; cd "$RUN"
log(){ echo "[$(date -u +%FT%TZ) $(date +%s)] $*" >> DRIVER03.log; }

{
  echo "utc=$(date -u +%FT%TZ) global_t0=$GLOBAL_T0 stop_at=$STOP_AT"
  sha256sum "$BIN" "$MANIFEST" "$PROFILE" "$GPU_PLAN" "$SHARD"
  git -C "${SRC_ENGINE_A:?}" rev-parse HEAD  # sanitization: was a private source-checkout path
} > identity.txt
printf 'GEN 64 9419\nGEN 64 14773,6353\nGEN 64 9419\nQUIT\n' > "$REQ"
sha256sum "$REQ" > requests.sha256
nvidia-smi --query-gpu=uuid,memory.used,memory.free --format=csv,noheader > gpu-baseline.csv
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader > gpu-baseline-apps.csv 2>&1 || true
grep -E 'MemAvailable|SwapFree' /proc/meminfo > mem-baseline.txt
EVT_BASE=$(awk '/^oom/{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0)
VMRSS_CEIL=$(( 24 * 1048576 ))
RSSANON_CEIL=$(( 8 * 1048576 ))
HOST_MIN=$(( 12 * 1048576 ))

check_admission(){
  local AV S SB EO
  AV=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$AV" -lt "$HOST_MIN" ] && { log "HOST-ADMISSION-FAIL MemAvailable=${AV}kB < 12GiB"; return 1; }
  S=$(awk '/SwapFree/{print $2}' /proc/meminfo)
  SB=$(awk '/SwapFree/{print $2}' mem-baseline.txt)
  [ "$S" -lt $(( SB - 65536 )) ] && { log "NEW-SWAP-STOP ${S}kB base=${SB}kB"; return 1; }
  EO=$(awk '/^oom/{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0)
  [ "$EO" -gt "$EVT_BASE" ] && { log "OOM-EVENT-STOP oom=$EO"; return 1; }
  return 0
}

sampler_pid=0
start_sampler(){
  local tag=$1 ep=$2
  sampler_pid=$( { while :; do
    CHILD=$(pgrep -P "$ep" 2>/dev/null | head -1)
    V=$(awk '/VmRSS/{print $2}' /proc/$CHILD/status 2>/dev/null || echo 0)
    A=$(awk '/RssAnon/{print $2}' /proc/$CHILD/status 2>/dev/null || echo 0)
    F=$(awk '/RssFile/{print $2}' /proc/$CHILD/status 2>/dev/null || echo 0)
    M=$(awk '/RssShmem/{print $2}' /proc/$CHILD/status 2>/dev/null || echo 0)
    MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    SF=$(awk '/SwapFree/{print $2}' /proc/meminfo)
    echo "$(date +%s) ${CHILD:-0} $V $A $F $M $MA $SF"
    if [ "$V" -gt "$VMRSS_CEIL" ]; then echo "$(date +%s) VMRSS-CEILING $V" >> $tag.violations.txt; kill -TERM "$ep" 2>/dev/null; break; fi
    if [ "$A" -gt "$RSSANON_CEIL" ]; then echo "$(date +%s) RSSANON-CEILING $A" >> $tag.violations.txt; kill -TERM "$ep" 2>/dev/null; break; fi
    if [ "$MA" -lt $(( 4 * 1048576 )) ]; then echo "$(date +%s) HEADROOM-STOP $MA" >> $tag.violations.txt; kill -TERM "$ep" 2>/dev/null; break; fi
    kill -0 "$ep" 2>/dev/null || break
    sleep 2
  done; } > $tag.samples.tsv & echo $! )
}
stop_sampler(){ [ "$sampler_pid" -gt 0 ] && kill -TERM "$sampler_pid" 2>/dev/null; wait "$sampler_pid" 2>/dev/null; sampler_pid=0; return 0; }

run_engine(){
  local tag=$1 cap_s=$2
  shift 2
  date +%s.%N > $tag.launch.epoch
  env OMP_NUM_THREADS=2 STRATA_IO_PF_THREADS=2 STRATA_ROUTE_TAIL_SKIP=0 STRATA_LOOKAHEAD=0 \
    STRATA_LOOKAHEAD_STATS=0 STRATA_FS_SLOTS=0 \
    STRATA_DUMP_FIRST_LOGITS="$RUN/$tag.first-logits" \
    timeout --signal=TERM --kill-after=15s $(( cap_s - 15 )) \
    "$BIN" --serve --native "$SHARD" --pack "$PACK" --gpu "$GPU_SELECT_A" --mmap-experts --greedy --pool-workers 2 \
    --remote-cpu-experts "$ENDPOINT_CPU_C_ON_A" --remote-cpu-deadline-ms 10000 \
    --remote-cpu-local-manifest "$MANIFEST" --remote-cpu-resident --remote-cpu-setup-ms 600000 \
    --expert-cache 1056 --expert-cache-per-layer --expert-profile "$PROFILE" \
    --expert-cache-device1 2263 --remote-expert-opt --vram-reserve-mib 1152 \
    --adapt-every 0 --adapt-swaps 0 --pcie-frac 0 --short-read 64 --prefill 64 \
    --prompt-cache 0 --spec 2 --suffix-draft 0 --lookup-chain 0 --pipeline-windows 0 --max-context 1024 \
    --dump-routing "$RUN/$tag.routing.bin" "$@" \
    < "$REQ" > "$RUN/$tag.out" 2> "$RUN/$tag.log" &
  local EPID=$!
  echo "$EPID" > $tag.engine.pid
  start_sampler "$tag" "$EPID"
  wait "$EPID"
  echo $? > $tag.rc
  date +%s.%N > $tag.end.epoch
  stop_sampler
}

after_phase(){
  local tag=$1 RC
  RC=$(cat $tag.rc)
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader > $tag.gpu-apps-after.csv 2>&1 || true
  nvidia-smi --query-gpu=uuid,memory.used,memory.free --format=csv,noheader > $tag.gpu-after.csv
  grep -E 'MemAvailable|SwapFree' /proc/meminfo > $tag.mem-after.txt
  log "PHASE-$tag rc=$RC gpu_apps_lines=$(wc -l < $tag.gpu-apps-after.csv)"
  [ "$RC" -eq 0 ]
}

# ---------- B0 ----------
[ -f STOP-03 ] && { log "STOP-FILE before B0"; exit 9; }
check_admission || exit 4
log "B0 start (cap 1200)"
run_engine B0 1200
after_phase B0 || { log "B0 FAILED - stopping dependent phases"; echo B0-FAILED > PHASE-VERDICT; exit 6; }
sleep 11
# ---------- B1 ----------
[ -f STOP-03 ] && { log "STOP-FILE before B1"; exit 9; }
check_admission || exit 4
log "B1 start (cap 600)"
run_engine B1 600
after_phase B1 || { log "B1 FAILED - stopping dependent phases"; echo B1-FAILED > PHASE-VERDICT; exit 6; }
echo B-PHASES-DONE > PHASE-VERDICT
log "B phases complete; waiting for G0-GATE"
# ---------- G0 ----------
GATE_DEADLINE=$(( STOP_AT - 1900 ))
while [ ! -f G0-GATE ]; do
  [ "$(date +%s)" -gt "$GATE_DEADLINE" ] && { log "G0-GATE timeout"; echo G0-GATE-TIMEOUT >> PHASE-VERDICT; exit 7; }
  [ -f STOP-03 ] && { log "STOP-FILE at G0 gate"; exit 9; }
  sleep 2
done
check_admission || exit 4
log "G0 start (cap 1800)"
run_engine G0 1800 --remote-gpu-experts "$ENDPOINT_GPU_B_ON_A" \
  --remote-gpu-plan "$GPU_PLAN" --remote-gpu-setup-ms 1200000 --remote-gpu-deadline-ms 10000
after_phase G0 || { log "G0 FAILED - stopping dependent phases"; echo G0-FAILED >> PHASE-VERDICT; exit 6; }
echo G0-DONE >> PHASE-VERDICT
log "G0 complete; waiting for G1-GATE"
# ---------- G1 ----------
GATE_DEADLINE=$(( STOP_AT - 1000 ))
while [ ! -f G1-GATE ]; do
  [ "$(date +%s)" -gt "$GATE_DEADLINE" ] && { log "G1-GATE timeout"; echo G1-GATE-TIMEOUT >> PHASE-VERDICT; exit 7; }
  [ -f STOP-03 ] && { log "STOP-FILE at G1 gate"; exit 9; }
  sleep 2
done
check_admission || exit 4
log "G1 start (cap 900)"
run_engine G1 900 --remote-gpu-experts "$ENDPOINT_GPU_B_ON_A" \
  --remote-gpu-plan "$GPU_PLAN" --remote-gpu-setup-ms 1200000 --remote-gpu-deadline-ms 10000
after_phase G1 || { log "G1 FAILED"; echo G1-FAILED >> PHASE-VERDICT; exit 6; }
echo G1-DONE >> PHASE-VERDICT
# ---------- final ----------
nvidia-smi --query-gpu=uuid,memory.used,memory.free --format=csv,noheader > gpu-final.csv
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader > gpu-final-apps.csv 2>&1 || true
grep -E 'MemAvailable|SwapFree' /proc/meminfo > mem-final.txt
echo "events_end oom=$(awk '/^oom/{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)" >> mem-final.txt
log "ALL-PHASES-COMPLETE"
exit 0
