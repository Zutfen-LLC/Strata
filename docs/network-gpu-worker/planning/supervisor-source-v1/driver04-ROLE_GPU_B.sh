#!/bin/bash
# --- sanitized historical source (supervisor-source-v1) ----------------------
# Strata bf51f09e full-model trial - ROLE_GPU_B driver (host identity removed).
# Sanitized excerpt of the private original driver04.sh: NOT a byte-identical
# original and NOT runnable deployment instructions. Lines tagged
# "# sanitization:" are the only intentional textual edits; all logic,
# thresholds, units, ordering, error paths and original defects are preserved.
# -----------------------------------------------------------------------------
# Usage: driver04.sh launch G0|G1   (fresh owned process per phase, same cache)
#        driver04.sh stop  G0|G1    (TERM owned wrapper, collect final counters)
# G0: new EMPTY owned cache at CACHE_B (cap 18432 MiB). G1: REUSE warm CACHE_B.
set -uo pipefail
RUN=${RUN_B:?}                 # sanitization: private evidence dir (was hardcoded path)
BIN=${BIN_GPU_B:?}             # sanitization: private worker executable path
PLAN=$RUN/plan_B.bin
CACHE=${CACHE_B:?}             # sanitization: private bounded cache dir
PORT=${GPU_PORT_B:?}           # sanitization: private approved loopback port
PHASE=${2:-}
ACTION=${1:-}
mkdir -p "$RUN"; cd "$RUN"
EXPECT_PLAN=$EXPECTED_GPU_PLAN_A_SHA  # sanitization: integrity pin removed (public docs must not carry placement proof)
case "$ACTION" in
launch)
  [ "$PHASE" = G0 ] || [ "$PHASE" = G1 ] || { echo "bad phase"; exit 2; }
  [ "$PHASE" = G0 ] && CAP=1900 || CAP=1000
  {
    echo "utc=$(date -u +%FT%TZ) phase=$PHASE cap_s=$CAP"
    sha256sum "$BIN" "$PLAN"
  } > $PHASE.identity.txt
  GOT=$(sha256sum "$PLAN" | awk '{print $1}')
  [ "$GOT" = "$EXPECT_PLAN" ] || { echo "PLAN-HASH-MISMATCH $GOT" | tee $PHASE.launch.log; exit 4; }
  if ss -tln | grep -q ":$PORT "; then echo "PORT-$PORT-BUSY" | tee -a $PHASE.launch.log; exit 5; fi
  # host/cgroup admission >= 4 GiB
  AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$AVAIL_KB" -lt $(( 4 * 1048576 )) ] && { echo "HOST-ADMISSION-REFUSED ${AVAIL_KB}kB" | tee -a $PHASE.launch.log; exit 4; }
  # disk >= 22.25 GiB (G0 creates cache) / >= 4 GiB ongoing
  if [ "$PHASE" = G0 ]; then
    DISK_NEED=$(( 23890755584 ))
    [ -e "$CACHE" ] && { echo "CACHE-EXISTS-REFUSING-NEW $CACHE" | tee -a $PHASE.launch.log; exit 7; }
    mkdir -p "$CACHE"
  else
    DISK_NEED=$(( 4 * 1073741824 ))
    [ -d "$CACHE" ] || { echo "WARM-CACHE-MISSING $CACHE" | tee -a $PHASE.launch.log; exit 7; }
  fi
  DISK_FREE=$(df -B1 --output=avail "$RUN" | tail -1)
  [ "$DISK_FREE" -lt "$DISK_NEED" ] && { echo "DISK-REFUSED free=$DISK_FREE need=$DISK_NEED" | tee -a $PHASE.launch.log; exit 4; }
  # GPU provisional admission: floor((F-4096MiB-256MiB-1598720)/5222400) >= 3289
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  F_BYTES=$(( F * 1048576 ))
  ALLOW=$(( (F_BYTES - 4096*1048576 - 256*1048576 - 1598720) / 5222400 ))
  {
    echo "gpu_free_mib=$F allowed_experts=$ALLOW need=3289"
    echo "MemAvailable_kb=$AVAIL_KB disk_free_b=$DISK_FREE disk_need_b=$DISK_NEED"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  } > $PHASE.admission.txt
  [ "$ALLOW" -lt 3289 ] && { echo "GPU-ADMISSION-REFUSED allowed=$ALLOW" | tee -a $PHASE.launch.log; exit 4; }
  # launch: exact budget CLI (reserve-mib 4096 per budget line for ROLE_GPU_B)
  date +%s > $PHASE.launch.epoch
  setsid nohup timeout --signal=TERM --kill-after=15s $(( CAP - 15 )) \
    env CUDA_VISIBLE_DEVICES=$GPU_SELECT_B_ORDINAL OMP_NUM_THREADS=2 \
    "$BIN" --plan "$PLAN" --cache-root "$CACHE" \
    --port "$PORT" --device 0 --cache-budget-mib 18432 --reserve-mib 4096 \
    --deadline-ms 10000 --setup-ms 1200000 \
    < /dev/null > "$RUN/$PHASE.worker.out" 2> "$RUN/$PHASE.worker.log" &
  WPID=$!
  echo "$WPID" > $PHASE.wrapper.pid
  sleep 2
  CHILD=$(pgrep -P "$WPID" | head -1)
  [ -z "$CHILD" ] && CHILD=$(pgrep -P "$WPID" | head -1)
  {
    echo "wrapper_pid=$WPID child_pid=$CHILD"
    echo "child_start=$(stat -c %Y /proc/$CHILD 2>/dev/null)"
    echo "child_exe=$(readlink /proc/$CHILD/exe 2>/dev/null)"
    echo "cap_s=$CAP"
  } > $PHASE.identity.txt.append
  cat $PHASE.identity.txt.append >> $PHASE.identity.txt
  # readiness: port listening (loopback)
  for i in $(seq 1 150); do
    ss -tln | grep -q ":$PORT " && { echo READY > $PHASE.STATUS; break; }
    sleep 0.2
  done
  [ -f $PHASE.STATUS ] || { echo NOT-READY > $PHASE.STATUS; tail -5 "$RUN/$PHASE.worker.log"; exit 1; }
  echo "WORKER-READY phase=$PHASE wrapper=$WPID child=$CHILD allowed=$ALLOW"
  # background 2s sampler until wrapper exit (RSS<=2GiB, disk>=4GiB, gpu free>=4096MiB post-READY)
  (
    RSS_CEIL=$(( 2 * 1048576 ))
    echo "epoch pid VmRSS_kb RssAnon_kb gpu_free_mib disk_free_b" > $PHASE.samples.tsv
    while kill -0 "$WPID" 2>/dev/null; do
      PID=$(pgrep -P "$WPID" | head -1)
      V=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
      A=$(awk '/RssAnon/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
      G=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      D=$(df -B1 --output=avail "$RUN" | tail -1)
      echo "$(date +%s) $PID $V $A $G $D" >> $PHASE.samples.tsv
      if [ "$V" -gt "$RSS_CEIL" ]; then echo "$(date +%s) RSS-CEILING $V" >> $PHASE.violations.txt; kill -TERM "$WPID"; break; fi
      if [ "$D" -lt $(( 4 * 1073741824 )) ]; then echo "$(date +%s) DISK-STOP $D" >> $PHASE.violations.txt; kill -TERM "$WPID"; break; fi
      sleep 2
    done
  ) > /dev/null 2>&1 &
  echo $! > $PHASE.sampler.pid
  ;;
stop)
  [ "$PHASE" = G0 ] || [ "$PHASE" = G1 ] || { echo "bad phase"; exit 2; }
  WPID=$(cat $PHASE.wrapper.pid 2>/dev/null || echo 0)
  SPID=$(cat $PHASE.sampler.pid 2>/dev/null || echo 0)
  [ "$WPID" -gt 0 ] && kill -TERM "$WPID" 2>/dev/null
  for i in $(seq 1 40); do kill -0 "$WPID" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$WPID" 2>/dev/null && kill -KILL "$WPID" 2>/dev/null
  [ "$SPID" -gt 0 ] && kill -TERM "$SPID" 2>/dev/null
  sleep 2
  {
    echo "utc=$(date -u +%FT%TZ)"
    echo "final_counters:"; grep -E "cuda-worker stopped|READY" "$RUN/$PHASE.worker.log" | tail -3
    echo "gpu_after:"; nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    echo "compute_apps_after:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    echo "worker_procs_after: $(pgrep -P $(cat $PHASE.wrapper.pid) | tr \n\  )"
    echo "cache_bytes: $(du -sb $CACHE 2>/dev/null | awk '{print $1}')"
    echo "cache_files: $(find $CACHE -type f | wc -l)"
  } > $PHASE.final.txt 2>&1
  cat $PHASE.final.txt
  ;;
*) echo "usage: driver04.sh launch|stop G0|G1"; exit 2;;
esac
