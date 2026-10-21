#!/bin/bash
# --- sanitized historical source (supervisor-source-v1) ----------------------
# Strata bf51f09e full-model trial - ROLE_CPU_C driver (host identity removed).
# Sanitized excerpt of the private original driver05.sh: NOT a byte-identical
# original and NOT runnable deployment instructions. Lines tagged
# "# sanitization:" are the only intentional textual edits; all logic,
# thresholds, units, ordering, error paths and original defects are preserved,
# including the strict SwapFree < admission_baseline - 65536 KiB stop test and
# the original cgroup-guard behavior. Original line ranges are given in
# ../SOURCE-MAP (README).
# -----------------------------------------------------------------------------
# Bounded: worker absolute cap 7185s (timeout TERM/15s). Fresh full arena.
# Retention after trial success is intentional: this driver and the worker's own
# supervisor enforce the idle-TTL (1800000ms) and absolute (7185s) bounds.
set -uo pipefail
RUN=${RUN_C:?}              # sanitization: private evidence dir (was hardcoded path)
BIN=${BIN_CPU_C:?}          # sanitization: private worker executable path
MANIFEST=${MANIFEST_C:?}    # sanitization: private SWLOCAL1 manifest path
ROOT=${LOCAL_ROOT_C:?}      # sanitization: private CPU tensor-source root
PORT=${CPU_PORT_C:?}        # sanitization: private approved loopback port
mkdir -p "$RUN"
cd "$RUN"
echo "DRIVER05-START utc=$(date -u +%FT%TZ) epoch=$(date +%s)" > DRIVER05.log

# --- identity receipts ---
{
  echo "utc=$(date -u +%FT%TZ)"
  sha256sum "$BIN" "$MANIFEST"
  echo "src_note=<private build receipt omitted during sanitization>"  # sanitization: was a private build-receipt id
} > identity.txt

# --- no existing arena / no listener ---
if ss -tln | grep -q ":$PORT "; then { echo "PORT-$PORT-BUSY - refusing to duplicate/take over"; echo PORT-BUSY > STATUS; exit 5; } >> DRIVER05.log 2>&1; fi
pgrep -f "strata-expert-worker --port $PORT" > /dev/null && { echo "WORKER-ALREADY-RUNNING"; echo WORKER-EXISTS > STATUS; exit 6; } >> DRIVER05.log 2>&1  # sanitization: port was hardcoded

# --- admission: >=120.4921875 GiB host AND tightest-cgroup headroom ---
AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
NEED_KB=$(( 126345216 ))   # 120.4921875 GiB in KiB
CGMAX=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
CGUSED=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
if [ "$AVAIL_KB" -lt "$NEED_KB" ]; then
  echo "ADMISSION-REFUSED MemAvailable_kb=$AVAIL_KB need_kb=$NEED_KB" | tee -a DRIVER05.log
  echo ADMISSION-REFUSED > STATUS; exit 4
fi
if [ "$CGMAX" != "max" ]; then
  CGROOM=$(( CGMAX - CGUSED ))
  [ "$CGROOM" -lt "$NEED_KB" ] && { echo "CGROUP-REFUSED room_kb=$CGROOM"; echo ADMISSION-REFUSED > STATUS; exit 4; }
fi
{
  date -u +"admission_utc=%FT%TZ"
  echo "MemAvailable_kb=$AVAIL_KB need_kb=$NEED_KB cgroup_max=$CGMAX"
  grep -E 'MemTotal|^Cached:|Buffers:|SwapTotal|SwapFree' /proc/meminfo
  echo "events_start: $(tr '\n' ' ' < /sys/fs/cgroup/memory.events 2>/dev/null)"
} > admission.txt
SWAP_BASE=$(awk '/SwapFree/{print $2}' /proc/meminfo)
EVT_BASE=$(awk '/^oom/{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0)
EVT_HIGH_BASE=$(awk '/^high/{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0)

# --- launch: exact budget CLI, detached, closed stdin, owned supervisor ---
date +%s > worker-launch.epoch
setsid nohup timeout --signal=TERM --kill-after=15s 7185 \
  env OMP_NUM_THREADS=2 "$BIN" --port "$PORT" --workers 2 \
  --memory-budget-mib 105870 --deadline-ms 10000 --local-setup-ms 600000 \
  --local-root "$ROOT" --local-manifest "$MANIFEST" --residency-ms 1800000 \
  < /dev/null > "$RUN/worker.out" 2> "$RUN/worker.log" &
WPID=$!
echo "$WPID" > worker-wrapper.pid
sleep 2
CHILD=$(pgrep -P "$WPID" | head -1)
[ -z "$CHILD" ] && CHILD=$(pgrep -P "$WPID" | head -1)
{
  echo "wrapper_pid=$WPID child_pid=$CHILD"
  echo "wrapper_start=$(stat -c %Y /proc/$WPID 2>/dev/null)"
  echo "child_start=$(stat -c %Y /proc/$CHILD 2>/dev/null)"
  echo "child_exe=$(readlink /proc/$CHILD/exe 2>/dev/null)"
  echo "child_cmd=$(tr '\0' ' ' < /proc/$CHILD/cmdline 2>/dev/null)"
  echo "absolute_cap_epoch=$(( $(cat worker-launch.epoch) + 7185 ))"
  echo "absolute_cap_utc=$(date -u -d @$(( $(cat worker-launch.epoch) + 7185 )) +%FT%TZ)"
  echo "idle_ttl_s=1800 (worker-enforced, NOT renewed by status probes)"
} > worker-identity.txt
echo LAUNCHED > STATUS
echo "WORKER_LAUNCHED wrapper=$WPID child=$CHILD" >> DRIVER05.log

# --- monitor: ~2s samples; ceilings RSS<=108024 MiB, headroom>=8 GiB, no new swap ---
RSS_CEIL_KB=$(( 108024 * 1024 ))
HEADROOM_KB=$(( 8 * 1048576 ))
echo "epoch phase pid VmRSS_kb RssAnon_kb RssFile_kb RssShmem_kb MemAvail_kb SwapFree_kb" > samples.tsv
VIOL=0
while :; do
  NOW=$(date +%s)
  EL=$(( NOW - $(cat worker-launch.epoch) ))
  [ "$EL" -ge 7245 ] && break
  kill -0 "$WPID" 2>/dev/null || { echo "WRAPPER-EXIT at ${EL}s" >> DRIVER05.log; break; }
  PID=$CHILD
  [ -n "$PID" ] && [ -d /proc/$PID ] || PID=$(pgrep -P "$WPID" | head -1)
  VRSS=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
  RANON=$(awk '/RssAnon/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
  RFIL=$(awk '/RssFile/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
  RSHM=$(awk '/RssShmem/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
  MA=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  SF=$(awk '/SwapFree/{print $2}' /proc/meminfo)
  echo "$NOW run $PID $VRSS $RANON $RFIL $RSHM $MA $SF" >> samples.tsv
  if [ "$VRSS" -gt "$RSS_CEIL_KB" ]; then echo "$NOW RSS-CEILING-EXCEEDED ${VRSS}kB" >> violations.txt; VIOL=1; fi
  if [ "$MA" -lt "$HEADROOM_KB" ]; then echo "$NOW HEADROOM-STOP MemAvail=${MA}kB" >> violations.txt; VIOL=1; fi
  if [ "$SF" -lt $(( SWAP_BASE - 65536 )) ]; then echo "$NOW NEW-SWAP SwapFree=${SF}kB base=${SWAP_BASE}kB" >> violations.txt; VIOL=1; fi
  EO=$(awk '/^oom/{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0)
  EH=$(awk '/^high/{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null || echo 0)
  { [ "$EO" -gt "$EVT_BASE" ] || [ "$EH" -gt "$EVT_HIGH_BASE" ]; } && { echo "$NOW CGROUP-EVENT oom=$EO high=$EH" >> violations.txt; VIOL=1; }
  if [ "$VIOL" -ne 0 ]; then
    echo "RESOURCE-STOP violation logged - TERM owned worker" >> DRIVER05.log
    kill -TERM "$WPID" 2>/dev/null
    echo RESOURCE-STOP > STATUS
    break
  fi
  sleep 2
done
# final state after loop (normal expiry or stop)
sleep 3
date -u +"monitor_end=%FT%TZ" >> DRIVER05.log
echo "events_end: $(tr '\n' ' ' < /sys/fs/cgroup/memory.events 2>/dev/null)" >> admission.txt
pgrep -f "strata-expert-worker --port $PORT" > /dev/null && echo "WORKER-STILL-RESIDENT $(pgrep -f "strata-expert-worker --port $PORT" | head -1)" >> DRIVER05.log || echo "WORKER-GONE" >> DRIVER05.log  # sanitization: port was hardcoded
echo MONITOR-DONE >> DRIVER05.log
