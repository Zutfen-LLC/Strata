#!/bin/bash
# --- sanitized historical source (supervisor-source-v1) ----------------------
# Strata bf51f09e full-model trial - ROLE_CONTROLLER ORCHESTRATOR (v3, host
# identity removed). Sanitized excerpt of the private original orchestrate.sh:
# NOT a byte-identical original and NOT runnable deployment instructions.
# Lines tagged "# sanitization:" are the only intentional textual edits; all
# logic, thresholds, units, ordering, error paths and original defects are
# preserved. The private relay route (ssh hosts, transport session names and
# user) is removed; the original is documented in the header comment below.
# -----------------------------------------------------------------------------
# One global monotonic 5400s wall. Serial B0/B1/G0/G1 with gates.
# Relay (two-session transport bridge, controller-mediated) - PRIVATE, REMOVED:
#   ROLE_ENGINE_A:CPU_PORT_C (-R on one transport session) -> ROLE_CONTROLLER:CTRL_PORT_C_FWD -> ROLE_CPU_C:CPU_PORT_C
#   ROLE_ENGINE_A:GPU_PORT_B (-R on a second transport session) -> ROLE_CONTROLLER:CTRL_PORT_B_FWD -> ROLE_GPU_B:GPU_PORT_B
#   # sanitization: original concrete ssh targets/users/ports replaced by role variables
set -uo pipefail
CT=${RUN_CTRL:?}               # sanitization: private controller evidence dir (was hardcoded path)
R03=${RUN_A:?}                 # sanitization: private engine-role evidence dir
R04=${RUN_B:?}                 # sanitization: private gpu-worker-role evidence dir
R05=${RUN_C:?}                 # sanitization: private cpu-worker-role evidence dir
SSH_ENGINE_A=${SSH_ENGINE_A:?}   # sanitization: private transport command for ROLE_ENGINE_A
SSH_GPU_B=${SSH_GPU_B:?}         # sanitization: private transport command for ROLE_GPU_B
SSH_CPU_C=${SSH_CPU_C:?}         # sanitization: private transport command for ROLE_CPU_C
CTRL_PORT_C_FWD=${CTRL_PORT_C_FWD:?}  # sanitization: private controller-side forward port for the CPU relay
CTRL_PORT_B_FWD=${CTRL_PORT_B_FWD:?}  # sanitization: private controller-side forward port for the GPU relay
GLOBAL_T0=$(date +%s)
STOP_AT=$(( GLOBAL_T0 + 5400 - 45 ))
echo "GLOBAL_T0=$GLOBAL_T0 stop_at=$STOP_AT utc=$(date -u -d @$GLOBAL_T0 +%FT%TZ)" | tee $CT/T0.txt
log(){ echo "[$(date -u +%FT%TZ) elapsed=$(( $(date +%s) - GLOBAL_T0 ))s] $*" | tee -a $CT/orchestrator.log; }
remain(){ echo $(( STOP_AT - $(date +%s) )); }
SSHOPT="-o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4"

abort(){
  log "ABORT: $*"
  echo "ABORT: $*" > $CT/ORCH-STATUS
  $SSH_ENGINE_A "touch $R03/STOP-03" 2>/dev/null
  $SSH_GPU_B "bash $R04/driver04.sh stop G0; bash $R04/driver04.sh stop G1" >/dev/null 2>&1
  # error cleanup: owned forwards + owned cpu-worker wrapper (no retention on failure)
  for f in fwdA_CPU_C fwdA_GPU_B fwdB_CPU_C fwdB_GPU_B; do
    P=$(cat $CT/runctl/$f.pid 2>/dev/null || true)
    [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null
  done
  $SSH_CPU_C "W=\$(cat $R05/worker-wrapper.pid 2>/dev/null); [ -n \"\$W\" ] && kill -TERM \$W 2>/dev/null" 2>/dev/null
  exit 1
}
mkdir -p $CT/runctl
[ "$(remain)" -gt 4900 ] || abort "wall misconfigured at start"

# ---------- bridge relays ----------
setsid nohup ssh -N $SSHOPT -L ${CTRL_PORT_C_FWD}:127.0.0.1:${CPU_PORT_C} $SSH_CPU_C_HOST_ARG > $CT/runctl/fwdA_CPU_C.log 2>&1 < /dev/null &
echo $! > $CT/runctl/fwdA_CPU_C.pid
setsid nohup ssh -N $SSHOPT -R ${CPU_PORT_C}:127.0.0.1:${CTRL_PORT_C_FWD} $SSH_ENGINE_A_HOST_ARG > $CT/runctl/fwdB_CPU_C.log 2>&1 < /dev/null &
echo $! > $CT/runctl/fwdB_CPU_C.pid
setsid nohup ssh -N $SSHOPT -L ${CTRL_PORT_B_FWD}:127.0.0.1:${GPU_PORT_B} $SSH_GPU_B_HOST_ARG > $CT/runctl/fwdA_GPU_B.log 2>&1 < /dev/null &
echo $! > $CT/runctl/fwdA_GPU_B.pid
setsid nohup ssh -N $SSHOPT -R ${GPU_PORT_B}:127.0.0.1:${CTRL_PORT_B_FWD} $SSH_ENGINE_A_HOST_ARG > $CT/runctl/fwdB_GPU_B.log 2>&1 < /dev/null &
echo $! > $CT/runctl/fwdB_GPU_B.pid
sleep 4
L1=$($SSH_ENGINE_A 'ss -tln | grep -c ":${CPU_PORT_C} "' || true)
L2=$($SSH_ENGINE_A 'ss -tln | grep -c ":${GPU_PORT_B} "' || true)
log "relay listeners on engine role: cpu=$L1 gpu=$L2"
{ [ "${L1:-0}" -ge 1 ] && [ "${L2:-0}" -ge 1 ]; } || abort "relay listeners missing on engine role"

w05(){ $SSH_CPU_C "$@"; }
w03(){ $SSH_ENGINE_A "$@"; }
w04(){ $SSH_GPU_B "$@"; }
status05(){ w05 "${WORKER_CONTROL_CPU_C:?} status --port ${CPU_PORT_C}" ; }  # sanitization: was a hardcoded private control-tool path

# ---------- launch cpu-worker driver (fresh arena) ----------
log "launching cpu-worker driver"
w05 "setsid nohup bash $R05/driver05.sh > $R05/driver05.out 2>&1 < /dev/null & echo \$!" | tail -1 > $CT/runctl/drv05.pid
sleep 5
ST=$(w05 "cat $R05/STATUS 2>/dev/null" || echo missing)
[ "$ST" = LAUNCHED ] || abort "cpu-worker driver STATUS=$ST"
for i in $(seq 1 90); do
  W=$(w05 "grep -c 'strata-worker: listening' $R05/worker.log 2>/dev/null || echo 0")
  [ "${W:-0}" -ge 1 ] && break
  [ "$(remain)" -gt 4600 ] || abort "cpu worker never listened"
  sleep 5
done
[ "${W:-0}" -ge 1 ] || abort "cpu worker not listening"
log "cpu worker listening (arena loads during B0 setup)"

# ---------- launch engine driver ----------
log "launching engine driver GLOBAL_T0=$GLOBAL_T0"
w03 "GLOBAL_T0=$GLOBAL_T0 setsid nohup bash $R03/driver03.sh > $R03/driver03.out 2>&1 < /dev/null & echo \$!" | tail -1 > $CT/runctl/drv03.pid
B0S=0; B1S=0
for i in $(seq 1 500); do
  V=$(w03 "cat $R03/PHASE-VERDICT 2>/dev/null" || echo "")
  if [ "$B0S" = 0 ] && w03 "test -f $R03/B0.rc"; then B0S=1; status05 > $CT/status-after-B0.json 2> $CT/status-after-B0.err; log "B0 complete (rc=$(w03 "cat $R03/B0.rc"))"; fi
  if [ "$B1S" = 0 ] && w03 "test -f $R03/B1.rc"; then B1S=1; status05 > $CT/status-after-B1.json 2> $CT/status-after-B1.err; log "B1 complete (rc=$(w03 "cat $R03/B1.rc"))"; fi
  [ "$V" = "B-PHASES-DONE" ] && break
  echo "$V" | grep -q FAIL && abort "engine-role B-phase failure: $V"
  [ "$(remain)" -gt 2450 ] || abort "B phases timeout"
  sleep 5
done
V=$(w03 "cat $R03/PHASE-VERDICT 2>/dev/null" || echo "")
[ "$V" = "B-PHASES-DONE" ] || abort "B phases incomplete: $V"
log "B phases DONE B0.rc=$(w03 "cat $R03/B0.rc") B1.rc=$(w03 "cat $R03/B1.rc")"

# ---------- G0 ----------
[ "$(remain)" -gt 2350 ] || abort "insufficient wall for G0"
w04 "bash $R04/driver04.sh launch G0" > $CT/G0-worker-launch.out 2>&1
grep -q WORKER-READY $CT/G0-worker-launch.out || { cat $CT/G0-worker-launch.out; abort "gpu-role G0 worker not ready"; }
log "gpu-role G0 worker READY (fresh empty cache-B); opening G0 gate"
w03 "touch $R03/G0-GATE"
G0S=0
for i in $(seq 1 420); do
  V=$(w03 "cat $R03/PHASE-VERDICT 2>/dev/null" || echo "")
  if [ "$G0S" = 0 ] && w03 "test -f $R03/G0.rc"; then G0S=1; status05 > $CT/status-after-G0.json 2> $CT/status-after-G0.err; log "G0 engine complete (rc=$(w03 "cat $R03/G0.rc"))"; fi
  echo "$V" | grep -q "G0-DONE" && break
  echo "$V" | grep -q FAIL && abort "G0 failed: $V"
  [ "$(remain)" -gt 640 ] || abort "G0 timeout"
  sleep 5
done
V=$(w03 "cat $R03/PHASE-VERDICT 2>/dev/null" || echo "")
echo "$V" | grep -q "G0-DONE" || abort "G0 incomplete: $V"
w04 "bash $R04/driver04.sh stop G0" > $CT/G0-worker-final.out 2>&1
log "gpu-role G0 worker stopped (final counters in G0-worker-final.out); 11s gap"
sleep 11
# ---------- G1 ----------
[ "$(remain)" -gt 600 ] || abort "insufficient wall for G1"
w04 "bash $R04/driver04.sh launch G1" > $CT/G1-worker-launch.out 2>&1
grep -q WORKER-READY $CT/G1-worker-launch.out || { cat $CT/G1-worker-launch.out; abort "gpu-role G1 worker not ready"; }
w03 "touch $R03/G1-GATE"
log "G1 gate open (fresh gpu-role process, warm cache-B)"
G1S=0
for i in $(seq 1 220); do
  V=$(w03 "cat $R03/PHASE-VERDICT 2>/dev/null" || echo "")
  if [ "$G1S" = 0 ] && w03 "test -f $R03/G1.rc"; then G1S=1; status05 > $CT/status-after-G1.json 2> $CT/status-after-G1.err; log "G1 engine complete (rc=$(w03 "cat $R03/G1.rc"))"; fi
  echo "$V" | grep -q "G1-DONE" && break
  echo "$V" | grep -q FAIL && abort "G1 failed: $V"
  [ "$(remain)" -gt 90 ] || abort "G1 timeout"
  sleep 5
done
V=$(w03 "cat $R03/PHASE-VERDICT 2>/dev/null" || echo "")
echo "$V" | grep -q "G1-DONE" || abort "G1 incomplete: $V"
w04 "bash $R04/driver04.sh stop G1" > $CT/G1-worker-final.out 2>&1
log "gpu-role G1 worker stopped"

# ---------- close unused GPU forwards; keep CPU control transport ----------
P=$(cat $CT/runctl/fwdA_GPU_B.pid 2>/dev/null); [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null
P=$(cat $CT/runctl/fwdB_GPU_B.pid 2>/dev/null); [ -n "${P:-}" ] && kill -TERM "$P" 2>/dev/null
log "GPU forwards closed; CPU control transport retained with cpu worker"
echo COMPLETE > $CT/ORCH-STATUS
log "ORCHESTRATION COMPLETE - cpu worker retained per approval; analysis next"
exit 0
