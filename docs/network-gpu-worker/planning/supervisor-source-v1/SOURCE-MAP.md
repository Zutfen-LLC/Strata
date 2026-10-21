# Source map: public sanitized files to original structure

Neutral labels only. Real file paths, hostnames and the private
original→sanitized byte mapping are held in the private transformation record
off-GitHub. "Complete" means every original line is represented; the only
textual differences are the tagged `# sanitization:` identifier edits listed
in each file's header and inline tags.

## driver05-ROLE_CPU_C.sh  (original: ROLE_CPU_C driver, 113 lines, complete)

| Function/section | Original lines | Notes |
| --- | ---: | --- |
| Header, bounds comment, set flags | 1-6 | cap 7185 s (timeout TERM/15s), idle-TTL 1800000 ms worker-enforced |
| Run/bin/manifest/root/port vars | 7-11 | all five identifiers sanitized to role variables |
| DRIVER05-START log | 12-14 | |
| identity.txt receipts | 16-21 | private build-receipt id replaced by omission marker |
| No-existing-arena / listener checks | 23-25 | exit 5 PORT-BUSY, exit 6 WORKER-EXISTS |
| Admission: MemAvailable >= 126345216 KiB | 27-35 | exit 4 ADMISSION-REFUSED |
| Admission: cgroup max/current headroom | 36-39 | skipped entirely when memory.max = "max" |
| admission.txt snapshot | 40-45 | SwapTotal/SwapFree echoed here (receipt) |
| SWAP_BASE + cgroup event baselines | 46-48 | THE admission baseline for the 64 MiB test |
| Launch: setsid nohup timeout wrapper | 50-58 | exact budget CLI; WPID = wrapper PID |
| Child resolve + worker-identity.txt | 59-71 | pid/start/exe/cmdline echoes; absolute-cap epochs |
| STATUS=LAUNCHED | 72-73 | controller polls this file |
| Monitor constants + TSV header | 75-78 | RSS_CEIL 108024 MiB, HEADROOM 8 GiB |
| Monitor loop: elapsed cap 7245 s | 80-83 | |
| Wrapper liveness kill -0 | 84 | |
| Child re-resolve via pgrep -P | 85-86 | covers child restart, not PID reuse |
| Sample: /proc status + meminfo | 87-93 | `|| echo 0` fallbacks preserved (original defect class: unreadable -> 0, cannot false-trigger) |
| RSS / headroom / NEW-SWAP tests | 94-96 | line 96: `[ "$SF" -lt $(( SWAP_BASE - 65536 )) ]` — single-sample host-wide LEVEL vs fixed launch baseline; strict `<`, so exactly 65536 KiB decline does NOT stop |
| cgroup oom/high event tests | 97-99 | inert where memory.events absent (file missing -> 0==0) |
| Violation -> single TERM to wrapper | 100-105 | STATUS=RESOURCE-STOP; loop breaks; NO KILL escalation, NO second confirmation |
| Final state | 108-113 | sleep 3, events_end append, WORKER-GONE residency check |

## driver03-ROLE_ENGINE_A.sh  (original: ROLE_ENGINE_A driver, 151 lines, complete)

| Function/section | Original lines | Notes |
| --- | ---: | --- |
| Header, vars, GLOBAL_T0/STOP_AT | 1-17 | wall = GLOBAL_T0+5400-45 |
| identity.txt (incl. git rev-parse) | 19-23 | source-checkout path sanitized |
| Fixed request bytes + baselines | 24-32 | request bytes are public in EXECUTION-BUDGET-bf51f09e.md |
| check_admission() | 34-44 | MemAvail >= 12 GiB; SwapFree vs mem-baseline.txt − 65536 KiB (same strict form); cgroup oom |
| start_sampler() / stop_sampler() | 46-65 | background subshell sampler; VMRSS 24 GiB, RssAnon 8 GiB, MemAvail 4 GiB floors; TERMs engine wrapper directly; stop via TERM+wait |
| run_engine() | 67-91 | per-phase: launch epoch, timeout TERM/15s (cap−15), full engine CLI (remote CPU/GPU endpoints), wait EPID, rc/end receipts, sampler stop |
| after_phase() | 93-101 | GPU apps/memory/mem receipts; verdict on rc==0 |
| B0 phase | 103-109 | STOP-03 guard, admission, cap 1200; failure -> PHASE-VERDICT B0-FAILED, exit 6 (dependent phases die) |
| B1 phase | 110-116 | cap 600 |
| G0 gate wait + phase | 118-130 | gate file G0-GATE, deadline STOP_AT−1900, cap 1800, remote GPU CLI |
| G1 gate wait + phase | 132-144 | gate file G1-GATE, deadline STOP_AT−1000, cap 900 |
| Final receipts | 145-151 | |

## driver04-ROLE_GPU_B.sh  (original: ROLE_GPU_B driver, 117 lines, complete)

| Function/section | Original lines | Notes |
| --- | ---: | --- |
| Header, vars, plan pin | 1-15 | EXPECT_PLAN value sanitized out (placement proof) |
| launch: phase validation, caps | 16-19 | G0 cap 1900, G1 cap 1000 |
| identity + plan hash gate | 20-25 | mismatch -> exit 4 |
| Port / host / disk admission | 26-40 | >= 4 GiB mem; G0 needs fresh cache + 22.25 GiB disk |
| GPU provisional admission formula | 41-50 | floor((F−4096MiB−256MiB−1598720)/5222400) >= 3289 |
| Launch + identity append | 51-70 | timeout TERM/15s (cap−15) |
| Readiness poll (port listen) | 71-77 | 150 × 0.2 s |
| Background sampler | 78-94 | RSS 2 GiB, disk 4 GiB; TERMs wrapper |
| stop: TERM -> poll 20 s -> KILL | 96-104 | the ONLY KILL escalation in the supervisor set |
| stop: final counters | 105-114 | cache bytes/files, GPU state |

## orchestrate-ROLE_CONTROLLER.sh  (original: controller orchestrator, 133 lines, complete)

| Function/section | Original lines | Notes |
| --- | ---: | --- |
| Header (relay route comment) | 1-6 | concrete route sanitized; topology preserved as role variables |
| Vars, GLOBAL_T0, log/remain, SSHOPT | 7-17 | |
| abort() | 19-31 | STOP-03 touch, driver04 stop G0/G1, TERM owned forward PIDs from pid files, TERM CPU wrapper; exit 1 |
| Relay bridge setup + listener check | 32-48 | four setsid ssh -N forwards; abort if listeners missing |
| Role wrappers + status05 | 50-53 | status05 uses private control tool |
| Launch CPU worker driver | 55-68 | STATUS==LAUNCHED gate; waits for "strata-worker: listening" in worker.log |
| Launch engine driver + B polling | 70-85 | 500 × 5 s; FAIL in PHASE-VERDICT -> abort; per-phase status snapshots |
| G0 section | 87-106 | driver04 launch G0 -> WORKER-READY gate -> touch G0-GATE -> poll -> driver04 stop G0 |
| G1 section | 107-125 | same pattern, G1-GATE |
| Close GPU forwards, keep CPU transport | 127-133 | COMPLETE status |

## Known-original-defect inventory (preserved deliberately)

- Unreadable /proc fields fall back to 0 (`|| echo 0`), conflating
  "unavailable" with "zero" (replacement monitor fixes this; fixtures cover
  unavailable-vs-zero).
- NEW-SWAP test is host-wide SwapFree LEVEL vs fixed launch baseline; swap-in
  resets the comparison; no debounce/confirmation sample.
- cgroup guard reads only the orchestrating process's own cgroup
  (/sys/fs/cgroup/memory.*), not the worker's; inert where memory.events is
  absent.
- driver05 resource stop sends a single TERM with no KILL escalation and no
  wait/reap of the wrapper.
- No launch-release (admission-gated SETUP) exists anywhere; engine connect
  is what starts arena load.
- No signal traps in any script; cleanup ordering depends on timeout
  wrappers, monitor loops and abort() alone.
