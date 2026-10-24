# Strata validation harness — sanitized source handoff (2026-10-24)

Source-only copy of the reusable Strata campaign harness for retrieval through
GitHub. Private original paths, configs, receipts, logs, model data, runtime
identifiers and credentials are NOT included; every such value is a
`__PLACEHOLDER__` to substitute before use. Nothing here is merged to main.

## Source map

| Path | Origin (local retention root) | Role |
| --- | --- | --- |
| `harness/campaign.py` | local harness retention root | Shared campaign core: identity binds, worker status/RPC, `run_arm` (engine session + 3 retained GEN requests + per-request affinity receipts), `wait_session` fail-fast, placement asserts |
| `harness/owner.py` | same | Engine broker: owns engine stdin, bounded drain, READY/DONE events, per-arm socket control, 32 MiB log caps, independent monitor |
| `harness/worker_owner.py` | same | Worker guardian: spawns `strata-expert-worker`, identity receipts, bounded logs, health heartbeat, cleanup on failure |
| `harness/monitor.py` | same | Independent per-process resource monitor (RSS/health receipts) |
| `harness/worker_control.py` | same | Worker control client: status / unload / stop (opcode 11) |
| `harness/clockmon.py` | same | 1 Hz read-only clock/thermal journal with STOP-file termination |
| `harness/freq_ctl.py` | same | cpufreq snapshot / set / verify transitions with receipts |
| `drivers/pr7_loadcmp.py`, `pr7_loadcmp2.py` | same `drivers/` | PR7 1/2/4-lane loading comparison drivers (config-driven; `pr7_loadcmp2.py` adds `wait_session` + recovery) |
| `drivers/pr7_build_gates_05.sh` | same | Detached serial build + focused ctest gates (05-style CPU builder) |
| `drivers/pr7-def4-build-gates-05.sh` | same | Default-four follow-up: 2-target incremental build + the two default-selection checks |
| `drivers/pr7_link_03.sh` | same | Incremental CUDA link gate with ldd check |
| `drivers/timed_run.py` | same | stdlib timing/RSS wrapper for build receipts |
| `drivers/recover_t2.py`, `watch_loadcmp.py` | same | One-shot recovery/observer utilities from the PR7 window (provenance) |
| `w13w14/w13w14.py` | live W13/W14 campaign root (sanitized) | RECOMMENDED worker-count comparison driver: stop/relaunch with N workers, authorized live pool/caller masks + verification, C0 four-thread load timed separately, one warm arm per count, P12-style token/entry identity asserts, decision + restore |
| `w13w14/as-run/w13w14.py` | same | Exact bytes executed 2026-10-24 (verdict below), sanitized paths only |
| `w13w14/config.template.json` | derived from as-run config | Every binding field as `__PLACEHOLDER__` |
| `REGRESSIONS.md` | this series | Concrete launcher defects + fix status (incl. the zombie-reap 420 s case) |

## As-run result (context only)

12-worker reference P12 14.9210 tok/s; W13 14.9018 (−0.1%, level); W14
13.7221 (−8.1%, pool shares the caller's physical core 0); 12 restored
resident. Token streams and worker_entries identical across all arms.

## Usage sketch

1. Copy `harness/*` + chosen driver into a fresh campaign root (mode 700).
2. Stage engine/worker/manifest/requests; hash them mechanically into
   `pins.json`-style preflight pins (never hand-copy digests).
3. Fill `config.template.json` → `config.json`; pin engine pool and loading
   lanes explicitly (`--local-load-threads 4`); keep engine and worker argv
   byte-identical across arms except the authorized single variable.
4. Run detached (`nohup setsid python3 -B <driver>.py`), poll state files
   with short foreground probes, stop on first failure.
