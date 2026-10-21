# Proposed Hermes gate — NOT AUTHORIZED / NOT EXECUTED

Request a separate explicit approval for this bounded no-model window. Do not
run it merely because the source is published. No full-model retry is included.

Bind privately: `ARTIFACT_COMMIT` (exact immutable integration commit from the
author's final receipt), `ROLE_CPU_C` (one approved Linux host), `ROLE_CONTROLLER`
(evidence coordinator), `FILES_ROOT` (new private download directory),
`PROBE_CONFIG` (new private JSON), `EVIDENCE_ROOT` (new private evidence directory).
No other execution target is permitted. No engine/GPU host operations, worker
services, listening ports, model/source assets, GPUs or pressure generation.

Use existing authorized GitHub authentication to fetch **only** the files in
`SHA256SUMS.integration-v1` plus that manifest from
`https://raw.githubusercontent.com/Zutfen-LLC/Strata/ARTIFACT_COMMIT/docs/network-gpu-worker/planning/`.
Limit transfer to256KiB total; retain paths relative to planning/. No clone,
external downloads/install or credentials in receipts. Verify the manifest's
full SHA from the final author receipt, then every file's SHA/size before use.
Stop on missing/mismatched bytes, unexpected main/feature drift, platform or
initial-namespace proof failure. Record exact candidate/config/file hashes and
interpreter/version/source provenance. Preserve all previous evidence.

Admission is read-only: confirm no conflicting owned probe, adequate host and
applicable ancestor headroom, pidfd/prctl/procfs/full cgroup-v2 support, and trusted
initial PID/mount/cgroup namespace receipts. Do not modify cgroups, limits,
security, network, services or swap. A true-root unsupported counter must be
reported explicitly; required unavailable coverage means STOP, never zero.

Stage1: run exactly the unchanged26 synthetic monitor cases and new44 integration
cases, serially, with no live signals/workloads. Existing reported26-case PASS
remains valid history; this is an explicit integration-regression gate, not a
cosmetic repeat. For each stage cap256MiB address space,30CPU-seconds,60wall-seconds,
8MiB log/output, one invocation, no auto-retry. The coordinator must own/pin each
new interpreter child and enforce wall deadlines with its own pidfd TERM then
KILL/wait, never a broad/group signal or unrelated PID.

Use explicit isolated bootstrap commands; `-I file.py` alone does not put adjacent
libraries on sys.path. Substitute only the private root path:

```sh
python3 -B -I -c 'import resource,runpy,sys; resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456)); resource.setrlimit(resource.RLIMIT_CPU,(30,30)); resource.setrlimit(resource.RLIMIT_FSIZE,(8388608,8388608)); sys.path[:0]=[sys.argv[1]]; runpy.run_path(sys.argv[1]+"/resource_monitor_fixtures.py",run_name="__main__")' FILES_ROOT
python3 -B -I -c 'import resource,runpy,sys; resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456)); resource.setrlimit(resource.RLIMIT_CPU,(30,30)); resource.setrlimit(resource.RLIMIT_FSIZE,(8388608,8388608)); sys.path[:0]=[sys.argv[1]+"/supervisor-integration-v1",sys.argv[1]]; runpy.run_path(sys.argv[1]+"/supervisor-integration-v1/test_integration.py",run_name="__main__")' FILES_ROOT
```

Require RC0, zero failures/errors/skips, exact counts and markers:
`RESOURCE_MONITOR_FIXTURES_PASS_V1 tests=26 live_signals=0 workloads=0` and
`SUPERVISOR_INTEGRATION_FIXTURES_PASS_V1 tests=44 live_signals=0 workloads=0`.
Stop at first failure; do not repair/run a second invocation in this window.

Stage2, only if Stage1 passes: create private PROBE_CONFIG with exactly
`namespaces` (trusted initial cgroup/mnt/pid device+inode pairs) and
`evidence_root` (new absolute EVIDENCE_ROOT/live path). No product bindings.
Run the pinned live_probe.py once with this isolated bootstrap:

```sh
python3 -B -I -c 'import runpy,sys; root,config=sys.argv[1:]; sys.path[:0]=[root+"/supervisor-integration-v1",root]; sys.argv=[root+"/supervisor-integration-v1/live_probe.py","--config",config]; runpy.run_path(sys.argv[0],run_name="__main__")' FILES_ROOT PROBE_CONFIG
```

Serial cases: real coverage+normal harmless-child exit; monitor exits; monitor
stalls after one synthetic coverage packet. The last two are expected refusals
of that owned job, not failed product assertions. All require complete cleanup
and genuine parent wait receipts. Probe code enforces128MiB address space and
10CPU-seconds per process,8MiB per output,45seconds internal overall bound.
Coordinator imposes60seconds wall plus at most5seconds cleanup. Maximum three
probe processes concurrently (guardian, one monitor, one harmless child), hence
384MiB aggregate address-space ceiling; coordinator/evidence tooling remains
outside that envelope and must stay below128MiB. No child allocates pressure.

Require RC0, the three `SUPERVISOR_OWNED_PROBE_CASE_PASS_V1` markers for
natural_exit/monitor_exit/monitor_stall, then
`SUPERVISOR_OWNED_PROBE_PASS_V1 cases=3 models=0 gpus=0 services=0`.
Require cleanup.json shows both direct children actually reaped, no incomplete
coverage/cleanup, no surviving owned tasks, memory returned to preflight range,
and no changed existing services/cgroups. Missing required host interface is a
blocked admission, not a waived pass. New swap decline/event, unexpected process,
timeouts, malformed evidence, log cap or ownership uncertainty stops the window;
signal only coordinator-owned pinned children and report exact incomplete work.

Window: at most300seconds total including admission, two synthetic invocations,
one live invocation and cleanup; no concurrent probes or retries. Preserve a new
REPORT.md, identities/config hashes, stdout/stderr with RC/start/end/monotonic
durations, JSONL coverage samples, cleanup receipts, byte counts, all hashes and
SHA256SUMS. Keep private mappings private; publish no raw operational logs.

This gate cannot establish production network/relay behavior, GPU guards,
live SEW1 retention, full agent/controller transport interaction or model
performance. Report those NOT RUN. Return the bounded evidence packet and STOP.
Do not start any real worker, engine, cache, model transfer/load or service,
retain CPU residency, merge PR #2 or activate any earlier model approval.
