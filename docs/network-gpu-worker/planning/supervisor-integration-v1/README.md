# Supervisor integration v1 — static candidate, not runtime approval

This is a new replacement supervisor, based on the complete sanitized historical
source at `ad921e0f09fa340bc138cc230202c1b7a6e0674d`. Its monitor libraries originate
at `a3b706deae4101e90592bed58701df20f9bd4da2`, with the correction linked below. It is not a
byte-for-byte patch to the historical shell drivers. See [SOURCE-COVERAGE.md](SOURCE-COVERAGE.md).

Current continuation: [PERIODIC-OBSERVATION-V1-20261021](../namespace-runner-v1/PERIODIC-OBSERVATION-V1-20261021.md).
It observes tasks/children without set convergence and preserves identity-safe
tracking. B0/G0 use one A smoke request; B1/G1 use matching A/B streams.
Effective phase waits/lifetimes clip to the remaining allowance. It supersedes historical
execution gates and time/usage balances below, with one new focused mocked entry
on ROLE_CPU_C followed automatically by the already authorized trial on PASS.
The author ran no tests, Python source, builds, model or fleet operations.

No new Python, syntax check, fixture, live probe, build, product executable or
model was executed by the author or reviewers. The user reported the original
26 monitor fixtures passed once at a3b706d, RC 0, no failures/errors/skips,
0.21 seconds. That report does not validate these integration sources. The
reported bundle digest is
`90af152c7d90f820d21eeb0ef372371c3c16f978ef7ee6c520a599d03feac1d9`.
No private evidence logs were inspected here.

The historical `supervisor-source-v1/`, original execution budget, metadata
planner, assignment and old monitor files are unchanged. PR #2 remains draft
and unmerged at `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`; these artifacts do not
change the implementation or extend the expired model-run approval.

## Components

| File | Purpose |
| --- | --- |
| policy.py | Explicit role policies, local stamps, release and phase gates |
| runtime.py | Parent-owned blocked exec, separate monitor, watchdog and subreaper |
| linux_owned.py | Initial-namespace identity, direct-child pinning, one pinned exec |
| commands.py | Historical product CLI, exact request bytes, isolated environment |
| outer.py | UUID-bound GPU management reads, actual cache-filesystem checks |
| cache_status.py | Public SEW1 read-only idle-TTL adapter, one-second total I/O bound |
| agent.py | Host-local guardian, challenge lease, phase receipts, finite retention |
| controller.py | Three required peers, serial phases and owned transport cleanup |
| test_integration.py | Synthetic fixtures; no live workloads or signals |
| test_procfs_task_race.py | Nine focused mocked cases for task reconciliation, failures and amended request bytes; NOT RUN here |
| test_procfs_child_churn.py | Twelve focused mocked lifecycle, safety, cleanup and request cases; NOT RUN here |
| live_probe.py | Future separately approved three-case harmless-child probe |

## Role policies and unchanged workload envelope

All quantities below are binary bytes. Each role has its own admission,
runtime host/cgroup headroom and process RSS policy; the engine never receives
the CPU arena policy. Every applicable cgroup-v2 ancestor's finite `memory.max`
and `memory.high` participates in the tightest headroom. Unlimited limits still
require readable event counters. Any increase of high/max/oom/oom_kill stops work;
counter decrease, identity/limit change or required missing data refuses coverage.

| Role | Admission | Runtime headroom | RSS ceiling | Extra memory ceiling |
| --- | ---: | ---: | ---: | ---: |
| ROLE_CPU_C | 123384 MiB (120.4921875 GiB) | 8 GiB | 108024 MiB | none |
| ROLE_ENGINE_A | 12 GiB | 4 GiB | 24 GiB | RssAnon 8 GiB |
| ROLE_GPU_B | 4 GiB | 2 GiB | 2 GiB | none |

The original **host SwapFree fixed-admission level guard** is retained: take
one baseline for the host session; stop on a single sample strictly below that
baseline minus 65536 KiB. Exactly 64 MiB decline does not stop; greater decline
does. This is cumulative net level decline, not a rolling activity measure.
The old budget did not specify this threshold/semantics; the historical driver
did. This amendment makes the previously implicit implementation choice explicit.
Existing occupied swap grants no additional allowance. VmSwap, pswpin/pswpout
and PSI remain diagnostic; unavailable/malformed values are not zero and do not
introduce a new stop policy. No 16 MiB worker threshold, debounce or larger host
swap allowance is introduced.

GPU-RESERVE-WARNING-V1 keeps strict local startup admission at 1152 MiB primary
and 512 MiB helper, including the initial pre-exec monitor sample. After exec-gate
release, those local reserve crossings are diagnostic
warnings, including during CUDA/serve startup, rather than termination. Sampling,
required telemetry/identity failures and cleanup baseline restoration remain
strict. Per-phase diagnostics record sampled minima and crossing counts; warning
records occur at most once per selected local GPU per phase. There is no new
runtime stop threshold or CUDA error detector. Existing product failures still
abort; actual CUDA OOM is possible. See the
[active policy and continuation](../namespace-runner-v1/GPU-RESERVE-WARNING-V1.md).
Remote GPU reserve remains a strict 4096 MiB, cache budget 18432 MiB, cold cache disk admission 22.25 GiB
and running disk floor 4 GiB on the cache filesystem. The provisional admission
is `floor((free - 4096MiB - 256MiB - 1598720) / 5222400) >= 3289`.
The fixed plan has 3289 remote experts, 17176473600 expert bytes, and SHA-256
`a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`.
The primary startup sizing correction fixes primary 960 (20/layer), helper 2247,
full CPU 21369, uncovered CPU 18080. Helper IDs can change when primary shrinks;
the new combined local holdings are a subset of the old local holdings, so the
unchanged remote plan remains disjoint. The CPU bitmap/partition identity is
freshly derived once, then identical across all four runs. See the
[current sizing amendment](../namespace-runner-v1/PRIMARY-STARTUP-SIZING-CORRECTION.md).
No metadata planner, owner selection, weight extraction or transfer occurs here.

The CPU CLI uses memory-budget 106428 MiB for 111597465600 arena bytes
(106427.63671875 MiB), two workers, setup 600000 ms,
residency 1800000 ms. Engine phases remain B0/B1/G0/G1 with caps
1200/600/1800/900 seconds; remote GPU G0/G1 caps are 1900/1000 seconds. Preserve
11-second gaps after B0/G0. Global cap is 5400 seconds, original stop/launch
cutoff 5355; leases conservatively consume transit/cleanup time inside it.
The current amendment supersedes full-maximum-envelope admission. Controller
waits and owned lifetimes clip to remaining time, preserving the 45-second
global reserve, five-second challenge transit and 15-second owned cleanup.
Fresh leases can only shorten deadlines. No positive safe interval means refusal.
CPU absolute cap remains original admission plus 7200 seconds. TERM starts with
15 seconds reserved; at most 14 seconds grace leaves up to one second for
pidfd KILL and actual wait/reap inside that cap. No deadline resets on attach.

GPU return accounting may take up to 30 seconds after owned tasks are reaped,
clipped by the original job/global bounds. This state services controller leases
and propagates failures. It never frees someone else's allocations or changes
the admitted free-memory baseline. A shared-host allocation can therefore refuse
the next phase safely. Cold cache must be a new directory; warm cache is reused;
no cache deletion or eviction is performed by this supervisor.

## Coverage, clocks and ownership

Supported: Linux with pidfd operations, prctl parent-death/subreaper support,
readable procfs, trusted receipts for initial PID/mount/cgroup namespaces, and
one full-root regular cgroup-v2 domain hierarchy with the memory controller.
V1/hybrid, hidden ancestors, threaded groups, changed namespaces, ambiguous
mounts or unavailable required interfaces block admission. True root event
interfaces are explicitly unsupported at that root only, not fabricated zero.
The original monitor bounds (16 processes, 256 tasks, 64 ancestors, bounded
reads and 64 MiB journals) remain. No support for new platforms is claimed.

Each host captures new admission/launch monotonic stamps and a unique domain.
Wall time is correlation only. Remote hosts exchange bounded remaining durations
on fresh five-second nonce challenges; raw monotonic clocks are never compared.
The first host baseline/events survive all serial phases. Initial local monitor
coverage, parent-owned workload/monitor pidfds and a fresh receipt are required
before the exec byte is released. Listener coverage means an owned listening
socket, **not loaded/READY residency**. Engine SETUP_BEGIN still initiates arena
loading. No standalone preload was invented.

Shared `OwnedProcess.children()` now performs two bounded observation passes.
Neither task sets nor child maps must converge within or between passes.
A children ENOENT is recoverable only when that task is absent from the
subsequent enumeration and pinned parent identity still verifies. The second
pass rereads tasks that can receive reparented children. All observed children
are retained for `OwnedTree` parent/identity pinning. A newly observed PID absent
at pidfd lookup or confirmed exited by its held pidfd grants no identity or
signal. Confirmed exited descendant handles are retired, keeping the 16-process
tracking cap without accumulating dead historical slots. EACCES,
malformed data, a still-present task missing children, missing leader/task
directory, process identity loss and unresolved ownership refuse coverage.
Churn alone is not a refusal. Failure
context includes the proc path, pinned PID/start/exe identity and scan task sets.
Descendant status/namespace/cgroup sampling and watchdog adoption distinguish
confirmed exit from required read failure. Partial cleanup discovery retains
already pinned descendants if their parent exits. These are periodic discovery
and identity-safe tracking/cleanup, not an atomic global procfs snapshot. No
total process-tree quiescence, sleep, deadline extension or ownership exemption.

The actual engine input write in `agent.py` calls `commands.requests(phase)`.
B0/G0 send `GEN 64 9419` then `QUIT`; B1/G1 send A then B, each GEN64, then QUIT.
`Controller` still requires a zero exit and complete cleanup for each phase;
its DONE event is a supervisor receipt, not a product request count. Evidence
must separately require one product DONE and `.first-logits.0` for each B0/G0,
two product DONEs and `.first-logits.0/.1` for each B1/G1. B0/G0 are startup/cache
smoke only. B1/G1 remain complete like-for-like comparison streams. See the current
amendment for the cumulative failed-request charge and active-time ledger.
Repeated-A coverage is dropped by this reduced schedule, not reported PASS.

The watchdog is parent of workload and monitor, and subreaper for their orphans.
All normal signals use held pidfds with identity checks; there are no group,
port or arbitrary-PID kills. The sole constructor-failure numeric signal is
`Popen.kill()` on that parent's still-unreaped newly created transport/helper,
before any poll/wait; its PID cannot have been reused. Failure to obtain the
required pidfd never permits workload release. Every owner cleanup is attempted
even if another fails. Reap receipts distinguish actual parent/adopted waitpid
results from descendants confirmed exited but already reaped by their parent.

A dedicated detached host guardian survives controller/monitor failure and
retains its local deadline. Active control EOF, expired lease, missing coverage,
monitor exit/stall, guard failure or cancellation stops owned work. No retry,
resizing or dependent phase follows a required-role failure. Abrupt guardian
death invokes direct-child PDEATHSIG; this is not a claim to survive arbitrary
kernel stalls or host failure. External interference with the trusted guardian
or cgroup is unsupported and stops the trial.

After explicit successful-trial retention approval, CPU lifetime is bounded by
the earliest of success plus 1800 seconds, public status's actual idle expiry,
and original admission plus 7200 seconds. Query latency and the challenge transit
allowance consume the grant. Engines/GPU workers/transports release first.
Controller exit cannot undo the committed finite grant. The proxy does not claim
to reap a detached guardian; after proxy exit, initial PID 1 reaps it. Failure
always stops owned CPU work. No CPU arena currently exists by assumption.

## Private binding contract — no real mapping belongs in GitHub

Agent JSON requires `role`, unique `run_id`, new absolute `run_root`,
`implementation` (exact bf51f09e full SHA), `binary_sha256`, `namespaces`,
`bindings`, and explicit boolean `retention_approved`. Non-CPU roles additionally
require absolute `gpu_tool` and ordered `gpu_query_ids` full physical UUIDs.
`namespaces` contains trusted initial host `cgroup`, `mnt`, `pid` device/inode
pairs, not a receipt inferred from an untrusted container's PID 1.

| Identifier | Semantic binding, privately supplied |
| --- | --- |
| BIN_CPU_C / BIN_ENGINE_A / BIN_GPU_B | Live verified candidate binary for that role |
| LOCAL_ROOT_C / MANIFEST_C | Existing CPU source root and exact full CPU manifest |
| NATIVE_SHARD_A / NATIVE_PACK_A | Existing engine nonexpert assets / expert pack |
| PROFILE_A / MANIFEST_A | Frozen verified profile / full CPU partition manifest |
| RUN_A | Same new private output root as the engine agent |
| GPU_PLAN_A / GPU_PLAN_B | Existing identical frozen binary plan at each role |
| EXPECTED_GPU_PLAN_A_SHA | Exact frozen plan SHA above, required at A and B |
| GPU_SELECT_A / GPU_SELECT_B | Ordered full UUID list identical to gpu_query_ids; primary then helper at A |
| CACHE_B | New cold cache directory, same retained path for warm G1 |
| CPU_PORT_C / GPU_PORT_B | Privately approved loopback listener ports |
| ENDPOINT_CPU_C_ON_A / ENDPOINT_GPU_B_ON_A | Existing approved private transport endpoints |

`commands.py` renders all product flags and request bytes. Its exact environment
replaces inheritance, including LD/STRATA/ISA/CUDA overrides. If a binary needs
additional runtime library bindings, stop for reviewed explicit bindings; do not
silently restore ambient environment. Engine --gpu passes the UUID list through
to CUDA_VISIBLE_DEVICES in pinned generate.cpp. NVML queries return UUID plus
free bytes, checked for exact distinct membership and reordered to that list.

Controller `launch(config)` accepts one run_id, at most four exact approved
`forwards` argv arrays, three `peers` role-to-existing-transport argv arrays,
and `retain` boolean. No SSH/network/security options are generated. No shell is
used. Live endpoint readiness, new-path permissions, namespace provenance,
binary/source/config receipts, manifest/profile identity, and ownership/absence
of conflicting services remain prerequisites for any later separately approved
window. No private worker_control helper is fetched or trusted.

## Validation gate

44 new synthetic cases are authored, not executed. The original 26 cases can be
run from their unchanged pins; the user-reported previous pass is preserved.
The proposed live probe exercises real proc/pidfd/cgroup sampling, natural exit,
monitor exit, stall escalation and parent reaping using only harmless children.
It does not validate production transport, GPU sampling, full agent retention,
SEW1 live worker status, product CLI startup or full-model behavior. Those remain
unvalidated and require a later approved window after this gate. No model retry
or continuation of the old approval is allowed. See [HERMES-NO-MODEL.md](HERMES-NO-MODEL.md).
