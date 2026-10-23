# FULL-Q8-SCALEOUT-V1-20261023

Execute the approved full-Q8 comparison through Hermes, not on Valinor/Geekom.
This handoff changes no product source. PR #3 stays unmerged. Product revision is
`9dd56a946b025bd56e4bd0174997d6b9e51f4e02` in **both** modes. Supplied focused
validation passed native ON/OFF, three-peer overlap, single-peer compatibility
and actual 04/02 CUDA exchanges. That evidence is not full-model scale-out proof.
The new helper here is statically reviewed, **execution NOT RUN** by its author.

Use the exact validated engine SHA-256
`92df6d5ccc859d6e2f2606517ca6c9ff6d13547cf54b19094a6c14a4fb057753`
and worker SHA-256
`4ae79b97ef4761088f87167a239a271f80025d8c2a507567aee1985378cb1576`.
Resolve their paths and existing owner/relay launch commands from the successful
MRGV1 receipt (report directory `strata-mrgv1-20261023-1792716062`, archive SHA
`8e7acb53e10c2a42ba0befdd321a48a7c890638ec1eec0c68ee245983a0e4307`).
Do not rebuild engine/workers, substitute the handoff revision as product identity,
or repeat passed fixtures. No dependency install, security change or new route.

## Workload and comparison

Keep the retained five tasks, exact plan SHA
`1c81fa79b9a26f66b0a4f7b2e6eae92858c82f47de827f08751b312948a21792`.
Resolve the plan and tokenizer metadata from the retained practical run receipts.
Do not retokenize, truncate, rewrite the plan, run its two probes, or add warmups.
Each phase is the five existing GEN32 task requests followed by QUIT.

Run **C0, S0, S1, C1** once each: current, scale-out, scale-out, current. C uses
03's two local 3060s plus the existing 04 remote 3090 and 05 CPU arena; S adds 02
device 0 (3060 Ti 8 GB) and device 1 (3060 12 GB). Both modes use the same engine,
full-Q8 assets, generation flags, profile, local ownership and 04 assignment.
Endpoint reduction order is always 04, 02 device 0, 02 device 1; C uses its prefix.
There are at most **20 started requests / 640 generated tokens**. Charge an
interrupted start its full 32-token allowance. Historical usage is 38 starts /
994 actual generated tokens; this is a new grant, not a reset. New maximum is
58 starts / 1,634 actual-plus-allowed tokens. Preserve actual and charged totals.
If an interrupted phase's exact accepted-start count is unavailable, conservatively
charge all five 32-token allowances for that phase; label the actual count unknown.

These fixed participants/assignments are experiment controls, not permanent
architecture limits. The intended later direction is dynamic membership and
placement informed by per-device usable memory, compute and communication cost,
with selective movement and safe routing activation. That work is not part of
this run. The product collection currently supports up to eight remote endpoints;
this experiment uses three heterogeneous remote devices on two remote hosts.

## Admission, resident service and placement

Use the existing approved controller and temporary relay pattern from MRGV1.
Resolve private SSH bindings, GPU UUIDs, tool paths and ownership receipts there;
do not publish them or invent new access. Record fresh read-only admission on
03/04/02 and 05's existing service. No provisioning campaign is needed.

On 05, use the retained `resident_cpu.live_status`/resident status procedure while
idle, with the original service config and independently owned guardian/monitor.
Require identity validity, fresh coverage, complete 111,597,465,600-byte arena,
21,369-expert partition and UINT64_MAX idle TTL. Match its retained CPU bitmap
and partition identity. Do not start, replace, restart or unload it. If absent or
incompatible, report that concrete blocker; no automatic cold-load fallback.
During an engine connection observe identities/resources only, not a second
competing cache-status connection. After each disconnect, recheck idle residency
and reuse counters. No new CPU allocation/load/network is expected.

Keep primary 960 (20/layer), helper 2,247, full CPU 21,369 and 04 remote 3,289.
Resolve the **post-sizing-correction** `cpu-full_C.bitmap` and its pinned hash,
not the obsolete 1,056/2,263 placement. The helper verifies the retained profile,
reconstructs these unchanged local owners, and compares the complete CPU bitmap.
04's exact 92,156-byte plan remains SHA
`a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578`.
Its 17,176,473,600-byte complete real cache must be retained, never cleaned/refilled
or copied wholesale. Fresh cache validation decides actual reuse.

On 02, obtain actual per-UUID free/total VRAM and host/cgroup/swap/disk headroom.
Use two new separate 0700 cache directories on the same admitted model filesystem.
Both cards must remain physically identified despite CUDA visibility remapping.
The helper reserves at least 1,536 MiB or 10% of total VRAM, whichever is greater,
plus the exact 1,598,720-byte workspace and 256 MiB allocator/driver margin. Weight
ceilings are additionally 6 GiB on device 0 and 10 GiB on device 1. These are
ceilings, not fit claims; actual free memory may lower them. Retain a combined
4 GiB filesystem free margin after both caches and metadata. Product admission
still checks actual allocation and post-load free memory; never weaken it to fit.

Create a private `admission.json` with these fields from real receipts:

```text
cpu_bitmap_sha256: full retained bitmap SHA
cache_filesystem_free_bytes: current free bytes of the common 02 cache filesystem
devices: [
  {index: 0, uuid: actual GPU0 UUID, free_bytes: integer, total_bytes: integer,
   cold_payload_bytes: actual transferred bytes, cold_setup_seconds: measured seconds},
  {index: 1, uuid: actual GPU1 UUID, free_bytes: integer, total_bytes: integer,
   cold_payload_bytes: actual transferred bytes, cold_setup_seconds: measured seconds}
]
```

The last two fields come from each same-route MRGV1 cold client's `transferred`
and `setup_ms/1000` counters, not the aggregate 12.188-second stage or cache-hit
bytes. This is conservative effective setup goodput including CUDA/context/cache
overhead, **not measured raw link bandwidth**. Retain those exact source lines.
Do not reuse 04's reported 106 Mbps as an unmeasured 02 link specification. If the
per-peer cold counters are missing or zero, report that missing budget input
before product launch, rather than inventing bandwidth or another fixture.

On 03/controller, with this immutable handoff checkout as `HANDOFF_SRC`, run:

```sh
TOOLS="$HANDOFF_SRC/docs/scaleout/scaleout.py"
python3 -B "$TOOLS" place --admission "$ADMISSION" --profile "$PROFILE" --cpu-bitmap "$CPU_BITMAP" --plan04 "$PLAN04" --out "$NEW_PLACEMENT_DIR"
"$GPU_MANIFEST" plan "$TRUSTED_MANIFEST" "$NEW_PLACEMENT_DIR/assigned02-0.tsv" "$NEW_PLACEMENT_DIR/02-0.plan"
"$GPU_MANIFEST" plan "$TRUSTED_MANIFEST" "$NEW_PLACEMENT_DIR/assigned02-1.tsv" "$NEW_PLACEMENT_DIR/02-1.plan"
```

`GPU_MANIFEST` is the existing accepted `strata-gpu-manifest` metadata CLI. Its
source is unchanged. If its ELF is unavailable, the only permitted preparation
build is that target, serially, in the retained 03 native-ON cache with donor ggml
and disconnected dependency fetching (120 seconds, 2 GiB RSS). No engine/worker
or dependency rebuild. Planning reads bounded metadata, never weight payloads.
The engine/product revalidates both new plans against the trusted manifest.
Retain plan SHAs/placement IDs/counts/bytes. Check pairwise disjointness and union
with unchanged local/04 owners. CPU serves 18,080 minus the new assigned counts;
its full resident bitmap remains unchanged. Selection uses the existing ranking
in order, first eligible device-0 entries then device-1 entries; this is capacity
placement, not a claim of optimal heterogeneous load balancing.

## Transfer budget and lifetimes

The helper derives each cold setup cap as
`ceil(1.5 * assigned_bytes / measured_effective_setup_goodput + 300)` seconds.
It accounts conservatively for a completely missing new assignment; valid cache
entries can reduce actual work. The 04 warm setup allowance is 300 seconds.
The common product setup-ms option is the largest per-worker cap, supported by
the existing worker CLI (at most 86,400,000 ms). Serial setup allowances sum into
the cold phase's outer cap; no shorter engine/worker/owner timer may contradict it.

`placement.json` gives the complete one-shot window, including C0/C1 startup
300 seconds each, S1 warm startup 600 seconds, four 240-second task batches,
45 seconds per phase cleanup, 300 seconds controller/admission margin and a
separate final 45-second cleanup reserve. The maximum is four hours; the actual
derived cap is normally smaller. If the measured estimate exceeds four hours,
report the concrete bytes/goodput/estimate before launch; do not silently shrink
the experiment or prescribe a knowingly inadequate timer. No running extension.

Start and retain one controller monotonic clock before relay/owned-process startup.
Use `window_s` as an upper bound and clip fresh lifetimes to its remaining time
minus cleanup reserve. Workers can stay listening across all four clients; an
engine disconnect releases session VRAM while preserving disk cache. A product
setup cap applies to a connection, not a service's entire lifetime. Never retain
the old 1,200-second setup deadline or a 1,500-second trial/observer lease here.

During cold S0, sample new 02 cache publication progress every ten seconds.
After the first missing entry, no newly published assigned bytes across **either**
02 cache for 600 seconds is a bounded transfer-stall refusal. A later idle peer
must not be declared stalled while the earlier peer's serial transfer progresses.
Before first progress allow 04's measured setup plus a 300-second margin; after
all missing entries are complete, use the remaining bounded setup allowance for
validation/VRAM loading. Retained partial entries are not assumed reusable;
product length/digest checks and missing bitmaps decide reuse on a later authorized
resume. Do not count STATUS traffic or unrelated network I/O as weight progress.

## Launch and measurement

Reuse MRGV1's successful process owner/monitor and temporary relay launcher,
with fresh roots and the derived caps above. Do not rerun the legacy serial
three-role controller: it hardcodes one GPU, the 04 plan, reserve and short leases.
The helpers here deliberately do not introduce another process manager.
Keep actual executable identities, true-parent/pidfd ownership, host/cgroup event,
swap/RSS and cleanup checks. 02's two workers together require at least 8 GiB
available host/cgroup memory at admission and 4 GiB ongoing headroom; each worker
RSS is capped at 2 GiB. Replace MRGV1's synthetic 1 GiB GPU-growth ceiling with
each admitted assignment + exact workspace + declared allocator margin; retain
the strict free-VRAM reserve. For the real 03 engine use its existing product
RSS/anonymous caps (24/8 GiB), not the tiny fixture's 2 GiB host cap. Retain the
other 03/04 substantive guards. Local 03 runtime
free-VRAM crossings remain warnings only, with strict initial admission; 04 keeps
its 4 GiB reserve, 02 keeps each newly declared strict reserve. No unrelated PID
may be signalled. Refresh required admission receipts through the approved channel,
without changing security or touching the resident owner.

Start one owned GPU worker per remote device using the validated worker ELF.
04 keeps its accepted plan/cache/budget (18,432 MiB cache, 4,096 MiB reserve).
02 uses its generated plan and `cache_budget_mib`/`reserve_mib` from placement.json:

```sh
"$WORKER" --plan "$WORKER_PLAN" --cache-root "$WORKER_CACHE" --port "$WORKER_PORT" --device "$VISIBLE_DEVICE" --cache-budget-mib "$CACHE_MIB" --reserve-mib "$RESERVE_MIB" --deadline-ms 10000 --setup-ms "$SETUP_MS"
```

Run that command inside the existing owner with the **derived full service cap**,
not an old fixture's 180-second timeout. Both 02 devices visible means --device
0/1; when each worker sees only its pinned UUID, both use --device 0. Record the
physical UUID-to-visible-ordinal mapping. Ports/caches/plans are independent.
Use only controller-mediated numeric loopback endpoints on 03, in 04/02-0/02-1
order. Copy only small plans and helper files automatically through the approved
transport. Missing assigned expert slices move through SGW1 from 03; do not mirror
the model or change any retained cache.

Build private `bindings.json` from the accepted practical engine bindings:
`BIN_ENGINE_A`, `GPU_SELECT_A`, `NATIVE_SHARD_A`, `NATIVE_PACK_A`,
`ENDPOINT_CPU_C_ON_A`, `MANIFEST_A`, `PROFILE_A`, plus `engine_sha256` and
`worker_sha256` exactly as above and `endpoints=[{label:"04",endpoint:...,plan:...,plan_sha256:...},
{label:"02-0",endpoint:...,plan:...,plan_sha256:...},{label:"02-1",endpoint:...,plan:...,plan_sha256:...}]`.
Plan paths are absolute host-03 paths. This config is private and must never be
committed. Render each phase once into a new directory:

```sh
python3 -B "$TOOLS" render --bindings "$BINDINGS" --placement "$PLACEMENT_JSON" --plan "$TASK_PLAN" --phase "$PHASE" --out "$PHASE_DIR"
```

The generated `launch.json` contains the exact engine argv/environment/stdin.
Run it through the retained MRGV1 owner with stdout in `PHASE_DIR/workload.out`,
bounded stderr and real exit/cleanup receipts. No shell reinterpretation of argv;
pass the JSON argv as an array. Use the identical render for both C and S; only
the remote endpoint collection differs. TRACE/CAPTURE/logpos/routing dumps are
absent. No speculative/quantizer/placement flag changes. Feed `requests.txt` once
and account for actual accepted starts; do not send requests again after errors.

Before each engine starts, create its empty stdout file and start this bounded
file observer on 03 under the same owner; it opens no model/socket and signals
no process:

```sh
python3 -B "$TOOLS" clock --output "$PHASE_DIR/workload.out" --receipt "$PHASE_DIR/output-clock.json" --seconds "$EFFECTIVE_PHASE_CAP"
```

It records READY and the five DONE arrivals with 50 ms polling jitter, excluding
startup from the warm READY-to-last-DONE wall. Stop the observer with its owner
on failed engine exit; retain partial receipts. Also retain per-request product
prompt/decode times and token counts. Record engine rc/identity and cleanup in
`exit.json` with keys `rc`, `phase`, `engine_sha256`, `plan_sha256`; populate rc
from the real wait/reap receipt, not an inferred DONE. Success requires actual
rc 0 plus five complete DONE records.

Run C0/S0/S1/C1 automatically while healthy and time remains. Separate S0 cold
transfer/persistence/cache validation/VRAM loading from its request timing. S1
must validate full assigned reuse and zero new weight transfer/persist; local
read/hash passes are still real and must not be called zero reads. C0/C1 04 cache
and all CPU attaches must also reuse. Use product cumulative counter deltas,
not cache mtimes. Capture per-worker requests/entries/service_ns/exchange time,
cache R+M=W accounting, source reads and positive actual participation. Overlapping
CPU/GPU timings cannot be added into a wall-time partition. service_ns includes
executor overhead, transfers, kernels and synchronization, not only kernels.

## Report, failure and closeout

Retrieve the unchanged evaluator automatically from retained commit
`98b5791d6042d3b25eac0bb5c5f6e372f78166a5` at
`docs/network-gpu-worker/experiments/paired_tasks.py`. Keep that path under a
checkout containing its existing `tools/strata_tokenizer.py` and `serve/frontend.py`
so its imports resolve. Its SHA-256 is
`8d2cc2cf6aff51c70157a6e1d22374fe14956acc5fd9be19625db6ca4cbb2587`.
For example, Hermes stages it automatically, without a user copy/paste step:

```sh
git -C "$HANDOFF_SRC" fetch origin 98b5791d6042d3b25eac0bb5c5f6e372f78166a5
mkdir -p "$HANDOFF_SRC/docs/network-gpu-worker/experiments"
PAIRED_TASKS="$HANDOFF_SRC/docs/network-gpu-worker/experiments/paired_tasks.py"
git -C "$HANDOFF_SRC" show 98b5791d6042d3b25eac0bb5c5f6e372f78166a5:docs/network-gpu-worker/experiments/paired_tasks.py > "$PAIRED_TASKS"
```

Do not invoke its prepare/report CLI (which regenerates the plan). This helper
uses its decoder and unchanged strict answer/clock checks, with a bounded DONE
reader that permits a legitimate zero CPU share after adding GPUs; it does not
invent CPU participation. Positive aggregate tier participation is reported from
actual product counters, or marked not demonstrated:

```sh
python3 -B "$TOOLS" report --evaluator "$PAIRED_TASKS" --plan "$TASK_PLAN" --pack "$TOKENIZER_METADATA" --root "$RESULT_ROOT" > "$COMPARISON_JSON"
```

Keep C0/S0/S1/C1 directories under RESULT_ROOT. Preserve actual decoded answers,
token IDs, strict results and OUTPUT_CAP_REACHED separately. Fenced code can be
substantively correct while failing the unchanged strict format check; do not
silently strip fences or execute generated Python. Compare the two warm samples
per mode and both order-reversed pairs. Report range/mean and all individual
prompt/decode/wall/TPS values; a two-sample smoke test does not establish general
scaling. If token lengths/answers differ, disclose differing work rather than
claiming equal-work speedup. Historical cross-mode NUMERICAL_FAIL remains; it does
not automatically cancel these healthy practical requests or become PASS here.

A real product/resource/identity fault stops dependent phases with cleanup; no
hidden inference retry. A pre-inference ordinary launcher correction may resume
the first unstarted phase within the original clock and grant. Preserve failed
attempts and successful phases. Never rerun completed C/S phases merely to obtain
a prettier report; interrupted started requests remain charged.

At normal/fault closeout, stop/reap only owned engines, GPU workers, new observers
and temporary relays; check GPU return by host/device. Leave healthy 05 worker,
monitor and guardian loaded, and retain 04 and new 02 complete/partial caches.
Numerical/task-format failure alone is not a CPU resource fault. Only an actual
resident service/resource safety fault warrants its owning service's normal
shutdown. Preserve all old evidence and report continuing residency explicitly.
No merge, permanent deployment, new credentials or persistent tunnel is included.
