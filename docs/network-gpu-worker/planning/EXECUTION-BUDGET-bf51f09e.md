# Proposed execution budget: not authorized

All identifiers have semantic definitions in [README.md](README.md). Bind real roles, paths, accelerator selections and endpoints only in a private approval packet. This sanitized proposal supersedes conflicting earlier planning templates; it authorizes no runtime work.

Implementation pin: `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`, tree `8e0cc2eda89f17a63182d1b273140288a1418b47`; main `34a59d83a51bdb3f0299d65306740bd357818715`; PR #2 draft/unmerged. The artifact revision is separate from that implementation. No runtime PASS or measured fit/speedup is asserted here.

## Fixed placement and resources

- ROLE_ENGINE_A owns two local accelerators: primary 1056 experts (22/layer), helper 2263, selected from the pinned profile. Explicit counts; no automatic sizing changes, peer/stage/adaptive changes or mid-session movement. Primary reserve 1152 MiB. Helper weights are `11818291200` bytes =11270.80078125 MiB, plus workspace/context and the helper's 512 MiB reserve. A management-interface free-memory sample does not establish post-context fit; actual candidate allocation/refusal is decisive. Never resize to turn a refusal into success.
- ROLE_ENGINE_A host admission: 12 GiB current host AND tightest-cgroup headroom. Separate stopping ceilings: VmRSS 24 GiB and RssAnon 8 GiB; record RssFile/RssShmem. Abort on remaining headroom below 4 GiB, exceeded ceilings, new swap/OOM/high-limit events or CUDA errors. Local expert source mappings can contribute `(1056+2263)*5222400=17333145600` bytes, about 16.14 GiB, especially on warm fills. One-blob staging does not bound mapped RSS. The total ceiling allows those clean mappings plus other runtime/mappings; it is not a reservation or fit guarantee. Clean file pages are reclaimable, so the proposed 8 GiB anonymous envelope plus 4 GiB reserve explains the admission threshold. Finite cgroup ancestors remain independent guards; unlimited cgroups use host MemAvailable. These are proposed limits, not historical peak measurements.
- ROLE_CPU_C: one newly approved full arena of 21257 blobs =`111012556800` bytes (103.3885002 GiB), including every ROLE_GPU_B duplicate. Worker weight cap 105870 MiB =`111012741120` bytes: slack 184320 bytes, less than one expert. Conservative admission `107000+16384 MiB=120.4921875 GiB` within BOTH host MemAvailable and tightest-cgroup headroom, including the 16 GiB reserve. RSS ceiling 108024 MiB; abort below 8 GiB remaining headroom or on new swap/limit events.
- Primary-clamp protection: explicit local counts cannot grow. If primary sizing clamps below 22/layer, the larger CPU complement exceeds the tight cap and CPU SETUP refuses BEFORE arena allocation. An undersized fixed helper fails before CPU setup. The profile fixes exact identities; counts alone do not. Successful sessions retain the entire approved bitmap/arena, with no shrink or CPU upload fallback. Never take over or duplicate an existing large arena; existing-service identity/control rights require specific approval.
- ROLE_GPU_B: exactly 3289 experts selected ONLY from the CPU complement. Payload W=`17176473600` bytes; grouped workspace `1598720` bytes; requested sum `17178072320` bytes. Runtime reserve 4096 MiB and planning margin 256 MiB. For a privately recorded management-interface free observation F bytes, provisional admission is `min(3289,21257,floor((F-4096*1048576-256*1048576-1598720)/5222400))`. Require it still permits 3289; otherwise stop without choosing a smaller plan. Actual post-context CUDA admission requires W+workspace+reserve within free_before and free_after>=reserve, before READY.
- ROLE_GPU_B host/cache: admission 4 GiB host/cgroup headroom; RSS ceiling 2 GiB; remaining headroom at least 2 GiB; reported weight staging at most 16 MiB. New empty owned cache with independent cap 18432 MiB; require 22.25 GiB disk free before launch (18 GiB cache+4 GiB reserve+256 MiB evidence). Abort below 4 GiB disk free or post-READY free accelerator memory, or on swap/limit/driver errors. No complete layer/shard/model download or host mirror. Retain the bounded cache by explicit decision; no accumulating roots or automatic deletion of prior assets.
- In every comparison use OMP_NUM_THREADS=2, engine pool workers=2, CPU workers=2 and STRATA_IO_PF_THREADS=2. The last setting freezes an I/O pool whose default would otherwise be eight. No concurrent model, compile, test, benchmark or broad model hasher. Sample owned memory/I/O, host/cgroup memory and accelerator state about every two seconds. No new cgroup/limit/driver/install/security/network configuration. These fixed worker counts do not recreate an unrestricted historical-core performance result.

Before ANY model or worker launch, verify live source/configuration/binary provenance, including BIN_GPU_B against its complete private receipt hash. A bundle hash alone is insufficient. Use already validated binaries; missing identity blocks launch, without rebuild/extraction/install authority. Recheck all actual admission observations and accelerator mappings during the separately approved execution window.

## Time, topology and retention

One global monotonic 5400-second active wall includes setup, analysis and cleanup. Stop launching and TERM owned active tasks at deadline minus 45 seconds; allow at most 15 seconds grace, KILL only matching remaining owned children, and finish/reap/report by deadline. Clip every nested process/transport/analysis operation to remaining time. B0/B1/G0/G1 engine hard caps, grace included: 1200/600/1800/900 seconds (sum4500); remaining900 seconds covers owned setup/gaps/evidence/cleanup. Remote-GPU process caps: G0=1900/G1=1000 seconds, nested within the global wall and stopped after each phase.

CPU idle TTL1800000 ms preserves the same arena BETWEEN all sessions. GPU-setup STATUS does not renew TTL; the1200000 ms setup cap must leave time for first successful inference. CPU process TERM at7185 seconds plus15 seconds grace fits an absolute7200-second cap. Default: stop the owned CPU worker at trial end. Optional explicit retention approval: keep ONLY that worker for at most30 minutes idle, bounded by its remaining absolute cap, with private PID/start/executable/expiry records and minimal bounded control transport. Release all engines/GPU allocations. Error cleanup stops owned work unless explicit retention approval covers that error.

Bind CPU_PORT_C, GPU_PORT_B, ENDPOINT_CPU_C_ON_A and ENDPOINT_GPU_B_ON_A through the already approved private transport. Workers remain loopback-only. Do not publish address/relay bindings, create trust, expose listeners or change network configuration. Check occupancy passively. Use a supervisor with empty/closed stdin and positive owned child/wrapper PID, executable and start-time checks. Never use kill-by-port, broad process matching or negative/group PID signals. No unrelated service may be stopped.

## Candidate CLI templates: execution approval required

These are review templates only. Bind every variable privately and record exact commands/RCs in new evidence. GPU_PLAN_A is OUT_A/plan_B.bin; GPU_PLAN_B is its later approved matching metadata copy. Supervisors supply bounded process ownership; no command below is a deployment instruction by itself.

ROLE_CPU_C foreground worker under its owned supervisor:

```sh
OMP_NUM_THREADS=2 timeout --signal=TERM --kill-after=15s 7185 \
  "$BIN_CPU_C" --port "$CPU_PORT_C" --workers 2 --memory-budget-mib 105870 \
  --deadline-ms 10000 --local-setup-ms 600000 --local-root "$LOCAL_ROOT_C" \
  --local-manifest "$MANIFEST_C" --residency-ms 1800000 \
  > "$RUN_C/worker.out" 2> "$RUN_C/worker.log"
```

Fresh ROLE_GPU_B process for each G phase, same plan/cache, phase cap1900/1000 seconds including grace:

```sh
CUDA_VISIBLE_DEVICES="$GPU_SELECT_B" OMP_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=15s "$((GPU_PHASE_CAP_S-15))" \
  "$BIN_GPU_B" --plan "$GPU_PLAN_B" --cache-root "$CACHE_B" \
  --port "$GPU_PORT_B" --device 0 --cache-budget-mib 18432 --reserve-mib 4096 \
  --deadline-ms 10000 --setup-ms 1200000 \
  > "$RUN_B/$PHASE.out" 2> "$RUN_B/$PHASE.log"
```

ROLE_ENGINE_A common function; every cap includes15 seconds grace:

```sh
run_engine() {
  local tag=$1 cap_s=$2
  shift 2
  env OMP_NUM_THREADS=2 STRATA_IO_PF_THREADS=2 STRATA_ROUTE_TAIL_SKIP=0 STRATA_LOOKAHEAD=0 \
    STRATA_LOOKAHEAD_STATS=0 STRATA_FS_SLOTS=0 \
    STRATA_DUMP_FIRST_LOGITS="$RUN_A/$tag.first-logits" \
    timeout --signal=TERM --kill-after=15s "$((cap_s-15))" \
    "$BIN_ENGINE_A" --serve --native "$NATIVE_SHARD_A" --pack "$NATIVE_PACK_A" --gpu "$GPU_SELECT_A" \
    --mmap-experts --greedy --pool-workers 2 \
    --remote-cpu-experts "$ENDPOINT_CPU_C_ON_A" --remote-cpu-deadline-ms 10000 \
    --remote-cpu-local-manifest "$MANIFEST_A" --remote-cpu-resident --remote-cpu-setup-ms 600000 \
    --expert-cache 1056 --expert-cache-per-layer --expert-profile "$PROFILE_A" \
    --expert-cache-device1 2263 --remote-expert-opt --vram-reserve-mib 1152 \
    --adapt-every 0 --adapt-swaps 0 --pcie-frac 0 --short-read 64 --prefill 64 \
    --prompt-cache 0 --spec 2 --suffix-draft 0 --lookup-chain 0 --pipeline-windows 0 --max-context 1024 \
    --dump-routing "$RUN_A/$tag.routing.bin" "$@" \
    < "$REQUESTS_A" > "$RUN_A/$tag.out" 2> "$RUN_A/$tag.log"
}
```

All four processes receive identical request bytes:

```text
GEN 64 9419
GEN 64 14773,6353
GEN 64 9419
QUIT
```

Sequence: B0 baseline new application-cold arena, `run_engine B0 1200`; clean exit/accelerator release/about11-second gap; B1 baseline resident reconnect, `run_engine B1 600`; clean exit; G0 three-role serving with a new empty cache; clean exit/stop remote GPU/release/about11-second gap; G1 with a fresh GPU process and the same warm cache. G0/G1 calls are:

```sh
run_engine G0 1800 --remote-gpu-experts "$ENDPOINT_GPU_B_ON_A" \
  --remote-gpu-plan "$GPU_PLAN_A" --remote-gpu-setup-ms 1200000 --remote-gpu-deadline-ms 10000
run_engine G1 900 --remote-gpu-experts "$ENDPOINT_GPU_B_ON_A" \
  --remote-gpu-plan "$GPU_PLAN_A" --remote-gpu-setup-ms 1200000 --remote-gpu-deadline-ms 10000
```

These calls occur in their separate supervised phases, not consecutively without the stated worker transitions. One engine and serial stream; at most12 requests/768 generated tokens, at most64 prompt tokens/request. EOS may shorten output; no automatic extension or retry. The CPU arena/bitmap stays identical throughout. Prompt cache must report zero reused tokens and full prompt reads. No automatic sampling, MTP/draft-model, dynamic/adaptive/peer/resident-mirror changes.

## Acceptance, numerical observations and evidence

Require exact disjoint/exhaustive primary_A/helper_A/assigned_B/cpu-uncovered_C ownership. CPU residency retains ALL cpu-full_C. Reconcile complete serving-route trace totals with CPU/GPU client entries and helper computed entries, including all verification work. Trace weights are placeholders, not real router weights. Require positive actual execution on both local GPUs, remote GPU and CPU. Require independent remote-GPU final requests/entries/service_ns matching client totals; normal EOF disconnect increments are classified separately. Client remote allocation/cache values are setup snapshots, not continuous telemetry. Missing final counters after forced termination remain incomplete evidence.

CPU arena allocations=1; later full cache hits/reused111012556800 bytes with fresh sessions and zero load/hash/allocation/expiry/unload deltas. CPU network weight bytes remain zero. G0 exact selective logical source reads, expert network payload and new persisted bytes each=W17176473600; assigned=cached=W; READY only after verified residency. G1 source expert reads/network expert bytes/new persisted bytes=0, reused=W; local verification/GPU reload are expected. Keep physical I/O, logical reads, expert payload, optional isolated transport-stream counts, disk bytes, staging, allocations and RSS separate. Neither physical-disk nor NIC bytes are implied equal to W.

Natural worker loss must fail the affected request cleanly with no partial-result mixing, successful completion from invalid partials or invisible CPU retry. Stop dependent phases; recovery requires a new separately approved CPU-only session. Do not inject full-model failures as an unapproved chaos test. Keep separate CPU compatibility, native-OFF/ON, ISA and fixture evidence; this artifact adds no runtime PASS.

First-window numerical observation uses STRATA_DUMP_FIRST_LOGITS, headerless float32 files with exactly4*vocab finite bytes each. For these short inputs the selected row is the last prompt row. Recover vocab from trusted bounded metadata or startup information; diagnostic write errors do not necessarily fail the process, so missing/short files fail the gate. There is no full remote-partial/all-decode-logit dump here. Predeclared SCREENING: centered-logit relative L1<=0.03 and KL(softmax(B1)||softmax(G1))<=0.03 nats at temperature1. Center each vector by its mean; L1 denominator=max(sum(abs(centered baseline)),1e-12); use stable double log-softmax. Report raw/centered maxabs, RMSE, top1/margin and token agreement. Within-mode cold/warm/repeated-A drift maxabs<=1e-5*max(1,maxabs(reference)). Expert-reference thresholds do not establish end-to-end bounds; do not loosen thresholds after observation. No CPU/GPU bitwise/final-token equality or full-parity/perplexity/all-position claim. All output IDs in range, normal finite completion required.

Principal timing comparison is B1 versus G1: same inputs/settings/partition. Record startup, CPU setup phases, GPU transfer/cache/READY, whole-engine wall, each DONE prompt/decode time and actual tokens, client request/wait, worker service times and helper staging/wait. B0 has an application-cold CPU arena; G0 has a warm CPU arena and cold GPU disk cache. Do not attribute their wall difference solely to offload. OS page cache is uncontrolled; no flush. EOS/trajectory differences limit causal comparisons. No universal speedup gate; quantify actual uncovered CPU serving and timing honestly. Retained profile, per-expert route frequency and aggregate owner timings can support later between-run critical-path/migration-cost decisions; this trial implements no optimizer or live movement.

Preserve prior assets/evidence. New evidence<=256 MiB/role and256 MiB/controller; no model blobs in archives or public commits. Stop engines and GPU workers; confirm owned PIDs gone and accelerator allocations return to recorded baseline within30 seconds, without stopping others. Default-stop CPU unless explicit bounded retention applies. Cache cap/retention is explicit; unassigned cleanup needs later approval on an inactive exclusively owned cache using the documented metadata CLI cleanup operation, never broad deletion. Record all commands/RCs, full source/binary/plan/input hashes, private bindings, route/logit/memory/transfer/timing data, exact expiry/hardcaps and cleanup. Stop at review/report: no merge, changed plan, longer trial or permanent service.
