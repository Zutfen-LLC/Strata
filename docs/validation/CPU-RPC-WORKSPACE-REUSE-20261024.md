# CPU RPC workspace reuse: Hermes handoff

Baseline: current main `ef9180b34af30ce5b2cac1e3129666ba21584e0b` (merged PR #8).
Branch: `feature/cpu-rpc-workspace-reuse`. Use the full immutable commit supplied with this handoff as
`CANDIDATE`; the commit containing this file pins the complete source. **All builds, executable tests,
inference and measurements are NOT RUN locally.** Static review cannot establish compilation or speed.
Main was merged additively at `bc949472cedbc9f98196eb9c8b9e295e531e2b9d`; no history was rewritten.
The sole shared source file, `expert_worker.cpp`, merged without conflict. Relative to the baseline,
the implementation remains the original buffer-reuse change: loading and harness code are common to
both sides. No PR #6 merge to main, deployment, fleet action, install or host setting change occurred.
Focused static review found no overlap defect between loader completion, scratch reset and pool use.
One evidence gap remains: tests inspect codec/client/tier capacities directly, but worker scratch and
receive capacities have no direct assertion. A synchronized private-implementation accessor would
expand this refresh, so the gap stays disclosed. No allocation or RSS measurement is claimed.

Current state supplied by the parent: 02 has eight 32 GB DIMMs (256 GB installed, 251.5 GiB usable,
2133 MT/s) and a healthy R12 resident using an earlier PR7 implementation worker. Preserve it while
builds and checks run. 05 has 94.2 GiB usable and must never receive the full 116,135,731,200-byte arena.
Use 05 for CPU builds/fixtures, 03 for CUDA engine/dispatch checks, and **02 only for real model work**.
The authorized trial uses turbo, inference workers 12, loading threads 4, and the unchanged placement:
primary 960 / helper 1378 / CPU 22238 experts, 116,135,731,200 CPU bytes. Read back that same baseline;
do not repeat the old RAM, capped/turbo, loader-count or multi-CPU campaigns.

The synchronous CPU client retains its inference payload, contiguous frame and receive payload.
The single serving worker retains decoded inputs, activation scratch, jobs, lookup, rows, partials,
encoded reply and send storage. Receive storage survives successive frames on one connection; upload
buffers above the inference cap are discarded after handling. Inference receive growth reserves the
validated length first, so an earlier small upload does not force geometric growth above that cap.
Session invalidation releases execution scratch, even when verified weights remain resident. New
setup rebuilds geometry and job pointers. The tier transfers the cleared previous reply's capacity to
its existing socket owner under the request mutex and publishes only after exchange completion.

Protocol bounds limit all retained inference sizes: T <= 8, K <= 10, H <= 2560, F <= 640 and experts <=
the existing manifest cap. Standard vector growth may retain capacity above the active size. This is
one bounded high-water mark per owner, never a request history or another weight arena. Required
zeroing, scalar codecs, arithmetic, grouping, reduction order, copied asynchronous inputs, deadlines,
socket settings, placement and scheduling are unchanged by intent. There is no new switch or default.
The existing CPU-worker feature remains opt-in. Byte and bit identity must still pass remotely.

## Retained inputs and bounded builds

Hermes uses its existing authorized controller and retained source/cache bindings. Do not run on
Geekom or Valinor. Recover exact paths, generator, compiler/ISA options, dependency revision and resource
guard from the latest accepted receipt. Use 05's CPU builder and 03's CUDA builder; runtime is on 02.
No new SSH trust, listener access, firewall/NIC permission or route is assumed. Missing access, cache,
config or asset is an explicit blocker. Do not install, clean or start a toolchain/provenance campaign.

Before changing a dedicated build checkout, require clean source and preserve old engine/worker ELFs,
full SHA-256 hashes and source/build receipts in a new result directory. Preserve dependencies, model
shards, manifests, partition files and unrelated services. First establish compiled-input equivalence
of each retained baseline engine and worker to `ef9180b3`: target source/header and generated inputs,
compiler/ISA flags, dependencies and link inputs from existing receipts/cache. A filename, explicit
loading flag or source label is insufficient. In particular, the PR7 implementation `723c5a45` predates
the four-thread default/header/CLI follow-up; `--local-load-threads 4` does not make those compiled
inputs identical. If equivalence is not established, make only the needed bounded incremental baseline
build at `BASE` and preserve it. Do not discard the running resident to perform this check or build.
Confirm worker ISA/runtime-library compatibility with 02; a build on 05 is not automatically portable.

On each builder resolve `SRC`, `BUILD`, `BIN`, `REV` and `CANDIDATE` from the retained private config.
`REV` is `BASE` for a needed baseline build or `CANDIDATE` for the candidate. `BUILD` must already name
that same `SRC` through `CMAKE_HOME_DIRECTORY`; `BIN` must be new and owned. Keep native-ON/OFF caches
separate. Fetch through the existing authorized path, then use the existing resource/time guard:

```sh
set -euo pipefail
: "${SRC:?dedicated clean source bound to retained cache}"
: "${BUILD:?matching retained build directory}"
: "${BIN:?new output directory for this revision}"
: "${CANDIDATE:?full published candidate SHA}"
: "${REV:?exact baseline or candidate SHA selected by config}"
BASE=ef9180b34af30ce5b2cac1e3129666ba21584e0b
test "${#CANDIDATE}" -eq 40
test "$REV" = "$BASE" || test "$REV" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
git -C "$SRC" cat-file -e "$CANDIDATE^{commit}"
git -C "$SRC" merge-base --is-ancestor "$BASE" "$CANDIDATE"
git -C "$SRC" diff --exit-code "$BASE" "$CANDIDATE" -- tools/validation-harness
git -C "$SRC" switch --detach "$REV"
test "$(git -C "$SRC" rev-parse HEAD)" = "$REV"
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "$SRC"
test ! -e "$BIN"
mkdir -m 700 "$BIN"
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN"
```

Preserve the retained generator, dependencies, options and toolchain; inspect configure output for
unexpected changes. Use one build job. Old ELFs must be outside output locations before configuration;
stop if they could be overwritten. Save command, stdout/stderr, elapsed/RSS, exit status and hashes.
Stop dependent work at the first failing gate. Transfer only needed engine/worker ELFs through the
existing path, and compare full source/destination hashes before execution.

## Focused correctness before model work

On 05, incrementally build only the worker and these targets. Run the candidate checks; preserve
existing equivalent-baseline receipts, or apply the same gates if its inputs were not validated:

```sh
cmake --build "$BUILD" --parallel 1 --target strata-expert-worker \
  expert_worker_test expert_worker_e2e_test expert_tier_test \
  worker_residency_test worker_service_residency_test worker_compat_test worker_local_test
ctest --test-dir "$BUILD" -V -j1 -R \
  '^(expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_residency_test|worker_service_residency_test|worker_compat_test|worker_local_binary_test|worker_loader_binary_4)$'
```

`expert_worker_test` includes codec gates. New cases alternate T8 -> T1 -> T2 -> T1 repeatedly, vary K
and masks, compare exact scalar-reference request/frame/reply bytes, and compare dense/zero/sparse
partials bitwise with `direct_native_partial`. They inspect warm codec/client/reply capacities, valid
work at a sequence refused for excess expert uses, same-connection smaller geometry, reconnect with
larger geometry, and unload. Existing refusal/malformed/deadline batteries remain. Tier cases cover
alternating shape/capacity/zero work, copied caller-input lifetime, ordered completion and error latches.
The two real-binary local-loader fixtures check the merged default and explicit four-thread path;
they are not another 1/2/4 performance campaign. All use disposable synthetic loopback services,
never the live resident service. No full model is loaded on 05.

On 03, configure its retained native-ON cache with the resolved bindings above, incrementally build
`strata worker_dispatch_test`, and run CTest `^(worker_dispatch_test|worker_dispatch_host_zero_test)$`
serially under existing guards. These check real tier/dispatch integration. Shared CPU codecs also
feed GPU protocol consumers: build/run existing `gpu_worker_test` and `gpu_frame_test` where enabled.
Preserve the applicable native-OFF `strata` link check in its separate retained cache. Report missing
HIP/SYCL/Windows/Metal compilation as NOT RUN; do not turn this into a platform campaign.

## Fixed config and nine measured requests on 02

Only after prerequisite builds/checks pass, use the merged versioned harness from `BASE`, directory
`tools/validation-harness/`. It is byte-identical in the candidate; verify that above and record the
static source hash. Preserve one private campaign config containing retained absolute model/ELF paths,
masks, the three tokenized GEN256 lines, output roots and policies. Do not edit helpers or sequencing
between arms. Use existing `bundle.py` to stage once on the execution host and the staged `run.py` to
run; preserve mechanical pins and baseline ELFs. No new wrapper or supervisor is needed.

On 02 read back turbo, post-swap DIMMs/capacity, topology/masks and named resident identity. Do not assert
absence: the healthy R12 service is expected. If its engine/worker compiled inputs are proven equivalent
to `BASE`, reuse that resident for A0. Otherwise keep it running through bounded builds/fixtures, then
perform the parent-authorized named owned replacement for the baseline pair. An engine change alone
cannot execute candidate worker code. Each actual worker replacement releases the arena and requires
the same partition read/verification again; record cold setup separately. No concurrent arenas,
gratuitous unload, network permission or policy change is authorized.

Ownership is a real binding, not a PID-only workaround. The merged CLI's `attach_root` mode is
retain-only and cannot mix later restarts or reap another launcher's child. Equivalent-resident reuse
therefore needs the retained owning launcher's supported handoff/reap transition after A0, while
keeping the same private campaign config and fixed sequencing. If that binding is unavailable, report
the exact owner/config gap before service action; do not unload merely to fit the CLI, fabricate a
Popen handle, or create a new sequence script. An ordinary launcher binding correction can be reviewed
if needed. A resident whose compiled inputs are not equivalent takes the normal owned-start path
below, after the existing owner performs the authorized stop/reap; do not probe port-free beforehand.

The private config uses existing fields only: `worker_argv` has `--workers 12` and
`--local-load-threads 4`; every arm has `workers: 12`, `loader_threads: 4`, the same caller mask,
12 pool masks, engine-main mask, placement markers, environment and runtime options. Keep existing
engine-pool/OMP settings; those are separate from worker and loader counts. Assets `engine` / `worker`
name the preserved baseline pair; `engine_candidate` / `worker_candidate` name B. Use this ordered
selection when a baseline worker must be started:

| Entry | Engine / worker asset | `restart` | Request indices |
| --- | --- | --- | --- |
| A0-load | engine / worker | true | [] |
| A0 | engine / worker | false | [0, 1, 2] |
| B-load | engine_candidate / worker_candidate | true | [] |
| B | engine_candidate / worker_candidate | false | [0, 1, 2] |
| A1-load | engine / worker | true | [] |
| A1 | engine / worker | false | [0, 1, 2] |

For proven resident reuse, omit only A0-load through that retained owner binding; A0 still has three
measured requests. Final/failure worker policies remain explicit in the same config; retain the final
baseline service. No generation warmups: the retained campaign had none. Each load entry runs
READY/QUIT with zero GEN; each measured entry sends the three retained GEN256 lines once. This is
exactly nine measured requests in A0/B/A1 order, with exact engine+worker pairs.

Keep the SAME turbo policy, clocks/power limits, pool size, affinity, physical topology, primary/helper
GPUs, transport/NIC state, NODELAY/timing flags, model, partition, context, prompt-token files/order,
sampling/seed and output cap. Read back policy before/after each arm; do not retune. Keep exact inputs;
do not retokenize. Use new output paths and selected ELFs only. Cold setup/reload is outside measured
request time; do not restart between warm requests. Keep output IDs/raw terminal statuses, failures,
slower runs, per-request decode counts/times, arm wall time, worker/client counters and available
RSS/CPU observations. Leave useful verified weights resident after the final baseline arm.

Compare all nine outputs/statuses first, including A0/B/A1 token/count identity and the retained
terminal-status reference. Keep the requested 256-token cap; preserve any genuine early completion.
Divergence, refusal, timeout, residency failure or policy/topology drift blocks a speed claim. Report
each arm and baseline drift; three requests per arm are a diagnostic, not a statistical guarantee.
No reruns to select a fast result. Existing timings overlap; do not sum phases or subtract unsynchronized
host clocks. Replay prebuilds frames and cannot measure this allocation change's inference benefit.
No multi-CPU trial or additional timing-on attribution campaign belongs here.

Historical context supplied by the owner, not reproduced here: native LAN + ITR0 9.31 tok/s versus
tuned SSH 7.593 / relay 8.733 was token-identical across campaigns/binaries. Earlier local02 capped
11.6508/11.6652 versus turbo 12.1749 tok/s was about +4.4% with matching tokens; sampled busy clocks
1.98 -> 2.49 GHz and pool elapsed 38.62 -> 37.89 s do not identify a sole bottleneck. Neither comparison
proves this optimization's benefit. Expect no particular speedup.

Return one flat report with exact source/ELF/config/harness hashes, passed/failed/NOT RUN gates, all
nine outputs and timings, cold setup separately, policy/topology identity, resident state left behind,
and exact blockers. Link retained private artifacts through the approved channel; publish no raw
private bindings/logs. A null or negative result closes this optimization round. Stop before merging
PR #6 or deployment; do not begin distributed multi-CPU testing.
