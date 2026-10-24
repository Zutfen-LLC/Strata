# CPU RPC workspace reuse: Hermes handoff

Base: merged main `fbd3bfd1e710c277b818ff37417ba6ee313a5621` (PR #5).
Branch: `feature/cpu-rpc-workspace-reuse`. Use the full immutable commit supplied with this handoff as
`CANDIDATE`; the commit containing this file pins the complete source. **All builds, executable tests,
inference and measurements are NOT RUN locally.** Static review cannot establish compilation or speed.
No merge, deployment, fleet access, package installation or host setting change was performed here.
Three independent focused static passes found no blocking defect in the codec/client, worker lifecycle,
or tier/spec/handoff. One evidence gap remains: tests inspect codec/client/tier capacities directly,
but worker scratch and receive capacities have no direct assertion. Their retention is established
only by reading the code until remote execution. No allocation or RSS measurement is claimed.

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

## Source and retained build gates

Hermes uses its existing authorized controller and retained source/cache bindings. Do not run on
Geekom or Valinor. Recover exact paths, generator, compiler/ISA options, dependency revision, resource
guard and runtime driver from the latest accepted receipt. Use the existing CPU worker builder and
CUDA engine builder (previously 05 and 03); the runtime topology comes from the current receipt, not
from those build roles. No new SSH trust, listener access, firewall/NIC permission or route is assumed.
Missing access, cache, driver or asset is an explicit blocker. Do not install, clean or start a new
toolchain campaign. No new supervisor or provenance framework is needed.

Before editing a dedicated build checkout, require clean source and preserve old engine/worker ELFs,
their full SHA-256 hashes and source/build receipts in a new result directory. Preserve dependencies,
model shards, manifests, partition files and all unrelated services. If a retained baseline predates
the merge commit, establish that its compiled source matches merged main; a matching name is not
enough. Otherwise make a bounded incremental baseline build at the base first and preserve it.
Confirm that the retained worker build's ISA and existing runtime libraries fit the selected runtime
CPU; a worker built for another host is not assumed portable. Transfer only the needed candidate ELFs
through the existing authorized path and compare full source/destination hashes before execution.

On each builder, resolve `SRC`, `BUILD`, `BIN` and `CANDIDATE` to recorded absolute paths. `BUILD` must
already name that same `SRC` through `CMAKE_HOME_DIRECTORY`. `BIN` must be a new owned directory.
Keep each native-ON/OFF cache separate. Fetch through the existing authorized path, then:

```sh
set -euo pipefail
: "${SRC:?dedicated clean source bound to the retained cache}"
: "${BUILD:?matching retained build directory}"
: "${BIN:?new candidate output directory}"
: "${CANDIDATE:?full published candidate SHA}"
test "${#CANDIDATE}" -eq 40
test -z "$(git -C "$SRC" status --porcelain)"
git -C "$SRC" cat-file -e "$CANDIDATE^{commit}"
git -C "$SRC" merge-base --is-ancestor fbd3bfd1e710c277b818ff37417ba6ee313a5621 "$CANDIDATE"
git -C "$SRC" switch --detach "$CANDIDATE"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "$SRC"
test ! -e "$BIN"
mkdir -m 700 "$BIN"
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN"
```

Preserve the retained cache's generator, dependencies and toolchain; inspect configure output for any
unexpected change. Use the existing resource guard and one build job. Keep the old binaries outside
the output locations before configuration; stop if outputs could overwrite them. Save command,
stdout/stderr, elapsed/RSS, exit status and hashes. Stop dependent work at the first failing gate.

## Focused correctness before model work

On the CPU builder, incrementally build only the worker and these existing targets:

```sh
cmake --build "$BUILD" --parallel 1 --target strata-expert-worker \
  expert_worker_test expert_worker_e2e_test expert_tier_test \
  worker_residency_test worker_service_residency_test worker_compat_test
ctest --test-dir "$BUILD" -V -j1 -R \
  '^(expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_residency_test|worker_service_residency_test|worker_compat_test)$'
```

`expert_worker_test` includes the codec gates. New cases alternate T8 -> T1 -> T2 -> T1 repeatedly,
vary K and masks, compare exact scalar-reference request/frame/reply bytes, and compare dense, zero
and sparse partials bitwise with `direct_native_partial`. They verify warm codec/client/reply
capacities, a valid request using the sequence refused for excess expert uses, same-connection setup
with smaller geometry, reconnect with larger geometry, and unload. Existing refusal/malformed/deadline
batteries remain. The tier adds alternating shape/capacity/zero-work checks; its retained tests mutate
caller buffers after post, order a second post behind the first, and refuse mergeable results on error.
These gates use disposable synthetic loopback services, never the live resident service.

On the retained CUDA builder, configure with its own resolved bindings as above and incrementally
build `strata worker_dispatch_test`; run CTest `^(worker_dispatch_test|worker_dispatch_host_zero_test)$`
serially under its existing resource/time guards. These check the real tier/dispatch integration.
The CPU wire codec is also used by GPU protocol consumers: on that same configured builder, build and
run `gpu_worker_test` and `gpu_frame_test` if those existing targets are enabled. Preserve applicable
native-OFF checks in its separate retained cache. Report missing HIP/SYCL/Windows/Metal compilation as
NOT RUN; do not claim those backends from the CPU/CUDA results or expand this into a platform campaign.

## One small warm baseline / candidate / baseline comparison

Only after correctness gates pass, use the retained runtime driver and its exact current workload.
Resolve the current engine and CPU-worker identities, binary paths/hashes and owning guardian before
any service action. This document grants no new live-worker replacement permission. The operator must
authorize the named replacements in its execution session. A running old worker cannot execute the
new worker code merely because the engine changed: full-candidate comparison needs both ELFs.
Replacing a worker releases its useful CPU weight arena and requires loading/verifying the same
partition again. Retain baseline ELFs for the final baseline arm and rollback. Avoid concurrent weight
arenas unless the existing plan explicitly provides the memory. Leave useful verified weights resident
between requests and after the final baseline arm; never unload merely to collect a report.

Use serial arms A0 (baseline engine + worker), B (candidate engine + worker), A1 (the identical saved
baseline engine + worker). Keep the SAME current CPU policy, clocks/power limits, pool size, affinity,
physical topology, local/remote GPUs, transport/NIC state, NODELAY/timing flags, model, partition,
context, prompt token files, order, sampling/seed and output cap in all three arms. Read back existing
policy before/after each arm; do not retune or run capped-versus-turbo arms. Reuse the driver's bounded
warmup and three measured requests per arm. Reuse exact tokenized inputs; do not retokenize from text.
Use new output paths and the selected ELF paths only. Separate each cold setup/reload and warmup from
measured request time. Preserve exact output IDs and raw outputs, failures and slower runs, per-request
decode tokens/time, arm wall time, worker/client counters and available process RSS/CPU observations.
Do not restart a healthy worker between warm requests. If required runtime inputs are unavailable,
report the missing binding instead of constructing a replacement driver.

Compare every output and terminal status first. Any divergence, refusal, timeout, residency failure or
policy/topology drift blocks a speed claim. Report each arm and baseline drift; with three requests
this is a small diagnostic, not a statistical guarantee. No reruns to select a fast result. Existing
timings overlap; do not sum phases or subtract unsynchronized host clocks. Replay prebuilds frames and
cannot measure this allocation change's inference benefit. Optional timing-on attribution belongs in a
separate later decision, not the speed arms. No multi-CPU trial is part of this round.

Context supplied by the owner, not reproduced here: native LAN + ITR0 9.31 tok/s versus tuned SSH
7.593 / relay 8.733 was token-identical across campaigns/binaries. On local worker02, capped
11.6508/11.6652 versus turbo 12.1749 tok/s was about +4.4% with matching tokens; sampled busy clocks
1.98 -> 2.49 GHz and pool elapsed 38.62 -> 37.89 s do not identify a sole bottleneck. Neither comparison
proves this optimization's benefit. Expect no particular speedup.

Return one flat report with exact source/ELF hashes, passed/failed/NOT RUN gates, all nine measured
outputs and arm timings, cold setup separately, CPU-policy/topology identity, resident state left behind,
and exact blockers. Link retained raw artifacts directly. A null or negative result can close this
optimization round. Stop before merge or deployment; do not begin distributed multi-CPU testing.
