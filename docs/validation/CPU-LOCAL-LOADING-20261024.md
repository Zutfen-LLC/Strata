# Bounded parallel CPU loading: Hermes handoff

Base: `fbd3bfd1e710c277b818ff37417ba6ee313a5621`, branch `feature/cpu-local-load-threads`.
Use the full immutable commit supplied with this handoff as `CANDIDATE`; the commit containing this
file pins the source. Related scope: [issue #4](https://github.com/Zutfen-LLC/Strata/issues/4), expanded
by the owner to include several disjoint expert readers. [PR #6](https://github.com/Zutfen-LLC/Strata/pull/6)
is separate and is not included in this branch. Do not combine their measurements.

**All builds, executable tests, inference and measurements are NOT RUN locally.** Both inference
hosts were reported powered off for a RAM swap while this source was prepared. No immediate remote
execution is requested. After they return, Hermes uses its existing authorized access, retained
caches/assets and runtime driver. No new network/security permission, package installation, host
setting change, supervisor, merge or deployment is part of this handoff.

Hardware update supplied by the owner: 02 now has eight 32 GB DIMMs (256 GB installed; post-boot
verification pending), and 05 reports 94.2 GiB. Neither has a resident model after shutdown. The former
05 partition of 116,135,731,200 bytes cannot fit on 05 now. Small fixtures/builds may still use 05;
all full-model loading and inference comparisons use **02's local worker** and the unchanged validated
placement, after the parent/Hermes verifies the new hardware and memory headroom. Do not reload that
partition on 05 or assume installed capacity equals available memory.

Three independent focused static reviews found no blocking implementation defect in loading
lifecycle, concurrency/memory bounds, or specification/tests/handoff. The failure-test limits below
remain explicit. Static review and `git diff --check` are not executable validation.

## Source, old binaries and caches

Resolve the current CPU builder's dedicated disposable source and matching retained cache from its
accepted receipt. Previously the CPU builder was 05; the CUDA builder was 03. Runtime roles and paths
must come from the current receipt. Preserve old worker/engine ELFs, source/dependency revisions,
build settings and full SHA-256 hashes in a new result directory. Keep source/assets/model shards,
trusted manifests, partition files, user changes and unrelated services intact. Do not retarget a
CMake cache to a different source path or clean/fetch dependencies. A missing compatible cache,
toolchain, source binding, runtime library or authorized access is a blocker.

Use one build job and the retained resource guard. Set `SRC`, `BUILD`, `BIN`, `CANDIDATE` to recorded
absolute paths on the selected builder; `BIN` must be new. After fetching the published commit through
the existing authorized path:

```sh
set -euo pipefail
: "${SRC:?dedicated clean source checkout}"
: "${BUILD:?matching retained CPU cache}"
: "${BIN:?new candidate output path}"
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
cmake --build "$BUILD" --parallel 1 --target strata-expert-worker worker_loader_test \
  worker_local_test worker_residency_test worker_service_residency_test worker_compat_test
```

Inspect configure output for unexpected dependency/toolchain changes. Do not overwrite saved ELFs.
Keep logs, exit status, elapsed/RSS and candidate hashes; stop dependent work at the first failing gate.

## Focused gates first

On the CPU builder, use only disposable synthetic fixture services:

```sh
ctest --test-dir "$BUILD" -V -j1 -R \
  '^(worker_loader_(threads|binary)_(1|2|4)|worker_local_test|worker_local_binary_test|worker_residency_test|worker_service_residency_test|worker_compat_test)$'
python3 "$SRC/tests/worker/worker_control_test.py" \
  --worker "$BIN/strata-expert-worker" --control "$SRC/tools/worker_control.py"
```

The three explicit loader configurations compare exact destination bytes/digests and guard bytes,
count loaded/verified bytes, and force cancellation/callback failure with every lane outstanding.
They immediately reuse the same arena/source after return, truncate an already opened file, and
exercise real service READY refusal, corrupt source, missing/corrupt roles, fallback, fresh setup,
resident reconnect with unchanged load/hash/allocation counts, and unload. Existing serial local,
residency and legacy protocol fixtures remain. CLI negative cases reject out-of-range/invalid counts.
No global allocation counting or storage speed claim is made by these page-cache-sized fixtures.
The three binary cases pass the explicit counts through the real worker CLI and local-source load.

Thread-launch exception cleanup is covered by static inspection, not an injected executable case.
Active service stop/disconnect/deadline polling at multiple lanes is not directly forced by the new
fixture; cancellation and exception reaping are tested at the synchronous loader seam. Retained
service cancellation/deadline fixtures still exercise the default serial path. Report these coverage
limits separately; do not turn fixture success into a claim of exhaustive failure testing.

On the existing CUDA builder, preserve its own source/cache/output bindings and incrementally link
`strata` with the new shared worker library. This requires no model launch. Use its separate retained
native-OFF cache if available; do not build a new environment for it. Report unavailable HIP/SYCL,
Windows/Metal and other platform compilation as NOT RUN. Transfer only the needed candidate worker
ELF through the existing approved path, require matching full source/destination hashes and verify
its ISA/runtime compatibility with the selected runtime CPU. A new candidate engine is unnecessary
for the loading measurement: keep the validated runtime engine and driver fixed.

## Independent 1 / 2 / 4 / 1 loading comparison

After 02 returns and correctness passes, verify its reported eight DIMMs and usable capacity, then
record the post-swap RAM inventory, CPU policy, pool
size/affinity, topology, storage/filesystem and available memory. Establish a fresh same-day baseline;
the pre-swap load times cannot serve as its matched control. Use 02's validated local-worker placement,
not the former 05 partition route. Confirm no model is resident initially; resolve the intended worker
service/guardian and absence or identity of its PID before service operations. The operator must authorize named startup/replacement
and reload actions in that execution session. Keep baseline ELFs and exact current options available
for rollback. A resident cache hit skips this work and is not a loading measurement.

Use the SAME candidate worker ELF, retained engine and startup/READY driver in four serial arms,
changing only `--local-load-threads 1`, `2`, `4`, then `1`. Keep model shards, trusted manifest hash,
ownership/partition, arena budget, inference pool count, CPU/NUMA/clock/power policy, route and setup
deadlines fixed. Each arm starts with an empty worker arena and reaches the same complete verified
partition. Replacement releases the prior arena; do not keep two full-model arenas or separate
full-model processes. Keep the existing bounded startup deadlines and outer owned timeout. Do not
add inference requests merely to measure loading or run the buffer-reuse A/B in these arms.

Disclose cache conditions for every arm: first load after boot, previously read shards, available
file cache/memory and observed read I/O counters from the existing driver/OS observations. Do not
drop global caches, reboot, change settings or assume cache residency from arm order. If the first
arm reads from storage and later arms use page cache, report them separately. Do not divide logical
expert bytes by wall time and call that device bandwidth. If comparable cache conditions cannot be
established within existing permissions, report a cache-confounded loading diagnostic and the blocker
to a storage comparison. No forced eviction or broad repeat campaign is authorized.

Record per-arm process source/ELF hashes, exact options, complete/ready state and identities, local
bytes/hits/misses, network weight bytes (zero for complete local sources), loaded/verified bytes,
arena bytes/peak/allocation count, read/verify work time, total successful READY wall time, elapsed/RSS
and failures. Read and hash counters are summed lane work; they overlap and may exceed wall time.
Separate resident-hit setup from fresh load. Preserve every arm, failures and the final 1-thread drift;
do not retry to select fast outcomes. Useful CPU weights stay resident after the final successful arm.

Return one flat report with passed/failed/NOT RUN gates, all four load arms, post-swap hardware and
cache disclosures, raw evidence links, state left behind and exact blockers. There is no promised
speedup or assumption that disk throughput scales with threads. Close this startup round separately
from inference buffer reuse before any distributed multi-CPU trial. Stop before merge or deployment.
