# Hermes handoff: parallel CPU expert workers

Use the exact 40-character `CANDIDATE` from the coding handoff only after it is explicitly published.
This coding task did not configure, build, run tests, contact the fleet, touch a service or load a model.
The existing real-network baseline is independent: do not stop it, reuse its ports, modify its checkout,
or mix its logs with this validation.

This is a serial, fixture-first gate. It preserves each host's existing generator, compiler, pinned ggml
checkout and native-ON options from its successful cache. Do not install, download, merge, deploy, or
invent a new toolchain path. Stop at the first failure and preserve the command, output, exit code,
candidate SHA and cache. Native-OFF, HIP, SYCL and Windows remain unvalidated by this handoff.

## 1. Resolve existing caches and pin source identity

Hermes supplies absolute, dedicated `SRC` and `BUILD` paths on each selected host. Use the CPU-only
native-ON caches on inferswarm02 and inferswarm05. Use the existing native-ON CUDA cache on the engine
host. Resolve source paths from `CMAKE_HOME_DIRECTORY`; do not guess them. Require clean sources and the
canonical Zutfen-LLC origin before advancing a dedicated checkout to the published candidate.

```sh
set -euo pipefail
: "${CANDIDATE:?published 40-character candidate SHA}"
: "${SRC:?dedicated source checkout}"
: "${BUILD:?existing configured build directory}"
test "${#CANDIDATE}" -eq 40
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:[^=]*=//p' "$BUILD/CMakeCache.txt")" = "$SRC"
test -z "$(git -C "$SRC" status --porcelain)"
case "$(git -C "$SRC" remote get-url origin)" in
  git@github.com:Zutfen-LLC/Strata.git|ssh://git@github.com/Zutfen-LLC/Strata.git|https://github.com/Zutfen-LLC/Strata.git|https://github.com/Zutfen-LLC/Strata) ;;
  *) echo 'STOP: unexpected origin'; exit 1 ;;
esac
git -C "$SRC" fetch origin "${PUBLISHED_BRANCH:?published branch}"
test "$(git -C "$SRC" rev-parse FETCH_HEAD)" = "$CANDIDATE"
git -C "$SRC" merge-base --is-ancestor 9bf66c042c06e1b3a1840d131b7444333ce9e502 "$CANDIDATE"
git -C "$SRC" switch --detach "$CANDIDATE"
```

Before configuring, copy `CMakeCache.txt` to a new owned log directory and record the generator,
compilers, `STRATA_GGML_DIR`, `STRATA_NATIVE_EXPERTS`, CUDA/HIP/SYCL settings and CUDA architectures.
Require native experts ON. On the engine host require the retained CUDA architecture/settings from its
successful cache; do not substitute the repository default. On the CPU hosts preserve CUDA/HIP/SYCL OFF.

## 2. CPU-only build and focused fixtures on 02, then 05

Run this block separately on inferswarm02 and inferswarm05, in that order. `RUN` must be a new log
directory outside source/build trees. The retained CPU caches use their recorded generator; no `-G` is
passed here. The commands are serial and use only synthetic fixtures.

```sh
set -euo pipefail
: "${RUN:?new owned log directory}"
mkdir "$RUN"
cp "$BUILD/CMakeCache.txt" "$RUN/cache-before.txt"
export CMAKE_BUILD_PARALLEL_LEVEL=1 CTEST_PARALLEL_LEVEL=1 OMP_NUM_THREADS=2
run_logged() {
  label=$1; shift; started=$SECONDS
  set +e
  "$@" >"$RUN/$label.log" 2>&1
  rc=$?
  set -e
  printf '%s\t%s\t%s\n' "$label" "$rc" "$((SECONDS-started))" >>"$RUN/results.tsv"
  return "$rc"
}
: >"$RUN/results.tsv"
run_logged configure cmake -S "$SRC" -B "$BUILD" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
run_logged build cmake --build "$BUILD" --target strata-expert-worker expert_tier_test worker_local_test \
  worker_residency_test -j1
run_logged tests ctest --test-dir "$BUILD" -V -j1 --no-tests=error \
  -R '^(expert_tier_test|worker_local_test|worker_local_binary_test|worker_residency_test)$'
sha256sum "$BUILD/strata-expert-worker" "$BUILD/expert_tier_test" >"$RUN/binaries.sha256"
cp "$BUILD/CMakeCache.txt" "$RUN/cache-after.txt"
git -C "$SRC" rev-parse HEAD >"$RUN/source.sha"
```

`expert_tier_test` must report real single-, two- and three-endpoint group checks; disjoint/exhaustive,
capacity and endpoint validation; sequential setup keepalive; every active request published before finish;
per-worker bitwise direct partials; ordered grouped numerical comparison; no-work; and mixed-success
all-or-nothing draining for refusal, stale identity, malformed reply, timeout and disconnect. Do not
describe a fixture pass as real-network or model validation.

## 3. Engine serial build and focused dispatcher fixtures

Only after both CPU stages pass, use the existing engine-host CUDA/native-ON build. Preserve its cache,
including its verified CUDA architecture and pinned ggml path.

```sh
set -euo pipefail
: "${RUN:?new owned engine log directory}"
mkdir "$RUN"
cp "$BUILD/CMakeCache.txt" "$RUN/cache-before.txt"
export CMAKE_BUILD_PARALLEL_LEVEL=1 CTEST_PARALLEL_LEVEL=1 OMP_NUM_THREADS=2
run_logged() {
  label=$1; shift; started=$SECONDS
  set +e
  "$@" >"$RUN/$label.log" 2>&1
  rc=$?
  set -e
  printf '%s\t%s\t%s\n' "$label" "$rc" "$((SECONDS-started))" >>"$RUN/results.tsv"
  return "$rc"
}
: >"$RUN/results.tsv"
run_logged configure cmake -S "$SRC" -B "$BUILD" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
run_logged build cmake --build "$BUILD" --target strata worker_dispatch_test -j1
run_logged tests ctest --test-dir "$BUILD" -V -j1 --no-tests=error \
  -R '^(worker_dispatch_test|worker_dispatch_host_zero_test)$'
run_logged device-plan env STRATA_VERIFY_DEVICE_PLAN=1 "$BUILD/worker_dispatch_test"
sha256sum "$BUILD/strata" "$BUILD/worker_dispatch_test" >"$RUN/binaries.sha256"
cp "$BUILD/CMakeCache.txt" "$RUN/cache-after.txt"
git -C "$SRC" rev-parse HEAD >"$RUN/source.sha"
```

Return one fixture report with exact host/source/build/cache/compiler/ggml identities, commands, exit
codes, elapsed times, named test outcomes, binary hashes and log locations. Keep passed, failed, blocked
and not run separate. Stop here for review.

## 4. Proposed full-model split comparison — review required before execution

Do not execute this section until the fixture report is reviewed and a new full-model window is approved.
In particular, do not touch the baseline session currently running with existing binaries.

The current inferswarm02 resident arena holds the complete 22,238-expert / 116,135,731,200-byte
complement, and inferswarm05 may hold the same exact partition. A split produces different partition
digests. Exact residency validation will correctly reject reuse, so the approved run must use a
controlled unload/expiry and fresh load on both workers. Do not weaken validation, treat the old full
arena as a superset, or reload either running service implicitly.

For an approved run, copy the verified baseline engine command and change only the candidate binary,
owned ports/log directory, and CPU endpoint arguments. Preserve the model, manifest, profile, GPU
placement, cache count, routing/window/speculation/KV settings, prompts and sampling. Resolve capacities
from each worker's actual RAM allowance and choose speed weights from a reviewed single-worker baseline;
do not assume the hosts are equal. Require `CAP02 + CAP05 >= 22238`.

```sh
: "${CAP02:?maximum expert blobs approved for inferswarm02}"
: "${CAP05:?maximum expert blobs approved for inferswarm05}"
: "${SPEED02:?positive reviewed relative speed}"
: "${SPEED05:?positive reviewed relative speed}"
test "$((CAP02 + CAP05))" -ge 22238
# Append to the otherwise identical verified candidate engine command. PORT02 and PORT05 are owned,
# already-verified controller/loopback endpoints; this handoff does not create or assume tunnels.
--remote-cpu-experts "127.0.0.1:${PORT02}@${CAP02}:${SPEED02}" \
--remote-cpu-experts "127.0.0.1:${PORT05}@${CAP05}:${SPEED05}"
```

Run the reviewed single-endpoint candidate control first, then the split candidate with the same inputs.
Compare request success, output/window hashes under the existing deterministic protocol, tokens/s,
per-worker entries, request/queue/exchange/wait timing, setup/reuse bytes and worker CPU service. The
multi-worker final reduction groups floating-point additions differently from one worker: require each
worker's fixture partial to remain bitwise correct, but use the project's reviewed end-to-end numerical
criterion for the full-model comparison rather than inventing a new threshold or demanding global
bitwise identity. Preserve both good and bad runs; make no speed claim from a single sample.
