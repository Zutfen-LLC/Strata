# CPU worker persistent client: Hermes validation handoff

This is the bounded remote-validation handoff for the additive CPU-worker client candidate based on
`9dd56a946b025bd56e4bd0174997d6b9e51f4e02`. Use the full immutable candidate SHA supplied with this
handoff. Do not substitute `main` or another upstream revision. The code and fixtures were statically
reviewed on Valinor and are **NOT RUN**. Do not build or run them on Valinor or Geekom.

The candidate changes only the engine-side CPU-worker tier. It keeps one request slot and one
asynchronous communication thread after the first nonempty post. It does not change the worker wire
protocol, expert math, reduction order, quantization, placement, worker count, socket settings or
resident-weight lifecycle. The existing resident worker on inferswarm02 must not restart, unload or
receive a signal. Do not access other fleet services. No merge or deployment is authorized.

## 1. Preserve identities and build serially

On the Hermes controller, retain the existing baseline report and its referenced prompt-token,
sampling, binary and output artifacts:

```sh
BASE_REPORT=/home/zutfen/strata-cpu-loopback-20261023T163101Z/REPORT.txt
test -s "$BASE_REPORT"
sha256sum "$BASE_REPORT"
```

Build on inferswarm03 by default. Resolve its existing dedicated source checkout and retained compatible
sm_86 CUDA 13.1 build cache from the accepted build receipt. The copied validated baseline engine on
inferswarm02 does **not** establish a matching source/build cache there. A missing 02 cache is not a
blocker. Use 02 for the build only if a matching existing cache is positively identified and current
resource headroom fits; never configure over the live worker's files.

Use no user working tree. Stop if the selected build source has unrelated changes, if its cache does not
identify its expected source/toolchain, or if the full candidate SHA is unavailable. Preserve every old
executable. On 03 bind `CANDIDATE`, `SRC03`, `BUILD03`, `RUN03` and a new empty `BIN03`; `RUN03` and
`BIN03` must be candidate-specific absolute paths. Fetch only the published candidate, detach at it,
and verify its base. Do not install or fetch dependencies, start a real model, or touch unrelated
services. Record available memory/cgroup headroom first and use the retained build's existing resource
guard; insufficient headroom blocks the build host without authorizing a broad cleanup or another host:

```sh
set -euo pipefail
: "${CANDIDATE:?full 40-character candidate SHA}"
: "${SRC03:?dedicated clean source checkout on inferswarm03}"
: "${BUILD03:?retained compatible sm_86 CUDA 13.1 cache on inferswarm03}"
: "${RUN03:?new candidate result directory on inferswarm03}"
: "${BIN03:?new candidate binary directory on inferswarm03}"
test "${#CANDIDATE}" -eq 40
test -z "$(git -C "$SRC03" status --porcelain)"
git -C "$SRC03" fetch origin "$CANDIDATE"
test "$(git -C "$SRC03" rev-parse FETCH_HEAD)" = "$CANDIDATE"
git -C "$SRC03" merge-base --is-ancestor 9dd56a946b025bd56e4bd0174997d6b9e51f4e02 "$CANDIDATE"
git -C "$SRC03" switch --detach "$CANDIDATE"
test ! -e "$RUN03" && mkdir -m 700 "$RUN03"
test ! -e "$BIN03" && mkdir -m 700 "$BIN03"
grep -E '^(CMAKE_HOME_DIRECTORY|CMAKE_GENERATOR|STRATA_ENABLE_CUDA|STRATA_NATIVE_EXPERTS|CMAKE_CUDA_ARCHITECTURES):' \
  "$BUILD03/CMakeCache.txt" | tee "$RUN03/cmake-cache-binding.txt"
cmake -S "$SRC03" -B "$BUILD03" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN03"
timeout --signal=TERM --kill-after=10s 1200s \
  cmake --build "$BUILD03" --parallel 1 --target strata expert_tier_test worker_dispatch_test
sha256sum "$BIN03/strata" "$BIN03/expert_tier_test" "$BIN03/worker_dispatch_test" \
  | tee "$RUN03/candidate-binaries.sha256"
```

The three CMake target names above are exact at the pinned source. Do not substitute historical make
names. Do not overwrite or relabel baseline engine/worker binaries. If the selected cache would place
outputs over old binaries despite the explicit runtime directory, stop and report that blocker rather
than reconfigure broadly.

## 2. Focused synthetic gates

Run serially before any model request. Preserve stdout, stderr, elapsed time, maximum RSS and exit code.
Do not repeat a failed gate to obtain a pass.

Use CTest's exact pinned names on the selected build host. The fixture services are dedicated and
isolated; they may listen only on their fixture-selected loopback ports. They must not attach to the 02
resident worker or any unrelated service.

```sh
timeout --signal=TERM --kill-after=5s 600s \
  ctest --test-dir "$BUILD03" -V -j1 -R '^expert_tier_test$'
timeout --signal=TERM --kill-after=5s 900s \
  ctest --test-dir "$BUILD03" -V -j1 -R '^worker_dispatch_test$'
timeout --signal=TERM --kill-after=5s 900s \
  ctest --test-dir "$BUILD03" -V -j1 -R '^worker_dispatch_host_zero_test$'
```

These are exactly the CTest names declared by the pinned `CMakeLists.txt`; the host-zero case already
sets `STRATA_DEC_BATCH=0` through CMake. If the optional verified 02 cache is selected instead, bind the
same variables with suffix `02` and substitute them consistently in sections 1 and 2. Do not mix source,
cache, runtime-output or test directories between hosts.

The tier fixture uses the real client/service and checks one persistent thread across repeated requests,
zero-work skip, copied input lifetime, post-before-finish ordering, typed refusal, malformed and stale
replies, disconnect, deadline, no mergeable partial after failure, and idle/in-flight cleanup. The two
dispatcher invocations keep the existing device- and host-zeroing coverage. Require exit code zero.
These are synthetic correctness checks, not model equivalence or performance evidence.

## 3. Retained-worker checks and six-request run

After all three fixture gates pass on the selected build host, copy **only** the candidate `strata`
engine from the default 03 build to a new owned candidate directory on inferswarm02 through the existing
approved controller-mediated transfer path. If the optional positively verified 02 cache was used,
keep its uniquely named engine in that new candidate directory and skip the cross-host copy.
Do not copy a worker, fixture binary, library, source tree or build cache. Record the source SHA-256,
copy under a candidate-specific filename, and require the destination SHA-256 to match. Existing 02
runtime libraries must satisfy the engine; an incompatibility is a blocker, not permission to install or
copy dependencies. Preserve the validated baseline engine separately.

For example, from the controller, bind `BIN03` and `RUN02` to recorded absolute paths on their named
hosts and use the existing host aliases:

```sh
: "${RUN02:?new owned candidate directory on inferswarm02}"
ENGINE02="$RUN02/strata-$CANDIDATE"
SRC_HASH=$(ssh inferswarm03 "sha256sum '$BIN03/strata'" | awk '{print $1}')
ssh inferswarm02 "test ! -e '$RUN02' && mkdir -m 700 '$RUN02'"
scp -3 "inferswarm03:$BIN03/strata" "inferswarm02:$ENGINE02"
ssh inferswarm02 "chmod 700 '$ENGINE02'"
DST_HASH=$(ssh inferswarm02 "sha256sum '$ENGINE02'" | awk '{print $1}')
test "$SRC_HASH" = "$DST_HASH"
```

If the retained controller uses a different approved transfer wrapper, perform the same one-file copy
and two-host hash comparison through that wrapper instead of inventing a route.

Before launching the candidate engine, identify the existing inferswarm02 resident worker from the
retained receipt. Confirm its PID, parent/guardian, `/proc/PID/exe`, start time, command line, loopback
listener and binary SHA match. Record the old engine and resident worker executable paths, `/proc` start
times and SHA-256 values separately from the candidate. Record worker status without unloading. A
missing or changed identity is a blocker: do not launch a replacement or cold-load weights.

Recover the exact six-request command, raw prompt-token files, request order, sampling parameters,
placement and GPU settings from `BASE_REPORT` and its retained artifacts. Do not reconstruct prompts
from text, retokenize them, alter seeds, or invent flags in this handoff. Change only the engine path to
`$ENGINE02` and candidate output paths. Run the six requests once, serially, against the same
existing localhost endpoint and resident worker. Preserve every request's exit status, token IDs,
output, decode timing and engine report. Do not run a cold reload, topology sweep or extra model query.

While the candidate engine is live, the communication thread is named `strata-cpu-io`. Record the
engine PID and inspect its task affinity without changing it:

```sh
for task in /proc/"$ENGINE_PID"/task/*; do
  test "$(cat "$task/comm")" = strata-cpu-io || continue
  printf 'tid=%s ' "${task##*/}"
  grep -E '^(Cpus_allowed_list|Mems_allowed_list):' "$task/status"
done
```

Require exactly one matching task after the first nonempty post. Compare its allowed CPUs with the
engine dispatcher/caller record and with the retained 12-worker pool/caller affinity. Report any shared
physical core; do not repin either process during this run. The engine's final aggregate line also gives
`communication thread ... id=<tid> starts=1`, queue time, exchange time and caller finish wait. These
timings overlap and are **not** additive wall-time partitions.

## 4. Evaluation and retention

Evaluate numerical/output equivalence before performance:

1. Compare request-by-request generated token IDs and terminal status with the retained baseline.
2. Report exact matches, divergences and the first divergent token separately. Do not discard divergent
   requests to improve the performance summary.
3. Only then compare all-six decode throughput and the already-defined token-identical subset with the
   retained `11.5656` and `11.5124` tok/s localhost-worker results. Keep the native-local `14.0190` and
   token-identical `14.0380` tok/s values as context, not as candidate results.
4. Correlate throughput with aggregate queue/exchange/finish-wait counters and observed affinity. Do not
   claim thread churn caused the whole gap merely because performance improved.

Retain the candidate binaries, hashes, source SHA, commands, six outputs, token comparisons, engine
report, affinity snapshot, resident-worker identity before/after, and failure logs. Confirm the worker
PID/start time/executable and resident status are unchanged after the run. Leave the worker and its
weights resident. Report PASS/FAIL/**NOT RUN** per gate and any exact blocker. Do not merge, deploy,
restart, unload, clean retained weights, or access another service.
