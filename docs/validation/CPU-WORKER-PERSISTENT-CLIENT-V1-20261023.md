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

Resolve the existing dedicated inferswarm02 source checkout and matching CUDA build cache from the
retained receipt. Use no user working tree. Stop if the source has unrelated changes, if the cache does
not identify its expected source/toolchain, or if the full candidate SHA is unavailable. Preserve every
old executable. Bind `CANDIDATE`, `SRC02`, `BUILD02`, `RUN02` and a new empty `BIN02`; `RUN02` and
`BIN02` must be candidate-specific absolute paths. Fetch only the published candidate, detach at it,
and verify its base:

```sh
set -euo pipefail
: "${CANDIDATE:?full 40-character candidate SHA}"
: "${SRC02:?dedicated clean source checkout on inferswarm02}"
: "${BUILD02:?existing matching CUDA build cache on inferswarm02}"
: "${RUN02:?new candidate result directory on inferswarm02}"
: "${BIN02:?new candidate binary directory on inferswarm02}"
test "${#CANDIDATE}" -eq 40
test -z "$(git -C "$SRC02" status --porcelain)"
git -C "$SRC02" fetch origin "$CANDIDATE"
test "$(git -C "$SRC02" rev-parse FETCH_HEAD)" = "$CANDIDATE"
git -C "$SRC02" merge-base --is-ancestor 9dd56a946b025bd56e4bd0174997d6b9e51f4e02 "$CANDIDATE"
git -C "$SRC02" switch --detach "$CANDIDATE"
test ! -e "$RUN02" && mkdir -m 700 "$RUN02"
test ! -e "$BIN02" && mkdir -m 700 "$BIN02"
grep -E '^(CMAKE_HOME_DIRECTORY|CMAKE_GENERATOR|STRATA_ENABLE_CUDA|CMAKE_CUDA_ARCHITECTURES):' \
  "$BUILD02/CMakeCache.txt" | tee "$RUN02/cmake-cache-binding.txt"
cmake -S "$SRC02" -B "$BUILD02" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN02"
timeout --signal=TERM --kill-after=10s 1200s \
  cmake --build "$BUILD02" --parallel 1 --target strata expert_tier_test worker_dispatch_test
sha256sum "$BIN02/strata" "$BIN02/expert_tier_test" "$BIN02/worker_dispatch_test" \
  | tee "$RUN02/candidate-binaries.sha256"
```

Do not overwrite or relabel the baseline engine/worker binaries. Record the old engine and resident
worker executable paths, `/proc` start times and SHA-256 values separately from the three candidate
outputs. If the retained cache would place outputs over old binaries despite the explicit runtime
directory, stop and report that blocker rather than reconfigure broadly.

## 2. Focused synthetic gates

Run serially before any model request. Preserve stdout, stderr, elapsed time, maximum RSS and exit code.
Do not repeat a failed gate to obtain a pass.

```sh
timeout --signal=TERM --kill-after=5s 600s "$BIN02/expert_tier_test"
timeout --signal=TERM --kill-after=5s 180s "$BIN02/worker_dispatch_test"
STRATA_DEC_BATCH=0 timeout --signal=TERM --kill-after=5s 180s "$BIN02/worker_dispatch_test"
```

The tier fixture uses the real client/service and checks one persistent thread across repeated requests,
zero-work skip, copied input lifetime, post-before-finish ordering, typed refusal, malformed and stale
replies, disconnect, deadline, no mergeable partial after failure, and idle/in-flight cleanup. The two
dispatcher invocations keep the existing device- and host-zeroing coverage. Require exit code zero.
These are synthetic correctness checks, not model equivalence or performance evidence.

## 3. Retained-worker checks and six-request run

Before launching the candidate engine, identify the existing inferswarm02 resident worker from the
retained receipt. Confirm its PID, parent/guardian, `/proc/PID/exe`, start time, command line, loopback
listener and binary SHA match. Record its status without unloading. A missing or changed identity is a
blocker: do not launch a replacement or cold-load weights.

Recover the exact six-request command, raw prompt-token files, request order, sampling parameters,
placement and GPU settings from `BASE_REPORT` and its retained artifacts. Do not reconstruct prompts
from text, retokenize them, alter seeds, or invent flags in this handoff. Change only the engine path to
`$BIN02/strata` and candidate output paths. Run the six requests once, serially, against the same
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
