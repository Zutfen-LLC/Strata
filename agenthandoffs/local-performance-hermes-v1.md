# Bounded local CUDA / Vulkan timing checkpoint

Prepared for parent review/dispatch only. No fleet run by this task. Draft PR
stacks on PR43 docs c76d03c7. Expert placement, reduction and precision are
unchanged: portable dense/coordinator layers on Vulkan0/1, all experts CPU,
host PLE. PR42's adapter and tests are carried from fb518b1b; its existing owned
native lifecycle is reused. There is no new service or persistent CLI mode.
Original logit-parity gates remain FAILED. Performance cannot waive them.

## Workload and limits

Use the **existing** frozen twelve-case suite from the completed v2 screen:
`/home/hermes/portable-quality-ac85ff3-v2/suite` on02 and
`/srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v2/suite` on05.
The adapter revalidates all request, manifest, scorer and vocabulary bindings
before selecting **case 1 only**: `Return only the integer: 17 * 23.`, with the
unchanged template. It issues exactly four identical GEN32 requests per engine,
fresh engine/state each time: case-01 excluded warmup; case-02..04 measured.
Keep early EOS and report actual counts; never pad, retry or run other cases.
This short prompt is a bounded latency sample, not a long steady-state TPS test.

Native starts one owned loopback CPU worker at 127.0.0.1:17431, LOAD once, then
four fresh local engines, then owned cleanup. An occupied port is a STOP. Portable
opens/maps/loads a fresh model each time; it cannot reuse a loaded CLI session.
No cache flush, mlock, cache archive or artificial persistence. One excluded
warmup permits warm OS/file caches; later portable trials still have new expert
verification tables. Native retains CPU worker state after LOAD. This distinction
must accompany results; this measures the existing local execution paths.

Portable: 300 s per case, 1200 s suite; external 1250 s deadline plus 15 s kill
allowance. All four trials, **including setup**, fit within the verified 1800 s
high-clock window. Build and synthetic tests occur before that window opens.
Native: existing startup900 / LOAD or READY1800 / request300 / stop420 s ceilings,
further bounded by external 1800 s arm deadline and 1320 s cleanup allowance.
The retained driver cleans up the engine before the adapter cleans up its
separately-sessioned worker. Before DONE this can use 420 + 420 = 840 s of timed
waits. After READY and DONE, engine_stage can call quit_engine: QUIT waits 420 s,
then stop_process_group can wait another 420 s before KILL; the worker can then
wait 420 s. Allow 1260 s plus 60 s margin, rather than killing the adapter before
worker cleanup. These are timed wait allowances; final reap after KILL relies on
the OS. This native-only allowance does not enlarge the AMD high-clock window.
No automatic retries. SIGTERM enters owned cleanup. Inspect STOP receipts first.

Preserve inherited portable six-shard working-set +32 GiB anonymous +8 GiB reserve
admission, cgroup available, swap and per-die guards, 0.2 s sampling, caps7680 MiB,
threads4, expert-workers12, split32/16 and strict placement checks. Preserve native
retained argv/env, prefill64, speculative/greedy flags and CPU entry-row policy.
Do not stop unrelated services or resident models. No remote GPU workers.

## Source and candidate build, before clocks

Parent exports the **complete committed tree** from this draft PR; transport it
with existing approved file transport. Do not select individual Python files:
quality.py imports tools/strata_tokenizer.py and validation-harness/harness helpers.
Freeze the export SHA and its tar SHA in the receipt. No fleet Git/network needed.

```sh
# Valinor, checkout containing the reviewed commit
REV=$(git rev-parse HEAD)
git archive --format=tar "$REV" > /tmp/strata-local-performance-source.tar
sha256sum /tmp/strata-local-performance-source.tar
```

Hermes uses fresh SOURCE05, BUILD05 and CHECKS05 directories below
`/srv/models/inferswarm-artifacts/local-performance-v1/`. Abort if output roots
already exist. Preserve the qualified PR43 binary at its original path and hash
cb71b55bd7309320a81446c117728e493ea1d19c8f1ec0d54c7ef2836c8734c3.
The candidate has a **new** SHA; do not label it the physically qualified binary.

Use the retained CMAKE executable, the same LLAMA_SRC pinned dependency and
MODEL_CMAKE_PREFIX05 / MODEL_CXX_FLAGS05 as the accepted PR43 build. No dependency
fetch/download or tool installation. Set these resolved variables in the receipt.

```sh
set -euo pipefail
umask 077
: "${SOURCE05:?complete export}" "${BUILD05:?fresh path}" "${CHECKS05:?fresh path}"
: "${CMAKE:?retained executable}" "${LLAMA_SRC:?retained dependency}"
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test ! -e "$BUILD05"
mkdir "$CHECKS05"
"$CMAKE" -S "$SOURCE05" -B "$BUILD05" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON -DSTRATA_PORTABLE_QSA=ON \
  -DSTRATA_BUILD_TESTS=ON -DGGML_VULKAN=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF \
  -DSTRATA_ENABLE_SYCL=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" -DCMAKE_PREFIX_PATH="${MODEL_CMAKE_PREFIX05:-}" \
  -DCMAKE_CXX_FLAGS="${MODEL_CXX_FLAGS05:-}" > "$CHECKS05/configure.log" 2>&1
"$CMAKE" --build "$BUILD05" --parallel 2 --target portable_timing_cli_test \
  > "$CHECKS05/build.log" 2>&1
"$(dirname "$CMAKE")/ctest" --test-dir "$BUILD05" --output-on-failure \
  -R '^portable_timing_cli_test$' > "$CHECKS05/ctest.log" 2>&1
sha256sum "$BUILD05/portable_model" > "$CHECKS05/binary.sha256"
```

Run the heavy synthetic test alone on its host. This test uses six sparse
synthetic shards, CPU execution, two prompt tokens and
early EOS. It compares timed/default stdout and every logit byte, and checks
actual counts and finite nonnegative timers. A Vulkan-linked test is compilation
and CPU-path evidence; it does not establish physical GPU correctness.

## Fleet commands, parent/Hermes only

Use existing PYTHON02/PYTHON05 with regex installed. SOURCE02 is the complete same
export on02. RUN02 may be a fresh private `/home/hermes/local-performance-v1`;
RUN05 must be fresh `/srv/models/inferswarm-artifacts/local-performance-v1/run`.
The adapter creates the output root with mode0700; do not precreate it. Hash both
existing suite files on both hosts and confirm the same hashes before either arm.

```sh
# 02: existing native engine and driver, no clock changes here
export GGUF02=/srv/models/qwen38-q8/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
export SUITE02=/home/hermes/portable-quality-ac85ff3-v2/suite
sha256sum "$SUITE02/requests" "$SUITE02/quality_manifest"
timeout --signal=TERM --kill-after=1320s 1800s \
  "$PYTHON02" -B "$SOURCE02/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF02" native --benchmark \
  --driver /home/hermes/strata-nd-repeat-20261027-run.driver5.py \
  --suite "$SUITE02" --out "$RUN02"
```

Native engine SHA931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89
and driver SHAb5f1a5c10a0337d6254776ad01a19dd787076f41fd114538ea01f28708a1ae2f
are enforced. Retained config inputs remain
`/home/hermes/NDon-strata-ndab-20261026T161728Z.input-config.json` and
`/home/hermes/strata-pr10-9ed478e-20261025-run/A0-load.config.json`.
No driver main, distributed arm, historical comparison or certificate operation.
Benchmark mode disables first-logit dumps on both engines.

Immediately before05 inference, Hermes owns the **verified** clock operation and
privilege. The two exact controls are
`/sys/bus/pci/devices/0000:07:00.0/power_dpm_force_performance_level` and
`/sys/bus/pci/devices/0000:0b:00.0/power_dpm_force_performance_level`.
Both were auto at handoff. Install EXIT/TERM/INT restoration **before** writing
high, verify both readbacks, sample every valid hwmon temp*_input each second,
and stop/restore auto on sensor failure, temperature >=80 C, failure, completion
or 30-minute deadline. Use a separate 2 s pp_dpm_sclk/pp_dpm_mclk/GPU-busy sampler;
retain it. Prior active samples1500/945 MHz are context, not the new result;
idle300/945 is insufficient. There is no retained prior guard-script path.
Do not broaden this to voltage, power limits or driver changes. If the verified
guard is unavailable, STOP; do not run an unguarded replacement.

Run this command **inside** Hermes's verified finite temperature/clock guard:

```sh
export GGUF05=/srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
export SUITE05=/srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v2/suite
sha256sum "$SUITE05/requests" "$SUITE05/quality_manifest"
BINARY_SHA05=$(sha256sum "$BUILD05/portable_model" | cut -d' ' -f1)
timeout --signal=TERM --kill-after=15s 1250s \
  "$PYTHON05" -B "$SOURCE05/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF05" portable --benchmark --binary "$BUILD05/portable_model" \
  --binary-sha256 "$BINARY_SHA05" --suite "$SUITE05" --out "$RUN05"
```

Verify both controls restored to auto after cleanup, and that active samples show
the intended profile. If interrupted, retain partial output and report STOP;
no retry, extra quality run, service restart or further measurement campaign.

## Timing fields and interpretation

`benchmark.json` retains status, four individual rows, excluded_warmup flags,
actual generated IDs/counts, binary/suite bindings and guards. `summary.json`
also remains. No pooled TPS is emitted. Per-trial comparison must first verify
status complete, same suite binding and token IDs/counts; report differences
explicitly and do not compute a speedup over unequal generated-token counts.
Report median/range over case-02..04 while retaining all three values.

Portable `case-NN/timings.json` schema1:

- setup_ms: before backend initialization through model open, admission and setup
  diagnostics, ending immediately before prompt processing. This excludes process
  startup/argument parsing, teardown, placement/timing file writes.
- prompt_ms / prompt_steps: every prompt token fully consumed, including the last.
- decode_ms / decode_steps: every generated token fully consumed, including EOS or
  the final max-new token. output_count is actual generated/consumed tokens.
- selection_prompt_ms / selection_prompt_steps: up to just before the last prompt
  step; N-1 steps. selection_decode_ms / selection_decode_steps: from there to the
  selection of the final generated output; G forward steps. terminal_feed_ms:
  the remaining final generated-token feed and its normal receipts.

Native `case-NN/result.json` preserves `setup_s` (launch to polled READY, ~2 s
poll granularity), `request_wall_s` (~1 s poll granularity) and raw DONE. Native
benchmark timing rows convert setup_s to setup_ms and carry DONE prompt_ms,
decode_ms, prompt_count and output_count without loading or polling added to TPS.
worker_startup_s and the separate `load.setup_s` belong to one-time native LOAD.
Neither setup measurement implies all mapped file pages are resident.

Static source inspection of fd35f803 generate.cpp: native prompt ends before the
last prompt forward step; decode includes that step and selection/emission of G
outputs (not a final Gth generated feed). Therefore compare native DONE with the
portable **selection_*** fields for the closest work boundary: prompt N-1 steps,
decode G steps. Retain portable full-consumption fields and terminal_feed_ms too.
Portable receipt/device-memory overhead remains included; native scheduling,
prefill batching and speculation remain as configured. These are engine-path wall
times, not isolated kernel rates or a platform-wide speedup. Count semantic
prompt tokens N separately from N-1 prompt forward steps. Compute phase rates
only with their stated counts and positive denominators. No load/setup in TPS.

## Retention and checkpoint

Keep full small private per-trial receipts, stdout/stderr, argv, placements,
raw DONE, hashes, source revision/export hash, build flags, clock/busy/temperature
samples and cleanup/restoration result. Do not post decoded answers or private
config/log payloads to Slack/Git. Record `du -sb` for source, build and results;
new05 bulky artifacts stay on /srv/models,02 small receipts may use /home/hermes.
No new weight copies, logits, resident cache archives or old-artifact deletions.
Source and build may be discarded later only by a separate authorized cleanup;
this task retains them. Parent receives the draft PR and exact revision for
review/dispatch. There is no merge or independent fleet launch.
