# Focused Hermes check: loader affinity phases

Base: main `ef9180b34af30ce5b2cac1e3129666ba21584e0b`.
Branch: `fix/harness-loader-affinity`; use the full immutable published head as `CANDIDATE`.
This is a separate harness-only patch. PR6 candidate `1094d9cc734868016d8b7997240dbc08197732f9`, its
binaries, and the active comparison stay unchanged. No C++ edits or model rerun are requested.

Source diagnosis: `launch_worker()` applied final caller affinity before SETUP. The serving caller
creates `local_loader.cpp`'s helper threads, which inherit its mask. The parent then supplied live
confirmation: argv selected four loading lanes, but caller mask was {0}, all three transient loader
helpers had Cpus_allowed_list 0, and the inference pool was idle. These are supplied observations,
not a run or private-log inspection by Codex. This is harness affinity inheritance, not a product
loader thread-count failure. A0-load completed READY/QUIT with zero GEN; the owner chose to finish
the current A0/B/A1 campaign unchanged because startup is outside inference timing and masks match.
Do not modify or interrupt it.

For future launches, cold SETUP gives the caller the union of configured caller/pool CPUs; it does
not expand beyond those configured CPUs. Pool placement stays the same. After READY, when loader
helpers have joined, apply final caller/pool masks before any GEN, also for zero-GEN load entries.
Restoration requires the retained engine-owner handle, broker/engine receipts and idle READY state;
it sends no STATUS/STOP/unload during the engine session. Worker task count must be back to N+2.
Warm resident compute affinity, engine affinity and A0/B/A1 policy remain unchanged. Wider eligibility
alone does not prove physical-core parallelism or a speedup.

**All executable checks are NOT RUN on Geekom/Valinor.** Local checks are source review, AST parsing,
exact staged-file/diff inspection and whitespace/publication checks. Historical lifecycle handoff and
receipts remain at the immutable base; they are not validation of this patch.

On the existing authorized Hermes controller, use clean exact-SHA source and the retained Python.
No GPU, native binary, model, package install, fleet action or system affinity/policy change is needed.
The inheritance regression changes only its own tiny child processes/threads and needs at least two
CPUs in the controller's allowed set. A skip must be reported as a skip, not an inheritance PASS.
Run this selection once under the existing owned output/timeout guard, with new bounded receipts:

```sh
set -euo pipefail
: "${SRC:?clean candidate source on controller}"
: "${CANDIDATE:?full published SHA}"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
(
  cd "$SRC/tools/validation-harness/tests"
  timeout --signal=TERM --kill-after=5s 120s python3 -B -m unittest -v \
    test_harness.HarnessTests.test_loader_inherits_setup_union_and_ready_restores_compute_masks \
    test_harness.HarnessTests.test_ready_affinity_precedes_gen_and_also_runs_for_zero_gen_load \
    test_harness.HarnessTests.test_port_probe_follows_owned_stop \
    test_harness.HarnessTests.test_worker_controls_never_queue_behind_engine_session
)
```

The real child must report four helper masks equal to the configured two-CPU union during setup,
then final singleton caller/pool masks after READY; the synthetic engine mask must stay unchanged.
An active request must refuse affinity restoration without changing a TID. The boundary case must
apply READY affinity before any GEN and even with no GEN; a refusal must prevent GEN. The launch
case must select setup phase. Keep setup and final affinity receipts separate.

Return one flat receipt with source SHA, duration, tests/pass/fail/skip/NOT RUN, exit status, cleanup
and first-failure logs. Preserve failures; do not repeat to select a pass. No automatic hardware stage,
model campaign, deployment or merge is authorized by this handoff. This correction is for subsequent
launches after validation, not an intervention in the running PR6 comparison.
