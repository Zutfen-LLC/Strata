# Frozen exploratory backend answer screen

This is a prospective run contract, **not a model result**. It addresses issue
[#23](https://github.com/Zutfen-LLC/Strata/issues/23), after PR #31's functional
integration and the additive native diagnostic at `e7715a002aef16620609a1587cf86f47da3dddc0`.
The original functional pin `5139cd08ae1ffa92542562ddadf3d704fa333257`, exact gates,
fixed numeric limits and retained failures remain intact. Parent-supplied CPU
versus Vulkan evidence (max absolute logit difference .698, relative L2 .0678,
same top-1) is not newly measured here and remains a failed numeric comparison.

## Frozen criteria

`quality.py` freezes twelve equally weighted cases: two each for arithmetic,
extraction, JSON structure, instruction following, short reasoning and basic
Python behavior. The code cases ask for output; generated code is never executed.
Expected answers, categories, scoring, template and bounds are in that source
before any candidate output. No prompt/expected-answer replacement, tolerance
adjustment or best-of selection is allowed after observing either arm. Existing
`tools/qwen35/parity.py` has a reusable no-thinking chat template, but its logit
parity task is not answer quality. The retained community capability scripts use
HTTP, longer generations and code execution; this screen reuses the actual
GEN/DONE harness and `tools/strata_tokenizer.py` instead.

Use greedy decoding, identical weights/quantization, coordinator binary, local
GPU placement, CPU backing, affinity, prompt order, context and speculative/window
settings across arms. Only the explicit Vulkan endpoint/plan and its mTLS/setup
options differ. Use the accepted T1..T8 launch settings; do not increase prefill,
short-read or speculative windows to fit these requests. Each prompt is at most
224 tokens; at most 32 new tokens gives at most 256 total. Require the retained
context to accommodate that bound. No sampling keys or seeds are added to GEN;
`--greedy` is required. Length completion is scored as returned, without extending
the budget or retrying a failed case. Surrounding whitespace and one terminal
protocol EOS may be removed. Exact answers allow nothing else. JSON must parse
as the expected value, with exact types, no duplicate keys, extra keys or prose.

A screen pass requires CPU 12/12, candidate 12/12 and actual Vulkan entries on
**every** candidate case. CPU failure makes regression attribution inconclusive,
including when candidate improves. Missing case participation makes the screen
inconclusive. All per-case scores and disagreements remain in the report. Token
and first-logit exact failures and unchanged `compare_logits.py` numeric results
are separate diagnostics; a screen pass never converts them to passes. This is
one run per arm, a bounded screen of one expert, not broad quality equivalence,
reproducibility, full GPU placement or production readiness.

Protocol/source/plan/entry identity checks and finite-row checks remain in the
engine and worker. Transport errors, unexpected identities, nonfinite results,
invalid token IDs, crashes, hangs and fallback are fatal. `evaluation_mode:
"quality"` is an explicit opt-in. It records reference-token/logit mismatches
rather than raising; default/`"exact"` keeps the original rejection. Quality
requires complete generation, first-logit capture and one ordered warm suite.
Its harness rejects empty/abnormal completions and nonfinite first logits during
the request sequence. The scorer refuses incomplete/failed runs and mismatched
assets/settings. It requires one Vulkan endpoint with stable model, layout,
placement and session identity, mTLS, zero failures and per-case entry deltas.

## Parent-only dispatch, no hardware execution in this change

Parent dispatches operator only after the separately approved existing mTLS
arrangement is available. Do not provision credentials, change trust/network or
use plaintext/forwards. Reuse approved NODELAY options where the accepted launch
receipt supports them. Do not install, download, merge, tune performance or disturb
unrelated services/resident models. Missing prerequisites stop this trial.

1. Stage this exact committed source in a fresh isolated location. Retain the
   successful coordinator/worker launch commands and binary hashes from the
   accepted PR31/e7715 run, plus original baseline, trace and arithmetic evidence.
   The C++ code is unchanged by this quality change: reuse the same accepted
   coordinator binary for both arms, with its exact build receipt. Do not infer a
   build pass from these Python tests or replace a running executable. Preserve
   the complete CPU resident on 02; establish its exclusive idle use through the
   existing receipt/ownership procedure, not a guessed PID.
2. On 05 select the same approved physical V340 die from retained inventory. Use
   the accepted real Q8 expert302 plan, trusted manifest and content-addressed
   blob already used in the accepted row test. Verify its **layer and expert pair
   from that plan**, not from the shorthand 302. Use only that one assigned expert;
   do not hunt for better routing or copy a whole model. Keep full CPU backing on
   02 and the existing disjoint placement/ordered reducer. Existing source hashes,
   plan admission and no-fallback rules apply. If the plan is not available, stop
   with that concrete prerequisite; do not make a synthetic plan.
3. Before CPU or candidate requests, use the existing tokenizer environment
   (including its existing `regex`; stop if missing). Prepare once, from the same
   existing GGUF metadata. This starts no model processes:

   ```bash
   python3 -B "$STRATA_SRC/tools/validation-harness/quality.py" \
     --gguf "$MODEL_GGUF" prepare --out "$NEW_PRIVATE_FROZEN_ROOT"
   ```

   The destination must not exist. Retain `requests` and `quality_manifest`
   unchanged. The manifest binds requests, vocabulary and scorer hashes. Freeze
   the two private configs now; retain their hashes before running either arm.
4. Derive two private configs from the **retained successful** model-offload
   harness config, with separate fresh roots. Add assets `requests` and
   `quality_manifest` from step 3, `evaluation_mode: "quality"` and
   `capture_first_logits: true`. Remove the trace-stop environment setting. Use
   one warm arm (`restart: false`, `requests: [0,1,2,3,4,5,6,7,8,9,10,11]`), attached
   idle resident, and `final_worker: "retain"`, `failure_worker: "retain"`.
   Preserve every other CPU/model/placement setting and all finite resource and
   time limits. Set `request_seconds` to the same finite accepted value in both
   configs; each arm is bounded by 12 such request deadlines plus finite startup
   and cleanup. Do not use the old trace-only reference assets as answer refs.
   For twelve cases omit `first_logits_reference` (its retained exact API is
   request-zero only); the scorer compares **all twelve** saved first logits.
   An optional candidate `reference` must point to the new CPU arm summary.
5. Run the CPU-backed arm first with no network GPU endpoint; keep group entry
   rows and full CPU backing. Retain all twelve outputs, including wrong answers.
   Only then run the candidate with exactly the accepted Vulkan endpoint/plan,
   mTLS identities, full CPU backing, entry rows and `external_gpu_rows: true`.
   Reuse an approved worker already in the accepted state or its exact retained
   successful launch command, tensor/cache limits, die and ICD environment. If
   starting an owned process is needed, record its actual `$!`/Popen PID and
   identities **before** any bounded wait; use the existing guardian/deadlines.
   Do not invent new host bindings or launch a replacement resident.

   ```bash
   python3 -B "$STRATA_SRC/tools/validation-harness/harness/bundle.py" --config "$CPU_QUALITY_CONFIG"
   python3 -B "$CPU_QUALITY_ROOT/run.py" --run-root "$CPU_QUALITY_ROOT"
   # Parent-admitted existing Vulkan worker must be READY before this stage.
   python3 -B "$STRATA_SRC/tools/validation-harness/harness/bundle.py" --config "$VK_QUALITY_CONFIG"
   python3 -B "$VK_QUALITY_ROOT/run.py" --run-root "$VK_QUALITY_ROOT"
   python3 -B "$STRATA_SRC/tools/validation-harness/quality.py" --gguf "$MODEL_GGUF" score \
     --baseline "$CPU_QUALITY_ROOT" --candidate "$VK_QUALITY_ROOT" --out "$NEW_PRIVATE_SCORE_JSON"
   ```

   Run commands only with concrete approved private bindings. An exception is an
   invalid trial, never an answer score. A report with exit 1 is an observed
   screening regression or inconclusive result; retain it. No automatic retry.
   Existing QUIT/reap/uncertain cleanup rules apply. Preserve residents on all
   failures. Stop after one CPU and one mixed suite.

## Private artifact and reporting contract

Keep frozen files, both configs and staging pins, source/binary/model/manifest/
plan hashes, build and successful launch receipts, physical die/driver/ICD
identity, mTLS SAN identities (no key material), actual PID/owner/monitor/reap
receipts, resource observations, stdout/stderr, all first-logit files and arm
summaries in private fresh run roots. Keep `NEW_PRIVATE_SCORE_JSON` there too.
It contains decoded answers and token IDs; **do not commit or publish it**.

The score report contains per-case category, correctness for each arm,
correctness disagreement, generated token IDs/position differences, timings,
finish reasons, Vulkan entry delta, exact first-logit result and all fixed numeric
metrics/limits. Return to parent the score counts, result label, exercised-case
count, any regressions/baseline failures, exact/numeric failures, finite timeout
settings, run/build identities and private artifact locations. Report one-expert
scope explicitly and keep timing descriptive; no speedup inference. One greedy
run does not establish repeatability. No full-fleet, full-GPU or model-wide claim.

## Host validation for this change

Linux x86-64, Python 3.14. No model assets or GPU were used. Passed:
`python3 -m unittest discover -s tools/validation-harness/tests -p 'test_*.py'`
(60 tests: 42 existing harness, 8 peer, 3 numeric, 7 quality), Python compilation
of changed modules, and `git diff --check`. Exact-reference rejection, trace mode
and process lifecycle regressions remain passing. New synthetic tests cover
frozen request binding, strict scoring, retained baseline/numeric failures,
participation/identity refusal and fatal nonfinite/invalid completion handling.

Initial restricted runs failed at fixture socket creation; the same host-only
tests passed with local sockets enabled. One combined run failed because an
existing randomized temporary path contained `__` and triggered the retained
placeholder guard; a fresh full run passed. Existing unclosed-file ResourceWarnings
remain. No unrelated fixture guard was changed.

Not run: real-tokenizer preparation, model generation, Vulkan hardware, multi-host
transport or new engine builds. This change touches only Python validation and
documentation. The draft remains pending parent-admitted execution of the above
contract; no quality result or arithmetic parity is claimed.
