# Frozen v2: 100-case paired backend screen

This is a separate prospective version, originally stacked on PR #34. It preserves
the original twelve cases, scoring and retained **inconclusive-baseline** verdict.
The shared comparison helper defaults to its original single-endpoint contract.
It does not rescore or publish old private answers. All v2 prompts,
golds, derivations, semantic rules, format rules and criteria below are frozen
before v2 CPU or candidate outputs. There is no hardware result at this revision.

## Inventory and rules

The auditable inventory is [quality_v2_cases.json](quality_v2_cases.json). Each
record has a stable ID, category, full prompt, gold, derivation and both scoring
rules. Its 100 prompts are distinct from the original twelve. These are authored
screening cases, not an independently sampled population or a statistical test.

| Category | Cases | Covered operations and example gold |
| --- | ---: | --- |
| Arithmetic | 17 | Addition, subtraction, multiplication, division, precedence, exponent, percentages, fraction addition, decimal addition, area, unit price, remainder, median, proportion and time conversion. `3/4 + 1/8` → `7/8`. |
| Extraction | 17 | Named fields, maximum row, address component, time, SKU, quantity, ordinal item, email, zero-stock row, room, brackets, mapping, date, unit, status and reference. Drum SKU → `D-62`. |
| Structured output | 17 | Typed objects, ordered/sorted/deduplicated arrays, Boolean, null, empty containers, nested objects, negative/zero integers and string versus number. `{"outer":{"inner":5}}`. |
| Instructions | 17 | Case, separators, reversal, initials, filtering, substitution, lines, word sorting, space removal, indexing, repetition, condition, delimiters, word count, capitalization and truncation. Reverse `planet` → `tenalp`. |
| Short reasoning | 16 | Implication, exclusion, temporal/spatial/weight order, exhaustive alternatives, converse refusal, elimination, relative days, existential/universal distinction, exclusive truth, integer interval and conjunction. A>B and C>A → `C`. |
| Python behavior | 16 | Length, floor division, modulo, append, slicing, substring count, range/sum, negative index, dictionary default, maximum, conditional expression, join, assignment order, set size, replacement and filtered sum. `'cobalt'[2:5]` → `bal`. |

No generated code is executed. Gold derivations can be read before running. Do
not replace hard, failing, unexercised or length-limited cases after results.

Semantic and format scores are independent Boolean values, one of each per case.
Both ignore surrounding whitespace and one terminal protocol EOS only. Internal
whitespace, explanations, Markdown fences and punctuation are not stripped.

- Arithmetic alone accepts exact rational equivalence of a standalone signed
  integer, decimal or fraction, using Python's standard-library `Fraction`.
  No exponent notation, prose extraction or floating tolerance is used. For
  example, `14/16` has the correct value for `7/8`, but fails the declared reduced
  fraction format. Integer formatting excludes a leading plus/leading zeroes;
  decimal formatting requires a decimal point; fractions require canonical
  integer spelling, positive denominator and coprime numerator/denominator.
- Explicit binary yes/no reasoning permits case-insensitive semantic equality.
  Its format still requires lowercase `yes` or `no`. `YES` can therefore be
  semantically correct and format noncompliant.
- Extraction, other instructions, other reasoning and printed program output
  use the exact expected text after the shared outer-whitespace/EOS rule. Case
  is preserved where it carries the requested answer, such as `SILENT` or `ROM`.
- Structured answers require the expected JSON value with exact types and keys;
  object key order and JSON whitespace may differ. Boolean and integer, or
  integer and decimal, are not interchangeable. Format compliance means one
  valid finite JSON value, without duplicate keys or prose. A wrong but valid
  JSON value can satisfy format while failing semantics. Other format rules are
  explicit ASCII regular expressions in the inventory, independent of gold.

## Paired screening criterion, fixed before outputs

Keep all 100 case pairs in the report. For semantics **and** formatting, report
baseline-correct/candidate-wrong regressions, candidate improvements, both-correct
pairs and shared failures, with counts and case IDs. Report each category and
both all-case and actually exercised subsets. Baseline imperfection is retained
and does not automatically invalidate the comparison. Improvements never offset
regressions. A zero-baseline score would be a weak capability anchor even if no
new regression is observed; absolute scores/shared failures remain visible.

Only case pairs with positive candidate Vulkan entry deltas count as evidence
about offload quality. Nonparticipating pairs remain in the all-case results and
are listed separately, never replaced. In precedence order:

1. Any exercised semantic regression → `screen-semantic-regression`.
2. Otherwise, any exercised format regression → `screen-format-regression`.
3. No exercised cases → `screen-unexercised`.
4. Some but not all cases exercised → `screen-coverage-limited`.
5. All 100 exercised with no paired semantic or format regressions →
   `screen-no-observed-regression` (exit 0).

All other report labels exit 1 and must be retained. A runtime/identity/scoring
exception is an invalid trial, not a quality score. No baseline 100/100 condition,
net-score threshold, adaptive margin, case replacement, retry or best-of result
is used. This is a descriptive regression screen on a small authored suite; it
is **not statistical noninferiority**, broad quality equivalence or readiness.
Exact token/logit and unchanged fixed numeric metrics remain separate diagnostics.
Known numeric failures remain failures even if the answer screen sees no regression.

## Phase 1 and bounded cost

Use the same accepted expert302 **layer/expert plan**, one physical V340 die on
05, full CPU backing on 02, identical existing weights/quantization and the same
accepted coordinator/worker binaries and launch receipts. Freeze the admitted
plan bytes during preparation; the scorer checks the candidate's actual plan
hash. The hash binds selection, not proof of plan contents: verify its one-expert
identity against the retained trusted plan/admission receipt before preparation.
Do not select experts from these new outputs. Preserve entry rows, ordered
reduction, resident reuse, mTLS, approved NODELAY behavior and all proven settings.
No certificate, access, driver, build, service or arithmetic change is needed.

Exactly **one CPU arm plus one candidate arm**, each with 100 requests, in fixed
inventory order: **200 requests total**. Each prompt is limited to 224 tokens and
each request to 32 new tokens, identically for both arms. Thus the worst case is
6,400 generated tokens and 51,200 prompt-plus-output token positions across both
arms. This does not count speculative internal evaluations and is not a runtime
estimate. Context must be at least 256; keep the accepted actual context and
T1..T8/window/speculative settings unchanged. Do not increase windows to run v2.
All gold strings fit 32 tokens under the synthetic byte-tokenizer test; real
preparation must pass the existing model tokenizer roundtrip and prompt bound.
Actual generations may hit length; score those returned answers without extension.

Freeze the same finite `request_seconds` and existing startup/shutdown/cleanup
limits in both configs. Each arm has at most 100 request deadlines plus those
finite lifecycle limits and bounded controls. No unbounded wait or automatic
retry is introduced. Retain actual Popen/`$!` identities before bounded waits.
First-logit storage is 200 raw F32 vocabulary rows, or `800 * vocabulary_size`
bytes (about 189.5 MiB at 248,320 entries), plus ordinary logs. Keep private run
roots and the existing independent resource monitor; do not infer a resource pass.

Phase 2 is only proposed: after separately reviewing phase-1 evidence, admit a
bounded larger real expert subset from trusted routing evidence within proven
memory budgets. Reuse the same frozen suite in a separately identified paired
trial. Broader placement, capacity work and dispatch are not implemented here.

## Minimal existing-harness handoff — parent dispatch only

Do not run hardware now. Reuse the successful v1 command receipts, approved mTLS
resources and idle-resident ownership procedure from [QUALITY.md](QUALITY.md).
Preserve original roots/results. Missing assets, dependencies or approval stop
this trial; no installs, credentials, plaintext substitutes or service replacement.

1. In a fresh isolated checkout at this PR's exact head, read all 100 inventory
   records and freeze the existing approved expert302 plan hash from its receipt.
   Use the already installed tokenizer environment and same existing model GGUF:

   ```bash
   python3 -B "$STRATA_SRC/tools/validation-harness/quality_v2.py" --gguf "$MODEL_GGUF" prepare \
     --placement-plan "$ACCEPTED_EXPERT302_PLAN" --out "$NEW_PRIVATE_V2_FROZEN_ROOT"
   ```

   This is metadata/token preparation only. The new root must not exist. Keep
   its requests and manifest unchanged. The manifest binds inventory, bounds,
   template, token IDs, vocabulary, admitted plan hash and scoring/tokenizer sources.
2. Derive private CPU and candidate configs from the successful v1 configs.
   Point their `requests` and `quality_manifest` assets to the new frozen files;
   add `quality_suite: "v2-100"` alongside explicit `evaluation_mode: "quality"`.
   Keep `capture_first_logits: true`, one warm arm with `requests` exactly
   `list(range(100))` expressed as the JSON integer array 0 through 99, and fresh
   distinct run roots. Preserve the same attached idle resident and both retain
   policies. Remove any trace-stop setting. Do not use old 12-case token refs or
   `first_logits_reference`; all 100 logits are compared offline. An optional
   candidate reference must point only to this v2 CPU arm summary. Freeze both
   configs before either arm; do not adjust argv or deadlines after seeing output.
3. Use the existing staging/run commands once for each arm, CPU first, and only
   start/attach the approved Vulkan endpoint through its retained success receipt:

   ```bash
   python3 -B "$STRATA_SRC/tools/validation-harness/harness/bundle.py" --config "$CPU_V2_CONFIG"
   python3 -B "$CPU_V2_ROOT/run.py" --run-root "$CPU_V2_ROOT"
   python3 -B "$STRATA_SRC/tools/validation-harness/harness/bundle.py" --config "$VK_V2_CONFIG"
   python3 -B "$VK_V2_ROOT/run.py" --run-root "$VK_V2_ROOT"
   python3 -B "$STRATA_SRC/tools/validation-harness/quality_v2.py" --gguf "$MODEL_GGUF" score \
     --baseline "$CPU_V2_ROOT" --candidate "$VK_V2_ROOT" --out "$NEW_PRIVATE_V2_REPORT"
   ```

   Use concrete approved bindings. Transport faults, changed/missing identity,
   invalid tokens/nonfinite results, crash, hang or fallback are fatal under
   existing guards. Preserve runtime failure and cleanup receipts; do not retry
   uncertain operations. Normal QUIT/reap and resident-preservation rules apply.

Keep decoded answers, tokens and reports **private**, with both root/config/pin
sets, frozen files, model/binary/manifest/plan identities, owner/monitor/cleanup
receipts, per-case logits, stderr/stdout, physical die/driver identity, finite
limits and resource observations. Publish no private run output. Return parent
case/category semantic and format scores, paired loss/gain/shared-failure IDs,
participation coverage, token/numeric diagnostics and private artifact locations.

## Host validation

Passed on Linux x86-64, Python 3.14:
`python3 -m unittest discover -s tools/validation-harness/tests -p 'test_*.py'` (67 tests, including seven new v2
policy tests), changed-module Python compilation and `git diff --check`. Local
fixture sockets were enabled for existing lifecycle/peer tests; no fleet or
model process was involved. Existing unclosed-file ResourceWarnings remain.
The original scorer's Git blob hash still equals its PR #34 base. Synthetic
reports test baseline imperfection, paired loss/gain separation, inactive-case
exclusion, independent format failures and changed plan/source refusal.

Not run: real-tokenizer preparation, hardware generation, multi-host transport,
new engine builds or phase 2. Only Python validation and documentation changed;
no absolute quality, arithmetic parity or runtime performance result is claimed.


## Multiple Vulkan endpoints — parent dispatch only

The same frozen 100 cases and scoring rules can screen 1..8 ordered Vulkan
endpoints. Repeat `--placement-plan` in the same order as the candidate's
`--remote-gpu-worker` arguments. A single plan still uses manifest `plan_sha256`
and report `placement_plan_sha256`. Multiple plans use manifest `plans` and report
`placement_plans`: ordered SHA-256, model/layout/placement identities, geometry,
assigned counts and bytes. Preparation reads bounded SGPUPLN1 metadata, checks its
runtime digest, common model/layout/geometry and disjoint assignments. These are
integrity/admission bindings, not authentication of binaries or physical devices.
The coordinator still validates the plans against its trusted source manifest.

The scorer verifies every endpoint's startup and post-request snapshots in
coordinator index order: mTLS/READY, model/layout/placement against its frozen
plan, stable nonzero session, assigned bytes, zero failures, and monotonic entry
and request counters starting at zero. Missing, duplicate, reordered, reset or
mismatched evidence invalidates the trial. No response is generated again.

Each case retains aggregate `vulkan_entries`; multi-endpoint reports add
`vulkan_entries_by_endpoint` in frozen order. `endpoint_participation` reports
each endpoint's identity, total entries, exercised case IDs, nonparticipating IDs
and paired semantic/format counts on that endpoint's exercised cases. One case
can exercise multiple endpoints; those subsets must not be added as independent
samples. Endpoint counters **cannot prove every assigned expert executed**.
Aggregate case participation drives the original paired criterion. Additionally,
a multi-endpoint trial with an entirely unused endpoint cannot receive
`screen-no-observed-regression`: it is `screen-coverage-limited`. Existing semantic
and format regression precedence is preserved. Every case need not use every
endpoint. Numeric limits and CPU answer identity requirements are unchanged.

Old single-plan schema fields remain supported. Source pins remain strict:
preserve old frozen manifests/results and use their original pinned scorer when
inspecting historical runs. Prepare a fresh manifest before a new paired run;
never rewrite an old manifest's source hashes to pass a newer scorer.

### Minimal two-die handoff

No hardware dispatch, second listener/access change or credential work is performed
by this change. The parent admits and dispatches the bounded run using existing
receipts, resource guards, CPU resident ownership and finite deadlines.

1. Split the **same previously admitted 960-expert union**, without selecting from
   answers, into two disjoint sets of 480: ten experts per layer in each set across
   48 layers. Verify their disjoint union equals the original assignments. Use
   `strata-gpu-manifest plan LOCAL_MANIFEST ASSIGNMENTS OUTPUT` separately for each.
   Each contains 2,506,752,000 weight bytes (2.3346 GiB); both total 5,013,504,000.
   Tensor caps must include padded graph allocations. Backend compute/pipelines,
   driver allocations and host/cache storage need separate existing headroom
   checks; neither capacity nor speedup follows from these weight totals.
2. Reuse the accepted worker binary twice, on independently verified Vulkan
   device indices for the two physical dies, with distinct approved ports and
   **separate cache roots** (the cache is exclusively locked). Use each worker's
   corresponding `--plan`, `--device`, admitted tensor/cache budgets and retained
   listener/source-IP/TCP settings. Record actual executed binary hashes and
   device mappings; a staged asset hash is not an executed-process identity.
3. Keep the existing coordinator and full CPU backing/settings unchanged. Supply
   two `--remote-gpu-worker vulkan@HOST:PORT PLAN` pairs and one exact expected
   `--remote-gpu-tls-peer SAN_DNS` per endpoint, in matching order. Each worker
   verifies the coordinator SAN through existing mTLS. Reuse only already
   authorized, valid certificates; check the retained expiry before dispatch.
   Missing/expired certificates or second-listener approval stop dispatch for the
   parent to resolve; do not mint credentials or weaken verification here.
4. Prepare in a new private root using the existing tokenizer/model metadata:

   ```bash
   python3 -B "$STRATA_SRC/tools/validation-harness/quality_v2.py" --gguf "$MODEL_GGUF" prepare \
     --placement-plan "$DIE0_PLAN" --placement-plan "$DIE1_PLAN" --out "$NEW_FROZEN_ROOT"
   ```

   Stage fresh CPU/candidate configs using that manifest and its unchanged 100
   requests. Use the existing bundle/run/score commands above, once per arm.
   Keep the two plan files as separately named staged assets and bind candidate
   argv to those staged paths in order. The existing scorer verifies their bytes
   against the frozen identities. Preserve old private trial roots/results.
5. Retain both endpoints' complete startup/post-request reports and physical-die,
   executable, certificate and resource receipts privately. Return endpoint
   participation plus the unchanged paired semantic/format verdict and numeric
   diagnostics. This small authored screen cannot establish systematic quality
   equivalence, CPU bit parity or readiness. No private addresses, keys, model
   answers or fleet results belong in the PR.
