# Fixed two-host CPU group: Hermes handoff

Base source: `c9cef19a35369d5a3721d11515d524ebd6476da1`. Candidate is the full published PR head
containing this file; bind `CANDIDATE` to that exact SHA, never a moving branch. Source, fixtures and
commands are authored. **Builds, executable tests and inference: NOT RUN on Geekom/Valinor.**
No speedup or numerical acceptance is established. No fleet work was dispatched by this coding task.

The change adds only the opt-in fixed CPU endpoint group, necessary per-host use of the existing
guardian/config driver, and focused fixtures. Wire messages, worker implementation and default
one-endpoint route stay unchanged. It does not add discovery, rebalancing or a network GPU mode.

## Prerequisites and focused checks

The original full-model comparison at 9ed478e5543885e019205512f2450ab4336d393e has completed.
The parent reports that the split arm failed every frozen first-logit screen and was slower; the
local baseline/restore first logits were bit-identical. This is supplied evidence, not execution
or raw-receipt inspection by this coding thread. **Do not rerun that campaign.** The next action
is only the bounded diagnostic below. The original commands later in this file remain historical
procedure, not authorization for another model load.

### Current next step: K10/T1 synthetic numerical discriminator

The parent reports identical ggml source bytes at the pinned revision, matching effective compiler
flags, and a fixed retained K2/T8 fixture passing both retained worker ELFs bitwise (20,480 floats).
Those results narrow the search but do not establish equality at every shape or explain the model
failure. This follow-up changes only tests, the fixture generator and the manual build target.
Production routing, arithmetic, protocol and all numerical thresholds are unchanged.

Use the existing CPU/native-ON build on the designated fixture host. Bind the full new diagnostic
source SHA as CANDIDATE, and verify clean source as below. WORKER_A and WORKER_B are existing,
compatible, preserved worker ELFs from the failed comparison. Keep their exact hashes and the same
fixed reference build/runtime environment in the private receipt. Do not rebuild or replace those
workers. No GPU or model assets are needed.

~~~sh
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
cmake --build "$CPU_BUILD" --parallel 2 --target cpu_group_numeric_diagnostic worker_local_test
~~~

There is deliberately **no CTest entry** for the manual diagnostic: its two ELF inputs must be
explicit. Under the existing owned process-tree/RAM/output guard, run these once, in order, with a
180-second wall cap and 2 GiB RAM cap per invocation (no automatic repeat):

~~~sh
OMP_NUM_THREADS=2 "$CPU_BUILD/worker_local_test" --binary "$WORKER_A" --load-threads 4
OMP_NUM_THREADS=2 "$CPU_BUILD/cpu_group_numeric_diagnostic" --worker-a "$WORKER_A" --worker-b "$WORKER_B"
~~~

The first invocation checks the shared generator's unchanged default two-expert fixture after its
bounded extension; it does not repeat both earlier cross-binary checks. The new diagnostic constructs
32 Q8_0 experts per layer at H2560/FF640 across two layers, with only20 CPU experts per layer. Every
T1 request has10 **distinct** IDs. Full synthetic CPU arenas are208,896,000 B per worker; each split
subset is104,448,000 B. The generated fixture payload is334,233,600 B, not a model-sized asset.
Each child has a256MiB arena budget,12 inference workers and4 loader threads; only fresh loopback
port0 children can be controlled. Output is a fixed number of metric lines; child logs cap at1MiB
each. Successful completion prints child reaping and zero real-model loads. Failure cleanup signals
only the children it forked, with bounded TERM then KILL/reap. Leave the real resident worker alone.
Do not set STRATA_WORKER_FIXTURE_KEEP; normal exit removes the synthetic files.

Seven fixed cases cover dense/shifted routing, sparse CPU/GPU ownership, work on each peer alone,
zero CPU work, and arithmetic cancellation. IDs0–9 have distinct seeded weights; IDs10–31 contain
adjacent identical expert pairs so the signed cancellation case has a known small residual.
Signed weights are a deliberate numerical stress case, not a router-policy change or normalization.
The diagnostic also checks copied queued inputs, inactive-peer INFER counts, cancellation before
connection, and valid reuse afterward. It does not claim to test in-flight native-kernel cancellation.

The comparisons run in this order:

| Stage label | What a failure distinguishes |
| --- | --- |
| common-pool-vs-independent-rows | Native twelve-worker row execution differs from independent single-entry native calls |
| peerN-batch-vs-common-rows | Changing the native batch membership changes a routed row, before weighting/reduction |
| binaryA/B-full, binaryA/B-maskedN, binaryA-vs-binaryB-full | Real retained worker arithmetic/masking differs at K10/T1 or from the common reference |
| subsetA/B-vs-common-batch-mask | Actual subset ownership/setup/slot mapping differs from a full-arena masked batch |
| group-vs-manual-peers | The production group differs from independent concurrent tier posts and configuration-order addition |
| group-vs-grouped-common-reference | The combined real group differs from the same grouping over independent native rows |
| reference-grouping-vs-unsplit, group-vs-unsplit, group-vs-actual-fullA | Quantifies reference/actual regrouping against both independent and actual full-worker baselines |

All row, masked-reply and same-grouped-reference gates are **bitwise**. Only grouping versus unsplit
uses the already-published 1e-4 + 4e-6*sum_abs_contributions envelope. Metrics report maximum
absolute/relative error, relative L2 (norm floor1e-30), RMSE, first differing index and hexadecimal
float values. Relative errors can be large for the deliberately near-zero cancellation result;
the fixed absolute/scale gate remains decisive. Non-finite results fail. A bits=0 result at an
unsplit comparison is reported even when its fixed envelope passes; do not hide it as bit identity.

Return the first differing stage and all fixed-case metrics, exit status, duration/peak memory,
exact diagnostic/worker hashes, and child/file cleanup. An all-pass result narrows the failure to
inputs or behavior absent from this fixture; it does **not** clear the model's failed logit screen.
Do not tune thresholds, rerun models, add seeds or start another campaign. Broader tracing needs a
separate decision after these results. **This diagnostic is authored and NOT RUN on the desktop.**

### Historical original validation procedure — do not repeat

Parent/Hermes owns execution and source/binary pins. Keep the healthy existing full worker on02
resident until all build/fixture gates pass. Do not probe, unload, stop or replace it for preflight.
Use retained caches/assets and existing build directories. Do not reconfigure policies or install
anything. On each selected checkout establish clean exact source first:

```sh
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
```

On the existing Hermes Python controller, run the synthetic suite once under its retained bounded
output guard. It needs Linux /proc and ordinary loopback sockets, no model, GPU or native binary:

```sh
cd "$SRC/tools/validation-harness/tests"
timeout --signal=TERM --kill-after=10s 240s python3 -B -m unittest -v test_harness test_peers test_numerics
```

The new peer tests exercise real tiny children: remote-role staging/pre-START binding, alternate
loopback readiness, fast STOP reconciliation, explicit abort/reaping, lost-channel uncertainty,
bounded transport stderr, and config-only local/split selection. Existing affinity/owner tests stay.
No local synthetic test demonstrates real network throughput or full-model correctness.

On05 use the existing CPU/native-ON test build; build-target names and CTest names are different:

```sh
cmake --build "$CPU_BUILD" --parallel 4 --target strata-expert-worker cpu_worker_group_test expert_tier_test expert_worker_test worker_local_test worker_service_residency_test
ctest --test-dir "$CPU_BUILD" --output-on-failure -R '^(cpu_worker_group_test|expert_tier_test|expert_worker_test|worker_local_binary_test|worker_service_residency_test)$'
```

The group fixture covers malformed/overlap/missing ownership, one/two peers, T8/T1/T2/T1, changing
K/masks, zero-work, copied input lifetime, concurrent fan-out, reversed completion, failure draining,
non-finite/overflow refusal, setup keepalive, resident reconnect and explicit geometry/partition
refusal. It compares grouped bits exactly. Its independent unsplit gate is frozen before execution:
`abs(delta) <= 1e-4 + 4e-6 * sum(abs(weighted entry contributions))`, with max absolute/relative,
scale-relative and RMSE printed. This scale covers cancellation; the existing single-tier bitwise
gates are unchanged. Preserve any first failure; do not loosen gates to pass.

On02, outside inference, use its retained CUDA/native-ON cache, ccache and approved bulk parallel8:

```sh
cmake --build "$CUDA_BUILD" --parallel 8 --target strata strata-expert-worker worker_dispatch_test cpu_group_dispatch_test
ctest --test-dir "$CUDA_BUILD" --output-on-failure -R '^(worker_dispatch_test|worker_dispatch_host_zero_test|cpu_group_dispatch_test)$'
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$CUDA_BUILD/strata" --native-experts on
```

The new CUDA fixture uses real group/local services, the real dispatcher and mapped combine at the
production K10 contract (T1 on its two-expert synthetic pack). It asserts no local weight fallback,
zeroed worker rows, exact grouped reference and exactly one mapped contribution. Multi-token
grouping is covered by the CPU fixture, old cache/worker merge gates by the existing CUDA tests.

On03 do the small incremental native-OFF boundary check with its existing build (no model):

```sh
cmake --build "$OFF_BUILD" --parallel 2 --target strata worker_disabled_dispatch_test
ctest --test-dir "$OFF_BUILD" --output-on-failure -R '^worker_disabled_dispatch_test$'
python3 -B "$SRC/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_BUILD/strata"
python3 -B "$SRC/tests/worker/worker_family_cli_test.py" --engine "$OFF_BUILD/strata" --native-experts off
```

These commands request no HIP/SYCL build. Their backend build/link validation remains NOT RUN until
the parent selects available hardware; do not claim it from CUDA or CPU results. If a named retained
build lacks the required configure options, report that exact prerequisite rather than silently
substituting a different backend. Stop after the first failed gate and keep receipts/cleanup.

## One local / split / local comparison, only after gates pass

Use candidate engine and matching candidate worker binaries for every arm: this compares ownership,
not different source releases. Retain the old baseline binaries separately. Bind one private config
to the [PEERS.md contract](../../tools/validation-harness/PEERS.md), stage mechanically once per host,
then run that static driver. No per-arm wrapper edits, automatic retries or generation warmups.
The six explicit load/warm entries yield exactly three retained GEN256 requests in each A0/B/A1.
Cold load/READY timing stays separate from inference. Loading cache conditions must be disclosed.

Engine plus RTX3060/3060Ti stay on02. Keep primary960/helper1378 placement, Q8 manifest/assets,
same retained three prompts, worker12 and loader4 on each participating host, turbo and identical
engine/worker masks/policy. Verify post-swap02 usable251.5GiB, 8x32GB,2133MT/s and05 usable94.2GiB
from parent runtime receipts. All22238 CPU experts occupy116135731200B local; B alternates them in
layer/expert order,11119 and58067865600B per peer.05 receives only that subset, never the116GB arena.
Budget each host from actual headroom; do not infer safety from installed RAM alone.

Use private actual numeric LAN bindings and the existing exact source allowlist. No new SSH/control
route, firewall/NIC/access permission or tunnel is assumed. If an authorized command channel from the
engine controller to the existing second-host guardian cannot be bound, stop before START and report
it. Do not invent addresses or change access to make the experiment run.

The current resident full bitmap cannot stand in for a split subset. After prerequisites pass,
parent names the owned worker replacement/handoff; the driver must own the retained guardian handle
before a restart. It refuses arbitrary PID adoption and occupied unowned ports. If current ownership
cannot be transferred, report the concrete handoff blocker before touching residency. Cold loads and
explicit replacement are necessary at A0/B/A1 partition changes; matching warm sessions reconnect
without reload. Final A1 stays resident;05 is stopped/reaped before A1. Keep useful weights until the
next named owned replacement, not an unrelated cleanup.

Report per-peer entries, requests, resident/loaded/verified bytes, allocation/unload/failure counters,
service and exchange timing, endpoint order, and group post-to-finish/caller-wait wall scopes. Peers
overlap, so never sum their elapsed scopes as wall time. Transport and slower peers can dominate;
equal expert counts do not establish balance. Report all nine request times and tokens, per-arm
aggregate and B versus both A0/A1 with drift disclosed. A flat or slower result closes this round;
do not add tuning arms or rerun until selecting a win.

## Numerical and correctness gates fixed before the run

Set `capture_first_logits: true` for every arm. It uses the existing
`STRATA_DUMP_FIRST_LOGITS` diagnostic: raw F32 first next-token row per matched prompt, `.0`/`.1`/`.2`.
This does **not** compare logits at every continuation position. Missing, malformed or non-finite
rows fail the comparison. Run the same source helper twice, A0 versus B and A0 versus A1:

```sh
python3 -B "$SRC/tools/validation-harness/compare_logits.py" \
  --baseline "$ENGINE_RUN_ROOT/A0-first-logits.0" "$ENGINE_RUN_ROOT/A0-first-logits.1" "$ENGINE_RUN_ROOT/A0-first-logits.2" \
  --candidate "$ENGINE_RUN_ROOT/B-first-logits.0" "$ENGINE_RUN_ROOT/B-first-logits.1" "$ENGINE_RUN_ROOT/B-first-logits.2"
python3 -B "$SRC/tools/validation-harness/compare_logits.py" \
  --baseline "$ENGINE_RUN_ROOT/A0-first-logits.0" "$ENGINE_RUN_ROOT/A0-first-logits.1" "$ENGINE_RUN_ROOT/A0-first-logits.2" \
  --candidate "$ENGINE_RUN_ROOT/A1-first-logits.0" "$ENGINE_RUN_ROOT/A1-first-logits.1" "$ENGINE_RUN_ROOT/A1-first-logits.2"
```

Frozen screening limits: max absolute logit error `<=1e-2 + 1e-4*max(abs(baseline logits))`, relative
L2 `<=1e-4`, max absolute softmax probability error `<=1e-3`, KL(A0||other) `<=1e-5`. Max relative
element error uses denominator floor1e-12 and is reported, not used as a cancellation-unstable gate.
These are prospective screening limits, not measured acceptance; no after-result threshold tuning.
Report top1 equality and full256 token arrays separately, including first divergence. Keep exact
baseline outputs. Bind the retained prompt task oracle before execution; record task correctness
separately from token equality and numeric pass. If those prompts have no task oracle, mark task
correctness UNASSESSED and leave full-model acceptance pending rather than inventing one.

Return one flat receipt with exact source/binary/config hashes, host policy/topology, commands,
pass/fail/skip/NOT RUN, first-failure output, cleanup/uncertainty, all nine measurements, numeric
metrics, token comparison and task verdict. No broad provenance framework, further campaign, merge
or deployment is authorized by this handoff.
