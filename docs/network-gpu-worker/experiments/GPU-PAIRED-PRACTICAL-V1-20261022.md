# GPU-PAIRED-PRACTICAL-V1-20261022 — prepared evaluation, not an execution grant

This closes seed tracing. Supplied four-way result at cdc21279 reproduced both
captured seed outputs and their joint delta exactly, using matching captured
local/helper anchors. Row arithmetic and grouping explain that seed difference.
They do not establish that the full-logit 3.5–5% relative L1 discrepancy is harmless
or is measured quality loss. Historical **NUMERICAL_FAIL remains unchanged**.
No summation, quantizer, kernel, placement or capture change is proposed here.

This additive preparation contains only a fixed task checker, an attached
launcher profile, focused mocked tests and this procedure. Product binaries are
reused, not rebuilt. All new checks and execution are **NOT RUN** by the author.
The parent must obtain approval for the proposed fresh model budget below.
Static Standards review found no documented breaches or material smells. Static
Spec review found an isolated-Python sibling import omission; it was corrected
before publication. Neither review found a remaining concrete blocker. They did
not run the fixtures, tokenize the real metadata or validate a model request.

## Questions, tasks and measurements

The five public tasks are fixed in paired_tasks.py. Each gets GEN 32, greedy,
the model's existing chat template with enable_thinking=false, and no hidden
warmup. Both modes receive identical token IDs in identical task order.

| Task | Required answer/check |
| --- | --- |
| 18 minus 7 plus 5 pens | Only `16` |
| Sort 9, 2, 11, 4 | JSON array `[2,4,9,11]` |
| Extract highest score from Mira:7; Jon:12 | JSON object with exactly name Jon and integer score 12 |
| All zogs blue; no blue thing red | Only `no`, case-insensitive |
| Python is_even(n) | AST equal to `def is_even(n): return n % 2 == 0` |

Whitespace and JSON key order may vary. Prose, fences, extra keys and alternate
code forms fail these declared strict checks. Generated code is parsed, never
executed. This small set tests elementary reasoning, extraction, formatting and
a narrow coding instruction; it is not a representative benchmark or a general
quality estimate. Preserve actual text and distinguish a wrong answer from a
strict-format failure. There is one B-then-G order and no statistical speedup claim.
Length-capped output is **OUTPUT_CAP_REACHED**, separate from TASK_CHECK_FAIL,
even if its partial text happens to pass the format check. Do not infer ordinary
wrong-answer quality from truncation. No longer replacement request is included.

Separate numeric phases use the first two tasks with their declared answer text
appended to the same rendered assistant prefix. Each supplies the complete fixed
sequence, then GEN 1. Existing STRATA_LOGPOS records teacher-forced probabilities
for supplied next tokens; the report verifies every position and target and scores
the answer suffix. Existing STRATA_DUMP_FIRST_LOGITS captures the next-token
distribution after that supplied answer. These are common-input measurements,
not later logits from independently generated continuations. No layer capture or
new evaluator/kernel is needed. Numerical probe timings are excluded from the
performance comparison. No perplexity/general likelihood claim is made from two
short reference answers.

Source capabilities: src/program/generate.cpp:13207–13254 (serial prompt windows
and STRATA_LOGPOS), :14108–14118 (remote CPU refuses a fresh segment over
--short-read), :15114–15129 (first logits), :15384–15422 (DONE fields),
src/core/verify.cpp:3188–3267 (native log probabilities), tools/strata_tokenizer.py
and serve/frontend.py:40–67 (existing tokenizer/template). The supported prompt
bound here is **64 tokens including chat formatting and supplied reference answer**.
Preparation refuses longer input; never truncate, silently change --short-read,
or invent a tokenizer/likelihood CLI. --prompt-cache 0 disables request prefix reuse;
the report requires REUSED/DONE resume 0 so probability positions remain complete.

Perf phases remove --dump-routing and STRATA_DUMP_FIRST_LOGITS. All phases omit
layer/expert capture, STRATA_VERIFY_TRACE, STRATA_VERIFY_ASYNC_LAUNCH and timing
profiling overrides. The default untraced product path is retained. Numeric phases
alone enable the two existing logit/likelihood hooks. No inference flags change.

Report each task's DONE prompt_ms and decode_ms, their sum as **engine-accounted
request time**, output tokens and tokens/decode-second, and weighted aggregate throughput.
The engine's r0 starts after parsing the request; d0 starts after prompt work.
These clocks exclude initial engine/CPU/GPU setup, the small untimed gap between
prompt and decode clocks, post-decode bookkeeping and the caller's queue/SSH time.
The watchdog additionally records READY and DONE observation times outside the
product. READY-to-last-DONE is warm batch wall time; adjacent completion intervals
approximate queued request service wall time and include polling/guard jitter.
They are not wire RTT or precise send-to-first-token latency. No hidden request
is spent to establish either clock. Preserve raw times and cap/truncation status;
shorter incorrect outputs must not be advertised as faster equivalent work.

DONE worker_entries and worker_req_wall describe the remote CPU share; worker wall
is milliseconds of exchange time, not full generation wall. Preserve the existing
remote-GPU client summary for each G engine (requests, entries, setup_ms,
request_ms, assigned/reused/transferred/persisted bytes), GPU worker service_ns and
resource samples. Setup/network validation is separate from steady-state exchange
and decode. CPU and GPU times can overlap; do not add them as a wall-time partition.
No new route or device trace is required to report these scopes.

## Proposed fresh budget and retained state

Propose **1500 seconds / 25 minutes**, T before controller/temporary-forward startup
through engine/GPU cleanup, with the existing real **45-second cleanup reserve**.
All fresh caps and leases clip to this window; no running extension. Four fresh
engines run these exact internal phases:

```text
B0: five task GEN 32 requests, then QUIT; resident CPU, no network GPU
B1: two supplied-answer GEN 1 probes, then QUIT; resident CPU, no network GPU
G0: the same five task GEN 32 requests, then QUIT; resident CPU and warm 04 cache
G1: the same two supplied-answer GEN 1 probes, then QUIT; same placement/cache
```

The GPU worker is owned once from G0 through G1, then stopped. Its existing service
resets the executor between client sessions, so allow **two roughly 105-second
warm GPU setup/validation passes**, not a claim of persistent GPU allocations.
CPU remains the same resident arena throughout. No cold CPU load is included.
Decode upper-bound planning for 320 task tokens is about 144 seconds at reported
2.9 B / 1.8 G tokens/s, plus four probe tokens. At most 896 supplied prompt tokens
across all 14 requests, four engine initializations, setup, gaps and cleanup get
the remaining headroom. Prompt ingestion speed is unmeasured for these tasks;
1500 seconds is a bounded practical proposal, not a prediction. If it expires,
stop safely and report incomplete pairs; no retries or replacement requests.

New allowance: **14 started requests / 324 maximum generated tokens**. Charge a
started interrupted request its full 32- or 1-token allowance. Historical 17
requests / 835 generated tokens stay recorded; maximum cumulative starts become
31 and charged generated-token ceiling 1159. Historical time remains 7541 s +
approximately 117 s + 271.3 s, with the latest supplied offline 10 s separately;
do not turn that approximate history into a fabricated precise total. Report
fresh active time and ongoing CPU-service residency age separately.

Preserve primary 960 / helper 2247 / CPU full 21369 / uncovered 18080 / remote
3289, CPU budget 106428 MiB, arena 111597465600 B, full 04 cache 17176473600 B,
and the 92156-byte plan SHA-256
a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578.
Keep the same accepted contiguous-frame engine and GPU-worker ELFs for both modes,
resolved from V2/combined retained receipts. The base implementation remains
bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5; the accepted experiment lineage includes
contiguous framing. Bind actual ELF hashes once; do not claim an experiment ELF
equals the base product commit. No builds or binary replacement are needed here.

## One complete Hermes continuation, after budget approval

1. Obtain this immutable revision and verify GPU-PAIRED-PRACTICAL-V1.SHA256SUMS
   from its repository root. Preserve historical reports, trace captures and
   previous manifests at their immutable revisions. Use existing ROLE_CONTROLLER
   and private role/transport receipts. No execution on either user desktop,
   new network/security settings, credentials, persistent tunnels, dependencies,
   full-model hashing/transfer, rebuild, merge or deployment. General prompts and
   source are public; configurations, paths, outputs and metadata stay private.

2. Reuse the V2 staged **absolute** SUP_ROOT: supervisor/shared guards at
   311561689bb4b1f8a109821601e3a6bda07067e6 plus supervisor-residency.patch then
   first-layer-supervisor.patch. Do not patch the live CPU guardian's files.
   Expected postimage SHA-256:

   ```text
   agent.py      4307f89265617cf18c6ac747b1b19193457172c93dfaa902db81e011ff924849
   policy.py     d3b1de8805aef47041df1b81726cec6e6c7623ef117df22c1420695bec660cb8
   controller.py 13d0bbc84e39761b17fe09500d763f6f7521b69fac314c1ed2472f3670a686c9
   ```

   If a new private stage is necessary, obtain that immutable planning revision
   and apply each published patch with git apply --check --unidiff-zero followed
   by git apply --unidiff-zero, in that order. Verify these postimages. No old
   controller invocation may start a CPU worker. Reuse existing passed periodic,
   warning, framing, handshake and replay results; none is a new prerequisite.

3. Run only the new mocked fixture on 05 using the already available Python:

   ```sh
   EXPERIMENTS="$SRC/docs/network-gpu-worker/experiments"
   "$PYTHON" -B -I "$EXPERIMENTS/test_paired_tasks.py" --supervisor-root "$SUP_ROOT"
   ```

   Require rc 0, PAIRED_TASKS_PASS_V1 tests=7 workloads=0, no failures/errors/skips.
   It enforces 256 MiB AS / 10 CPU seconds / 20 wall seconds, creates only small
   private synthetic files, and forbids subprocesses, forks, sockets and PID
   signaling. Limit its output to 1 MiB with the existing bounded launcher. No
   build, GPU, worker or model request is used. Ordinary in-scope launcher/checker
   corrections may be made and reported without another preparation gate; retain
   exact source differences and do not hide a product failure or expand requests.

4. On 05, resolve PACK_META, PYTHON, EXPERIMENTS and SUP_ROOT from
   retained bindings. PACK_META
   names existing extracted tokenizer metadata, not GGUF tensors or the arena.
   If those five small files exist only on 03, copy only them through the existing
   approved transport into a private metadata directory. Use existing regex and
   Jinja2 dependencies; if absent report the concrete prerequisite, do not install.
   Create PLAN in a new 0700 directory and run:

   ```sh
   (
     ulimit -v 1048576
     ulimit -t 30
     timeout --signal=TERM --kill-after=5s 60s "$PYTHON" -B -I "$EXPERIMENTS/paired_tasks.py" --pack "$PACK_META" prepare --output "$PLAN"
   )
   ```

   Require PAIRED_PLAN_READY requests=14 generated_tokens_max=324 prompt_tokens_max=64.
   This uses the real tokenizer/template, validates exact token bounds and keeps
   the metadata hashes/IDs private. Copy identical PLAN bytes to the existing
   private role stages and bind their SHA-256; no re-tokenization on another host.
   If a prompt exceeds 64 tokens, stop before product launch and report the task;
   do not spend a partial run, silently shorten it, or increase the supported cap.
   Preparation and final reporting each have <=60 wall / 30 CPU seconds, <=1 GiB
   AS, <=1 MiB output, under existing 05 host/swap/headroom controls alongside the
   independently guarded arena. No model tensor reads or generation occur here.

5. Resolve the live service using the retained resident_config and receipt, not
   historical numeric PIDs. Preserve exact host/boot/namespaces, worker/monitor/
   guardian identity and independent resource coverage; require complete idle
   111597465600-byte arena with UINT64_MAX idle TTL and exact partition. Keep the
   admission receipt age rule; the supplied refresh was 2026-10-22 12:43:34 UTC,
   same boot. Refresh a stale required receipt only through the existing approved
   administrator channel. Do not alter service ownership files, restart, stop,
   unload, duplicate, cold-load or waive a failed guard. If absent/incompatible,
   report and stop. Keep 04 cache intact; both G setups must validate full reuse
   W=17176473600 with new source-weight reads/network/persist zero. Do not mirror,
   replan, delete or refill the cache to make the result pass.

6. Create fresh private configs/run roots using the retained V2 role bindings.
   Every new config binds diagnostic=GPU-PAIRED-PRACTICAL-V1-20261022,
   diagnostic_revision=<this immutable commit>, supervisor_commit=311561689bb4b1f8a109821601e3a6bda07067e6,
   approved_window_s=1500, plan=<absolute local PLAN>, plan_sha256=<identical PLAN
   SHA-256>, and execution_approved=true **only after the parent grants this budget**.
   Engine/GPU configs retain implementation=bf51f09e, gpu_completion=true,
   retention_approved=false, service_residency=false and actual retained ELF hashes.
   RUN_A must equal the new engine run_root so probe paths and reports agree.
   CPU observer config role=ROLE_CPU_C names the original resident_config, not a
   CPU-start config. Retained namespace/resource/plan bindings stay unchanged.

   Replace only the retained peers' executable profile paths/arguments. 03/04:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --agent-config "$ROLE_CONFIG"
   ```

   05 CPU peer:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --observe-config "$OBSERVER_CONFIG"
   ```

   Controller keeps exactly ROLE_CPU_C / ROLE_ENGINE_A / ROLE_GPU_B and at most
   four approved temporary controller-mediated loopback forwards. Reuse exact
   private endpoint/SSH mappings; this document grants no new network authority.
   Record T before startup and run once:

   ```sh
   "$PYTHON" -B -I "$EXPERIMENTS/paired_profile.py" --supervisor-root "$SUP_ROOT" --controller-config "$CTRL_CONFIG" >"$EXECUTION_RESULT"
   ```

   It automatically runs all four phases after focused PASS. Engines must report
   READY, 5/2/5/2 normal DONEs respectively and rc 0; shorter normal stop is allowed
   and reported, length must reach its cap. Resource/ownership/product faults stop
   through existing cleanup. Wrong task answers and historical/new numerical
   screening failures are report outcomes, not a reason to skip the healthy paired
   task evaluation or destroy the resident arena. No retry or hidden warmup.

7. Stop owned engines/GPU worker/temporary transports normally. Confirm GPU
   baselines/listener cleanup by host/role; 05 intentionally remains resident and
   listening with its original guardian/monitor and fresh live status. Do not
   claim all Strata processes/ports are gone. Preserve existing explicit resident
   status/reconnect/unload/stop procedure; no signal authority comes from old PIDs.

   Retrieve only new bounded workload.out/workload.log, output-clock.json, four
   probe logits (4*993280 B total), two short .logpos files, watchdog/cleanup
   receipts and result JSON through the established route. Keep these private;
   never copy model/cache trees. On 05 set B_ROOT/G_ROOT to the retrieved 03 engine
   run-root layout (they may be the same directory) and run:

   ```sh
   (
     ulimit -v 1048576
     ulimit -t 30
     timeout --signal=TERM --kill-after=5s 60s "$PYTHON" -B -I "$EXPERIMENTS/paired_tasks.py" --pack "$PACK_META" report --plan "$PLAN" --b-root "$B_ROOT" --g-root "$G_ROOT" >"$PAIRED_RESULT"
   )
   ```

   Report per-task text/check/finish/tokens, matched task timing pairs and batch
   throughput; distinguish strict-format failures from substantive wrong answers.
   Include two reference-answer NLL comparisons and same-prefix next-logit
   centered rel-L1 / KL(B||G). Historical centering is by each vector's mean,
   denominator max(sum(abs(centered B)),1e-12), stable double softmax, unchanged
   .03/.03 thresholds. Record fresh numerical outcomes without redefining the old
   failed verdict. Do not compare later generated states as identical inputs or
   use generated-token agreement alone as quality. Report setup/steady-state
   counters and their scopes, CPU same-arena hits with zero new load/allocation/
   hash/network, full warm GPU cache reuse, warnings/minima and cleanup/residency.
   Preserve incomplete/fault evidence if any; never call a missing pair PASS.

This is one prepared practical comparison, not an authorization to execute it.
The only remaining approval is the proposed fresh model window/request budget.
