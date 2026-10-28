# Two frozen-suite passes in one resident process

`portable_resident_v1.py` keeps its original `--tokens-file --max-new` mode:
four identical requests, request 1 excluded, tensor caps `2404,2404`, and
load/request/child deadlines of 180/180/960 seconds.

Use `--suite --gguf --reference-root` instead to run the existing twelve frozen
prompts twice in order. The existing vocabulary, template, gold, scorer and
32-token budget are verified by `portable_quality_v1.frozen()`. This mode enables
expert timings, sends exactly 24 GEN requests and one QUIT to a single child,
and uses 180/600/3600-second load/request/child deadlines. Normal QUIT/reap has
15 seconds within the child deadline; forced owned cleanup has at most 10 seconds.
There is no additional warmup, restart, retry or engine reset command. The engine
resets sequence state before each GEN and retains its expert providers.

Example with already retained inputs (no build or download):

```sh
python -B tools/validation-harness/portable_resident_v1.py \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA256" \
  --suite "$SUITE" --gguf "$GGUF" --reference-root "$FRESH_PORTABLE_ROOT" \
  --plan "$PLAN0" --plan "$PLAN1" --expert-tensor-mib "$TENSOR_CAPS" \
  --manifest "$MANIFEST" --out "$NEW_RESIDENT_ROOT"
```

`--gguf` is the existing first model shard used by the child and tokenizer.
`--model-root`, if supplied, must be its parent. `--max-new` and `--tokens-file`
are excluded in suite mode. Two tensor caps may be supplied explicitly; workspace
and reserve stay `128,128` and `512,512` MiB. Device selection, split and total
VRAM caps use the existing resident options and admission checks.

The fresh reference must contain a complete twelve-case `summary.json`, plus
`case-01` through `case-12`, each with its original `argv.json` and `stdout.jsonl`.
Its parent must contain the original `checks/input-identities.sha256`. Before
launch, the adapter checks suite and binary identities, case order, each original
prompt/budget/context/cap argument, both independent retained plan hashes, and
every full token/greedy/stop sequence through the existing `completion()` parser.
Reference summary tokens and stops must match the raw records. Missing or foreign
evidence stops preparation; it never triggers reconstruction or native inference.
Keep the reference and plans immutable during execution.

Each request must match its own fresh reference exactly, participate on both
devices, and have valid consumed-row phase counts and finite nonnegative timing
and work counters. Source invalidation is forbidden throughout. Pass 1 populates
the cache: any of its twelve prompts may discover CPU experts. Pass 2 must have
zero verification, copy, digest and invalidation work in every request. A mismatch,
resource failure, timeout or unexpected child exit stops the campaign and cleans
up only the owned child. External monitors are outside the adapter's ownership.

Outputs retain the bounded raw stdout/stderr, argv, placements and all 24
`request-N.json` records with pass/case/request IDs. `pass-1/summary.json` and
`pass-2/summary.json` contain twelve indexed token/finish records when complete;
either can be passed as `--portable-root` to the unchanged quality adapter's
`score` command with the original native reference. A pass summary describes its
validated requests; top-level `summary.json` separately records child exit,
owned cleanup and resources. Inspect both before accepting a campaign.

Per-case `timing` retains reset, prompt/decode, selection, terminal feed and
nested model/provider/pool/executor deltas. Pass totals report consumed rows,
generated tokens and request-loop wall seconds (including receipt validation and
writes, excluding load and QUIT). Enclosing and nested timers overlap: do not
sum them or claim pooled TPS or an end-to-end fresh-load speedup. Matching a frozen
functional screen does not establish full numerical parity or broad quality.

The existing 0.2-second resource sampling and host/cgroup/swap/device guards
remain active through load, requests and cleanup waits. Stdout is capped at
1 MiB and stderr at 16 MiB; partial receipts survive STOP. Operator temperature,
clock, exact global-baseline and ownership observations remain separate. A
monitor-only baseline delta must not cause monitor shutdown or inference retries.

Host-only checks (no model or hardware):

```sh
python -m unittest discover -s tools/validation-harness -p 'test_portable*.py'
python -m unittest discover -s tools/validation-harness/tests -v
```

The first includes frozen binding/mutation, per-case reference identity, all 24
prompt/done IDs, legitimate first-pass discoveries, warm-zero rejection, invalid
counts/timers, no continuation after failure, owned cleanup, and legacy repeats.
The second uses disposable local sockets and processes for existing harness
regressions. Existing real sparse-model A/B/A, reset/replay and poison controls
remain unchanged; they require the separately built fixture binaries.

## Fresh references with dense weights and session state resident

Use the same qualified binary and frozen placements for both commands below.
Paths must be absolute; `GGUF` must be the first shard in `MODEL_ROOT`. Create a
new campaign directory, with `checks` already created and no existing reference
or resident output directory. Pin the actual binary and both plan files before
running; retain these immutable inputs and receipts privately:

```sh
sha256sum "$BINARY" "$PLAN0" "$PLAN1" > "$CAMPAIGN/checks/input-identities.sha256"
python -B tools/validation-harness/portable_quality_v1.py --gguf "$GGUF" portable \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA256" \
  --model-root "$MODEL_ROOT" --manifest "$MANIFEST" --suite "$SUITE" \
  --expert-plan "0:$PLAN0" --expert-plan "1:$PLAN1" \
  --expert-tensor-mib 2404,2404 --expert-workspace-mib 128,128 \
  --expert-free-reserve-mib 512,512 \
  --dense-resident-weights --gdn-resident-state --out "$CAMPAIGN/fresh"
python -B tools/validation-harness/portable_resident_v1.py \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA256" \
  --suite "$SUITE" --gguf "$GGUF" --manifest "$MANIFEST" \
  --reference-root "$CAMPAIGN/fresh" --plan "$PLAN0" --plan "$PLAN1" \
  --expert-tensor-mib 2404,2404 --dense-resident-weights --gdn-resident-state \
  --devices 0,1 --split 32 --device-cap-mib 7680 --out "$CAMPAIGN/resident"
```

The first command sends twelve GENs in twelve fresh processes. The second sends
cases 1–12 twice in one child: 36 GENs in all. Do not pass `--benchmark` or
`--single-case`, add warmups, reuse references from another binary/policy, or
retry a failed case. Both adapters forward the three opt-in residency flags;
unspecified flags remain off. Every reference case must contain each enabled
flag exactly once and each disabled flag zero times, including the older
`--gdn-resident-weights` option. A policy mismatch stops before child launch.

The commands retain context 256, threads 4, CPU workers 12, host cap 32768 MiB,
device cap 7680 MiB per die, workspace 128 MiB and reserve 512 MiB per die.
Use the existing external guarded execution for the resident command: outer
3650 seconds, finite clock guard 3900 seconds, existing temperature/sensor stops
and owned cleanup. The adapter enforces load 180, request 600, child 3600 seconds.
The fresh adapter retains its existing case 600/suite 3600-second bounds.
These commands do not install or manage the external guard.

After checking top-level resident status, exit code and cleanup, score each pass
with the unchanged scorer and retained native quality baseline:

```sh
python -B tools/validation-harness/portable_quality_v1.py --gguf "$GGUF" score \
  --suite "$SUITE" --portable-root "$CAMPAIGN/resident/pass-1" \
  --native-root "$NATIVE_REFERENCE" --native-sha256 "$NATIVE_SHA256" \
  --out "$CAMPAIGN/score-pass-1.json"
python -B tools/validation-harness/portable_quality_v1.py --gguf "$GGUF" score \
  --suite "$SUITE" --portable-root "$CAMPAIGN/resident/pass-2" \
  --native-root "$NATIVE_REFERENCE" --native-sha256 "$NATIVE_SHA256" \
  --out "$CAMPAIGN/score-pass-2.json"
```

Full same-backend rows and stops establish reset/replay parity. Gold scoring
separately screens answer accuracy; native/portable token equality is reported
but is not a requirement. Neither result establishes numerical parity between
backends. A native score below 12 remains inconclusive under the existing scorer.
