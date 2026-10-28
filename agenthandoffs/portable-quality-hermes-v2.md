> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Frozen twelve-case answer screen — corrected owned lifecycle, v2

**Completed; historical dispatch instructions below are not a request to rerun.**
See [portable-quality-results-v1.md](portable-quality-results-v1.md): both10/12,
identical tokens on all12, runtime success, scorer inconclusive-baseline.

Supersedes v1. No staged bundle, pins.json, attached resident, compiler build or
new service framework is needed. Reuse the retained driver's actual owned local
lifecycle: start one loopback CPU worker, LOAD once, run twelve local requests,
stop/reap owned engines and that worker. Unrelated processes remain untouched.

Exact handoff files at this published PR42 head:
`tools/validation-harness/portable_quality_v1.py`, its host-only test, unchanged
`quality.py`, `portable_model_smoke_v1.py`, and existing tokenizer/helpers imported
by quality.py. A source export is sufficient; no Git checkout on fleet required.

Pins:

| Artifact | SHA256 |
| --- | --- |
| PR43 existing portable binary | `cb71b55bd7309320a81446c117728e493ea1d19c8f1ec0d54c7ef2836c8734c3` |
| Existing native engine | `931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89` |
| Retained driver5 | `b5f1a5c10a0337d6254776ad01a19dd787076f41fd114538ea01f28708a1ae2f` |
| Frozen prompts/gold/template/scorer, quality.py | `01a1a6c4dc6c1a5f94f005316e34b7983699362fb5f3a4edb3811597892e38f1` |
| Existing tokenizer, strata_tokenizer.py | `e0cd8d6ae38dda0a4ad7c14f812eb64d5fd6f4fed9c920d4427a14bf585dc9b3` |
| Corrected adapter, portable_quality_v1.py | `e5058d27c2e2c4afe7881de0322be44f0a9845900072a7ae0bf5102ec6b83b01` |

The PR43 binary is already qualified at source
`ac85ff3fdf0834c13437b69b1568d8704e9b0717`; use it unchanged. The driver is
already on02 at `${PRIVATE_ARTIFACT_A27E9A14}`, also
shared as Slack F0CCH9D1CPP. Its exact source was read/hash-verified locally.
Importing it loads definitions only. The adapter never calls its `main`,
distributed arm, historical replay/comparison, OpenSSL/certificate check or TLS
setup. It reuses `prepare_argv`, original LOAD and unchanged `engine_stage`
READY/GEN/DONE/QUIT handling, marker checks and process-group cleanup.

## 1. Freeze once on05, before either output

Use existing tokenizer interpreters with regex; no install. Let SOURCE05/SOURCE02
be exports of this exact published handoff head. Existing host environment and
Vulkan device visibility remain as in the accepted PR43 pair. Fresh private
artifact directories, mode0700; do not reuse prior output roots.

```sh
set -euo pipefail
umask 077
export SCREEN05=${PRIVATE_ARTIFACT_1245348D}
export GGUF05=${PRIVATE_ARTIFACT_ED0D8ADE}
mkdir "$SCREEN05"
"$PYTHON05" -B "$SOURCE05/tools/validation-harness/quality.py" --gguf "$GGUF05" \
  prepare --out "$SCREEN05/suite"
sha256sum "$SCREEN05/suite/requests" "$SCREEN05/suite/quality_manifest" \
  "$SOURCE05/tools/validation-harness/quality.py" \
  "$SOURCE05/tools/validation-harness/portable_quality_v1.py" > "$SCREEN05/frozen.sha256"
```

Freeze the resulting complete request/manifest hashes in the receipt. Parent's
existing approved file transport may relay only these small suite files/source
exports between hosts. Place exact suite/requests and suite/quality_manifest on02
at `${PRIVATE_ARTIFACT_135B0956}`; verify
both file hashes against05. Never regenerate or change prompts/gold between arms.
The earlier10/12 native result is historical observed evidence, not ground truth
or this paired baseline: different ELF and no matched current suite results.

## 2. One owned native local baseline on02

No inference process was active at parent receipt1793156196.030979. Immediately
before launch the adapter refuses an occupied127.0.0.1:17431 without stopping it.
An occupied port is a concrete stop, never authorization to kill another service.

```sh
export SCREEN02=${PRIVATE_ARTIFACT_1245348D}
export GGUF02=${PRIVATE_ARTIFACT_38696482}
"$PYTHON02" -B "$SOURCE02/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF02" native \
  --driver ${PRIVATE_ARTIFACT_A27E9A14} \
  --suite "$SCREEN02/suite" --out "$SCREEN02/native"
```

Reads only the driver's known retained inputs:
`${PRIVATE_ARTIFACT_D2184BFE}` and
`${PRIVATE_ARTIFACT_4739C8B0}`.
These are inputs to the same retained local-argv derivation, not an assertion
that the NDon configuration was the actual local launch. GPU-peer options are
removed by the original driver; actual local argv is recorded for every case.
Native engine/manifest hashes and original worker734fabb8 prefix are checked;
full worker/config hashes are recorded. Existing worker argv/env, profile, pack,
prefill64, speculative/greedy flags and loopback CPU entry-row policy remain.
No certificates are read or changed and no remote GPU endpoint is launched.

One owned resident is preloaded once. Twelve fresh local engines each consume
one frozen GEN32 request, matching the portable CLI's fresh-state isolation.
No generation during LOAD, warmup or automatic retry. Existing driver bounds
apply: worker startup900s, LOAD/engine READY1800s, request300s, QUIT/stop420s.
On request/setup failure the retained function reaps its engine, then the adapter
stops/reaps only its owned worker and saves STOP/error receipts. SIGTERM/interrupt
also enters owned cleanup. Keep finite first-logit rows (~12MiB) on02 for the
existing validity check; no complete token-position logit dumps.

## 3. One portable suite on05, existing PR43 binary

```sh
"$PYTHON05" -B "$SOURCE05/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF05" portable \
  --binary ${PRIVATE_ARTIFACT_23A84F79} \
  --binary-sha256 cb71b55bd7309320a81446c117728e493ea1d19c8f1ec0d54c7ef2836c8734c3 \
  --suite "$SCREEN05/suite" --out "$SCREEN05/portable"
```

Exactly12 serial fresh-state cases, context256, max-new32, threads4/workers12,
host PLE, genuine CPU experts, contiguous Vulkan0/1 split32/16, caps7680MiB each.
Existing six-shard working-set+32GiB+8GiB admission and0.2s host/cgroup/swap/VRAM
guards apply. Strict48-layer/head placement,24 BF16 load roles, prompt/greedy/EOS
receipts checked. Case600s/suite3600s limits; no retries or logits. Owned-child
failure cleanup and STOP receipts persist. No source/build/weights/cache archives.

## 4. Score on02 and stop

Relay only portable/summary.json to a fresh private directory PORTABLE_RESULTS02
through existing approved transport. Source receipts/raw answers stay private.

```sh
"$PYTHON02" -B "$SOURCE02/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF02" score --suite "$SCREEN02/suite" \
  --portable-root "$PORTABLE_RESULTS02" --native-root "$SCREEN02/native" \
  --native-sha256 931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89 \
  --out "$SCREEN02/score.json"
```

Scoring calls unchanged quality.correct() with original gold. Both12/12:
narrow screen-pass (exit0). Native12/12 with portable misses: screen-regression
(exit1). Native below12/12: inconclusive-baseline (exit1), report both scores
fairly, not baseline answers as truth. Runtime/identity/cleanup failures are
independent fatal stops. Numerical parity remains FAILED explicitly in every
score; no threshold changes or broad-quality claim. Report aggregate scores,
case IDs, stops, hashes and cleanup/resource receipts; keep decoded answers and
private score JSON off Slack/Git. No extra runs for bad answers.

## Local evidence

Seven adapter tests passed, including fake owned-worker success/request failure,
occupied-port refusal with no launch/stop, portable pressure cleanup, strict
scoring and all three score outcomes. Fake-process success/failure checks of
the **exact pinned driver5 engine_stage** also passed locally (no child launched).
Existing15 supervisor and7 quality tests passed; help/diff checks passed.
No C++ build, fleet run, model payload load or service operation by this task.
