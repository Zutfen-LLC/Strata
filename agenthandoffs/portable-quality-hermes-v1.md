# Frozen twelve-case functional screen — Hermes v1

No model build. Use PR43 source `ac85ff3fdf0834c13437b69b1568d8704e9b0717`
and its already qualified `portable_model` under
`/srv/models/inferswarm-artifacts/portable-expert-ac85ff3/build/` on05. Parent
must supply its exact **qualified binary SHA256 from the pair receipt**; do not
turn a freshly calculated hash into new qualification. The adapter uses the
unchanged CLI; source inspection of PR43 confirms no extra reuse switch is needed.
This handoff source pin is the full published PR42 commit accompanying this file.

Suite/scorer: `tools/validation-harness/quality.py`, SHA256
`01a1a6c4dc6c1a5f94f005316e34b7983699362fb5f3a4edb3811597892e38f1`.
It owns all12 prompts, gold answers, chat template, strict scorer, max-new32 and
max-prompt224. Existing tokenizer `tools/strata_tokenizer.py` SHA256
`e0cd8d6ae38dda0a4ad7c14f812eb64d5fd6f4fed9c920d4427a14bf585dc9b3`.
Adapter `tools/validation-harness/portable_quality_v1.py` SHA256
`e57aa5491abf421d860d3df08a4e9792cd46f6ae29c9834915be0dfe6f6b4aca`.
No new prompt/scorer, package install, weights, gold changes or numerical limits.
The existing metadata/tokenizer Python environment (with regex) is required.

## Baseline evidence and concrete prerequisite

Located the original12-case native receipt: Slack C0C5MK91H3L,
thread1793054475.545389, final1793055522.698579. Private evidence02:
`/home/hermes/amd-quality-pr34-20261026/score.json`, with config/pins/suite nearby.
It scored10/12 (cases9/12 failed), mixed11/12; result inconclusive-baseline.
Its native ELF was `dbea1269...`, not the selected current native ELF SHA256
`931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89`.
Thus it is historical evidence, not a matched current baseline to reuse silently.
The run5 first-logit38-token row is also not a twelve-case answer baseline.

Required narrow native input: **the path of a complete retained staged native
bundle** for the selected current02 native binary/settings, with `config.json`,
`pins.json`, attached resident identity root and existing helpers/assets. It must
have one native arm, an explicitly attached idle CPU resident, restart=false and
retain on success/failure. Known run5 receipt root is
`/home/hermes/strata-nd-repeat-20261027-run5/local/`, but this task has not verified
that it is such a bundle. Do not invent `pins.json`, an attach-root, PID, port or
launch policy if it is not. Hermes/parent can provide the exact retained bundle
path; that small prerequisite is needed before the paired dispatch. No broad
search, resident restart, network/security change or new driver is authorized.

If a genuinely matched twelve-case bundle already exists, `score` verifies its
pinned files, selected executable, frozen request IDs/vocabulary/scorer/manifest,
complete ordered arm and finite first-logit rows. It can then be reused without
native inference. Otherwise stage/run one native12-request arm as below, using
existing harness lifecycle and retained resident controls. No warmup or rerun.

## 1. Freeze suite and both arm inputs before generation

On05, use a small existing source export of this handoff commit; no C++ build.
`SOURCE` denotes that export, `PYTHON` the already working tokenizer interpreter.
Fresh results live entirely on the artifact volume. Run with `umask077`.

```sh
set -euo pipefail
umask 077
: "${SOURCE:?exact published handoff source export}"
: "${PYTHON:?existing tokenizer Python with regex}"
: "${QUALIFIED_PR43_BINARY_SHA256:?exact binary SHA from accepted PR43 pair receipt}"
export SCREEN05=/srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v1
export GGUF05=/srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
export PORTABLE05=/srv/models/inferswarm-artifacts/portable-expert-ac85ff3/build/portable_model
mkdir "$SCREEN05"
"$PYTHON" -B "$SOURCE/tools/validation-harness/quality.py" --gguf "$GGUF05" \
  prepare --out "$SCREEN05/suite"
sha256sum "$SCREEN05/suite/requests" "$SCREEN05/suite/quality_manifest" \
  "$SOURCE/tools/validation-harness/quality.py" \
  "$SOURCE/tools/validation-harness/portable_quality_v1.py" > "$SCREEN05/frozen.sha256"
```

The resulting exact files are `suite/requests` (12 GEN32 lines) and
`suite/quality_manifest`. The latter binds complete request/vocabulary/scorer
SHA256 values; return those hashes before runtime. A historical receipt only
supplied request prefix4fa19236; do not invent its missing full digest.
Use these exact two small frozen files on02 through the parent's existing
approved file transport; byte-verify against `frozen.sha256`, no re-tokenization.
Freeze binary SHA/config receipts as well. Keep source hashes with the receipt.

## 2. Native arm on02, only if no genuinely matched retained arm

Set `SOURCE02` to the same handoff source export, `PYTHON02` to the existing
interpreter, `NATIVE_TEMPLATE02` to the prerequisite verified staged bundle, and
`GGUF02` to the existing metadata shard:
`/srv/models/qwen38-q8/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf`.
Create the fresh02 artifact root and place the frozen suite at its `suite/`.
Before launch, parent/Hermes confirms exclusive idle use of the attached resident
using its existing identity procedure; no guessed PID or service restart.

```sh
export SCREEN02=/srv/models/inferswarm-artifacts/portable-quality-ac85ff3-v1
export NATIVE_SHA256=931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89
"$PYTHON02" -B "$SOURCE02/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF02" stage-native --template-root "$NATIVE_TEMPLATE02" \
  --native-sha256 "$NATIVE_SHA256" --suite "$SCREEN02/suite" --out "$SCREEN02/native"
# Review native.input.json: same argv/env/model/pack/GPU/CPU policies; only new
# request/suite paths, fresh output root, quality capture and arm request indices.
# It refuses restart/stop policies. It does not launch during stage-native.
"$PYTHON02" -B "$SCREEN02/native/run.py" --run-root "$SCREEN02/native"
```

The existing finite lifecycle/request deadlines, idle/identity checks, masks,
CPU resident retention and owned-engine cleanup remain in force. The native
harness writes12 finite first-logit rows (~12MiB) as its existing runtime guard;
no extra dumps or new logging framework. Do not use timeout to kill the lifecycle
controller and leave brokers behind. Stop on any failure, preserve evidence.

## 3. Portable arm on05; one sequence, fresh process per case

```sh
"$PYTHON" -B "$SOURCE/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF05" portable --binary "$PORTABLE05" \
  --binary-sha256 "$QUALIFIED_PR43_BINARY_SHA256" \
  --suite "$SCREEN05/suite" --out "$SCREEN05/portable"
```

The adapter runs exactly12 serial CLI requests, fresh state each time,
context256/max-new32, four threads,12 CPU expert workers, host PLE, explicit
Vulkan0/1 contiguous32/16 split and7680MiB separate caps. It imports the existing
supervisor's actual host/cgroup/swap/VRAM admission and0.2s guards, including
six-shard working-set+32GiB+8GiB admission. It validates all48 layers+head and
24 BF16 load markers, contiguous positions, exact consumed prompt, greedy IDs,
EOS and normal stops. It emits no logits. Max600s per case/3600s whole portable
suite; no retries. On failure it terminates/reaps only its owned child group,
keeps STOP summary and starts no next case. Per-case stdout/stderr capped8MiB;
only compact per-case resource extrema, argv, placements and token receipts.
At most3072 consumed tokens/384 generated tokens. This is not a speed benchmark.

## 4. Score using original strict rules

Run on02 with both small portable results and the native bundle accessible via
already approved transport. Preserve `case-*/placements.tsv` and raw receipts05;
only portable `summary.json` is needed alongside the native bundle to score.
`PORTABLE_RESULTS02` is the copied private directory containing that summary.

```sh
"$PYTHON02" -B "$SOURCE02/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF02" score --suite "$SCREEN02/suite" \
  --portable-root "$PORTABLE_RESULTS02" --native-root "$SCREEN02/native" \
  --native-sha256 "$NATIVE_SHA256" --out "$SCREEN02/score.json"
```

No clean native12/12: `inconclusive-baseline` (exit1); report both scores and
case IDs. Native12/12 with portable misses: `screen-regression` (exit1); stop
expansion and retain first failing case. Both12/12 with healthy execution:
`screen-pass` (exit0), narrowly for these12 tasks. No broad quality-equivalence
claim. Original full-model numerical parity remains FAILED in every report.
Runtime/placement/identity failure is a separate fatal result, not a wrong answer
or a resumable suite. No prompt/gold/limit edits after output; no answer-driven retry.
Keep decoded answers/private score off GitHub/Slack; return aggregate, case IDs,
stop counts, hashes and resource/placement/cleanup receipts. No heavy new artifacts.

## Local adapter validation

Six host-only tests passed: frozen-suite mutation refusal, prompt/greedy/EOS
receipt validation, native config preservation/resident-policy rejection, owned
child cleanup and STOP persistence on pressure, all three fixed score outcomes,
and strict existing scorer behavior. Existing15 supervisor and7 quality tests
also passed. No model, fleet, network or GPU used by these tests. CLI help and
diff checks passed. No build of PR43 or change to its provider.
