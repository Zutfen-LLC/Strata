# Frozen native-method answer screen

`portable_quality_v1.py portable` can forward the three native method options
without changing its frozen twelve prompts, tokenizer/template, scorer, context
256, maximum 32 generated tokens, resource guards or case/suite deadlines.
This is the existing fresh-process loop: each case starts a new model session.
It uses `--tokens` with `--max-new 32`, equivalent to the frozen GEN32 request;
it does not combine those switches with stdio. No engine changes or new harness.

Declare all three choices explicitly and enable native grouped-expert
observation in both arms:

```
control:   --no-native-router --no-native-shared --no-native-moe-combine --expert-native-q8-observation
candidate: --native-router --native-shared --native-moe-combine --expert-native-q8-observation
```

These adapter options are booleans. It generates fresh private product receipt
prefixes under each `case-NN/native-receipts` directory. The product receives
`--native-router PREFIX`, `--native-shared PREFIX` and `--native-moe-combine`
only when enabled. Both arms retain the same native expert implementation and
frozen placement/CPU misses. Omission of all new options preserves the previous
adapter argv, default receipt parsing and summary fields. Partial declarations
are rejected before model launch. This declared screen excludes benchmark,
single-case, chunking, head elision and executed-head dumps.

Use the same existing product binary for both commands, with the strict native
build already qualified on the selected devices. This adapter-only change does
not require rebuilding the engine. Preserve the verified process-local driver
and SDK environment, frozen model/manifest/20-expert plan, device order, split,
threads and caps. Keep the established external thermal/clock supervisor.
Prepare each arm in a new output directory; do not retry a failed case or
change plans after observing answers.

```bash
python3 -B tools/validation-harness/portable_quality_v1.py --gguf "$GGUF" portable \
  --binary "$BINARY" --binary-sha256 "$BINARY_SHA256" \
  --model-root "$MODEL_ROOT" --manifest "$MANIFEST" --suite "$FROZEN_SUITE" \
  --expert-plan "0:$FROZEN_20_EXPERT_PLAN" --expert-tensor-mib 256,256 \
  --expert-workspace-mib 128,128 --expert-free-reserve-mib 512,512 \
  --devices 0,1 --pci "$VERIFIED_PCI_ORDER" --split 32 --device-cap-mib 7680,7680 \
  --no-native-router --no-native-shared --no-native-moe-combine \
  --expert-native-q8-observation --out "$NEW_CONTROL_ROOT"
# After successful control exit and cleanup, use the identical common arguments,
# replacing only the three selections and output root:
# --native-router --native-shared --native-moe-combine --out "$NEW_CANDIDATE_ROOT"
```

Each arm runs cases 1–12 once, in order, with no extra warmup. Existing bounds
are 600 seconds per case and 3600 seconds per suite, plus owned termination/reap.
An external per-arm guard can retain the previously used 3650-second outer and
3900-second clock-monitor bounds; it remains responsible for temperatures,
clock policy and exact baseline observations. The adapter stops at the first
protocol, nonfinite, resource, child or receipt failure and retains partial
results. Samples of direct child/host resources do not prove whole-tree coverage.

The adapter records explicit `native_methods` and bounded plan SHA/placement
bindings, checks plan identity before/after every case, and requires actual
Vulkan expert participation and complete row checks. Enabled router/shared
completion calls and rows must each equal 48 times consumed tokens; OFF fields
must be absent. The final stop receipt supplies these counters without requesting
a timing mode. Combine has no separate product completion counter: the adapter
requires its effective-module receipts and successful complete generation.

Effective modules are checked for expected layer/name coverage, bounded files,
SPIR-V headers and SHA256. Extra failed-planning owner receipts are retained and
hashed; unrequested method modules are refused. These checks are not a shader
validator. Use the matching SDK `spirv-val --target-env vulkan1.2` on retained
modules during operator evidence review. No runtime model activations are saved
by these shader-receipt prefixes.

Score each complete arm with the unchanged `score` command and retained native
reference. It keeps the original gold, categories and 12/12 baseline rule;
token equality is reported separately. An incomplete arm or failed native
baseline remains inconclusive. A screen pass is not broad quality, numerical
parity, complete native placement or performance qualification.

Host-only tests (no model load or GPU):

```bash
python3 -m unittest discover -s tools/validation-harness -p 'test_portable*.py'
```

They exercise the actual twelve-case loop with mocked model children, 24
requests through a compiled exact product parser prefix stopped before backend
setup, fresh child ownership/reset boundaries, receipt uniqueness, plan/counter/
module/participation failures and no continuation after failure. Existing default
quality, resident and scorer checks remain unchanged.
