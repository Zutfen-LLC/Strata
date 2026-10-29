# Local CUDA and Vulkan product comparison

This is a prospective recipe and **unconfirmed static analysis**, not a hardware
result. It accompanies the fresh adapter's device selection change and issue
[#65](https://github.com/Zutfen-LLC/Strata/issues/65). No runtime code changes.
The baseline is main `45d0493640b43e6264bd6d52da02a0ae5bf061cf`, tree
`b9f526a1e7787a3c88a0f1af55c58f206a219ebd` (the merged PR79 tree).
PR75 overlap and PR80 device combine are outside this recipe.

## Device selection and preflight

`portable_quality_v1.py portable` now shares `device_arguments`,
`device_selection`, `verified_devices`, `sample_vram` and `admit_vram` with the
resident adapter through `portable_model_smoke_v1.py`. Exactly two devices are
supported. Omitted options retain devices `0,1`, the existing AMD PCI defaults,
split `32`, and a `7680` MiB cap on each device. Explicit selection is required
on another host. Selection does not choose experts or generate placement plans.

The binary's `--list-vulkan` output must map each ordered logical index to its
selected physical PCI address. Software devices, missing mappings and invalid
capacity stop before inference. VRAM sampling uses AMD sysfs or NVIDIA
`nvidia-smi` queried by PCI address, never NVIDIA's unrelated logical index.
Receipts contain the verified mapping, selected split/cap and per-request VRAM
baseline. Placement verification uses those same logical indices and split.
The shared cap bound remains 1–8192 MiB; admission still requires the entire cap
free on each selected physical device. Host/cgroup admission, the 8 GiB reserve,
swap checks, output limits, deadlines and owned-child cleanup remain in force.

Before any hardware run, retain private source/build/dependency/binary hashes,
model and pack provenance, actual Vulkan index/PCI inventory, driver/ICD,
placement plans and resource baseline. Preserve other services. Keep existing
finite outer guards, continuous host/swap/VRAM and thermal/sensor monitoring,
clock restoration and owned-process cleanup. This Python adapter alone does not
supply the external temperature/clock guard. Stop on first failure; do not raise
caps, change tolerances or retry to obtain a pass.

Latest GDN projection/causal chunk and QSA residency/reuse policies still need
focused NVIDIA qualification. Build and run existing FFN, GDN, PLE, QSA, BF16
and head fixtures on each selected NVIDIA device with their strict
backend/numerical and route/state gates before any production-model capture.
Include `portable_head_test --vulkan INDEX` to qualify the raw integer status
and scheduled nonfinite head guard on that backend. Use the other fixtures'
supported device options (`--device INDEX --expect-device DESCRIPTION`, BF16
`--vulkan INDEX`). BF16 and head lack `--expect-device`; bind those runs to the
external verified Vulkan index/PCI inventory.
Record actual binary hashes and metrics; then qualify the exact opt-in policy
with the existing fresh executed-head capture/comparator and functional checks.
Previous AMD qualifications remain evidence for unchanged runtime on AMD; this
harness change does not require repeating that campaign or establish NVIDIA
support. Do not substitute llama.cpp for either product arm.

Prepare the frozen suite once from retained tokenizer metadata:

```sh
python3 -B "$SRC/tools/validation-harness/quality.py" \
  --gguf "$GGUF" prepare --out "$SUITE"
"$PORTABLE" --list-vulkan
```

For fresh case1 capture, with variables bound to private retained inputs:

```sh
python3 -B "$SRC/tools/validation-harness/portable_quality_v1.py" \
  --gguf "$GGUF" portable --binary "$PORTABLE" --binary-sha256 "$PORTABLE_SHA" \
  --model-root "$MODEL_ROOT" --manifest "$MANIFEST" \
  --devices "$VK_INDICES" --pci "$PCI_IDS" --split "$LAYER_SPLIT" \
  --device-cap-mib "$CAP_MIB" \
  --expert-plan "0:$PLAN0" --expert-plan "1:$PLAN1" \
  --expert-tensor-mib "$EXPERT_MIB" --expert-workspace-mib 128,128 \
  --expert-free-reserve-mib 512,512 \
  --dense-resident-weights --gdn-resident-state --gdn-resident-weights \
  --gdn-production --qsa-production --qsa-reuse-attention \
  --prompt-head-elision --prompt-chunk-size 8 \
  --gdn-batch-input-projections --gdn-causal-chunk \
  --single-case --dump-executed-logits --suite "$SUITE" --out "$NEW_CAPTURE"
```

This uses context256, max-new32, threads4, expert-workers12 and host-cap32768
MiB. The plan prefixes `0` and `1` are executor slots, not physical GPU indices.
For fresh timing, replace `--single-case --dump-executed-logits` with
`--benchmark`: four fresh engines, first excluded. Capture and timing are
separate runs; dumps are not performance results. Preserve exact argv and policy
receipts. `executed_logits_v1.py` compares portable executed-head captures using
explicit baseline/candidate chunk sizes and unchanged gates; it is not a native
CUDA capture adapter. Matching generated tokens does not establish logit parity.

## Existing resident interfaces

The native engine already supports one loaded process with repeated requests.
In `src/program/generate.cpp`, the `--serve` loop accepts
`GEN <max_new> <comma-separated IDs>`, emits `T` tokens and `DONE`, and accepts
`QUIT`. Weights, expert arena and GPU tier load once. With `--prompt-cache 0`
and `--conversation-cache-mib 0`, `resume` stays zero, and `session_zero` clears
sequence state before each request, including layer-split stages. No explicit
RESET command or new serving layer is needed. Require `RESUME 0`, zero reused
prompt tokens in DONE, and complete prompt accounting in every retained log.

`harness/campaign.py:run_arm` launches one session outside its request loop and
can send request indices `[0,0,0,0]` through that engine. Use existing
`harness/bundle.py` and `harness/run.py` with a private config derived from the
retained working local native config. Keep its verified model/pack, fixed expert
profile, same-host CPU worker, finite monitor and affinity settings. Disable
remote GPU/other-host routes. Set `--serve --native`, `--greedy`,
`--prompt-cache 0`, `--conversation-cache-mib 0`, context256, and matched prefill8.
For the serial-decode comparison explicitly use `--spec 0 --suffix-draft 0
--lookup-chain 0 --pipeline-windows 0`; keep adaptation disabled and the fixed
expert placement unchanged. Confirm these policies in actual startup receipts.
Use one zero-GEN load entry and one `restart:false` warm entry with requests
`[0,0,0,0]`, excluding its first request offline. The harness finishes/reaps the
engine after all four, rather than between requests. Do not select a repeated
reference summary with duplicate indices as `reference`; compare retained
per-request tokens/stops offline. Keep raw DONE/stdout receipts for stop and
reuse fields that the default timing summary omits.

```sh
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$NATIVE_CONFIG"
python3 -B "$NATIVE_RUN_ROOT/run.py" --run-root "$NATIVE_RUN_ROOT"
```

Portable's existing `portable_resident_v1.py` uses `--stdio`; its CLI resets the
session before each GEN while retaining model/expert residency. Its default
mode takes `--tokens-file` containing the frozen case1 ID array, `--max-new 32`,
`--binary`, `--binary-sha256`, `--out`, two `--plan` paths and the same selected
`--devices`, `--pci`, `--split`, `--device-cap-mib`, model/manifest and opt-in
policies above. It sends four requests and excludes the first. The existing
suite mode instead compares fresh same-policy references with two passes over
all twelve prompts. Do not substitute that workload for the four-request case1
comparison. Neither resident interface needs modification for this task.

```sh
python3 -B "$SRC/tools/validation-harness/portable_resident_v1.py" \
  --binary "$PORTABLE" --binary-sha256 "$PORTABLE_SHA" --out "$NEW_RESIDENT" \
  --tokens-file "$CASE1_IDS_JSON" --max-new 32 --expert-timings \
  --plan "$PLAN0" --plan "$PLAN1" --expert-tensor-mib "$EXPERT_MIB" \
  --model-root "$MODEL_ROOT" --manifest "$MANIFEST" \
  --devices "$VK_INDICES" --pci "$PCI_IDS" --split "$LAYER_SPLIT" \
  --device-cap-mib "$CAP_MIB" \
  --dense-resident-weights --gdn-resident-state --gdn-resident-weights \
  --gdn-production --qsa-production --qsa-reuse-attention \
  --prompt-head-elision --prompt-chunk-size 8 \
  --gdn-batch-input-projections --gdn-causal-chunk
```

The older `portable_quality_v1.py:owned_native` benchmark is different: it keeps
one loaded CPU worker/cache but calls driver `engine_stage` once per request,
launching fresh engines. Fresh portable mode also launches a new model process
per request and reloads/verifies its expert sources. Thus those fresh timers
have asymmetric setup/verification lifecycles even after warmup; do not label
them a warm resident backend comparison. Report setup/load/verification
separately from prompt, decode and end-to-end request time. Portable reset time
and native reset timing scopes also differ; preserve raw scopes, wall time and
original units rather than adding nested timers.

## What must match

Freeze before execution: model shard and native pack provenance/quantization,
vocabulary and prompt ID bytes/order, expert layer/ID sets and CPU/GPU placement,
layer split, resident budgets, CPU workers/threads/affinity/NUMA, context,
prefill and speculation policies, head elision, greedy decoding, max-new/EOS,
clock/power policies, memory caps/reserves, run order and exclusions. Verify
actual routing and residency receipts, not just equal slot counts. Different
native/portable storage formats need recorded provenance to the same weights.
If a cap, placement or policy cannot be matched, stop calling the comparison
controlled and state the difference explicitly; do not silently alter either arm.

First compare NVIDIA Vulkan with native CUDA on the same selected NVIDIA cards.
Then compare fully local AMD Vulkan with the declared local NVIDIA CUDA product
configuration. Keep CPU work on each declared local host; no distributed worker
substitution. Different hardware or placement is a **product-configuration
comparison**, not isolated backend causality. Even same-card CUDA/Vulkan timings
compare different engine implementations and must disclose remaining policies.

Retain each request's exact IDs, generated sequence, normalized and raw stop,
prompt/output/logical/executed/head/skipped counts where exposed, prompt/decode
and request wall times, warm-zero expert load/verify/copy work, admission and
sampled resource/temperature receipts, failures and cleanup. Mark unavailable
counters unavailable. Quality scoring, token equality and numerical logit gates
are separate outcomes. Use short counterbalanced arms and report individual
warm measurements before aggregates; no performance claim follows from this
source-only recipe.
