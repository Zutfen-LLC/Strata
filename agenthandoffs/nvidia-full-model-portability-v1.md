# Bounded NVIDIA full-model staging and NVIDIA02 mechanism receipt

This slice starts at PR45 `3d7e20e1f9c7e173b468191251826a5e49ab4a47`.
It changes device observation/selection only. No numerical qualification is
added: NVIDIA tiny resident probes do not qualify the full model. Intel has no
validated telemetry adapter and stops with an explicit adapter requirement.

The parent selected NVIDIA02 for the recorded smoke below. For any future run,
the parent selects the host after review; do not assume03 has weights.
Use the existing six Q8 model shards and matching retained SWLOCAL1 manifest
in place; no model copies. Put fresh source/build/receipts/evidence on a verified
secondary-disk artifact root with sufficient free space. Retain source commit,
GGML pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`, binary hash, and the
unchanged loader/ICD environment. Build the candidate's `portable_model` with
its existing portable FFN/GDN/QSA/model and Vulkan options, using already
available dependencies. No package, driver, security or service changes.

## Recorded NVIDIA02 mechanism smoke

The parent supplied this completed-run receipt and its
[primary evidence](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793168735004679).
This documentation task did not execute or independently reproduce the run.

- Source: `fb2ea3c92093eacc8663d2b90b88813ec2bc4076`.
- Evidence root on 02: `/home/hermes/nvidia-full-model-fb2ea3c`.
- One full-model mechanism smoke: 210.3 seconds, exit 0, 38 prompt tokens,
  generated IDs `[71093, 12305, 198]`, stop `max_new`. No expert plans were used.
- Logical Vulkan0: PCI `0000:02:00.0`, RTX3060Ti. Logical Vulkan1:
  PCI `0000:03:00.0`, RTX3060. Split 32; each device cap 7680 MiB.
- Strict placement checks: 3223 operations on Vulkan0 and 1606 on Vulkan1;
  no fallback.
- Minimum effective host/cgroup available memory: 260172148736 bytes.
  Peak process RSS: 41373507584 bytes. Sampled aggregate VRAM peaks:
  1307574272/1329594368 bytes in logical-device order. Zero swap growth.
- Manifest pin, six shard sizes, headers and loader admission passed.
  No full model rehash was performed.
- Cleanup: owned process gone, no compute applications, VRAM returned to
  baseline 9 MiB/1 MiB in logical-device order.
- A source filename typo was corrected before transfer; no inference retry.

Retained SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Executed binary | `98db98aeb768ab28b5934e034519fde572c929818e9b9fb850b267d2da7163f1` |
| Source archive | `8bdd2339f2d273b9e99a111f7cbafc2644a99a6987cce65bedbb92316d14af8b` |
| GGML archive (commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`) | `cb7b39ca7e6fc5b9adc79d0ae0f14367207b59ae6a859d9a46156d1a3c697889` |

Result: **NVIDIA02 full-model mechanism smoke PASS at the source above**.
The generated prefix agrees with the retained IDs, but full numerical and
quality qualification remain pending. This run is not a speed comparison and
does not qualify local expert residency or Intel. The memory peaks are sampled
aggregate observations, not exact transient peaks or per-process GPU accounting.

## Preflight and single smoke recipe (no new run authorized)

Resolve these variables on the parent-selected host. `LOGICAL_DEVICES` contains
indices from this candidate's **GGML logical** enumeration, not raw Vulkan or
nvidia-smi indices. `PCI_DEVICES` contains the corresponding ordered physical
bus IDs. An existing GGML_VK_VISIBLE_DEVICES setting can remap logical indices;
keep it identical for listing and launch. Missing PCI extension information
is a STOP, not a reason to guess an index.

```sh
set -euo pipefail
umask 077
: "${SOURCE:?candidate source}" "${BINARY:?candidate portable_model}"
: "${MODEL_ROOT:?existing six Q8 shards}" "${MANIFEST:?matching retained manifest}"
: "${RECEIPTS:?fresh receipts path on verified secondary disk}"
: "${EVIDENCE:?new evidence path on verified secondary disk}"
mkdir "$RECEIPTS"
git -C "$SOURCE" rev-parse HEAD > "$RECEIPTS/source.sha"
sha256sum "$BINARY" > "$RECEIPTS/binary.sha256"
"$BINARY" --list-vulkan > "$RECEIPTS/vulkan.tsv" 2> "$RECEIPTS/vulkan.log"
nvidia-smi --query-gpu=pci.bus_id,uuid,name,memory.used,memory.total --format=csv \
  > "$RECEIPTS/nvidia.csv"
```

Review that listing and select two actual GPUs, excluding llvmpipe/lavapipe.
For03 only, the previously observed physical IDs were01:00.0/03:00.0; these
are context, not a host selection or an inferred Vulkan mapping. Verify current
identity, existing model/manifest paths, free disk, and no interfering GPU load.
Then perform exactly one run; do not retry a failure:

```sh
: "${LOGICAL_DEVICES:?two verified GGML indices, comma separated}"
: "${PCI_DEVICES:?two matching domain:bus:device.function IDs, comma separated}"
python3 "$SOURCE/tools/validation-harness/portable_model_smoke_v1.py" \
  --binary "$BINARY" --model-root "$MODEL_ROOT" --manifest "$MANIFEST" \
  --devices "$LOGICAL_DEVICES" --pci "$PCI_DEVICES" \
  --split 32 --device-cap-mib 7680 --timeout-seconds 1800 --evidence "$EVIDENCE"
```

The fixed prompt, three generated tokens, precision, placement checks and expert
source contracts remain unchanged. The cap ceiling remains 8192 MiB; there is no
reason to raise it for this staging smoke. Admission checks each cap against
GGML's physical capacity and current aggregate telemetry free memory. Memory
observations use AMD sysfs bytes or NVIDIA PCI-addressed nvidia-smi MiB counters
converted to bytes. NVIDIA observations have MiB granularity. These are sampled
aggregate counters (including other users), not exact transient peaks or
per-process allocation accounting. A failed query (5-second timeout per GPU),
invalid observation, capacity drift or unsupported vendor stops and reaps the
owned model process; guards are never disabled. Host/cgroup full working-set
admission, 32 GiB anonymous cap, 8 GiB reserve, swap-growth and timeout checks
remain active.

For the existing one-token local expert workflow, the same optional `--devices`,
`--pci`, `--split`, `--device-cap-mib`, `--model-root`, and `--manifest` arguments
are accepted by `portable_local_expert_smoke_v1.py`. Keep its existing plan and
binary-hash requirements and 300-second budget. Plans must match the selected
slot placement; this change does not generate or reinterpret plans.

Retain the complete STOP or mechanism report, argv, placement report, memory
samples, stderr, logits, exit and cleanup observations. A mechanism pass remains
`numerical/quality verdict pending`; compare against the retained numerical
reference separately. The NVIDIA02 run above establishes only that recorded
mechanism result. Intel full-model execution remains unvalidated.

For any future run, prerequisites remain: a reviewed exact candidate built on the chosen host,
verified GGML-to-PCI mapping with VK_EXT_pci_bus_info, working NVIDIA memory
telemetry, and existing full model plus matching manifest with sufficient
host/cgroup headroom. Host03 model availability is unestablished. Intel needs
an actual supported aggregate memory telemetry adapter plus hardware validation;
`vram(pci)` is the narrow extension point and currently fails closed for Intel.
