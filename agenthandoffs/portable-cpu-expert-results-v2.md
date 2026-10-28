# Portable CPU expert reuse validation — v2

Provider/CLI/test source: `fb9b4904e9d7b7eb10f38e0c0f8220ed87ef7125`,
parent `59ce0bc5aae81b8cd4f3f17924e8b1cbb16a243d`. These clean CMake
checks supersede v1's manually linked publication checkpoint. No provider,
quantizer, native kernel, route or reduction changes were needed for this
validation. The follow-up adds only the supervisor comparison flag/test and
documentation. All old evidence and build directories remain intact.

Environment: local Linux x86_64, GCC 16.2.1, existing CMake 3.31.6. Fresh
`validation/cpu` and `validation/vulkan` build trees under the isolated task-9
checkout. Both rebuilt the required GGML, native CPU and portable libraries
from source; neither links retained task-8 libraries. Used the existing
`/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d` source snapshot
and retained SPIR-V headers, with dependency fetching disconnected. This
local snapshot lacks Git metadata; its path/pin comes from the retained
prior build setup, not a fresh independent Git provenance check.

## Passed

Both clean configurations used Release, STRATA_NATIVE_EXPERTS,
STRATA_PORTABLE_MODEL/FFN/GDN/QSA and STRATA_BUILD_TESTS ON; CUDA/HIP/SYCL
OFF. CPU used GGML_VULKAN OFF; Vulkan used GGML_VULKAN ON plus existing
`${BUILD_ARTIFACT_D2E5D17A}` include/prefix settings. Each build used
two jobs and actual CMake targets:

```text
portable_cpu_experts_test portable_model_test portable_model
portable_head_test portable_device_admission_test portable_eos_metadata_test
portable_ffn_test portable_gdn_test portable_ple_test portable_qsa_test
portable_qsa_gguf_test portable_qsa_bf16_test
```

The Vulkan build additionally built portable_{ffn,gdn,ple,qsa}_vulkan_test.
All targets completed successfully. Both generated CLIs passed `--help` and
advertise `--expert-dispatch-oracle`.

Exactly one selected CTest invocation per build:

```sh
ctest --test-dir BUILD --output-on-failure \
  -R '^portable_(cpu_experts|model|head|device_admission|eos_metadata|ffn|gdn|ple|qsa|qsa_gguf|qsa_bf16)_test$'
```

- CPU: 11/11 passed, 44.56 seconds total; full model test 29.24 seconds.
- Vulkan-linked: 11/11 passed, 43.46 seconds total; full model 28.12 seconds.
  These CTests run their CPU cases; device execution is recorded separately.
- In each build the real Q8 native-row test records hits=20,
  verifications=43, copied_bytes=digest_bytes=224563200, native_batches=6.
  The ten-expert repeated batch adds no copy/digest bytes. New expert misses,
  reorder, layer/model isolation, retained mapping use, ordinary metadata
  change/corruption and failed verification retry pass.
- In each build the 48-layer/full-vocabulary synthetic model compares two
  sequential tokens byte for byte against the former staging oracle. The
  second token has 480 expert hits. Existing scalar logits, late layer-10
  corruption, poison/reset/replay, generation, EOS and context checks pass.
  Model report remains mapped=188214088960, host_weights=4831936000,
  staging=52224000 bytes. No F32 expert arena or cap increase.

Explicit device checks used only software llvmpipe LLVM23.1.1, Vulkan0,
selected with `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json` and
`GGML_VK_VISIBLE_DEVICES=0`. Each executable ran once with timeout180:

```sh
portable_{ffn,gdn,ple,qsa}_vulkan_test --device 0 --expect-device llvmpipe
portable_{head,qsa_bf16,device_admission}_test --vulkan 0
```

All seven passed: actual strict scheduled Vulkan execution and CPU/scalar
checks for FFN/GDN/PLE/QSA/head; exhaustive BF16 bit widening and exact F32
binding parity; rejection of distinct handles for one physical device.
QSA includes the high-score spare-key/selection regression. This one-device
software setup does not qualify two physical dies or real-model placement.

`python3 -m unittest discover -s tools/validation-harness -p
 test_portable_model_smoke_v1.py -v`: all 16 tests passed. The new test
checks that only the oracle flag changes child argv after normalizing output
destinations, and both modes still reject failed host admission before Popen.
Python compile/help checks and `git diff --check` passed.

The v2 recipe's shell blocks pass `bash -n`; its embedded Python compiles.
Synthetic comparison controls passed: identical 41-row data accepted;
one-byte difference, source_invalidations and native-batch mismatch rejected.
Those controls are fixture data only, not real-model evidence.

## Failed attempts and limits

The initial clean configure command found no `cmake` on PATH and did not
configure a build. Selecting the already installed CMake3.31.6 resolved it;
no tool was installed. Builds emit existing GGML deprecation warnings in
portable QSA code; there are no outstanding build/test failures. The v1
missing-PLE manual link failure remains recorded in v1 and was not hidden.

Not run by this coding task: physical V340, real model, clean CUDA/HIP/SYCL
builds or timing comparison. This task made no fleet access, deployment, model
modification or real-model inference launch. Parent/operator subsequently supplied
PASS for the exact ac85ff3 physical pair; see the attributed
[physical receipt](portable-cpu-expert-physical-results-v1.md), including all
41 byte-identical logit rows, unchanged native work and timing caveats.
The failed CUDA numerical screen remains failed. Local CTest duration is
validation cost, not model-performance evidence.

The matched parent-dispatched recipe is `portable-cpu-expert-proof-v2.md`.
It keeps immutable sources, first-use corruption rejection, lifetime-owned
mappings, bounded staging/host caps, all runtime guards and no retries.
Metadata checks do not provide adversarial/concurrent-mutation protection.
Native jobs read mapped bytes rather than the verified staging snapshot.
Tests mutate fixtures only between completed dispatches; they exercise ordinary
change detection and corruption rejection, not concurrent-write safety.
Parent supplied an independent narrow source GO at fb9b4904 with no concrete
P1/P2 finding; this is attributed review, not another execution result.

Local raw logs, binary hashes and checksums: task-9/strata/validation,
including cpu/vulkan-{configure,build,ctest}.log,
{ffn,gdn,ple,qsa,head,qsa_bf16,device_admission}-strict-vulkan.log,
supervisor-tests.log and evidence.sha256. They remain outside Git. The
committed receipts are summaries of those actual outputs, not fabricated
hardware reports.
