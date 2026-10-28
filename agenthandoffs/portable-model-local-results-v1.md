# Portable model local verification — v1

Scope: opt-in serial local Strata product path, stacked on corrected PR41
`1d8eae1c3906962932b77e366f324c51aa89819d`. No fleet execution took place.
Operational stages: `portable-model-hermes-v1.md`. Main interface and limits:
`docs/PORTABLE_MODEL.md`. All portable flags remain OFF by default.

## Passed locally

- CPU-only build of portable_model, portable_head_test and portable_model_test.
  Seven targeted CPU CTests passed in35.28s after adding layer-tagged reports
  and actual H2560/HC4/rank320 head fixtures.
- Sparse six-shard synthetic model runs all48 layers, layer1 host PLE,
  genuine nonzero Q8 CPU experts, full synthetic vocabulary248064, independent
  scalar logits, fresh-token HC, a layer10 selected-expert digest failure,
  poison/no mixed-state retry/reset/replay, EOS and bounded greedy/context
  stops. Its unique mapped file sizes total188194264224B; anonymous copied
  weights4773486080B; bounded expert staging52224000B.
- Fresh Vulkan build with pinned ggml. Strict software-Vulkan head execution
  at actual H2560/HC4/rank320,64 vocabulary rows, independent scalar checks,
  no scheduled CPU fallback. Final head log reports actual device scheduler
  reservation3604480B and conservative private workspace bound13275136B.
- Existing strict Vulkan FFN/GDN/PLE/QSA tests passed sequentially on llvmpipe
  (LLVM23.1.1,256bits). QSA includes width2051/tie cases and exact corrected
  high-score first-key spare regression at cached_cells2055.
- Final rebuilt head/model plus GGUF reader/split/geometry CTests passed
  (five tests,21.88s), after the largest-head-matrix workspace correction.
- Four report-admission tests passed: intentional host route/merge is allowed;
  hidden CPU tensor fallback, head on the wrong die, and QSA projection
  without attention execution are rejected. These tests validate reporting,
  not physical device execution.
- Default-OFF configure and target listing contain no portable model/head
  targets. CUDA/HIP/SYCL options remain OFF and their source paths unchanged.
- Supervisor --help and Python compilation, git diff --check.

The Vulkan-linked seven-CTest CPU suite passed in35.79s, recorded in
`${BUILD_ARTIFACT_74001DCE}`. The final new head workspace
bound was additionally exercised by strict Vulkan head execution. The model
CLI is linked with Vulkan but the whole assembled model has not run on
Vulkan locally.

## Exact local build context

Source: task-8/strata, branch codex/portable-model-session.
CMake: `/tmp/cmake-3.31.6-linux-x86_64/bin/cmake`.
Pinned dependency: `/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d`.
CPU build: `${BUILD_ARTIFACT_83FD5685}`.
Vulkan build: `${BUILD_ARTIFACT_4E1E2E30}`.
Default build: `${BUILD_ARTIFACT_3C594F8C}`.

Vulkan configure used Release, all portable flags and tests ON,
STRATA_NATIVE_EXPERTS=ON, GGML_VULKAN=ON,
FETCHCONTENT_FULLY_DISCONNECTED=ON,
CMAKE_PREFIX_PATH=${BUILD_ARTIFACT_D2E5D17A}, and
CMAKE_CXX_FLAGS=-I${BUILD_ARTIFACT_36492828}

The first configure/build omitted that include flag and failed compiling
spirv/unified1/spirv.hpp; it was corrected using the already-existing header
prefix from prior builds, with no install. An initial device test without
an explicit ICD found no device; the accepted existing local software ICD
selection below exposed llvmpipe. No physical GPU access was claimed.

Strict local device environment:
```
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json
GGML_VK_VISIBLE_DEVICES=0
```
Executed with that environment, sequentially:
```
timeout 90s ${BUILD_ARTIFACT_8915F53B} --vulkan 0
timeout 120s ${BUILD_ARTIFACT_855E6BF1} --device 0 --expect-device llvmpipe
timeout 120s ${BUILD_ARTIFACT_81477536} --device 0 --expect-device llvmpipe
timeout 180s ${BUILD_ARTIFACT_5DB22BD4} --device 0 --expect-device llvmpipe
timeout 180s ${BUILD_ARTIFACT_C2C4F360} --device 0 --expect-device llvmpipe
```

Logs:
- `${BUILD_ARTIFACT_4A4CD819}`
- `${BUILD_ARTIFACT_99B50F4D}`
- `${BUILD_ARTIFACT_C958B8DB}`
- `${BUILD_ARTIFACT_1096FE3E}`
- `${BUILD_ARTIFACT_23BBD0B1}`
- `${BUILD_ARTIFACT_E8F5126A}`
- `${BUILD_ARTIFACT_4F6FF6D3},gdn,ple,qsa}-vulkan.log`
- `${BUILD_ARTIFACT_74001DCE}`
- `${BUILD_ARTIFACT_E6536328},ctest}.log`
- `${BUILD_ARTIFACT_6A7C3BD7}`
- `${BUILD_ARTIFACT_262A9BCC},targets}.log`

## Pending and limits

No first real-weight GGUF binding/run or physical simultaneous dual-V340
execution has been performed by this task. Parent/operator owns stage dispatch
and numerical interpretation. The final recipe uses the retained38-token
prompt, context64, max-new3, explicit contiguous32/16 layer placement, head on
final die, host PLE and in-process CPU experts. It preserves full-vocabulary
logits and IDs and records actual scheduler assignments, per-backend buffer
reservations, Vulkan memory snapshots and sampled aggregate per-PCI VRAM.
It does not pool two8GiB dies or label budget numbers as measured VRAM peaks.

Driver pipelines/command pools and unrelated users are outside the owned
allocation ledger. The supervisor records baseline/observed growth and uses
separate die caps with measured free headroom. Sampled aggregate peaks can
miss brief transients; owned scratch/workspace bounds remain explicit.

The SWLOCAL1 identity is retained expert evidence, not a full-model hash
manifest. Selected expert blobs are checked while staging; no175GiB setup
hashing/weight scan. There are no GPU-resident expert optimization claims.
Recurrent state and residuals still use host/device transfers. Synthetic
screening tolerances were not changed to fit candidate results. Real logits
use the existing compare_logits.py fixed metrics/limits; matching a prefix is
not a full-model quality verdict. Raw weights/logits stay off Git.
