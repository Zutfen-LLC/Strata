# Local residency implementation checks

Scope: the opt-in local provider and direct-source/borrowed-backend Vulkan
executor additions in this commit, stacked on64634517. Local Valinor Linux
x86_64, GCC16.2.1, retained CMake3.31.6 and existing GGML/SPIR-V snapshots. No
fleet SSH, physical GPU inference, clocks, model mutation, new dependencies or
service changes. The source tree contains no generated model/cache payloads.

Passed:

- Root Release CPU and Vulkan builds of portable_model,
  portable_local_experts_test, portable_timing_cli_test,
  portable_cpu_experts_test and portable_model_test, two build jobs. Vulkan also
  builds strata-gpu-manifest. Uses the previously configured disconnected GGML
  snapshot at `/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d`, with
  `/tmp/strata-spirv-headers-prefix` for Vulkan. That local dependency snapshot
  lacks Git metadata; its retained path is provenance, not a new hash audit.
- Four selected CTests in each configuration:
  `ctest --test-dir validation/{cpu,vulkan} --output-on-failure -R '^portable_(local_experts|timing_cli|cpu_experts|model)_test$'`.
  CPU4/4,45.45 s; Vulkan-linked4/4,44.24 s. Heavy tests ran serially.
  After the final row-metric normalization/diagnostic precision edits, the
  local_experts and timing_cli targets were rebuilt and those two focused tests
  were rerun in each configuration; their raw receipts remain alongside these.
- New provider test uses two injected executors to exercise mixed hot/cold slots,
  original route order, actual offload counters, cross-device duplicate refusal,
  tensor/free-memory refusal, malformed shapes, duplicate/nonfinite replies,
  failure without CPU replay, source metadata changes and owned cleanup.
  It also uses the existing executor's explicit CPU graph substitution to load
  directly from mapped selected blobs, perform indexed same-backend repeat
  inference, reject shape errors, and free partial preparation after a selected
  digest mismatch. No disk GpuCache is constructed in this direct-load test.
- Existing CPU expert/model differentials retain byte-identical oracle rows and
  logits, poison/reset/context/EOS checks. Timed/default CLI stdout and every
  synthetic logit byte still match. No no-residency arithmetic was changed.
- Existing tests/amd standalone Vulkan-linked targets strata-vulkan-q8-expert
  and strata-vulkan-expert-worker build. amd_q8_expert_host and
  amd_vulkan_adapter_host passed, including resident indexed-order and lifecycle
  tests. amd_vulkan_loopback_host passed after running with local socket access.
- `python -m unittest discover -s tools/validation-harness -p 'test_portable*py'`:
  30/30 passed. New tests cover exactly16 total assignments per die, strict
  expert backend receipts and owned-child pressure cleanup with STOP retention.
- CLI, new smoke wrapper and quality adapter help checks; diff whitespace check.

Failed/blocked attempt: the existing loopback test initially failed at socket()
inside the restricted network sandbox. The same test passed0.40 s with the
required local socket permission. No socket/protocol code was changed to pass it.
An initial new-target build command ran before CMake regeneration and reported
no make rule; regeneration resolved it. Compilation also caught the request's
uint16 ID vector conversion, which was corrected before these passing builds.

Not run: physical Vulkan expert inference or row-error/answer qualification,
meaningful-VRAM placement/performance, CUDA/HIP/SYCL compilation. The common
GpuExecutor interface adds a default unsupported direct-loader method; CUDA
source and execution are unchanged, but those backend builds remain unverified.
Vulkan-linked host tests execute CPU substitution or fake executors and must not
be represented as physical GPU offload. Keep the PR draft pending independent
review and parent/Hermes hardware dispatch.

Raw configure/build/test/help receipts and local build trees remain untracked in
`validation/`. The complete-source recipe is
[portable-local-residency-hermes-v1.md](portable-local-residency-hermes-v1.md).
No merge, deployment, unrelated cleanup or fleet launch occurred here.

Python-only receipt follow-up: smoke summaries now retain compact resource
baselines/last snapshots, sampled VRAM/host/process/swap extrema and final reaped
child exit_code on STOP as well as success. The focused pressure-failure test
checks that the failing sample and -15 reaped result survive. All30 Python tests
pass after the change. No C++ source changed or rebuild was required.
