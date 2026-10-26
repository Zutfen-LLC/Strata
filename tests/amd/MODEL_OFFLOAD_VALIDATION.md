# Model offload integration: local host validation

Environment: Linux x86-64, GCC 16.2.1, Release, retained ggml source
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, `STRATA_AMD_PROBE_VULKAN=OFF`,
`STRATA_WORKER_TLS=ON`, installed OpenSSL 3.6.5. No dependencies were downloaded.

Configure the standalone `tests/amd` project with `STRATA_GGML_DIR` pointing to
that retained source. Build `strata-vulkan-q8-expert`,
`strata-vulkan-expert-worker`, `amd_gpu_worker_regression`,
`amd_gpu_group_regression`, and `amd_tls_transport_regression`.

Passed:

- `ctest --test-dir ../build-amd-host --output-on-failure`: six tests. The Vulkan
  graph backend is explicitly substituted with CPU. All eight supported token
  counts return exact indexed same-token direct-hybrid rows over the existing
  mTLS service/client path. One prepare/two fetches, stable resident allocations
  and zero owned counters on release. Oversized windows and private plaintext
  listeners/clients fail. CUDA-contract host regressions remain passing.
- The existing probe's capture path: synthetic T2/T1 capture, trusted source/plan
  and staged content digests, exact returned rows, resident reuse, release and
  foreign/corrupt metadata refusals. These are synthetic checks, not real-model
  acceptance evidence.
- `python3 -m unittest discover -s tools/validation-harness/tests -p test_harness.py`:
  42 tests, including imported private-endpoint transport regressions, explicit
  Vulkan selection, duplicate physical endpoint refusal, and private Vulkan mTLS
  requirements. Existing tests emit ResourceWarnings but finish OK.
- Vulkan-enabled `c++ -std=c++17 -fsyntax-only` checks for the executor and probe
  against the retained ggml headers, without shader compilation or GPU execution.
- `git diff --check`.

The initial sandboxed socket runs failed at `socket()`; authorized reruns with
local sockets available passed. Two new configuration-test expectations were
corrected to use actual disposable TLS fixture files and reset the reply-mode
fixture before asserting refusal. No production gate was changed.

Not run: Vulkan hardware/shader build, real captured model rows, coordinator
CUDA build, HIP/SYCL builds, model prompt/decode, multi-host transport or resource
measurements. The PR stays draft. The only GPU/model execution instructions are
in [MODEL_OFFLOAD_HANDOFF.md](MODEL_OFFLOAD_HANDOFF.md), for later parent admission.
CPU bit parity remains unestablished; no numerical threshold was relaxed.
