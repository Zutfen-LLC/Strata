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

## Additive first-entry native arithmetic diagnostic

`--capture-native-check` adds only the first intersecting entry's native
reference and four stage metrics to the existing capture mode. The original
5139cd08 commit and its outputs remain separate and immutable.

Passed locally:

- Retained native host build: new `--native-isa-check` correctly exited nonzero
  with actual flags=23 (AVX2/FMA/F16C plus AVX-VNNI), despite cache VNNI=OFF. Its
  synthetic capture test and public CLI refusal checks passed; no scratch service
  was created by the refused native-check CLI.
- Separate CPU-only Release build using `STRATA_PORTABLE=ON`, GGML AVX-VNNI and
  AVX512 options OFF: ISA preflight exited 0 with flags=7, non-VNNI AVX2/FMA/F16C.
  GCC 16.2.1, the same retained ggml pin, installed OpenSSL 3.6.5, no downloads.
  The three graph/adapter/loopback CTests passed. The narrow capture test verifies
  exactly one record0/entry0/expert1 observation, four raw F32 stage lengths,
  bit-difference counts and numeric metrics recomputed from saved bytes, packed
  equalities, original capture completion and public CLI artifacts. This is a
  synthetic CPU graph substitution, not a Vulkan/native parity result.
- The narrow public-CLI capture CTest passed again in each ISA configuration
  after adding its final assertions. Vulkan-enabled probe syntax and whitespace
  checks passed. Existing unrelated compiler warnings remain.

Not run: new diagnostic on real expert302 or Vulkan hardware. No model, trace,
fleet, credential or access operations. Use
[NATIVE_ROW_HANDOFF.md](NATIVE_ROW_HANDOFF.md) once after parent dispatch. The
initial model trace and CPU-only control outcomes described there are supplied
parent results, not newly executed or independently verified here.
