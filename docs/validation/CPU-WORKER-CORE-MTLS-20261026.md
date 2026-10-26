# CPU worker core mTLS transport: validation and build handoff

Stacked on PR #27, verified remote head
`935c2415d23b8ff2dd6fb9fbd7a81f00bd4f926a`, issue #26. This branch adds CPU-worker
transport to the existing shared TLS seam. It makes mTLS the remote product policy,
requires explicit insecure development mode for plaintext, and enables accepted
CPU/GPU NODELAY. Local `127.0.0.1` CPU resident use remains local-only. Existing
protocol bytes, codecs, arithmetic, packed eligibility and caches are preserved.
This is a source/configuration migration; running services are not reconfigured.

## Local evidence

Linux x86-64, GCC 16.2.1, Release, OpenSSL 3.6.5. Native experts ON; CUDA/HIP/SYCL
OFF. Reused already-installed dependencies and local ggml sources at the existing
repository pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Models: zero. TLS material:
only the existing auto-deleting `TlsFixture` in local temporary directories.

Passed:

- TLS-enabled host builds: `strata-expert-worker`, `cpu_tls_test`,
  `cpu_packed_wire_test`, `worker_local_test`, `worker_residency_test`,
  `worker_lan_test`, `worker_socket_test`, `cpu_worker_group_test`,
  `gpu_worker_test`, `gpu_worker_group_test`, `tls_transport_test`, `gpu_frame_test`.
- 12/12 focused CTests in 43.58 seconds. The same CPU TLS test was then extended
  with cold chunked upload; its focused rerun passed (1/1). No production source
  changed after the 12-test run.
- CPU mTLS: real native services/client and two-member coordinator group, cold
  chunked upload, local source load/READY, repeated inference with one session,
  exact native output bits, original indexed-row reduction, cache status, verified
  resident reattachment, wrong SAN on both peers, untrusted/expired/missing
  certificate configuration, CN-only leaf, plaintext mismatch both directions,
  partial encrypted header deadline, valid reconnect after refusal, and certificate
  file replacement observed at the next handshake. Existing sessions retain their
  authenticated TLS state during replacement.
- Existing packed/raw wire suite runs both plaintext local and mTLS modes. It
  checks exact raw encoding, packed useful bytes, raw/packed native result parity,
  negotiation, malformed/stale/replayed/unowned requests and cancellation.
- Shared TLS regression retains actual missing-client-certificate handshake,
  expired/untrusted mutual identities, TLS 1.2 refusal, fragmented 512 KiB exchange,
  finite handshake/read/write budgets, stalled writes, trickle reads, cancellation
  and disconnect failure. GPU fake-compute regression retains persistent session,
  upload/cache reuse, identity failure and plaintext mismatch cases.
- Separate TLS-disabled host build of CPU packed/local/group and GPU worker/group
  targets; 5/5 focused CTests in 22.33 seconds. Explicit CPU TLS configuration fails
  before connecting with a clear `STRATA_WORKER_TLS=ON` requirement.
- Five actual CPU worker CLI refusal checks: incomplete TLS paths, empty value,
  missing value, invalid peer identity and old remote plaintext configuration.
  TLS-disabled CPU worker CLI also reports the required build option.
- Host C++20 syntax checks of GPU worker main and coordinator main with native and
  network CUDA paths enabled, existing CUDA headers and the normal version define.
  These are not CUDA/HIP/SYCL link builds or device execution.
- `git diff --check`.

Initial failures, corrected:

- Sandbox socket creation was unavailable; the authorized disposable loopback
  rerun passed. This was not a transport assertion failure.
- Initial development build attempts found a missing TLS header, test target
  registration order and a test helper name collision; corrected before tests.
- The first host coordinator syntax command omitted the existing CUDA compiler
  header include directory; the command with both runtime/compiler headers passed.

Not run: full CUDA/HIP/SYCL/Metal/Windows builds, full repository suite, real
cross-host transport, model/device numerical or performance tests, actual
operator enrollment/renewal, active removal/revocation and fleet rollout.
No package installs, persistent credentials, system trust/access edits, fleet
service changes, hardware campaign, merge or deployment were performed.

## Reproduce the bounded local checks

Use existing local dependencies only. Set `GGML_SOURCE` to an existing checkout at
the repository pin. Do not let this command fetch packages or sources on a fleet
host without separate authorization.

```sh
cmake -S . -B build-mtls -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_WORKER_TLS=ON -DSTRATA_GGML_DIR="$GGML_SOURCE"
cmake --build build-mtls --target strata-expert-worker cpu_tls_test \
  cpu_packed_wire_test worker_local_test worker_residency_test worker_lan_test \
  worker_socket_test cpu_worker_group_test gpu_worker_test gpu_worker_group_test \
  tls_transport_test gpu_frame_test -j 6
ctest --test-dir build-mtls --output-on-failure \
  -R '^(cpu_tls_test|cpu_packed_wire_test(_mtls)?|worker_local_test|worker_residency_test|worker_lan_test|worker_socket_test|cpu_worker_group_test|gpu_worker_test|gpu_worker_group_test|tls_transport_test|gpu_frame_test)$'
```

Configure a separate `build-plain` with identical options and `STRATA_WORKER_TLS=OFF`.
Build `cpu_packed_wire_test worker_local_test cpu_worker_group_test gpu_worker_test
gpu_worker_group_test strata-expert-worker`. Run:

```sh
ctest --test-dir build-plain --output-on-failure \
  -R '^(cpu_packed_wire_test|worker_local_test|cpu_worker_group_test|gpu_worker_test|gpu_worker_group_test)$'
```

## Build-only handoff

Pin this draft PR's exact head; base is #27's head above. Use a fresh owned build
folder and already-installed toolchains/dependencies. Preserve AMD integration
PR #31, other branches, all services and resident weights. Keep the draft state.

For the existing CUDA build configuration, enable `STRATA_WORKER_TLS=ON` and build
`strata-expert-worker`, `strata-cuda-expert-worker` and `strata`, plus `cpu_tls_test`,
`cpu_packed_wire_test`, `tls_transport_test`, `gpu_worker_test` and
`gpu_worker_group_test` where supported. Preserve existing model/kernel feature
options; do not retune anything. Report exact source hash, compiler/OpenSSL
versions, command, return code and linked SSL/crypto libraries. Host-only tests
above are permitted in disposable loopback fixtures; do not run GPU/model test
variants, launch workers against fleet endpoints or replace resident binaries.
If an installed dependency is absent, stop that build and report the exact missing
component. Do not install packages, create operator certificates or alter trust.

HIP/SYCL/Metal/Windows compilation remains targeted follow-up validation where
available; it must not disturb the separate AMD integration. Parent/user approval
handles any later fleet credentials, runtime qualification or deployment.

[Configuration and migration](../network-gpu-worker/MTLS.md) explain explicit
plaintext development flags and old-binary incompatibility. The
[certificate lifecycle specification](../network-gpu-worker/CERTIFICATE-LIFECYCLE.md)
separates today's path-based transport behavior from planned trusted supervisor
enrollment, node-held keys/CSR exchange, short-lived issuance and renewal. No
lifecycle controller or automatic session-expiry enforcement is claimed.
