# Native worker mTLS local validation, 2026-10-26

Scope: issue #26, stacked on direct-TCP PR #24 at
`9e1cb200a4c7e7e19316b186272c18bb7b630f84`. This is a development-only transport
slice. No fleet credentials, host settings, packages, model runs or hardware
measurements were changed. PR #10, PR #24 and AMD PR #25 are separate work.

Environment: Linux, GCC 16.2.1, system OpenSSL 3.6.5, Release, native experts ON,
CUDA/HIP/SYCL OFF. Existing local llama.cpp sources matched the repository pin
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. No dependency installation was needed.

Passed:

- TLS-enabled host build of `tls_transport_test`, `gpu_worker_test`,
  `gpu_worker_group_test` and `gpu_frame_test`; 4/4 CTests passed in 10.76 seconds.
- Separate TLS-disabled host build of the three GPU tests; 3/3 CTests passed in
  10.00 seconds, including rejection of a nonempty TLS config without TLS support.
- TLS transport checks: exact mutual SAN identities, untrusted/expired certificates
  in both directions, absent client certificate, CN-only certificate, invalid
  domain-prefix/wildcard peer configuration, TLS 1.2 rejection, three 512 KiB
  request/reply exchanges after one handshake, fragmentation, disconnect failure,
  finite handshake/read/write budgets, cancellation, stalled writes and trickle
  reads that cannot renew the total I/O budget.
- GPU protocol checks with fake compute: actual client/service setup, cold upload,
  READY, status, two inference requests with one session, warm cache reconnect,
  identity failure latching and plaintext/TLS mismatch refusal in both directions.
- Actual worker CLI linked to a temporary no-GPU executor: five refusals for
  empty, incomplete or invalid TLS arguments. No GPU work was invoked.
- Worker-main C++17 syntax check and coordinator C++20 syntax check with native
  network worker paths enabled, using existing local CUDA headers. This is a host
  syntax check, not a CUDA link/build or model run.
- `git diff --check`.

Initial environment failures:

- CMake's automatic dependency fetch could not use configured SSH authentication.
  Configuration succeeded using the existing pinned local source instead.
- Unprivileged sandbox CTests could not create sockets. The authorized loopback
  rerun passed. No transport assertion failure was observed in that first run.
- The first coordinator syntax command used C++17 and omitted STRATA_VERSION;
  existing C++20 `ends_with` and version references rejected that command. The
  corrected C++20 command with the normal version definition passed.

Not run:

- Full CUDA worker/coordinator compilation or device execution; HIP/SYCL/Metal or
  Windows builds; the full repository test suite; cross-host mTLS; hardware/model
  numerical parity or timing. The PR must remain draft pending relevant backend
  checks. No encrypted throughput claim is made.
- Real CA/key provisioning, signing, distribution or deployment. Only auto-deleted
  local test PKI fixtures were generated. No fixture private material is committed.

See [configuration, test commands, limits, benchmark and permission plan](../network-gpu-worker/MTLS.md).
