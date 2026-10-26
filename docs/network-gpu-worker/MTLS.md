# Persistent mutual TLS for native workers

Plain direct TCP has source-IP filtering but no encryption or certificate identity.
For encrypted native GPU worker connections, build with `STRATA_WORKER_TLS=ON`.
This uses installed OpenSSL 3.5 or newer; CMake does not install it. TLS 1.3 is the
minimum. The minimum library version follows the [OpenSSL support policy](https://openssl-library.org/policies/releasestrat/). Both sides require a trusted certificate chain and the configured exact
SAN DNS identity. Common-name fallback, wildcard names, domain-prefix names,
resumption and early data are disabled. No plaintext retry occurs after TLS failure.
Identity checks use [OpenSSL verification](https://docs.openssl.org/3.5/man3/SSL_set1_host/);
nonblocking I/O follows [OpenSSL retry rules](https://docs.openssl.org/3.5/man3/SSL_get_error/).

Native CPU client/service/coordinator transport and GPU transport reuse the same
`TlsConfig`/`TlsSession` implementation. CPU binary framing, raw/packed activations,
indexed-row reduction, cache and local resident attachment contracts are unchanged.
HIP/SYCL/Metal/Windows GPU network support is not added.

## Build and configure

Add `-DSTRATA_WORKER_TLS=ON` to the native worker build. The build option remains
OFF for dependency compatibility; a TLS-disabled binary rejects TLS configuration
and refuses remote transport unless plaintext development mode is explicit.
New remote CPU and all GPU connections require mTLS by default. Purely local CPU
connections to `127.0.0.1` may retain plaintext resident attachment; this is local-only,
not inter-node encryption. Do not use loopback SSH forwards to imply secure product
inference transport. SSH is for administration; inter-node inference uses mTLS.

For each GPU worker, add these arguments to its existing plan/cache/listener args:

```text
--tls-ca /approved/test-ca.pem
--tls-cert /approved/worker-chain.pem
--tls-key /approved/worker-key.pem
--tls-peer coordinator.test.example
```

For the coordinator, add:

```text
--remote-gpu-tls-ca /approved/test-ca.pem
--remote-gpu-tls-cert /approved/coordinator-chain.pem
--remote-gpu-tls-key /approved/coordinator-key.pem
--remote-gpu-tls-peer worker-a.test.example
--remote-gpu-tls-peer worker-b.test.example
```

Supply one peer name per `--remote-gpu-worker HOST:PORT PLAN`, in the same order.
The legacy single endpoint uses one peer name. All endpoints use the same explicit
CA and coordinator certificate/key. These names identify certificate SANs; routing
still uses the existing numeric IPv4 endpoints, with no DNS discovery. Worker
`--tls-peer` identifies the intended coordinator. Existing bind/source-IP policy
still applies. Each peer must possess its own approved leaf certificate/key.

Any TLS configuration requires all four values. Empty CLI values are errors. The
key must be a readable PEM private key with permissions limited to its process
account; encrypted keys fail without an interactive password prompt. No system
trust store is read or modified by this feature. Files are loaded when connections
are created, so an existing session does not pick up changed trust or certificates.
No enrollment/renewal controller or online revocation checking is included.

CPU workers accept the same four TLS arguments. The coordinator uses
`--remote-cpu-tls-ca`, `--remote-cpu-tls-cert`, `--remote-cpu-tls-key` and one
`--remote-cpu-tls-peer` per CPU endpoint, in order. See
[certificate lifecycle and supervisor design](CERTIFICATE-LIFECYCLE.md) for identity
mapping, rotation, expiry diagnostics, removal and implemented versus planned work.

Accepted CPU and GPU sockets now enable TCP_NODELAY by default, after source-IP
admission; existing `--tcp-nodelay` remains accepted. Coordinator sockets already
enable it. This is the intended latency policy. The prior matched mTLS GPU pair at
PR #27's `935c2415` reported 8.15 to 9.05 TPS with exact tokens/first logits; that
supplied result does not establish CPU-worker performance. No new campaign is run.

A connection performs one handshake before SGW1 HELLO, then carries setup, uploads,
status and repeated inference frames under TLS. Frame bytes, numerical code, caches,
batching and thread ownership are unchanged. Handshake time uses the RPC deadline
and the client's remaining total setup budget. TLS reads/writes reuse the socket
layer's monotonic total budget, cancellation and poll helpers, including either
OpenSSL WANT_READ/WANT_WRITE direction. Partial progress never renews the budget.
EOF or failed/partial I/O ends the session. Cleanup does not wait for close_notify.
The socket BIO uses the existing MSG_NOSIGNAL policy without changing process
signal handling. Transport selection is shown as `mTLS` or `plaintext-test` in logs.

## Migration and deliberate development mode

Old remote launch configurations that omit TLS now fail with a configuration
error. Add explicit TLS paths and identities on both peers and build both with TLS
support. Old binaries do not understand these flags: upgrade both ends before
launching. A TLS/plaintext wire mismatch fails authentication/protocol I/O within
finite deadlines; it never selects a compatible but weaker mode automatically.
SEW1/SGW1 frame contents have not changed, so no protocol version bump is needed.

Disposable plaintext tests must use worker `--plaintext-test` and coordinator
`--remote-cpu-plaintext-test` or `--remote-gpu-plaintext-test` as appropriate.
The API equivalents are `plaintext_test=true` on worker/tier options and the CPU
client's explicit `connect_transport` call. TLS and plaintext-test flags are mutually
exclusive. Plaintext tests are insecure even on a private network. Merely omitting
TLS is not a remote development opt-in. Local-only CPU loopback use remains available.
No running service or deployed configuration is changed by this source migration.

## Local checks

[CPU/core validation and build handoff](../validation/CPU-WORKER-CORE-MTLS-20261026.md)
contains the current CPU/GPU bounded suite and migration limits. The original GPU
commands below remain useful for that subset.

With a local checkout of the repository-pinned llama.cpp revision
`3cf03257f219afbe7334045ff7c6a06ac68c627d` in `$GGML_ROOT`:

```sh
cmake -S . -B build-mtls -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_WORKER_TLS=ON -DSTRATA_GGML_DIR="$GGML_ROOT"
cmake --build build-mtls --target tls_transport_test gpu_worker_test \
  gpu_worker_group_test gpu_frame_test -j 4
ctest --test-dir build-mtls \
  -R '^(tls_transport_test|gpu_worker_test|gpu_worker_group_test|gpu_frame_test)$' \
  --output-on-failure
```

The tests generate disposable local certificate fixtures in a private `/tmp`
directory, remove them on normal completion and never print private material.
They do not contact fleet hosts. `tls_transport_test` covers both mutual identities,
untrusted/expired peers in both directions, missing client certificates, CN-only
certificates, TLS 1.2 rejection, partial reads/writes, persistent reuse, deadlines,
cancellation and disconnects. `gpu_worker_test` exercises the actual GPU protocol,
client, worker, cache and session logic with a fake executor; it is not CUDA hardware
validation. It also rejects plaintext/TLS mismatches in both directions. Repeat the
three existing GPU tests in a separate `STRATA_WORKER_TLS=OFF` build to check the
compatibility path and refusal of TLS configuration by a TLS-disabled binary.

On Linux, `tls_io_diagnostic` provides a bounded, model-free transport comparison:

```sh
cmake --build build-mtls --target tls_io_diagnostic -j 4
build-mtls/tls_io_diagnostic --rounds 16 > /approved/local-output.csv
```

The executable uses actual SGW1 framing with pre-encoded, codec-validated synthetic
inference/entry-row payloads for `(T,K)` shapes `(1,1)`, `(1,2)`, `(1,8)` and `(8,10)`.
It compares plaintext/TLS with accepted-worker TCP_NODELAY off/on in two trials.
The coordinator-side NODELAY policy stays on. Each case uses one persistent
connection, excludes handshake and two warmups, and checks payload bytes exactly.
The fixture does not run GPU dispatch, model arithmetic or cache work.

Linux linker wrappers collect per-side socket send/receive calls, successful size
ranges/byte counts, EAGAIN counts, poll calls and time inside poll. They preserve
errno and exist only in the diagnostic executable; production I/O has no new
instrumentation. TLS still uses the normal mutual chain/exact-SAN checks and the
existing disposable local fixture. A short correctness run is registered with
CTest, with no performance threshold. Counts/timings are instrumented loopback
observations and cannot establish LAN throughput or hardware TPS recovery. Keep
raw results local until their publication is authorized.

## Future matched benchmark and permissions

No hardware/model run or fleet certificate provisioning has been performed for this
slice. After backend compilation and review, compare plaintext / mTLS / plaintext
on the same 1 GbE topology and the same TLS-enabled worker/coordinator binaries.
Keep model, plans, warm caches, CPU residency, thread settings, batching, deadlines,
prompt0 GEN256 and `STRATA_CUDA_Q8_CPU_REFERENCE=1` fixed. Use distinct per-arm logs,
record binary hashes and endpoint/peer mappings, and keep TLS handshakes/setup out
of steady-state timing while reporting their costs separately. Compare every output
token and first-logit byte before interpreting prompt/decode/wall timing. Stop on a
mismatch or any authentication error. Do not claim encryption cost from the prior
SSH/direct result; that comparison measured the full forwarding configuration.

Before that run, obtain explicit approval for all of the following:

- The named coordinator and GPU-worker hosts/accounts, certificate SAN identities,
  certificate lifetimes and account-owned file paths.
- Creating per-process test private keys on those hosts, signing clientAuth and
  serverAuth leaf certificates with an approved test CA, and distributing only the
  required public CA/certificate chains. Name the signing host and who controls
  its CA key. Do not copy private keys into logs, bundles or the repository.
- Installing the approved test files with account-limited key permissions, and
  launching temporary worker/coordinator processes with the explicit TLS paths.
- The bounded model/benchmark run and cleanup of only its owned temporary processes
  and test credentials afterward.

No approval to change machine-wide trust, firewall rules, SSH, services, deployment
or fleet defaults is implied. If those changes prove necessary, stop and request
that exact additional permission separately.
