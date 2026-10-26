# Persistent mutual TLS (opt-in development slice)

Plain direct TCP has source-IP filtering but no encryption or certificate identity.
For encrypted native GPU worker connections, build with `STRATA_WORKER_TLS=ON`.
This uses installed OpenSSL 3.5 or newer; CMake does not install it. TLS 1.3 is the
minimum. The minimum library version follows the [OpenSSL support policy](https://openssl-library.org/policies/releasestrat/). Both sides require a trusted certificate chain and the configured exact
SAN DNS identity. Common-name fallback, wildcard names, domain-prefix names,
resumption and early data are disabled. No plaintext retry occurs after TLS failure.
Identity checks use [OpenSSL verification](https://docs.openssl.org/3.5/man3/SSL_set1_host/);
nonblocking I/O follows [OpenSSL retry rules](https://docs.openssl.org/3.5/man3/SSL_get_error/).

This slice covers the POSIX native CUDA coordinator/GPU-worker path, including
parallel GPU groups. CPU-worker connections remain unchanged. The reusable
`TlsConfig`/`TlsSession` interface lives beside the shared socket helpers; a future
CPU-worker integration should pass it through the CPU frame reader/writer rather
than add another TLS implementation. HIP, SYCL, Metal and Windows network GPU
support is not added.

## Build and configure

Add `-DSTRATA_WORKER_TLS=ON` to the existing native CUDA build. The option defaults
to OFF so existing binaries and fleet launch configuration keep their behavior.
A binary built without TLS rejects a nonempty TLS configuration.

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
No enrollment, reload/rotation process or online revocation checking is included.

A connection performs one handshake before SGW1 HELLO, then carries setup, uploads,
status and repeated inference frames under TLS. Frame bytes, numerical code, caches,
batching and thread ownership are unchanged. Handshake time uses the RPC deadline
and the client's remaining total setup budget. TLS reads/writes reuse the socket
layer's monotonic total budget, cancellation and poll helpers, including either
OpenSSL WANT_READ/WANT_WRITE direction. Partial progress never renews the budget.
EOF or failed/partial I/O ends the session. Cleanup does not wait for close_notify.
The socket BIO uses the existing MSG_NOSIGNAL policy without changing process
signal handling. Transport selection is shown as `mTLS` or `plaintext-test` in logs.

Omitting all TLS options retains plaintext solely for compatibility and explicitly
selected local/matched tests. It is not the recommended production configuration.
Do not expose workers on an untrusted network or change deployed defaults based on
this draft. This does not encrypt CPU-worker traffic, which still needs its existing
trusted transport.

## Local checks

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
