# Optional CUDA entry-row transport

Issue #18, first bounded slice on draft PR #10. Base: `5b508f68c0969f07f018d97bfda3945637fb74d0`.

`GpuWorkerTier::Open::entry_rows=true` selects unweighted F32 rows. All peers in a
`GpuWorkerGroup` must use the same mode. After successful `finish`,
`entry_rows(peer)` exposes an `EntryRowsReply`: ascending original request-entry
indices and one unweighted row per selected entry. These are original T×K indices,
not peer-local or expert-group indices. The caller owns any later ordered assembly
and reduction. No production mixed CPU/GPU assembly is added here.

The CUDA executor reuses the rows it already downloads before its weighted loop.
It restores the original entry association without changing kernels, activation
quantization or the existing weighted arithmetic. The default remains weighted
partials; default HELLO, request and reply layouts stay unchanged.

Row mode requires HELLO capability `Cuda|EntryRows` and uses distinct
`GPU_INFER_ROWS` / `GPU_INFER_ROWS_ACK` operations. Unsupported peers or executors
fail before setup; there is no weighted fallback. The reply wraps the existing
entry-row codec in session/model/layout/placement and request-digest bindings.
Both service and client validate exact row count/indices, shape, sequence and
finite values. Only the new reply operation admits the bounded entry-row payload;
the weighted reply cap stays unchanged. Existing socket owners, copied requests,
deadlines, idle probes, failure latching and drain-before-commit remain in use.

The mixed CPU-group plus network-CUDA CLI guard is unchanged. CPU backing versus
execution masks, resident reuse bindings and one CPU/GPU ordered reduction are
follow-on work. This slice does not require a resident reload.

## Local validation

GNU C++ 16.2.1, CPU-only build, existing local ggml sources; no dependency download
or installation. Configure with `STRATA_BUILD_TESTS=ON`,
`STRATA_NATIVE_EXPERTS=ON`, `STRATA_ENABLE_CUDA=OFF` and `STRATA_GGML_DIR` pointing
to that existing source checkout.

```sh
cmake --build build-rows --target gpu_frame_test gpu_worker_test gpu_worker_group_test -j 4
ctest --test-dir build-rows --output-on-failure -R '^(gpu_frame_test|gpu_worker_test|gpu_worker_group_test)$'
```

All three tests passed. Coverage includes maximum-size byte-identical row transport
(including negative zero), original-index coverage, out-of-order peer completion,
independent ordered reference versus the existing CPU reducer, truncated/duplicate/
missing/wrong-index/stale/nonfinite/oversized replies, weighted-only negotiation
refusal, no invisible retry, group drain and allocation release. The existing
weighted fixtures pass alongside the row cases. The initial sandboxed test attempt
failed because socket creation was unavailable; the successful run used permitted
synthetic loopback sockets.

No CUDA compiler is installed locally. **The changed CUDA source and CUDA-only
fixture are uncompiled and unexecuted here.** The draft remains unqualified for
hardware use. No model, worker or fleet operation was performed.

## Targeted CUDA follow-up

Use an existing configured CUDA build with native experts and tests enabled and
its already available dependencies. Set `CUDA_BUILD` to that build directory;
preserve its existing architecture and compiler settings. These commands build
and run synthetic fixtures only, without model loads:

```sh
cmake --build "$CUDA_BUILD" --target strata-cuda-expert-worker gpu_frame_test gpu_worker_test gpu_worker_group_test gpu_worker_cuda_test gpu_worker_dispatch_test -j 4
ctest --test-dir "$CUDA_BUILD" --output-on-failure -R '^(gpu_frame_test|gpu_worker_test|gpu_worker_group_test|gpu_worker_cuda_test|gpu_worker_dispatch_test)$'
```

The extended CUDA fixture checks direct versus transported CUDA row bytes and an
independent ordered reduction of those rows against the existing CUDA weighted
result. These checks are pending. Wire exactness does not imply CPU/CUDA kernel
or activation-quantizer equality, full-model equality or a throughput gain.
