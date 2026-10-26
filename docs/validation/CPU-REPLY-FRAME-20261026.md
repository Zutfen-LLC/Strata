# Reply frame copy reduction — issue #13

The duplicate copy remained at published base `66cd5d82cbb448cb727cd6caf522ac414f60e76c`.
Successful CPU worker replies encoded rows or weighted partials into `scratch.payload`, then copied
that payload into `send_buffer`. This patch encodes directly into `send_buffer`, reserves 16 bytes
for the header, and fills the header after encoding. The payload-only codecs remain available.

The worker still calls the same controlled contiguous send with the same deadline, cancellation,
partial-send handling and active-reply residency rule. Errors still send an empty payload with the
same status. No arithmetic, protocol, receive path or request framing changed. No socket count or
network-byte reduction is claimed. Other work and resident processes were untouched.

## Local measurement

Linux 7.2.9-2-cachyos, Intel Core Ultra 9 185H, GCC 16.2.1, Release CPU-only build.
The existing `cpu_row_codec_bench` fixture uses synthetic finite rows, H=2560, K=10, T=1/2,
64 warmups and 4096 measured iterations. No model, socket or expert execution is timed.
Baseline was measured before editing. Its isolated payload-to-frame copy cost was 2.674 us for
102,440 payload bytes and 4.193 us for 204,860 bytes. Weighted payload copies were 0.076/0.156 us.

The fixture now compares complete old encode-plus-frame and direct-frame paths in the same
executable. Three invocations were run. Median of each phase's per-invocation mean, in microseconds:

| Reply | T | Payload bytes | Old encode + frame | Direct frame |
| --- | ---: | ---: | ---: | ---: |
| Weighted | 1 | 10,256 | 0.228 | 0.108 |
| Weighted | 2 | 20,496 | 0.473 | 0.230 |
| Indexed rows | 1 | 102,440 | 11.763 | 10.422 |
| Indexed rows | 2 | 204,860 | 25.134 | 20.015 |

All individual phases and runs are in [the raw evidence directory](reply-frame-20261026).
These are warm local codec measurements, with visible run variability. Phases run sequentially;
this does not model cold cache, Ethernet, backpressure or distributed overlap. A code improvement
is not a demonstrated TPS gain. The plausible benefit is small; no fleet model run is requested.

## Copy and storage accounting

Source inspection shows one full payload-to-frame copy removed per successful reply. Encoding
still copies output float bits into the final frame. Indexed T=1/2 saves 102,440/204,860 copied
bytes per reply; weighted T=1/2 saves 10,256/20,496. The header and sent byte counts are unchanged.

The separate retained payload vector is removed. A fresh worker needs one fewer payload backing
allocation on first use; vector growth/capacity rounding depends on the allocator and prior traffic.
Warm buffers already reused their allocations: do not claim fewer warm heap allocations. The
fixture checks unchanged capacities after warmup in both paths. It retains both buffers for A/B,
so its total scratch capacity is not a measurement of production worker memory savings.

## Validation

Passed: Release build of `cpu_row_codec_bench`, `expert_worker_test`, `cpu_worker_group_test`,
`worker_socket_test`, and `worker_socket_operation_test`; all five selected CTest checks outside
the network-restricted sandbox. Frame parity covers independent little-endian reference bytes,
T=1/2 representative replies, sparse/nonzero masks, empty and maximum row replies, signed zero,
subnormals and finite boundaries, weighted permitted nonfinite bits, and refused row infinity/NaN,
invalid geometry and missing selected rows. Existing service and socket checks cover exact
transported output, refusal/session behavior, fragmented I/O, backpressure and cancellation.

Failed environment attempt: the sandbox denied socket creation. Four socket-dependent tests failed;
`expert_worker_test` subsequently segfaulted after failed setup. The same suite passed outside the
sandbox. Both logs are retained. No assertion failure was observed in the successful rerun.

Not run: CUDA/HIP/SYCL builds, other host architectures, fleet execution, live model or TPS tests,
and interaction with try-I/O-first. Changed worker sources belong to the CPU-only
`strata_expert_worker` target; no GPU backend files changed. The PR remains draft.

Reproduce from a source directory (an extracted archive is sufficient):

```sh
cmake -S . -B build-reply -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/existing/llama.cpp
cmake --build build-reply --target cpu_row_codec_bench expert_worker_test \
  cpu_worker_group_test worker_socket_test worker_socket_operation_test -j 6
ctest --test-dir build-reply --output-on-failure \
  -R '^(expert_worker_test|cpu_worker_group_test|worker_socket_test|worker_socket_operation_test|cpu_row_codec_bench_check)$'
build-reply/cpu_row_codec_bench
```

Local validation reused the already present llama.cpp snapshot
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` through `STRATA_GGML_DIR`, without downloading or
modifying dependencies. No fleet host or resident model was accessed.
