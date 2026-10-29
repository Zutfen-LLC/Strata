# Optional shared expert and CPU scheduling

`portable_model --overlap-shared-cpu` splits the FFN into route-ready, shared,
and merge graphs. The default keeps the original serial graph. This implements
only the CPU/shared scheduling slice of #68; it adds no grouped GPU experts,
parallel GPU dispatch, prompt batching or changed arithmetic.

The route-ready graph computes HC mixing, injection and router logits once.
Its mixed F32 output stays allocated in its retained scheduler. The shared graph
uses a separate `OP_NONE` tensor bound to that allocation with
`ggml_backend_tensor_alloc`. It has no view or source edges into the producer
graph. Both graphs and the borrowed backend remain alive through completion.
The shared scheduler is submitted with `ggml_backend_sched_graph_compute_async`
after route readback. The caller runs the existing synchronous CPU expert pool,
then the existing resident GPU requests serially. Ownership masks are determined
before any dispatch. Empty ownership sets are skipped. Rows keep their original
slots, remain unweighted and are accumulated in ascending slot order as before.
The coordinator explicitly joins shared work before readback and merge. Errors
join outstanding graphs before returning; reset and destruction also drain work.
The residual and route are published only after the complete merge succeeds.

The extra scheduler is included in the same aggregate graph cap. Its actual
backend allocations are included in memory admission, along with an extra 2 MiB
metadata arena and CPU work bound. The alias owns no additional activation
allocation; it borrows the producer scheduler's retained output. Caps are not
raised by this option.

## Execution limits

Inspection of pinned ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d` confirms that
the asynchronous scheduler submits backend work without its synchronous wrapper
wait. Vulkan graph execution submits command buffers and normally returns before
completion. This permits overlap with CPU work; it does not prove overlap on a
particular device or configuration. Input transfers, backend fallback,
`GGML_VK_DISABLE_ASYNC`, `GGML_VK_SERIALIZE_SUBMISSIONS`, Vulkan result checking
and performance logging can serialize execution. No speed improvement is claimed.
The harness records the presence of the relevant runtime environment switches
without changing them. Record the build's `GGML_VULKAN_CHECK_RESULTS` and other
Vulkan compile settings separately from `CMakeCache.txt`.

## Fixture-only qualification

Build with the existing pinned toolchain and dependency configuration. Build
`portable_model`, `portable_ffn_test`, `portable_local_experts_test` and
`portable_ffn_vulkan_test`. Run the CPU tests and affected portable Python
harness tests after the build completes. The scheduling mock tests submission,
join, failure cleanup and eight matrix products; it is not a hardware timeline.

For each chosen Vulkan device, run the existing bounded fixture command:

```sh
portable_ffn_vulkan_test --device INDEX --expect-device DESCRIPTION
```

The fixture internally crosses overlap OFF/ON and dense residency OFF/ON. Require
exit zero and four final FFN passed summaries. Each combination checks eight
matrix products, F32/Q8 parity, route IDs and weights, reset, callback rollback,
strict placement and graph budget rejection. Retain binary hashes, exact device
identity, build settings, runtime environment presence and resource/cleanup
receipts. Do not change environment settings to force a favorable outcome.
This fixture does not run a real model or establish performance. A later physical
CPU/GPU timeline must demonstrate concurrent shared compute and cold CPU expert
work before any performance claim or default change.

Fresh and resident harnesses accept the same option and bind it in their
receipts and reference argv. Reference outputs from a different overlap policy
are rejected. CPU/shared overlap can make component timers overlap, so compare
request wall time and exposed latency rather than summing component timers.
