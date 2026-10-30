# Portable expert cache ownership

The Vulkan expert executor packs persistent Q8 weights into stable slots and
owns one reusable execution graph and scratch buffer per executor/device.
The graph selects a slot in the frozen plan's order. It preserves original
T*K entry indices, native host Q8 activation boundaries, host SwiGLU, GPU
matvec ordering and synchronization. There is no placement or numerical switch.
Standalone per-expert graphs remain in offline tests as the old-path reference.

## Native mapping

- `ExpertCache::open_sized` in `src/core/expert_cache.cpp` packs aligned slots
  into one logical device allocation. `src/worker/vulkan_expert_cache.hpp` uses
  GGML's context allocation helpers to pack whole weight slots into blocks
  bounded by the backend's maximum buffer size. The fixed Q8 projection sizes
  preserve 256-byte slot alignment, with backend alignment/padding included.
  This block plumbing is required by GGML/Vulkan buffer limits; it is not a new
  placement policy or allocator framework.
- Native session scratch in `src/core/session.cpp` and helper execution resources
  in `RemoteExperts::open` in `src/core/remote_experts.cpp` belong to the session
  or helper, rather than individual experts. Portable keeps execution metadata,
  activation quantization vectors and the five execution tensors once per
  executor. Per-slot GGML weight descriptors remain necessary for backend
  tensor binding. Returned rows own their values; scratch references do not
  escape. One serial caller owns prepare/infer/release, as before.
- Native slot selection and residency remain the source of the existing plans.
  This change keeps exact IDs, disjoint ownership, source/digest verification,
  model/schema checks and original entry order. It does not regenerate plans.

Grouped/asynchronous dispatch and CPU/helper overlap remain separate parity
gaps. Portable still executes individual entries serially. This change does
not port native grouped kernels or alter numerical ordering to imitate them.

## Accounting and failure behavior

Before any owned backend tensor allocation, checked size arithmetic and GGML's
dry-run calculation admit the sum of aligned persistent weight blocks and
once-owned execution scratch. Each actual allocation must match its estimate.
The scratch owner and every weight block respect the backend's buffer limit.
The same host blob vector stages successive uploads. There is no buffer per
expert and no scheduler/backend per expert.

`gpu_weight_bytes` remains uploaded payload bytes. `gpu_scratch_bytes` is the
single execution buffer. `gpu_allocated_bytes - gpu_scratch_bytes` is aligned
persistent weight storage; padding is not mislabeled as transient scratch.
`gpu_allocation_count` counts weight blocks plus the execution buffer, not
prepared experts. The failure-only completed-entry diagnostic consequently
uses preparation progress. Guard predicates, sampling baselines, caps,
workspace allowances and reserves are unchanged. The full packed allocation
is charged before the second reader guard and the final completion guard.

Partial allocation/preparation failures release owned buffers. Slots cannot
be computed before upload or uploaded again while resident. An upload exception
poisons partial preparation until release/reprepare. Compute failure synchronizes
pending work before propagation; the existing model session owns poisoning and
explicit reset/replay. Execution scratch has no recurrent state, and the next
complete execution overwrites it, preserving the executor retry contract. GGML tensor upload
returns no status, so process-fatal backend/driver errors are not converted into
recoverable upload errors. Tests inject a recoverable exception at that seam.
Borrowed backends outlive the executor; no storage is shared across executors.

These tensor counters exclude backend compute/upload workspace, shader and
driver allocations. Removing duplicated transient tensors does not attribute
an earlier device heap-growth excess or establish complete-model admission.
No real-model retry or memory-policy adjustment is part of this change.

## Focused component check

Build `portable_expert_cache_test` in a CPU or Vulkan configuration with the
unchanged GGML pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
With no arguments it uses the CPU backend and an instrumented real allocation
seam. The Vulkan build also supports this synthetic component-only check:

```sh
build-vulkan/portable_expert_cache_test --device INDEX --expect-device SUBSTRING
```

Replace both placeholders using a verified device identity. The check alternates
three distinct experts, signal/zero inputs and slot order, requiring byte equality
at gate, up, hidden and output against the previous same-backend implementation.
It does not load a model or claim cross-backend equality. CPU substitution alone
cannot establish GPU execution correctness.

CPU tests count actual backing allocations and bytes, including a forced
partial final block, backend padding, scratch reuse, exact-budget acceptance,
one-byte excess rejection before allocation, arithmetic overflow, bad/duplicate
IDs, allocation/upload/compute failures, poisoning, teardown and separate
backend/session storage. Existing local-expert T1..8 ordering, mixed ownership,
source validation and synthetic model/placement regressions remain applicable.
