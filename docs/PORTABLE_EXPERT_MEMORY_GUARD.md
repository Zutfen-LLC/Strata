# Resident expert memory-guard diagnostics

This is a source-level ownership analysis and diagnostic addition, not a change
to native behavior, expert placement, memory policy or allocation lifetimes.
It does not identify the allocation responsible for a hardware refusal.

## Snapshot order and attribution

`PortableModelSession::Impl` in `src/core/portable_model.cpp` builds the dense
layers, shared GDN/QSA resources and output head before constructing
`LocalExperts`. Expert budgets are admitted after those ledgers are final.
The session validates unique physical accelerator devices before construction.

`LocalExperts` first validates every plan without making device allocations.
It then prepares executors sequentially. Each device's `free_before` is sampled
immediately before that device's executor factory. The next device takes its
own snapshot after the previous executor finishes. The current Vulkan factory
borrows the selected backend; it does not create another Vulkan context.

During preparation, the reader guard runs before each asset is loaded, and a
final guard runs after preparation. `VulkanExecutor::prepare_impl` increments
`gpu_allocated_bytes` after each completed `ExpertGraph` allocation. Thus the
next reader sees completed tensor charges. Source loading and digest checks use
host memory. The model's reset occurs after all expert executors are prepared.
This ordering does not establish a bookkeeping defect that includes earlier
dense construction in a preparation-time growth delta.

The pinned Vulkan free-memory query reads device heap budgets/usages, rather
than allocation-owner counters. Allocations made after the snapshot by the same
backend or other session work can affect the estimate. Later inference guards
also include any post-snapshot dense runtime growth. The delta cannot attribute
an excess to expert tensors, backend/driver resources or unrelated work; moving
the baseline or granting extra allowance would change policy without evidence.
The no-memory-budget-extension fallback reports heap size instead, another
reason to avoid treating this estimate as measured ownership or proof of OOM.

## Native allocation lifetimes

Native `SessionState` in `include/strata/core/session.hpp` and `session_init` in
`src/core/session.cpp` own the shared GDN, QSA, MoE and block scratch arenas plus
separate layer state. Expert weights have separate persistent cache ownership:
`ExpertCache::open`/`open_sized` in `src/core/expert_cache.cpp` allocate cache
storage; `RemoteExperts::open`/`close` in `src/core/remote_experts.cpp` own the
helper's cache and fixed input/output/metadata/scratch resources.

Portable `ExpertGraph` in `src/worker/vulkan_expert_graph.hpp` instead retains
weights, input/intermediate/output tensors and metadata per expert. Its actual
backend tensor-buffer sizes are charged by `VulkanExecutor`; backend/driver
allocations are outside that tensor counter. This existing allocation-lifetime
parity gap is separate work. No native-backed lifetime correction is established
by the guard operands alone, and this diagnostic change does not attempt one.

## Failure fields

Both existing failure prefixes are retained. On failure only, the message adds
accelerator slot, backend/device name and description, device ID (or `unknown`),
phase, zero-based preparation entry and layer/expert IDs, completed/assigned
entry counts, `free_before`, `free_now`, total, charged tensor bytes, tensor cap,
workspace cap, growth, allowance, excess, reserve and reserve shortfall. All
memory operands are bytes. Outside an active preparation read the entry and
layer/expert fields are `none`; `phase` identifies completion or inference.
Backend identity queries are diagnostic plumbing only. Reported guard operands
come from the original free-memory sample, not the identity query.

The predicates remain `free <= total && free >= free_reserve` and
`max(0, free_before - free_now) <= charged_tensor_bytes + workspace_cap`.
There is no baseline reset, allocation credit, allowance, retry or new CLI.
Host tests cover distinct per-device baselines, the exact growth boundary and
one-byte excess during preparation on both slots, reserve refusal, zero-clamped
growth, cleanup and the existing indexed-row/executor lifecycle regressions.
