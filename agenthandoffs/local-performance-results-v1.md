# Physical local CUDA / Vulkan timing result

Completed on October 28, 2026 by Hermes, not by this coding task. Primary
[Hermes result and cleanup report](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793161012072129).
Code source: ef2434a5ebad85f82139e5c13d0aeaf32d700a23. Recipe corrections:
909641bf0585c8c3aee89b4c5f05306040184f04. Candidate portable binary SHA256:
`8429fc3a82c86f4556d5b26adcf63c59e5372c774d857d94f1c0eab25782fb0a`.
The original qualified PR43 binary remains unchanged. The isolated physical-host
synthetic timing differential passed 1/1 in 24.88 s before inference.

**These are two unequal engine configurations, not a backend performance verdict.**
Portable has all experts on CPU; native has GPU-offloaded experts. Matched
expert placement/work is required before inferring an AMD/Vulkan performance
deficit from the difference.

Both arms completed four valid trials: one excluded warmup and three measured
fresh engines, frozen case 1, GEN32, early EOS. Every trial generated four tokens,
including EOS; token-sequence hashes matched across all eight trials. No trial
failed and no rerun occurred. Native ran on02 with its retained owned CPU worker;
portable ran on05 with two V340L dies, dense/coordinator Vulkan work, CPU experts
and host PLE. Placement and numerics were not changed for measurement.

All reported measured-trial durations below are **milliseconds**, in trial order.
Warmup is excluded from this table, retained in the raw receipts.

| Phase | Trial 1 | Trial 2 | Trial 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Native setup | 10003 | 8002 | 8002 | 8002 |
| Native DONE prompt | 1581.4 | 1530.9 | 1538.6 | 1538.6 |
| Native DONE decode | 350.1 | 354.9 | 358.1 | 354.9 |
| Portable setup | 2091.4 | 1979.7 | 1973.5 | 1979.7 |
| Portable selection prompt | 74150.7 | 73345.0 | 73887.4 | 73887.4 |
| Portable selection decode | 9930.8 | 10026.1 | 9971.9 | 9971.9 |
| Portable terminal feed | 2310.1 | 2267.6 | 2342.0 | 2310.1 |

These use the recipe's closest matching selection boundaries: the last prompt
forward step is part of decode, which selects four generated outputs. Portable
also feeds the final generated token; that extra work is reported separately.
Its full-consumption prompt/decode records are retained. Native READY setup uses
approximately two-second polling, and native CPU LOAD/cache lifetime differs
from portable fresh model opens. Setup is not a loading-speed comparison or part
of TPS. This short EOS-ending sample is not steady-state throughput. Matching
output hashes do not resolve the original FAILED logit-parity gates.

**Placement differs.** Portable has exactly zero GPU-resident
experts: all 48 layers' selected expert evaluations run on CPU, while only dense
and coordinator work runs on the two GPUs. Native uses the retained local CUDA
configuration plus the owned CPU worker at127.0.0.1:17431, CPU resident entry-row
mode and original-order reduction; the driver strips remote-GPU options and
retains prefill64 and the existing speculative/greedy settings. Native startup receipts now confirm CUDA0 held 960 experts (20 per layer,
4.67 GiB reported), CUDA1 held 1,378 additional experts (6.70 GiB), for 2,338
GPU-resident experts total. READY INFO reports expert_cache_mib=11644 and
expert_cache_primary_mib=4781. The CPU worker held the remaining 22,238 of
24,576 experts, 116,135,731,200 bytes resident and verified. The measured case
reused that resident CPU cache.

The exact expert-placement argv included `--expert-cache 960
--expert-cache-per-layer --expert-cache-device1 1378 --vram-reserve-mib 1152`,
`--expert-profile /home/hermes/strata-local-q8-20261023T152939Z/expert-profile.bin`,
`--adapt-every 0 --adapt-swaps 0 --pcie-frac 0 --remote-expert-opt`, and the
retained CPU resident entry-row worker configuration. Worker settings included
memory-budget-mib110756, workers12 and local-load-threads4.

For measured case02, CPU receipts show 768 successful requests and 11,294 routed
expert entries; CUDA1 shows 1,221 entries across 498 active-layer launches. A
decode counter reports 124 GPU cache hits /124 lookups plus154 GPU entries over
PCIe or another GPU, out of278 routed entries in that counter's scope. These
scopes differ and must not be summed into a total CPU/GPU work fraction. Total
primary-GPU expert dispatch is not established by these excerpts. Configured
environment values are not process readbacks after exit. Source:
[Hermes native placement receipts](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793161378436649).
The timing difference remains a comparison of unequal expert placements.

Both dies recorded 1500 MHz GPU / 945 MHz HBM in all 189 retained two-second
telemetry samples within the guarded high window. Sensor peaks were 49 C and
63 C. These samples establish the recorded clock profile, not continuous GPU
utilization or which stage consumed wall time.

Benchmark runtime exit was 0, its saved receipt was 0 and benchmark.json was
complete. Exit255 belonged to the telemetry SSH wrapper terminated during owned
sampler cleanup; it was not an inference failure. The privileged guard logged
restoration at 04:15:28 UTC and exited 0. Independent readback at 00:16:30 EDT
confirmed both dies auto, no owned inference/guard/telemetry process remained,
and native port17431 was closed. No artifacts were deleted.

Evidence remains private on02 at `/home/hermes/local-performance-v1` and on05 at
`/srv/models/inferswarm-artifacts/local-performance-v1`. These directories retain
individual receipts, argv, stdout/stderr, placement, source/build identity and
clock/temperature/cleanup evidence. This task read the linked report and its
thread; it did not SSH hosts or independently retrieve private raw files.

## Narrow source diagnosis; no measured bottleneck yet

The median selection decode is about 2.493 s per forward step (9971.9 / 4), not
an independently measured time for every token. The terminal feed is separately
about 2.31 s. Aggregate timings cannot attribute either duration to a stage.

Three plausible costs follow from the current source:

- **CPU expert path.** [CpuExperts::run](../src/core/portable_cpu_experts.hpp)
  calls a 12-worker native pool for ten experts at each of 48 sequential layers:
  480 expert evaluations per token. Reuse skips copies/digests only after each
  layer/expert's first verified use in that fresh engine. A new selected expert
  still copies and digests 5,222,400 bytes; an entirely unseen token's selection
  can therefore copy/hash 2,506,752,000 bytes. This is a static worst-case byte
  count, not an observed miss count or bandwidth. Kernel work, memory reads,
  verification/page faults and pool waits are all candidates. First-use copy/FNV
  traffic is separate from repeated CPU kernel reads of already verified mapped
  weights. Either may be served from RAM/page cache. A mapped access does not
  prove physical disk traffic: no page residency, major-fault, process read_bytes
  or block-I/O measurement is present in the supplied report. Minor faults also
  do not imply disk reads. Report physical disk/page-cache behavior as unknown.
- **Graph/backend path.** [FFN](../src/core/portable_ffn.cpp) executes separate
  pre/post graphs for every layer. [GDN](../src/core/portable_gdn.cpp) has an
  attention graph, while [QSA](../src/core/portable_qsa.cpp) also constructs,
  schedules and allocates a temporary attention graph per token/layer, alongside
  its fixed projection/combine phases. [schedule](../src/core/portable_graph.hpp)
  walks nodes and records assignments. T=1 issues many small graph submissions;
  backend dispatch, synchronization, scheduler/allocation and actual dense GPU
  computation are plausible costs. The pinned GGML implementation synchronizes
  inside ggml_backend_sched_graph_compute; it is not merely enqueue timing.
- **Transfers and host work.** GDN explicitly uploads and downloads recurrent
  state, history and residuals every step. Its 36 layers each have 786,432 F32
  state elements: state alone implies about 216 MiB per token across both
  directions, before history/residual/FFN transfers. FFN reads routing and shared
  outputs, computes route/merge on the host, writes post inputs, then reads the
  next residual. QSA stages active history and intermediate tensors; the head
  reads full logits. Host allocations, finite scans, PLE gathers and placement
  aggregation add work. This establishes mechanisms, not their measured shares.

The existing CLI emits cumulative `expert_work position=... hits=...
verifications=... copied_bytes=... digest_bytes=... native_batches=...
source_invalidations=...` after each token to each case's stderr.log. Difference
adjacent rows, with the first row measured against zero, to recover per-token
work. The supplied report/thread exposes no values from those lines and no
stage timing log. They must not be treated as observed here. Read these existing
receipts before new inference; no aggregate benchmark repeat is needed.

## Product-relevant next step

The native residency evidence above closes the startup-placement gap. Parent
has selected portable local GPU expert residency as the next product slice,
reusing the existing Vulkan executor, rather than adding a native all-CPU
control solely to construct another baseline. No new control or campaign is
proposed here. A later backend inference still requires matched expert work.

The existing `PortableExpertRows` seam and Vulkan `infer_rows` indexed unweighted
reply allow a fixed hot-expert plan with CPU cold rows and the unchanged
original-order F32 merge. A tiny explicit per-die budget must include dense
allocations, executor tensor buffers, metadata/staging, backend-private runtime
allowance and free reserve; the executor's tensor cap alone is not total VRAM.
Vulkan hybrid Q8 is explicitly not the CPU bit-parity contract, so unchanged
weight precision/quantization boundaries and merge order do not establish row
or logit parity. Numerical gates remain mandatory. This is feasibility only.

## Single cheapest useful stage split, after placement is explicit

Propose one opt-in **per-token stage ledger**, reusing existing synchronous
boundaries, with three disjoint accumulators plus an unassigned host remainder:

1. **CPU expert wall time first:** CPU provider wall time around `experts->run` in portable_model.cpp. Also expose
   deltas of the pool's already-collected ms_multi_gu, ms_multi_q and
   ms_multi_down counters; provider minus pool then isolates verification/source
   checks/quantization/other provider overhead approximately, without another
   pool profiler. Keep existing verification/copy/hit byte/count deltas.
2. Graph/backend wall time around existing graph-compute calls and temporary
   QSA graph construction/schedule/allocation. This includes backend-internal
   copies and synchronization, not isolated kernel time.
3. Explicit transfer API wall time and requested byte counts around existing
   tensor get/set calls in GDN, QSA, FFN and head. Keep them outside graph timing;
   ordinary host vector construction/finite scans remain in the remainder.

Total step time minus those nonoverlapping categories exposes host remainder;
one compact receipt per token suffices. No GPU events, added synchronization,
per-op dump, placement/precision change or backend rewrite is needed. A timer
only around the CPU callback would be smaller but could not distinguish graph
work from explicit transfers. This ledger distinguishes those operational
categories; it cannot separate kernels from internal scheduler copies, or pool
barriers from arithmetic without later targeted evidence.

This is a proposal only: no instrumentation code, new fleet run or profiler was
added at this checkpoint. Parent owns the next bounded measurement dispatch.
