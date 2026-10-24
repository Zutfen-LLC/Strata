# Fixed remote CUDA worker groups

The opt-in native CUDA/POSIX engine can use 1–8 remote GPU workers alongside its
existing local GPUs and CPU worker. Each remote GPU runs one
`strata-cuda-expert-worker` process with its own device, loopback port, immutable
plan and cache directory. SGW1 and the existing worker executable are unchanged.
This adds endpoint concurrency; it does not discover hosts or choose placement.

Repeat `--remote-gpu-worker HOST:PORT PLAN` in the desired reduction order:

```text
--remote-gpu-worker 127.0.0.1:17432 plan-3090.bin
--remote-gpu-worker 127.0.0.1:17433 plan-3060ti.bin
--remote-gpu-worker 127.0.0.1:17434 plan-3060.bin
```

These are illustrative engine-side forwarded ports. Establish only authorized
temporary relays; the worker itself still binds loopback. Numeric IPv4 addresses
are required, as with the existing controlled connector. The old
`--remote-gpu-experts HOST:PORT --remote-gpu-plan FILE` pair is a one-worker
shorthand. Mixing the shorthand with the new collection is rejected. Both forms
require the existing CPU endpoint, trusted local manifest and other native-model
options. The CPU partition and its residency identity remain unchanged.

Every plan must match the trusted model, layout and blob digests. Remote plans
must be disjoint from each other and local GPU ownership, and their union must
lie inside the CPU partition. CPU requests execute only the remainder of that
partition. Only missing/invalid assigned assets transfer; each worker retains
its own validated cache. Two processes must not share a cache directory because
the cache holds an exclusive lock. Do not assume equal VRAM budgets: admission
includes that worker's assigned weights, scratch and runtime reserve.

For the first proposed three-peer topology, retain the remote 3090 on
inferswarm04, then use inferswarm02 device 0 (RTX 3060 Ti, 8 GB), then device 1
(RTX 3060, 12 GB). This order follows the supplied hardware report; x16 links,
driver 610.57.04 and CUDA 13.1 readiness are not Strata performance proof. Device
ordinals are relative to `CUDA_VISIBLE_DEVICES`; keep both 02 devices visible
when using `--device 0` and `--device 1`. Use separate plans and budgets.

Setup is serial to bound upload staging. Once READY, each endpoint's single
communication thread owns its socket, sends idle STATUS probes and receives its
requests. This keeps earlier peers alive during later setup and idle intervals.
Configure worker and engine frame deadlines consistently; a keepalive cannot
overcome an arbitrarily shorter server deadline or an unresponsive route.
Keepalives do not extend setup/inference deadlines or CPU residency TTLs.

The existing asynchronous CPU request is submitted first. The group copies
inputs and queues every participating GPU before waiting for replies. Results
are retained by configuration index, then merged after the existing helper sum
in endpoint order, with CPU last. Arrival order never sets floating-point
reduction order. One endpoint preserves the existing reduction order; moving
experts among several endpoints can change grouping and therefore rounding.
No kernel, quantizer, wire format or numerical threshold changes here.

A failed peer latches the group failed. Already-started exchanges finish/fail
under their existing bounded deadlines; socket owners are joined before sockets
are closed by the caller. No incomplete network sum reaches the device merge,
and there is no automatic reassignment or retry. Engine shutdown closes its
connections; normal worker ownership/cleanup and explicit CPU retention policy
still apply. Do not treat the diagnostic controller finishing as an instruction
to unload a healthy resident CPU arena.

The focused fixtures and remote-only instructions are in
[MULTI-REMOTE-GPU-V1-20261023.md](validation/MULTI-REMOTE-GPU-V1-20261023.md).
They exercise generated weights and zero GEN requests. New implementation and
fixtures are authored and statically reviewed, **NOT RUN** at publication.
Historical single-worker numerical screening failures remain disclosed; this
extension makes no new full-model quality, speed or scaling claim.
