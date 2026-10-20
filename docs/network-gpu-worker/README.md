# Fixed network CUDA expert placement

This is an unvalidated, opt-in candidate for one RTX 3090 worker on **inferswarm04**.
The engine remains on inferswarm03 (its two RTX 3060 GPUs keep their existing ownership).
Inferswarm05 keeps its existing resident CPU arena and full CPU partition. It may retain
duplicate weights; only requests outside the frozen GPU assignment execute on it.

No configuration, compilation, tests, benchmarks or model loads were run on Geekom.
No operations were run on an inferswarm host. A draft PR and independent static reviews
are preparation for the [Hermes fixture gate](HERMES.md), not runtime acceptance.

## Supported first slice

The existing CPU worker's supported layout is uniform plain Q8_0/Q8_0, without
Hadamard transforms or scale tails, within its existing geometry bounds. This slice
uses that same layout and the existing CUDA `native_expert_grouped` kernels. It does
not pretend that `RemoteExperts` is a network service: that class remains the
same-process helper for other local GPUs.

The new SGW1 protocol explicitly negotiates CUDA execution and q8_1 activations with
F32 results. The existing SEW1 CPU protocol and native-bitwise contract are unchanged.
An incompatible peer is an error; there is no upload-mode or CPU inference fallback.
The new service is POSIX/CUDA only. CPU-only/native-ON builds can build the asset and
protocol fixture sources without a CUDA dependency. Native-OFF builds exclude them.
There are no HIP, gfx906, SYCL, Windows or full-model parity claims.

The assignment is configured at startup on both peers, checked against the engine's
trusted local manifest, and frozen for the session. Every selected pair must be in
the existing CPU partition and outside all local GPU caches. A different model,
layout or placement requires closing the session and restarting with the new plan.
No discovery, live join, rebalancing or CPU arena shrinking is implemented.

## Selective acquisition

The operator starts from an existing trusted `SWLOCAL1` manifest and an explicit
list of `(layer, expert)` pairs. The selective plan contains only assigned entries,
their expected lengths/digests and required model/layout/geometry identity. Plan
generation reads bounded metadata, not model weights. This avoids repeating the
older one-time whole-model manifest preparation as part of GPU setup.

`GpuAssetSource` uses the existing GGUF layout: for expert `e`, each role range is
`layer_role_offset + e * role_bytes`. It reads exactly the gate, up and down slices,
concatenated in the native blob layout, for a missing assigned expert. It does not
download or stage a layer or shard. The engine must already have its trusted model
source; this is not a whole-model downloader. Files may span different GGUF shards.

Create the plan on the source host from explicit zero-based pairs:

```sh
# assigned.txt contains only the chosen LAYER EXPERT pairs, one per line.
strata-gpu-manifest plan existing-trusted.local assigned.txt fixed-3090.plan
sha256sum fixed-3090.plan
```

Copy only that small plan to the worker. The operator creates an empty dedicated
cache directory, then starts `strata-cuda-expert-worker --plan fixed-3090.plan
--cache-root CACHE --device 0 --port 7432 --cache-budget-mib 24576 --reserve-mib 1024`.
The final two values are independent operator limits, not measured recommendations.
The engine adds `--remote-gpu-experts NUMERIC_FORWARD_HOST:PORT --remote-gpu-plan
fixed-3090.plan` to its existing supported serial remote-CPU configuration. It also
requires that configuration's trusted `--remote-cpu-local-manifest`. An operator
must approve the real-model window before using that engine configuration.

The worker reports a missing bitmap after rechecking only assigned cache blobs.
The engine reads and transfers only those missing/corrupt entries. A restart reuses
verified assigned files; assignment additions fetch only missing additions. Cache
keys bind model, layout, layer, expert, length and expected digest. `Digest128` is the
existing two-pass noncryptographic FNV integrity check; the trusted manifest and
owned SSH transport remain the trust boundary. No authentication claim is made.

The worker's dedicated, operator-created cache directory uses relative generated
names, directory descriptors and no-follow regular-file opens. New files publish
atomically only after complete length and digest verification. Incomplete uploads
cannot become usable assets. Existing complete assigned files are reverified before
readiness, and stale temporary files are not hits.

Disk retention and VRAM residency have separate bounds. The disk budget counts all
retained regular files, including old assignments and incomplete temporary files;
setup reserves space for new assets before accepting them. Capacity refusal preserves
other assets. Nothing is automatically evicted. The explicit cleanup operation removes
only this cache format's unassigned assets/abandoned temporary files, with exclusive
cache ownership; unrelated files are preserved. Use a fresh owned directory for
fixtures. Close the worker before intentional retention cleanup.

Host staging is bounded by one expert blob (at most 8 MiB under the compiled cap),
plus bounded protocol frames and inference buffers. It never mirrors the assignment
in host RAM. Counters distinguish assigned bytes, verified cached/reused bytes,
network weight payload bytes, source bytes returned by selective reads, bytes written
to cache, retained disk bytes, host staging and actual CUDA allocations. Source-read
bytes are logical `pread` payload bytes; filesystem readahead and physical device
I/O need separate OS measurements. Protocol framing/SSH overhead is separate from
weight payload transfer.

## Capacity, readiness and failure

The CUDA executor initializes the chosen device, queries available VRAM, accounts for
the complete assigned weights and its actual fixed inference scratch/metadata/input/
output allocations, and subtracts an explicit runtime reserve. Each CUDA allocation
and copy is checked. An estimated 18–20 GB assignment is not a promise of fit: free
memory, reserve and actual allocations decide. Setup refuses before inference on
overcapacity or allocation failure. Use the worker's actual counters plus an external
GPU/RSS sample to assess untracked runtime and allocator overhead.

READY requires every assigned blob to be length/digest verified and copied to GPU
memory, with completed CUDA synchronization. Only then may inference run. Disconnect
invalidates the session and GPU allocation lifetime; complete disk files may survive.
A fresh connection sets up a new session with new request sequencing.

Requests and replies bind model, layout, frozen placement, session, sequence and the
request identity. Unassigned entries, stale sessions, malformed lengths, nonfinite
values, duplicate or mismatched results are errors. The engine masks each execution
owner's rows and merges its already-weighted partial exactly once. Both remote tiers
must finish successfully before any remote partial is accumulated for the window.
Failure latches the tier dead, aborts the affected request and does not silently retry
on 05. A newly started CPU-only engine session is the recovery path. There is no
transparent failover or mid-session placement change.

The first network-GPU post is synchronous, so there is no claim of network/CPU overlap.
During selective GPU startup the engine probes the already-ready CPU session so its
socket does not time out while weights transfer. Status probes do not renew the CPU
residency TTL. Expiry or loss cancels GPU setup; the configured TTL must cover setup
and the intended session. This does not resize or replace the CPU arena.
The service is one connection and one request at a time, with bounded setup and frame
deadlines. Kernel execution and ordinary filesystem system calls cannot be preempted
by those deadlines; Hermes uses an owned outer process timeout. The listener binds
loopback; use existing trusted SSH forwards for cross-host access, not a new firewall
or public listener.

## Numerical acceptance declared before execution

CUDA q8_1 and CPU-native activation rounding differ. No CPU/GPU bitwise equality is
expected. The reference is the existing `src/kernels/native_expert_parity.cpp`: GGML's
own weight dequantizer, float expert evaluation with double dot accumulation, and
the existing direct grouped CUDA path. The fixture gate must record, not assume:

* Relative L1 error of CUDA network output versus the dequantized float reference
  at most `3e-2`, the existing native-expert parity threshold. Report max absolute
  error and RMSE as well, per case and aggregate. A near-zero reference needs an
  absolute-error report rather than a misleading relative claim.
* Network CUDA output versus the same-device direct grouped CUDA calculation with
  the identical entry-order weighting: relative L1 at most `1e-6` and max absolute
  error at most `1e-5`. Transport should not introduce quantization.
* Mixed-owner merge versus a reference assembled from the same local-GPU, remote-GPU
  and CPU per-entry results in the documented merge order: relative L1 at most
  `1e-6`, max absolute error at most `1e-5`. Separately compare the full result to the
  dequantized reference with the `3e-2` bound and report cancellation-sensitive cases.
* Existing CPU direct-versus-transport bitwise fixtures must retain their contract.
  Ownership masks, entry counts, request identities and reply cardinality are exact
  structural checks, independently of floating-point tolerances.

These are acceptance requirements for Hermes, not measurements. Test coverage and
static findings are recorded with the candidate; no runtime PASS is claimed here.

The eventual real-model gate requires a separately authorized window after fixtures
pass. It must show that 04 executes assigned experts while 05's per-request entry
count includes only uncovered work, with source/network/disk/GPU/RSS/numerical/timing
evidence and unchanged 05 arena/partition identity.
