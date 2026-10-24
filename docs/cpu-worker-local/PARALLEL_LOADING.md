# Parallel local weight loading

The CPU worker reads and verifies independent experts with four loading lanes by default. Use
`--local-load-threads 1` for the existing serial, file-sorted loader, or explicitly select 2 or 4.
Values 1–4 are accepted. `--workers` still controls inference only.
The count is local configuration, not a new wire field. Upload-only setup and manifest preparation
are unchanged. The owner selected four as the default after the supplied Hermes comparison below.
The default-selection follow-up is statically reviewed; its executable checks are **NOT RUN locally**.

The loader keeps the existing batch limit of 32 experts and writes into the one budgeted final arena.
Each lane owns a contiguous range of experts in that batch, reads each expert's present roles, then
runs its unchanged forward/reverse FNV chains. Another lane may read while this lane hashes. There
are at most four lanes including the serving thread, three helper threads and four simultaneous
regular-file reads of at most 256 KiB each. No expert payload is staged in a second buffer. Small
batch metadata and native thread stacks are additional to the arena budget and count toward process
memory. File cache also uses system memory. More threads are not a promise of more disk bandwidth.

The serving thread validates placement and file metadata first, then freezes descriptor/stamp tables
until all lanes join. Parallel reads use `pread` and check file identity, size and timestamps before
and after each read. Hashes still cover every present byte. Missing roles still require whole-expert
fallback; a corrupt present role cannot hide behind an absent one. Counters and successful hit flags
are published only after join. A failure poisons the entire batch and prevents READY or resident
reuse. A new setup session is required to try again. No resident lifetime or protocol rule changes.

Stop, disconnect, unexpected pipelining and deadlines are checked between bounded reads/hash chunks.
An individual regular-file syscall can still stall in the OS; threads must join before the arena or
descriptors are released. Keep the existing owned outer timeout. Replacing a worker loses its arena
and requires a reload; enabling the flag on disk cannot change a running worker.

With multiple lanes, existing read and verification counters report **summed lane work time**.
They can overlap each other and exceed elapsed setup time. The worker's `setup_s` at `local ready`
reports elapsed time from local session setup through successful READY checks, including allocation
and first touch. The engine's READY handshake is another elapsed scope. Do not add these timings or
subtract clocks across hosts. Loaded bytes count completed read chunks; verified bytes count bytes
processed by each forward/reverse pair once, including work before failure.

Hermes supplied results at `723c5a45cd0fe289cb57c7c3d1e504c068bd0cf8`: on 02 with eight 32 GB DIMMs,
251.5 GiB usable RAM at 2133 MT/s, engine-to-READY took 206.8 s / 114.6 s / 68.5 s with 1 / 2 / 4 lanes.
The observed four-lane time was 66.9% lower than one lane. Each completed the same 116,135,731,200-byte
arena, with zero inference. These are supplied results, not local execution or an independently read
report. T4 process counters were `read_bytes=0`, `rchar=116135731200`; this is not a cold-storage speedup
claim. T1/T2 cache residency was inferred only. Per-TID captures failed, and overlapping lane-time sums
do not prove distinct physical cores. No successful load was repeated; wrapper failures were retained.

The [Hermes handoff](../validation/CPU-LOCAL-LOADING-20261024.md) records the supplied receipt and the
two small default-selection checks still needed. It keeps this change separate from
[inference buffer reuse PR #6](https://github.com/Zutfen-LLC/Strata/pull/6). Keep the existing T4 worker
resident; this default change does not require a new full-model campaign.
