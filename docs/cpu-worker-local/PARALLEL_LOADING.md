# Parallel local weight loading

The CPU worker can read and verify independent experts concurrently. Add `--local-load-threads 2`
or `--local-load-threads 4` to its existing local-source command. Values 1–4 are accepted; the default
is 1 and uses the existing serial, file-sorted loader. `--workers` still controls inference only.
The count is local configuration, not a new wire field. Upload-only setup and manifest preparation
are unchanged. This candidate is source/static-review only; builds and executable tests are **NOT RUN**.

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

The [Hermes handoff](../validation/CPU-LOCAL-LOADING-20261024.md) defines focused 1/2/4-thread gates and
an independent loading comparison after the hosts return. It keeps this change separate from
[inference buffer reuse PR #6](https://github.com/Zutfen-LLC/Strata/pull/6). No speedup or runtime
compatibility has been established for this candidate.
