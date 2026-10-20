# Worker-local expert setup candidate

Candidate branch: `inferswarm/worker-local-experts` in `Zutfen-LLC/Strata`.
Exact base: `7a590706f20bc976d5bf094be3dfd0c461cf8869` (`inferswarm/setup-chunks-1m`).
This is a pre-PR hardware-test candidate for InferSwarm #348/#349. No hardware trial,
hardware-host connection, PR, broad CI or merge is part of this coding task.

## Configuration and integrity contract

The worker defaults to upload-only setup. To enable local setup, its operator must provide both
`--local-root DIR` and `--local-manifest FILE`. The engine must separately provide
`--remote-cpu-local-manifest FILE`. Local setup uses a numeric IPv4 endpoint; use
`--remote-cpu-experts 127.0.0.1:7431` through the controller-owned forwards. This permits a bounded,
nonblocking connection without an unbounded DNS lookup. Upload-only host resolution is unchanged.

`strata-worker-manifest` reuses `expert_layout_load` for native index v2..v4 and `GgufFile` for tensor
headers. It accepts only uniform, unrotated Q8_0/Q8_0 at the worker's geometry. It checks each role's
name, type, shape, tensor length, absolute shard offset and file bound. It handles v4 per-role shard
names, including the real pack's split-role layers 17 and 35. It does not repack weights or create
`experts.bin`. Preparation reads all expert slices once through one blob of staging. It closes the
header mappings before bounded payload reads. It reports bytes after each layer and has an explicit
wall budget plus SIGINT/SIGTERM cancellation. The temporary output is not a usable completed manifest.

The manifest records geometry, quantization, the native index digest, shard basenames and sizes,
three role ranges per layer, one whole-expert digest per slot, and role digests for incomplete slots.
The manifest's full 128-bit digest is negotiated. Its first 64 bits become the session's model id.
The setup also binds the frozen ownership bitmap and its partition id. The engine checks the exact
native index, slice mapping and file inventory without reading expert payloads. It requires
`--native`, `--mmap-experts`, and the actual GGUF source; a pack with `experts.bin` is refused before
GPU expert-cache fills. Thus an alternative engine expert source cannot silently bypass this contract.

**Trust and reuse:** create the manifest from the verified, complete, immutable model copy. Pin the
manifest with `sha256sum`, retain that receipt with the existing model-copy receipt, and transfer the
small manifest through the dedicated controller to the other host. Verify the same SHA-256 there.
The operator supplies this trusted manifest; the protocol does not authenticate a manifest supplied
by an attacker. Use the existing verified copy receipts to establish that inferswarm03 and
inferswarm05 have the same model. Filenames, sizes and timestamps alone do not prove that identity.
Do not reuse a manifest with a different model or rewritten native index. Regenerate once after an
intentional asset change, rather than scanning 188 GB on every iteration.

The expert/role and wire identity digests are the existing two-pass FNV-1a `Digest128`, with no reduced
byte coverage. These are noncryptographic integrity checks, as in upload setup. The SHA-256 receipt
pins the trusted manifest outside the protocol. The worker hashes every actual assigned local expert
before it can become ready. It does not trust timestamps instead of those hashes. Descriptor checks
also reject size, inode, mtime or ctime changes during setup. A successful local load proves expert-byte
integrity under this existing non-adversarial contract; it does not certify unrelated dense tensors.

## Loading, fallback and bounds

New `SEW1` message types negotiate local-v1 explicitly. `LOCAL_SETUP` carries version, fallback flag,
manifest digest and the existing setup manifest. Its ACK carries the session, blob size, upload cap,
local version and batch cap. `LOCAL_LOAD` carries only a session and at most 32 owned `(layer, expert)`
pairs. Its ACK carries ordered hit flags and cumulative local byte/hit/miss/read/verify/setup counters.
No remote request contains a path or byte range. An old worker closes on this unknown message; the
new client reports failed local-v1 negotiation. A new unconfigured worker returns `local-unavailable`.
Neither case silently falls back to a full upload. Existing SETUP/UPLOAD/STATUS and inference frames
retain their layouts and bounds.

For each batch, the worker sorts source ranges by file and offset. It reads at most 256 KiB per syscall
directly into the budgeted assigned arena. There is no second weight mirror or disk copy. Metadata is
bounded: at most 64 basename files, 96 layers, 1024 experts/layer and an 8 MiB manifest. Paths containing
separators, overlapping ranges, overflow and out-of-bounds slices are rejected. `openat` beneath an
operator-opened root refuses symlinks and nonregular files. The operator owns the root and its parents;
the protocol cannot select another root. The service loads only the assigned bytes. Ordinary OS file
cache, native execution workspace and allocator overhead remain separate from the arena budget.

Only `ENOENT` is a local miss. Wrong size/type, permission errors, mutation and digest mismatch are
fatal. If a role is absent, present roles are still read and checked against their role digests.
For a genuinely incomplete expert, fallback uploads the whole expert through the existing bounded
UPLOAD path, not just its absent role. To allow this, **both** worker `--local-upload-fallback` and engine
`--remote-cpu-local-fallback` must be set. Upload can fill only a slot already reported absent; its
whole-expert digest must match the trusted manifest. Local hits cannot be overwritten. With all
assigned experts present, no engine blob fetch, payload hash or weight upload occurs.

READY still requires every assigned expert to be complete and verified. Local corruption poisons the
batch; only a new setup session can retry. Inference framing, arithmetic, ownership, sequence checks,
merge order and failure latching are unchanged. By default disconnect and restart void the arena and readiness.
The separate [bounded residency option](../cpu-worker-residency/README.md) allows a complete partition
to survive disconnect with a fresh session negotiation; a process restart still starts empty.

Local setup has a 600000 ms whole-session budget by default on each end (`--local-setup-ms` on the
worker, `--remote-cpu-setup-ms` on the engine). The existing 10000 ms I/O setting also bounds each batch.
The client checks the total budget before connection, during negotiation, before every header/payload
wait, during uploads and before readiness succeeds. Controlled socket waits observe cancellation at
most every 50 ms. The worker checks stop/disconnect/deadline between 256 KiB reads and 64 KiB hash
segments. An individual regular-file syscall is not preemptible if the OS stalls it; an outer owned
process timeout remains the hard wall bound. No failed setup exposes a partially ready partition.

Engine startup reports network weight bytes separately from local bytes, hits/misses, connect/setup,
local read/hash, fetch/upload and READY wall phases. Worker `local ready` and stop logs record local
read/hash/whole-setup times. Legacy STATUS `setup_weight_bytes` continues to count network weights;
its wire layout is unchanged. Local counters reset for each local session; legacy network counters
remain cumulative for the worker process. Use one worker process per trial for direct attribution.

## Validation and remaining checks

Geekom, Linux x86-64, GCC 16.2.1, Release CPU-only; existing pinned ggml source cache
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. The final six suites passed **6/6 in 21.09 s**.
The all-local binary smoke read **20,889,600 B**, reported **0 network weight B**, four hits and no
misses. These are small generated fixtures in ordinary OS file cache, not SATA/model measurements.
An initial restricted-sandbox run could not create sockets. All reported passing socket runs used
an allowed loopback environment; no production refusal was bypassed.

Both exact-base compatibility pairs passed the real e2e fixture: actual base client/current worker,
and current client/actual base worker. The former compiled the base client source and header from
`git show 7a590706:...` and linked that client before the current service archive. The latter used the
retained exact-base worker binary, first required local-v1 negotiation to fail clearly with no upload
fallback, then reconnected and completed ordinary upload, readiness and inference. The base worktree
remained clean at the exact base SHA. This is wire compatibility at this base, not a binary C++ ABI
promise or a claim about arbitrary historical peers. [Review record](evidence/review.md).

Reproduce the CPU checks with the existing ggml cache (set `GGML_SRC` to its local checkout):

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR="$GGML_SRC"
cmake --build build-cpu --target strata-expert-worker strata-worker-manifest worker_local_test \
  expert_worker_test expert_worker_e2e_test expert_tier_test worker_socket_test -j2
ctest --test-dir build-cpu --output-on-failure \
  -R '^(worker_local_test|worker_local_binary_test|expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_socket_test)$'
```

Geekom CPU validation uses generated GGUF Q8_0 fixtures and the real client/service/tier. The separate
binary smoke uses the real `strata-expert-worker` CLI. The fixtures cover upload/local/direct bitwise
parity at T=1/4/8, all-local zero network weights, missing-only fallback, split roles, corruption of a
present role beside an absent role, sparse ownership, duplicate/not-owned requests, wrong identity,
truncation, invalid ranges, path escape, symlinks, incomplete readiness, budget refusal, total setup
expiry, cancellation, delayed negotiation and reconnect. Retained results are in [evidence](evidence/).

Geekom has no CUDA compiler. No CUDA/HIP/SYCL build or GPU execution is claimed. The actual engine
source-selection guard and startup glue therefore require the incremental CUDA build and bounded
full-model run below. CPU fixtures do not prove full-model output parity, SATA throughput, two-GPU
participation or full-partition RSS. The supplied 18.7 MB/s two-expert network smoke is not a sustained
limit. The user's 400 MB/s SATA expectation is not a measured guarantee. This candidate claims no
hardware startup speedup before that trial.

See [HERMES.md](HERMES.md) for portable controller instructions. No Geekom artifact is needed there.
