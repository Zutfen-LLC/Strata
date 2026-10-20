# Remote CPU-expert correction: Hermes retest and asset preparation

Candidate branch: `inferswarm/two-host-trial-correction` in
[Zutfen-LLC/Strata](https://github.com/Zutfen-LLC/Strata). Base:
`17ecafca61065444b54c97a0d1b0bae4280769b0`, including worker
`484235ff60e08181f5fbc2d0084291a8111f4e7f`. Use the **full correction SHA in the handoff**,
verify it against the remote branch, then detach at that SHA. Do not substitute main.
[Findings and local evidence](cpu-worker-trial-correction/README.md).

This document separates three later Hermes windows: test retest (no model needed), bounded asset
preparation, and full-model execution. The coding task performed none of those host operations.
Asset preparation and model execution require their own explicit scope approval. No installations,
driver changes, new SSH trust, permanent mounts, or old llama.cpp worker rebuild are part of this recipe.

Run each shell block in a fail-fast Bash shell (`set -euo pipefail`). Any refusal stops that window;
retain its logs and report the blocker before issuing dependent commands.

## 1. Test-only retest on inferswarm 05 and inferswarm 03

On the controller, record the full correction SHA from the handoff and verify it:

```sh
: "${CANDIDATE:?Set the full 40-character correction SHA from the handoff}"
test "$(git ls-remote https://github.com/Zutfen-LLC/Strata.git refs/heads/inferswarm/two-host-trial-correction | cut -f1)" = "$CANDIDATE"
```

Set the same `CANDIDATE` value on **each host**, then reuse only the dedicated, clean trial checkout from the first attempt. Set `SRC` to
that checkout and `BUILD` to its already configured build directory. Neither variable is a geekom path.
Discover them from the retained build receipt or `CMAKE_HOME_DIRECTORY` in that build's CMakeCache.txt.
Save old logs/build receipts before rebuilding; stop if the checkout has unrelated changes.

```sh
: "${SRC:?Dedicated existing trial source checkout}"
: "${BUILD:?Existing trial build directory, absolute path}"
cd "$SRC"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = 17ecafca61065444b54c97a0d1b0bae4280769b0
# If this is not the old dedicated trial checkout, make a fresh scoped checkout instead; do not reset it.
git fetch origin inferswarm/two-host-trial-correction
test "$(git rev-parse FETCH_HEAD)" = "$CANDIDATE"
git switch --detach "$CANDIDATE"
```

On **inferswarm 05**, preserve its successful Makefiles generator and CPU ISA settings. Explicitly keep
all GPU backends off; use `-j 2`, which passed the first host build. No assets are prerequisites.

```sh
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF
cmake --build "$BUILD" --target strata-expert-worker expert_worker_test expert_worker_e2e_test expert_tier_test worker_socket_test -j2
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^(expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_socket_test)$'
python3 tools/worker_trial_adverse.py --build "$BUILD" --out "$SRC/retest-adverse-$CANDIDATE"
```

On **inferswarm 03**, reuse the successful CUDA sm_86 Ninja build and its pinned ggml source/cache.
Do not change CUDA 13.1.115/GCC 14.2.0 just to run this correction. Verify the cache still names architecture 86
before building; preserve all other existing configuration choices. This is an incremental test build,
not the former 172-step clean engine rebuild.

```sh
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$BUILD" --target worker_dispatch_test -j2
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^(worker_dispatch_test|worker_dispatch_host_zero_test)$'
# One additional policy check, a separate process because the engine caches these environment values:
STRATA_VERIFY_DEVICE_PLAN=1 timeout 120 "$BUILD/worker_dispatch_test"
```

The first dispatcher test covers default GPU zeroing; the second sets `STRATA_DEC_BATCH=0` and covers
host zeroing. Both verify actual plan membership/pointers/tokens, real row-copy masks, real partial-sum
combine, T=1/4/8, default-off local work and a dead worker. Native cache matvecs are emulated on the
host; no full-model numerical or GPU-matvec equivalence claim follows from this test.
Stop on the first failed command, retain stdout/stderr and exit status, and report it. No repeat-to-PASS.

## 2. Storage attribution and inventory (read-only; before approval to copy)

Run these checks on **inferswarm 03**, recording the path, source device, filesystem and exact bytes:

```sh
findmnt -T /srv/models -o TARGET,SOURCE,FSTYPE,OPTIONS
findmnt -T / -o TARGET,SOURCE,FSTYPE,OPTIONS
df -B1 --output=source,target,size,used,avail / /srv/models
lsblk -b -o NAME,SIZE,ROTA,FSTYPE,MOUNTPOINTS
```

The user reports older 03 measurements of roughly 1.8 TB free on the `/srv/models` sdb1 SSD and roughly 163 GiB
free on root. The first trial's unattributed “157 G free” does **not** establish that all of 03 lacks space.
No current mount was inspected in this coding task. Do not infer that the model directory is mounted
just because it exists: `findmnt -T` may report `/` when the SSD is unmounted. Require the recorded,
expected model filesystem before writing there; stop on a different or missing mount.

Trial inventory is user-reported: six Q8 shards totaling **175.6 GiB**, already on **inferswarm 05** and
**inferswarm 01**; no Qwen bytes on 03; no native pack found fleet-wide. Select one already verified
source set. Reuse its member names, sizes, digests and provenance from the retained trial manifest.
No full-model rehash is required merely to repeat known evidence; hash new destination copies as part
of transfer verification. If source provenance/metadata are missing, retain that limitation and resolve
it before treating files as verified. No 03→05 SSH trust exists or is assumed.

## 3. What the pinned native pack actually requires

Source basis: `tools/iq_pack.py::{Model,index_standalone,expert_layout,main}`;
`src/core/expert_source.cpp::{FileExpertSource::open,open_native_gguf}`;
`src/program/generate.cpp`'s `--native` shard discovery and worker setup.

| Required on 03 | Why / size |
| --- | --- |
| All six original Q8 GGUF members, original basenames together | `--native` discovers every split member; dense projections, embedding/head, PLE and expert setup refer into them. Approximately 175.6 GiB; exact sum comes from source member stats. Header-only stand-ins do not satisfy current loaders. |
| `native_experts.txt`, `index.txt` | Small reference tables; Q8 experts map their existing GGUF slices (v 3/v 4 per-role shard offsets). |
| `dense.bin`, `conversions.json` | Non-expert float tensors in the loader's required form. This requires reading/writing selected tensor data, so metadata-only extraction alone is insufficient. Exact size depends on the real source tensor headers and required conversions. |
| `tokenizer/` | Small metadata export, required by the Python text front end; raw-token engine trial uses frozen token IDs. |
| `data/expert-profile.bin` from this source revision | Shipped 48×512 STRP profile,196632 B. Required to fill the GPU caches; record its SHA 256. |

Do **not** pass `--experts-bin`: plain Q8_0/Q8_0 experts can be read from the original shards. That avoids
an additional **128345702400 B (119.53125 GiB)** expert copy on disk. A new native pack is not expert
requantization. `--compat-bf16`, if the real dense tensors require it, rounds only the small projections
the loader needs; it must be explicitly included in asset approval and its conversions retained.
Do not silently reinterpret Q8 weights or add a different worker quant. Hadamard/other worker layouts
remain unsupported. No full cold-expert RAM mirror is needed on 03; setup streams through one 5222400 B
worker staging buffer plus the file source's bounded staging and ordinary reclaimable page cache.
Do not enable `--resident-cpu-experts` or a RAM-preloaded expert source.

On 05, set `SHARD1` to the known first Q8 member from inventory, then run this **header/stat-only** estimate
from the pinned Strata source. It reads no tensor payload and creates no pack. Python numpy is needed to
import the pack helpers; do not install packages without approval.

```sh
: "${SHARD1:?Absolute first Q8 shard from the retained inventory}"
export SHARD1
python3 - <<'PY'
import json, os, pathlib, re, sys
sys.path.insert(0, 'tools')
import gguf_reader as G
from iq_pack import FLOAT, NATIVE_PLE_KEY, NOT_IN_PACK, form_of, is_expert, check_split
first = pathlib.Path(os.environ['SHARD1']).absolute()
m = re.search(r'-00001-of-(\d{5})\.gguf$', first.name)
assert m and int(m[1]) == 6, 'expected the retained six-shard set'
paths = [first.with_name(first.name[:m.start()] + '-%05d-of-%05d.gguf' % (i, 6)) for i in range(1, 7)]
expert = dense_bound = 0
files = []
layers = set()
for p in paths:
    g = G.GGUFFile(p)
    files.append(g)
    for t in g.tensors:
        n = t.expected_bytes()
        assert n is not None and g.data_start + t.offset + n <= p.stat().st_size, (p, t.name)
        if is_expert(t.name):
            assert t.type_id == 8, (t.name, t.type_name)
            expert += n
            layers.add(int(t.name.split('.')[1]))
        elif t.name not in NOT_IN_PACK:
            form = form_of(t.name)
            native = t.type_name not in FLOAT and (form is None or
                      (t.name == 'blk.1.ple_key.weight' and t.type_name in NATIVE_PLE_KEY))
            if not native:
                # Conservative on-disk upper bound, including alignment and index-kind3 F32 storage.
                dense_bound += (max(n, t.elements * 4) + 255) // 256 * 256
check_split(files)
assert len(layers) == 48 and expert == 128345702400, 'unexpected expert geometry/payload'
print(json.dumps(dict(shards=[dict(name=p.name, bytes=p.stat().st_size) for p in paths],
    input_bytes=sum(p.stat().st_size for p in paths), expert_payload_bytes=expert,
    expert_layers=len(layers), dense_bin_upper_bound_bytes=dense_bound), indent=2))
PY
```

Use the resulting exact source bytes and conservative dense bound to request a **new** directory on the
verified 03 model filesystem. Budget `source_bytes + dense_bound + 2 GiB` for pack metadata, tokenizer,
logs and small temporaries, plus at least 10% free-filesystem reserve; build space is additional. This is
an allocation bound, not a measured dense.bin size. Do not stage a second archive/model copy on root.

After separate asset approval, transfer the six named verified members through the controller's
existing approved copying path; keep the original files on 05/01. Where no approved copying command is
recorded, use existing controller access with `scp -3` (two controller-mediated SSH connections) rather
than creating 03→05 trust. Resolve the real source/destination paths from the inventory, preserve
basenames, and scope the command to those six members; no directory-wide recursive sweep. For example,
from the controller, with both directories explicitly set and approved:

```sh
: "${Q8_SOURCE_DIR:?Existing approved source directory on inferswarm05}"
: "${Q8_DEST_DIR:?New approved directory under the verified /srv/models filesystem on inferswarm03}"
# MEMBERS is an explicit six-line file of basenames from the verified manifest; no wildcards.
test "$(wc -l < MEMBERS)" -eq 6
while IFS= read -r member; do
  case "$member" in */*|*..*|'') exit 2;; esac
  scp -3 "inferswarm05:$Q8_SOURCE_DIR/$member" "inferswarm03:$Q8_DEST_DIR/$member" || exit
 done < MEMBERS
```

A reused approved streaming copy is preferable to storing a controller copy. No new permanent mount,
firewall rule, key distribution or trust change is proposed. Link 04/other hosts are not participants.

On 03, with approved source shard and fresh pack directory:

```sh
: "${SHARD1:?First copied and verified Q8 shard on inferswarm03}"
: "${PACK:?New approved pack directory on the model filesystem}"
test ! -e "$PACK"
/usr/bin/time -v python3 tools/iq_pack.py --gguf "$SHARD1" --out "$PACK" > pack-prepare.log 2>&1
# If preflight established that dense rounding is required, approval must name --compat-bf16;
# include that flag in the single preparation command above and retain conversions.json.
test ! -e "$PACK/experts.bin"
awk '!/^#/ && NF {if ($2 != 8 || $3 != 8) bad=1; n++} END {exit bad || n != 48}' "$PACK/native_experts.txt"
stat -c '%n %s' "$PACK/dense.bin" "$PACK/index.txt" "$PACK/native_experts.txt" data/expert-profile.bin
sha256sum data/expert-profile.bin "$PACK/index.txt" "$PACK/native_experts.txt" "$PACK/conversions.json"
```

Time estimates are arithmetic, **not measured throughput**:175.6 GiB at 100 MiB/s is about 30 min; at 250 MiB/s,
12 min. Allow up to 2 h for one transfer+verification+pack window, stop on failure or limit. Pack preparation
reads selected non-expert tensor bytes and writes dense.bin; it does not rewrite 119.5 GiB of experts.
Estimate its I/O time as `(selected_read_bytes + dense_bytes) / measured_disk_bytes_per_second`, plus
conversion/tokenizer CPU time. Exact dense bytes and duration cannot be supplied without these source
headers/data; retain `time -v`, output sizes and conversion records. A full extra experts.bin would add
119.5 GiB writes and is deliberately excluded. Source identity checks are separate from throughput.

**Minimal later asset approval:** read-only inventory/header checks on 05/01 and 03; one controller-mediated
copy of the six existing Q8 members (at most the verified source sum, approximately 175.6 GiB) into one new 03
model directory; one native reference pack without experts.bin, bounded by the header estimate above,
with dense compatibility conversion only if explicitly named; destination verification and retained
logs;2 h wall cap. No full-model execution, model download, requantization or fleet-wide asset work.

## 4. Memory and placement for a separately approved model window

Each expert is 5222400 B =4.98046875 MiB. All 24576 experts total 128345702400 B =119.53125 GiB.
For **actual distinct disjoint GPU cache slots** `G0` and `G1`, the worker arena is exactly
`(24576 - G0 - G1) * 5222400 B`. Use actual startup/cache manifests and the frozen complement, not requested
slots or assumed cache capacity. For illustration only:2200 cached experts leave 108.8310 GiB;
3400 leave 102.9945 GiB. Neither is a measured placement on the two 3060s.

Read `MemTotal` and `MemAvailable` from `/proc/meminfo` on 05, and any applicable cgroup memory cap.
“128 GB” and 128 GiB differ. Require the assigned arena **plus a chosen 16 GiB reserve** within currently
available RAM/cgroup headroom. The reserve covers worker threads, metadata, activations, staging and
host activity. `--memory-budget-mib` caps the arena, not total RSS: set it to the computed arena rounded
up toMiB only when the additional reserve fits. If it fails, report the exact shortfall and cache
requirement; do not conclude “119.5 cannot fit 128” or enable swap silently.

On 03, both GPU UUIDs/free VRAM, dense/model geometry, context/KV buffers and cache reserves must fit.
Use a budgeted **positive** `PRIMARY_SLOTS` value for `--expert-cache` and
`--expert-cache-device1 auto` with the shipped profile, then record the actual frozen caches and
complement at startup. Primary `--expert-cache auto` is currently refused by the remote-mode guard
(the parser's -1 auto sentinel reaches that guard before auto sizing); do not use it in this trial.
Choose `PRIMARY_SLOTS` from the approved dense/KV/prefill/context VRAM budget, with the normal reserve,
and stop if that budget cannot be established before launch. There is no remote-worker **plan-only startup command** in
this candidate: the partition print is followed immediately by setup upload. Do not start the full model
just to size it under a test-only or asset-preparation approval. For a later runtime approval, bound the
worker arena in advance using available RAM and stop on typed over-budget refusal. No retries or
placement sweep are implied by a refusal.

## 5. Later owned forward and full-model request (separate authorization)

Use existing controller access. The worker remains 127.0.0.1-only. One controller-owned local forward
from 05 and one reverse forward to 03 avoid 03→05 SSH trust. Port 7431 must be unused on the controller and
both hosts. Run these in a controller shell with an owned private log directory, keep PIDs, and trap
cleanup; these commands do not run during test-only retest or asset preparation:

```sh
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -N \
  -L 127.0.0.1:7431:127.0.0.1:7431 inferswarm05 > forward05.log 2>&1 &
forward05=$!
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -N \
  -R 127.0.0.1:7431:127.0.0.1:7431 inferswarm03 > forward03.log 2>&1 &
forward03=$!
trap 'kill "$forward03" "$forward05" 2>/dev/null; wait "$forward03" "$forward05" 2>/dev/null' EXIT INT TERM
```

Before launch record `lscpu`, RAM/cgroup limits, `nvidia-smi` UUID/free VRAM for both 3060s, `nvcc --version`,
source/binary/profile identities, disk receipts, environment and owned ports. Keep the controller alive
for both forwards. On 05, use the corrected worker binary and computed arena limit:

```sh
: "${BUILD05:?Corrected worker build on05}"
: "${ARENA_MIB:?Computed complement arena limit, with separate headroom}"
timeout --signal=TERM --kill-after=15s 7200 "$BUILD05/strata-expert-worker" \
  --port 7431 --memory-budget-mib "$ARENA_MIB" --deadline-ms 10000 > worker.log 2>&1
```

Keep it in an owned foreground SSH session or record the PID of its owned wrapper. Setup can take many
minutes while individual 64 KiB chunks remain within their I/O budgets.103 GiB at 100 MiB/s is at least 17.6 min,
plus digest, copy and controller-forward overhead. Upload is stop-and-wait: the 3400-cache /
102.9945 GiB illustration needs 1694080
64 KiB upload/ACK exchanges. At 1 ms per exchange that adds about 28.2 min; at 5 ms,141 min, before bandwidth
and hashing costs. These are estimates, not measured RTTs. The example runtime wrappers cap the whole
window at 2 h; approve it only if the actual path estimate plus reserve fits. Otherwise report the concrete
setup-time blocker before running; do not raise frame deadlines blindly to hide idle stalls.

The real engine accepts **GGUF shard 1** for `--native` and a distinct directory for `--pack`; it serves
raw token commands over stdin/stdout, not an HTTP text API. A minimal fixed feasibility workload can use
`Tokenizer.from_gguf(...).encode('Hello')`, require 1..64 token IDs, and freeze `GEN 4 <comma-separated IDs>`.
This is a raw text continuation check, not a chat-quality benchmark. Native serve requires
`--spec 2` or larger and positive `--prefill`; use `--spec 2 --suffix-draft 0 --lookup-chain 0` and no
`--mtp`. MTP is optional in this pinned source, and without either drafter decode verifies one token
per round. No separate MTP artifact is required. `--prefill 64` satisfies serve initialization; the
remote short-read guard still rejects any request that actually needs the batched-prefill path. Use the exact same flags/IDs for any feasible local control.

```sh
: "${BUILD03:?Corrected CUDA engine build on03}"
: "${PRIMARY_SLOTS:?Budgeted positive CUDA0 cache count from the approved placement}"
test "$PRIMARY_SLOTS" -gt 0
: "${SHARD1:?Verified first Q8 member on03}"
: "${PACK:?Prepared matching native reference pack on03}"
export SHARD1
python3 - <<'PY' > trial-input.txt
import os, sys
sys.path.insert(0, 'tools')
from strata_tokenizer import Tokenizer
ids = Tokenizer.from_gguf(os.environ['SHARD1']).encode('Hello')
assert 1 <= len(ids) <= 64
print('GEN 4 ' + ','.join(map(str, ids)))
print('QUIT')
PY
STRATA_ROUTE_TAIL_SKIP=0 STRATA_LOOKAHEAD=0 STRATA_LOOKAHEAD_STATS=0 STRATA_FS_SLOTS=0 \
  timeout --signal=TERM --kill-after=15s 7200 "$BUILD03/strata" generate --serve \
  --native "$SHARD1" --pack "$PACK" --gpu 0,1 --mmap-experts \
  --remote-cpu-experts 127.0.0.1:7431 --remote-cpu-deadline-ms 10000 \
  --expert-cache "$PRIMARY_SLOTS" --expert-cache-per-layer --expert-profile data/expert-profile.bin \
  --expert-cache-device1 auto --remote-expert-opt \
  --adapt-every 0 --adapt-swaps 0 --pcie-frac 0 --short-read 64 --prefill 64 \
  --spec 2 --suffix-draft 0 --lookup-chain 0 --max-context 1024 < trial-input.txt > engine.out 2> engine.log
```

The request must complete with finite output; both GPU tiers and worker requests/service time must be
nonzero in the counters, independently of cache residency. Report failure if this fixed prompt does
not exercise a tier; do not keep trying prompts to obtain participation. Record setup wall time/bytes,
client request wait, worker CPU service, inference bytes, RAM/RSS/page cache/VRAM peaks and source-read
deltas separately. No client wait time is GPU kernel time. A matching local-control model run, a deliberate
worker-stop failure probe and any output-token comparison each need to be named in that later runtime
approval. Mock fixtures establish neither hardware parity nor full-model quality.

Stop and clean up only owned processes/tunnels on success, error or timeout. Preserve first failures,
all logs, exit codes, full source SHA and cache/model/profile identities. Return the test-only result
first; asset-preparation readiness and still-unrun hardware checks belong in separate fields.
