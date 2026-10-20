# Hermes controller recipe: worker-local candidate

Use this recipe from the dedicated controller with its existing access to **inferswarm03** and
**inferswarm05**. Do not add host trust, install packages, rebuild old llama.cpp workers, change drivers,
reset unrelated checkouts or run broad CI. Keep the first failure and stop dependent work.
Use fail-fast Bash (`set -euo pipefail`) and save stdout, stderr, exit codes and full source SHA.
These are commands for Hermes; the coding task did not connect to either host or perform this trial.

## 1. Fetch and incrementally build the existing dedicated checkouts

On the controller, set `CANDIDATE` to the full published SHA in the handoff. Verify:

```sh
: "${CANDIDATE:?Full 40-character published candidate SHA}"
test "${#CANDIDATE}" -eq 40
test "$(git ls-remote git@github.com:Zutfen-LLC/Strata.git refs/heads/inferswarm/worker-local-experts | cut -f1)" = "$CANDIDATE"
```

On each host, through the controller's existing SSH connection, supply `CANDIDATE`, `SRC` and `BUILD`.
`SRC` is its existing dedicated clean trial checkout; `BUILD` is its existing absolute build directory.
Recover them from the retained build receipt or `CMAKE_HOME_DIRECTORY`. Do not use Geekom paths.
Preserve prior logs and confirm this is the dedicated trial checkout before switching it:

```sh
: "${SRC:?Existing dedicated trial checkout on this host}"
: "${BUILD:?Its existing configured build directory}"
: "${CANDIDATE:?Same full published SHA}"
cd "$SRC"
test -z "$(git status --porcelain)"
test "$(git remote get-url origin)" = git@github.com:Zutfen-LLC/Strata.git
# An HTTPS origin for the same Zutfen-LLC/Strata fork is also valid; verify it explicitly if used.
git fetch origin inferswarm/worker-local-experts
test "$(git rev-parse FETCH_HEAD)" = "$CANDIDATE"
git merge-base --is-ancestor 7a590706f20bc976d5bf094be3dfd0c461cf8869 "$CANDIDATE"
git switch --detach "$CANDIDATE"
```

On **inferswarm05**, preserve its successful Makefiles generator, CPU ISA and pinned ggml settings:

```sh
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF
cmake --build "$BUILD" --target strata-expert-worker strata-worker-manifest \
  worker_local_test expert_worker_test expert_worker_e2e_test expert_tier_test worker_socket_test -j2
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^(worker_local_test|worker_local_binary_test|expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_socket_test)$'
```

On **inferswarm03**, preserve the successful CUDA sm_86 Ninja build, CUDA 13.1.115/GCC 14.2.0 and
the pinned ggml source/cache. Confirm `CMAKE_CUDA_ARCHITECTURES` is still 86. Do not clean the build:

```sh
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$BUILD" --target strata strata-expert-worker worker_local_test worker_dispatch_test -j2
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^(worker_local_test|worker_local_binary_test|worker_dispatch_test|worker_dispatch_host_zero_test)$'
STRATA_VERIFY_DEVICE_PLAN=1 timeout 120 "$BUILD/worker_dispatch_test"
```

These incremental builds and small loopback fixture checks need no model read. The binary smoke makes
four 5,222,400-byte fixture experts, uses the real local-source CLI, and requires zero network weights.
Do this first. A compiler failure in generate.cpp blocks the full-model trial.

## 2. One bounded manifest preparation, then reuse

Hermes supplies the actual existing shard paths. On inferswarm03 the completed reference pack is
`/srv/models/strata-q8-38bb39ee97821de2c9009abb7e93950eec396e66/native-pack/`.
On inferswarm05 the model is below the existing `/srv/models/issue300-qwen38-q8-...` directory.
Do not infer its suffix or copy/download the model again. Use the existing verified six-shard copy
receipts (total **188,225,033,248 B**), inventory and native pack receipt. The model has 48 layers,
512 experts/layer and uniform Q8_0/Q8_0. Each expert is **5,222,400 B**. In this preparation, all expert
slices total **128,345,702,400 B**, less than the whole shard sum. Do not hash every full shard again.
Stop if the verified identity receipts are missing; names/sizes alone do not establish a trusted model.

Through the controller, copy only the exact `native_experts.txt` from the 03 pack into a new dedicated
metadata directory on 05 (`PREP05`, supplied by Hermes). Keep its SHA-256 receipt. Preserve all v4 shard
columns, including split-role rows 17 and 35. Keep the metadata file byte-identical to the 03 index.
No `dense.bin` or expert data copy is needed. If an already trusted matching local manifest exists,
verify its retained SHA-256 and reuse it. Do not regenerate it for each build.

On inferswarm05, set `SHARD05` to its existing verified shard 1 and `PREP05` to that metadata directory.
Use an owned log directory outside the clean source checkout. The first preparation has a 20-minute
budget and one blob of weight staging. It reads the original slices, writes only a small manifest,
and stops on signal or budget expiry. It does not allocate the full worker partition.

```sh
: "${SHARD05:?Hermes-supplied verified shard 1 on inferswarm05}"
: "${PREP05:?Dedicated metadata directory containing the exact native_experts.txt from 03}"
: "${RUN05:?Owned absolute log directory on inferswarm05}"
ROOT05=$(dirname -- "$SHARD05")
test ! -e "$PREP05/local.manifest"
/usr/bin/time -v timeout --signal=TERM --kill-after=15s 1215 \
  "$BUILD/strata-worker-manifest" "$PREP05" "$SHARD05" "$PREP05/local.manifest" \
  48 512 1200 --prepare > "$RUN05/manifest.out" 2> "$RUN05/manifest.log"
sha256sum "$PREP05/native_experts.txt" "$PREP05/local.manifest" > "$RUN05/manifest.sha256"
```

The output includes the complete original-slice indexing and per-expert/role digests. Through the
controller, transfer `local.manifest` and its SHA-256 receipt to an owned metadata directory on 03.
Verify the **manifest SHA-256 value** against the receipt on both hosts. The receipt's 05 absolute
filename is not a 03 path; use `sha256sum` on the 03 destination and compare the value. Retain the
source index SHA-256 separately. No Geekom file is involved. Freeze the assets until the trial ends.

Report preparation wall time, expert bytes read, hash cost included in preparation, output bytes and
exit code separately from every subsequent worker setup. This one-time full expert scan is deliberate.
Normal iterations reuse the trusted small manifest and scan only the worker's assigned bytes on 05.

## 3. Bound the full-partition feasibility window before launch

Use actual disjoint GPU cache occupancy, not requested cache counts. For `G` distinct GPU-owned slots,
the worker arena is `(24576-G)*5222400` bytes. The existing frozen-complement startup calculation remains
in force. There is no plan-only full engine startup command. Bound the arena before launch; the worker
refuses a complement over its configured budget. Do not launch a full model just to discover a budget
in a fixture-only window. Require the arena plus a **16 GiB reserve** within current `MemAvailable` and
cgroup headroom on 05. `--memory-budget-mib` caps the arena, not total RSS. Do not enable swap.

Record both 3060 UUIDs/free VRAM, RAM/cgroup limits, storage device/mount, source/binary/profile SHAs,
ports and environment. Choose a positive `PRIMARY_SLOTS` from the existing dense/KV/context VRAM budget.
Use `--expert-cache-device1 auto`; primary `--expert-cache auto` is refused by the existing remote guard.
The pack must have no `experts.bin`, and runtime uses `--mmap-experts`. Keep the original fixed placement,
window, quantization, speculation and KV settings in [CPU_WORKER_RECIPE.md](../CPU_WORKER_RECIPE.md).

Use one owned worker process and the controller-owned local/reverse forwards from that recipe:
controller `-L 127.0.0.1:7431:127.0.0.1:7431 inferswarm05`, then controller
`-R 127.0.0.1:7431:127.0.0.1:7431 inferswarm03`. Keep their PIDs and cleanup trap. Use existing trust;
there is no inferswarm03-to-inferswarm05 SSH dependency. Verify port 7431 is free on all three machines.

On inferswarm05, through an owned controller SSH session:

```sh
: "${ARENA_MIB:?Budgeted complement arena limit with separate 16 GiB reserve}"
# SHARD05, ROOT05, PREP05, BUILD and RUN05 are the verified 05 values above.
timeout --signal=TERM --kill-after=15s 1200 "$BUILD/strata-expert-worker" \
  --port 7431 --workers 0 --memory-budget-mib "$ARENA_MIB" --deadline-ms 10000 \
  --local-root "$ROOT05" --local-manifest "$PREP05/local.manifest" --local-setup-ms 600000 \
  > "$RUN05/worker.out" 2> "$RUN05/worker.log"
```

Use all-local mode for this feasibility trial: omit both fallback flags. Any miss stops the trial
instead of starting a 103 GiB network upload. A later explicitly bounded missing-only trial may enable
`--local-upload-fallback` and `--remote-cpu-local-fallback`; the small fixtures already exercise that path.

On inferswarm03, use its existing build and assets, with the transferred manifest (`MANIFEST03`). Prepare
the same fixed short input as the earlier recipe; the actual service takes raw token commands:

```sh
: "${SHARD03:?Hermes-supplied verified shard 1 on inferswarm03}"
: "${MANIFEST03:?Transferred manifest with verified SHA-256}"
: "${PRIMARY_SLOTS:?Positive budgeted CUDA0 cache count}"
: "${RUN03:?Owned absolute log directory on inferswarm03}"
PACK03=/srv/models/strata-q8-38bb39ee97821de2c9009abb7e93950eec396e66/native-pack
export SHARD03
python3 - <<'PY' > "$RUN03/trial-input.txt"
import os, sys
sys.path.insert(0, 'tools')
from strata_tokenizer import Tokenizer
ids = Tokenizer.from_gguf(os.environ['SHARD03']).encode('Hello')
assert 1 <= len(ids) <= 64
print('GEN 4 ' + ','.join(map(str, ids)))
print('QUIT')
PY
STRATA_ROUTE_TAIL_SKIP=0 STRATA_LOOKAHEAD=0 STRATA_LOOKAHEAD_STATS=0 STRATA_FS_SLOTS=0 \
  timeout --signal=TERM --kill-after=15s 1200 "$BUILD/strata" generate --serve \
  --native "$SHARD03" --pack "$PACK03" --gpu 0,1 --mmap-experts \
  --remote-cpu-experts 127.0.0.1:7431 --remote-cpu-deadline-ms 10000 \
  --remote-cpu-local-manifest "$MANIFEST03" --remote-cpu-setup-ms 600000 \
  --expert-cache "$PRIMARY_SLOTS" --expert-cache-per-layer --expert-profile data/expert-profile.bin \
  --expert-cache-device1 auto --remote-expert-opt \
  --adapt-every 0 --adapt-swaps 0 --pcie-frac 0 --short-read 64 --prefill 64 \
  --spec 2 --suffix-draft 0 --lookup-chain 0 --max-context 1024 \
  < "$RUN03/trial-input.txt" > "$RUN03/engine.out" 2> "$RUN03/engine.log"
```

The total local setup cap is ten minutes; the owned full window cap is twenty minutes. This is a
feasibility bound, not a prediction. Do not raise the deadlines after a failure or retry to obtain a
PASS. A local read rate near the user's expected 400 MB/s remains unmeasured until this window.

## 4. Required report and cleanup

Separate (1) incremental builds/fixture checks, (2) one-time manifest preparation, (3) engine dense/GPU
cache startup, (4) manifest validation and worker arena allocation, (5) local reads and verification,
(6) READY, and (7) the short request. The tier's connect/setup phase includes negotiation and worker
arena allocation; local wall includes protocol waits plus read/hash work; worker read/hash counters
separate those CPU/disk phases. Do not call the whole engine launch local-load time.

Require zero network **weight** bytes, local hit count equal to actual assigned count, no misses,
verified READY, and finite output. Protocol metadata still travels. Report worker CPU service and
engine wait separately. Require actual requests/service from the worker and participation from both
GPU tiers on this fixed prompt. If a tier is not exercised, report that limit without a prompt sweep.
Record worker RSS, RAM/cgroup headroom, page cache/source-read deltas and VRAM peaks. The arena byte
calculation is not a measured RSS peak. Keep disk capability estimates separate from observations.

Full-model numerical parity, a local-control run, a worker-stop model probe and any larger workload
are separate checks if included in Hermes's runtime scope. The CPU fixtures do not replace them.
Stop and clean up only the owned engine/worker wrappers and tunnels on completion, error or timeout.
Retain the first failure, exit codes, all logs, published SHA, manifest/index/profile identities and
actual frozen placement. Return the small fixture result first and the still-unrun checks explicitly.
