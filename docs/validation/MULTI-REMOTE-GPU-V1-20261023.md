# MULTI-REMOTE-GPU-V1-20261023

This is a focused remote validation handoff for the additive multi-endpoint
candidate based on main `202542af5128ee6ba1842d615dde4baf97899125`. The immutable
commit containing this file pins the source. `MULTI-REMOTE-GPU-V1-20261023.sha256`
pins the changed files. New code and fixtures are **NOT RUN** at publication.
This handoff prepares execution; it is not permission to launch it. Obtain the
parent's focused fleet-execution approval before running these commands.

No full model, GEN, engine inference, CPU-arena load, production placement
change or retained-cache operation is included. Historical model usage and
NUMERICAL_FAIL remain unchanged. Do not run these commands on Valinor/Geekom.
Do not contact or signal 05's intentionally live CPU worker, monitor or guardian.
Preserve 04's complete 17,176,473,600-byte cache and all unrelated services.

## Bounds and roles

Use the existing authorized controller and temporary SSH relay pattern. Resolve
its role/transport identities from retained private receipts; do not publish
credentials or invent SSH mappings. There is no new listener beyond loopback,
security setting, privilege, dependency installation or transport route here.
If an approved controller-to-02 binding is missing, report that prerequisite;
do not improvise access. The 02 readiness report is supplied environment evidence,
not a Strata test result.

The proposed focused session is at most 1,800 seconds including preparation and
45 seconds reserved for cleanup. Serial incremental build: 900 seconds maximum.
Local fixtures: 45/180/180 seconds. Synthetic remote step: 180 seconds including
worker/relay startup and teardown. Optional cached native-OFF check: 180 seconds,
only if it fits the remainder. No extensions or repeated model requests exist.
Ordinary launcher fixes may be corrected and recorded within these bounds.

Use one build job and modest output (at most 8 MiB per build, 1 MiB per fixture).
Retain elapsed/RSS/exit status. Stop the owned build if RSS exceeds 3 GiB. The
CPU-only group fixture enforces 768 MiB address space, 20 CPU seconds and bounded
files internally. Real CUDA fixtures need CUDA's virtual address space: do not
apply that address-space cap to them. Stop a CUDA fixture/worker if resident host
memory exceeds 2 GiB or its tiny synthetic workload uses over 1 GiB GPU memory
above the recorded prelaunch baseline. Preserve 1 GiB free VRAM reserve and
ordinary ownership/identity-safe TERM/KILL cleanup. Do not change cgroups or
privileges to implement these limits; use existing approved process supervision.

## 1. Incremental build on 03

Resolve `SRC03` and `BUILD03` from the retained accepted 03 build receipt. They
must be its disposable clean source checkout and matching CMake cache (actual
Unix Makefiles, Release, sm86, CUDA 13.1, native ON), not a user working tree.
Resolve `COMMIT` to the full immutable candidate SHA supplied with this handoff.
Create `RUN03` as a new owned 0700 output directory, set `BIN03=$RUN03/bin`, and
record those absolute bindings. Preserve existing ELFs and dependency caches.
Check `CMAKE_HOME_DIRECTORY`, generator, native/CUDA flags and existing donor
ggml path in `CMakeCache.txt`; no clean configure or dependency fetch is allowed.
If the retained cache/source binding is unavailable, report exactly that rather
than constructing a new toolchain campaign.

On 03, using those nonempty resolved variables:

```sh
git -C "$SRC03" status --porcelain
git -C "$SRC03" fetch origin "$COMMIT"
git -C "$SRC03" switch --detach "$COMMIT"
cmake -S "$SRC03" -B "$BUILD03" -DSTRATA_BUILD_TESTS=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN03"
timeout --signal=TERM --kill-after=5s 900s cmake --build "$BUILD03" --parallel 1 --target strata gpu_worker_group_test gpu_worker_dispatch_test strata-cuda-expert-worker
```

Stop before switching if the first command shows unrelated changes. Preserve
the existing cache's toolchain and donor ggml; the configure must not download
anything or alter dependency source. Verify the four outputs are in `BIN03` and
retain their SHA-256 values. Building `strata` checks the actual CLI/integration;
it is not launched against a model. The worker protocol/kernel source is
unchanged; a candidate worker ELF is built only for isolated synthetic testing.

## 2. Focused local gates, then continue automatically on PASS

Run exactly these on 03, serially, with the limits above:

```sh
timeout --signal=TERM --kill-after=5s 45s env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$BIN03/gpu_worker_group_test"
timeout --signal=TERM --kill-after=5s 180s env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "$BIN03/gpu_worker_dispatch_test"
timeout --signal=TERM --kill-after=5s 180s env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "$BIN03/gpu_worker_dispatch_test" --single
```

Require rc 0 and `GPU_WORKER_GROUP_V1 failures=0 endpoints=3 models=0 cuda=0`,
then `MULTI_GPU_DISPATCH_V1 workers=3 failures=0` and `workers=1 failures=0`.
The first uses real SGW1/cache/session handling with controlled executors: three
exchanges enter while CPU work is still in flight, replies are deliberately
released 2/0/1, caller buffers are changed after submission, earlier workers
survive slow later setup/idle, duplicate/model/owner conflicts refuse, and failed
peers drain without retry. The CUDA fixture uses the actual dispatcher, real CPU
service, three real CUDA executors on one card and actual device merge. It also
injects an inference failure after CPU submission and checks no committed remote
sum. Its primary local-GPU rows are emulated to keep it small. This proves the
seam, not a full two-local-plus-three-remote model topology or speedup.

If a retained native-OFF cache exists, resolve `OFF_BUILD03`/`OFF_BIN03` without
changing the native-ON cache. Update only its same disposable source binding,
set its candidate output directory, retain native OFF, and incrementally build
`strata worker_disabled_dispatch_test` serially within 180 seconds. Run:

```sh
timeout 15s "$OFF_BIN03/worker_disabled_dispatch_test"
timeout 30s python3 "$SRC03/tests/worker/worker_disabled_cli_test.py" --engine "$OFF_BIN03/strata"
```

These exercise the new flag and opaque group type without worker dependencies.
If that cache is absent or the check does not fit, report native-OFF execution
unrun; do not start a fresh build or replay unrelated suites.

## 3. Tiny three-host CUDA exchange after local PASS

Continue in the same authorized session; do not make another dummy benchmark
an approval-only gate. Generate plans on 03 (the directory must not yet exist):

```sh
timeout 30s "$BIN03/gpu_worker_group_test" --emit-plans "$RUN03/plans"
```

The generated plans assign one synthetic Q8_0 expert each, 5,222,400 bytes per
worker (15,667,200 total), at H=2560/F=640. They cannot name retained real-model
weights. Copy only these three small plan files and the built worker ELF plus
its already-available runtime libraries through the existing approved mechanism
to new owned fixture roots on 04 and 02. Do not install dependencies. If the ELF
cannot use existing libraries on a host, report that concrete prerequisite.

Use this fixed logical order:

| Index | Host | Physical GPU | Plan | Worker loopback port |
| --- | --- | --- | --- | --- |
| 0 | inferswarm04 | existing RTX 3090 | peer-0.plan | 27432 |
| 1 | inferswarm02 | GPU 0, RTX 3060 Ti 8 GB | peer-1.plan | 27433 |
| 2 | inferswarm02 | GPU 1, RTX 3060 12 GB | peer-2.plan | 27434 |

Confirm these illustrative ports are free first; choose other owned free ports
and record bindings if needed, never stop an existing listener. Keep both 02
GPUs visible so device 0/1 follows the supplied physical order. For each row,
resolve `WORKER`, `PLAN`, `CACHE`, `PORT`, `DEVICE` in its host's owned root. Make
each `CACHE` a separate new 0700 directory, bounded to 32 MiB. Launch under the
existing owner/guardian with an absolute 180-second expiry, recording executable
identity and true parent. The command inside that owner is:

```sh
timeout --signal=TERM --kill-after=5s 180s "$WORKER" --plan "$PLAN" --cache-root "$CACHE" --port "$PORT" --device "$DEVICE" --cache-budget-mib 32 --reserve-mib 1024 --deadline-ms 10000 --setup-ms 30000
```

Bind temporary relays through the existing controller so three distinct numeric
loopback endpoints on 03 (`EP0`, `EP1`, `EP2`) reach the corresponding worker
ports. Use the retained relay construction, not direct public worker binding.
Once all three loopback listeners/relays are identity-verified, run on 03:

```sh
timeout --signal=TERM --kill-after=5s 125s env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$BIN03/gpu_worker_group_test" --peers "$EP0" "$RUN03/plans/peer-0.plan" "$EP1" "$RUN03/plans/peer-1.plan" "$EP2" "$RUN03/plans/peer-2.plan"
```

Require rc 0, `GPU_WORKER_GROUP_REMOTE_V1 peers=3 windows=3`, and
`GPU_WORKER_GROUP_EXTERNAL_V1 failures=0 client_cuda=0 remote_cuda=1 models=0 GEN=0`.
Each peer must have three successful requests/entries, finite nonzero partials,
the expected independent assigned/cache/transfer accounting and no client
failures. The mode validates exact synthetic plan bytes before any upload.
Retain per-peer reports plus aggregate three-window elapsed; summed per-peer
request times overlap and are not an end-to-end wall-time partition. Do not
assert a speedup from three tiny windows. Genuine parallel entry/order/failure
behavior is checked deterministically by the local controlled fixture above.

## 4. Closeout and report

Close only this session's identity-pinned worker/relay processes and listeners.
Confirm their GPU allocations return to recorded baselines. Retain the bounded
fixture logs/plans/counters and remove only owned synthetic caches if desired.
05's resident service and 04's real cache remain intentionally untouched.
EOF-related worker failure counters must be distinguished from failed inference;
do not assert an EOF cause without its log. Report PASS/FAIL/NOT RUN per gate,
actual commands/SHAs, elapsed/RSS/GPU peaks and exact missing prerequisites.

Stop on a real fault after bounded cleanup. Record launcher corrections and failed
attempts truthfully; no hidden extra model requests, new network settings or
full-model follow-up. No merge/deployment is authorized by this handoff. A later
real-model trial needs approved unequal placement plans and its own budget.
