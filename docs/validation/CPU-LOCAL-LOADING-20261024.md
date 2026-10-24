# CPU local loading: default-selection handoff

Branch: `feature/cpu-local-load-threads`, [PR #7](https://github.com/Zutfen-LLC/Strata/pull/7).
Implementation validated by Hermes: `723c5a45cd0fe289cb57c7c3d1e504c068bd0cf8`, based on merged main
`fbd3bfd1e710c277b818ff37417ba6ee313a5621`. Use the full follow-up commit supplied with this handoff as
`CANDIDATE`; the commit containing this file pins its source. The follow-up changes the loading default
from one to four, help/docs and default-selection checks. Explicit 1/2/4 overrides remain. Loading
stays separate from inference `--workers`. Loader algorithms, protocol and residency are unchanged.
[PR #6](https://github.com/Zutfen-LLC/Strata/pull/6) remains separate and is not included here.

All builds, executable tests and inference are **NOT RUN locally** on Geekom/Valinor. Supplied remote
results below apply to `723c5a45`, not to execution of this follow-up. No fleet SSH, installation, host
settings change, merge or deployment was performed here.

## Supplied Hermes receipt

The parent supplied this summary; its raw report has not been transferred or read locally:
`/home/hermes/strata-pr7-loadcmp-723c5a45-20261024T193318/REPORT.txt` on 02.

- On 05: CPU build, 11 CTests and `worker_control_test` passed.
- On 03: native-ON and native-OFF CUDA `strata` linked.
- On post-swap 02: eight 32 GB DIMMs, 251.5 GiB usable, 2133 MT/s. The approved experiment was shortened
  to exactly 1 / 2 / 4 loading lanes. Engine-to-READY was 206.8 s / 114.6 s / 68.5 s, completing the same
  116,135,731,200-byte arena with zero inference. T4 observed startup was 66.9% lower than T1. The owner
  authorized four as the default on this evidence; it is not a speed guarantee for other hardware.
- T4 process counters: `read_bytes=0`, `rchar=116135731200`. No device reads were charged to that
  process; do not present this as cold-storage throughput. T1/T2 cache residency was inferred only.
- Per-TID captures failed. Summed lane read/hash times overlap wall time and do not prove that the
  lanes ran on distinct physical cores. Wrapper failures were preserved; no successful load repeated.
- The final T4 worker remains resident on 02. Preserve it and its useful weights. 05 has 94.2 GiB
  usable and must never receive this full arena load; small synthetic fixtures can still run there.

The earlier [implementation handoff](https://github.com/Zutfen-LLC/Strata/blob/723c5a45cd0fe289cb57c7c3d1e504c068bd0cf8/docs/validation/CPU-LOCAL-LOADING-20261024.md)
is historical. Its proposed fourth, repeated T1 arm was not performed and must not run automatically.
Do not repeat the full-model comparison to validate a default literal. The default-selection follow-up
does not require replacing the resident worker, reloading weights or a fresh inference campaign.

## Explicit configuration, fixed driver

Use the retained Hermes driver's existing config. Keep the driver unchanged; edit its existing fields
for the exact candidate, source/cache/output paths, target selection and ordered checks. Do not invent
a new config schema, harness or supervisor here. Record the config path/hash alongside the driver
path/hash and receipt. A missing config binding is a blocker, not permission to reconstruct a runner.

The explicit selection for this follow-up is:

| Config choice | Required value |
| --- | --- |
| Host | 05, synthetic fixtures only |
| Source | Exact `CANDIDATE`, clean dedicated checkout bound to retained cache |
| Build | `strata-expert-worker`, `worker_local_test`, one job; retained toolchain/dependencies |
| Check 1 | `worker_local_binary_test` once, omitted loading flag |
| Check 2 | `worker_loader_binary_1` once, explicit `--local-load-threads 1` |
| Automatic retries/repeats | Disabled; stop dependent work on failure |
| Full-model/inference campaign | Disabled; leave 02's resident worker untouched |

This table specifies values for the existing config, not a new parser or execution loop. The existing
explicit 2/4 CTest cases remain available; do not append them or a serial repeat automatically. Their
implementation was already in the supplied validation receipt.

## Minimal remote checks

Recover `SRC`, `BUILD`, `BIN`, `CANDIDATE` from that config. Preserve old ELFs/hashes and logs. Require
the dedicated source to be clean, update it through the existing authorized path, and require
`CMAKE_HOME_DIRECTORY` to equal `SRC`. Keep native flags, generator, ISA options and dependencies from
the retained cache. Do not retarget it, clean dependencies, install packages or change host settings.
Use a new candidate-specific output directory and the existing resource/timeout guards. For example,
with those exact nonempty bindings already verified:

```sh
set -euo pipefail
: "${SRC:?source path from existing driver config}"
: "${BUILD:?retained cache path from existing driver config}"
: "${BIN:?new output path from existing driver config}"
: "${CANDIDATE:?full follow-up SHA from existing driver config}"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
test "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$BUILD/CMakeCache.txt")" = "$SRC"
test ! -e "$BIN"
mkdir -m 700 "$BIN"
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN"
cmake --build "$BUILD" --parallel 1 --target strata-expert-worker worker_local_test
ctest --test-dir "$BUILD" -V -j1 -R '^worker_local_binary_test$'
ctest --test-dir "$BUILD" -V -j1 -R '^worker_loader_binary_1$'
```

The default case asserts the service option defaults to four, the real CLI reports four loading lanes,
and the independent inference pool remains explicitly two workers. The override case asserts the
serial path is selected with the same two inference workers. Both retain the existing small four-expert
READY, zero-network-weight, bitwise native parity and shutdown checks. These are synthetic fixtures,
not full-model loading. Do not run a broad suite, CUDA/full-model rerun or settings sweep for this delta.

Keep exact source/ELF hashes, config/driver identities, command outputs and exit status in the flat
report, with passed/failed/NOT RUN separate. The follow-up default checks remain pending until Hermes
runs them. Existing failure coverage limits remain: active multilane service stop/disconnect/deadline
and partial thread-launch failure are not directly injected; callback cancellation/reaping is tested
at the loader seam. Wider HIP/SYCL/Windows/Metal coverage is unestablished. Stop before merge or deployment.
