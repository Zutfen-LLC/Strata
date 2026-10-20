# Hermes: validate the native-OFF correction

The worker/residency revision **16d065e23211f2d422c56a3497a8be200c1e1d4d** has
[user-reported CPU/CUDA and full-model validation](evidence/hermes-16d065e-receipt.md).
The later correction head has **not** been configured, built or executed. All commands below are for
Hermes; Geekom remains code edits and static inspection only. Do not operate the old resident worker
or tunnels, change drivers, merge, launch broad CI, or start a network GPU worker.

## Scope and identity gate

The static P2 prediction was that `STRATA_NATIVE_EXPERTS=OFF` leaves shared dispatcher calls to worker
symbols whose libraries CMake omits. Startup also constructed the tier and reporting referenced it.
This correction guards the worker include, ownership queries, requests, joins, construction and reports.
A disabled build refuses all six worker CLI options before model loading or GPU initialization, and
refuses an externally attached worker before dispatch accesses source/activation/device state. The
main preflight runs before family dispatch. Qwen3.6's own parser also refuses actual worker flags after
the existing GGUF metadata probe, because serve-mode unknown-option consumption differs from the main
parser. Its engine construction, model payload loading and GPU setup have not started at refusal.
Native-ON retains the worker path. CUDA and shared HIP sources are affected; HIP is untested. SYCL has
separate migrated sources and is not claimed to contain this feature.

The [independent static review record](evidence/native-off-review.md) records findings and dispositions.
The worker service, client, wire protocol and resident lifecycle are unchanged from tested 16d065e.
The needed checks are an actual native-OFF engine link/refusal check plus native-ON dispatch and small
worker regressions. A new 103 GiB load or full-model trial is not required by this diff.

Set `CANDIDATE` to the exact new 40-character correction SHA in the handoff. On the controller:

```sh
set -euo pipefail
: "${CANDIDATE:?New full correction SHA, not 16d065e}"
BRANCH=inferswarm/persistent-cpu-weight-residency
test "${#CANDIDATE}" -eq 40
test "$(git ls-remote git@github.com:Zutfen-LLC/Strata.git "refs/heads/$BRANCH" | cut -f1)" = "$CANDIDATE"
```

On 03 and 05, use their existing dedicated checkouts, preserve old logs/build receipts, and require a
clean source tree and canonical fork. Fetch only the branch, verify `FETCH_HEAD=$CANDIDATE`, and check
that both 16d065e and main `e88ee39d1d6be6e3364183af3b39b62a7ed6f211` are ancestors before detaching
at the new head. Stop on drift; do not reset or overwrite unrelated work. The pre-correction main range
contains **seven commits**, including five worker prerequisites and two residency commits. Integration
must include that entire range and the additive correction, not just the two residency commits.

Save source SHA, configure/build commands, return codes, binary SHA-256 and CTest names/results under
new owned log directories outside source. Use `-j1`, `CTEST_PARALLEL_LEVEL=1`, `OMP_NUM_THREADS=2`, current
memory/cgroup headroom and outer timeouts. Keep the first failure and stop dependent checks.

## 1. Native-OFF CUDA build in a separate directory on 03

Supply absolute `SRC03`, **known-good native-ON** `ONBUILD03`, new `OFFBUILD03`, and `RUN03`.
Require `OFFBUILD03` to be absent and different from `ONBUILD03`; never toggle the successful ON cache
to OFF. Recover `CXX03`, `NVCC03` and any host compiler/toolchain overrides from the retained ON
`CMakeCache.txt` and receipt. Preserve sm_86 and the existing compiler environment; do not install
or change tools. Check current headroom before the serial compile.

```sh
: "${SRC03:?Dedicated clean correction source, absolute}"
: "${ONBUILD03:?Existing successful native-ON Ninja sm_86 build, absolute}"
: "${OFFBUILD03:?New distinct native-OFF build directory, absolute}"
: "${RUN03:?New owned log directory outside source}"
: "${CXX03:?C++ compiler from the retained ON cache}"
: "${NVCC03:?CUDA compiler from the retained ON cache}"
test "$OFFBUILD03" != "$ONBUILD03"
test ! -e "$OFFBUILD03"
export CMAKE_BUILD_PARALLEL_LEVEL=1 CTEST_PARALLEL_LEVEL=1 OMP_NUM_THREADS=2
# Carry any previously explicit CMAKE_CUDA_HOST_COMPILER/toolchain override into this command too.
cmake -S "$SRC03" -B "$OFFBUILD03" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$CXX03" -DCMAKE_CUDA_COMPILER="$NVCC03" \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF -DSTRATA_NATIVE_EXPERTS=OFF \
  -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_ORCA_Q4KS_MMQ=OFF -DSTRATA_BUILD_TESTS=ON \
  > "$RUN03/off-configure.log" 2>&1
timeout --signal=TERM --kill-after=15s 1800 \
  cmake --build "$OFFBUILD03" --target strata worker_disabled_dispatch_test -j1 \
  > "$RUN03/off-build.log" 2>&1
ctest --test-dir "$OFFBUILD03" -V -j1 -R '^worker_disabled_dispatch_test$' \
  > "$RUN03/off-dispatch.log" 2>&1
python3 "$SRC03/tests/worker/worker_disabled_cli_test.py" --engine "$OFFBUILD03/strata" \
  > "$RUN03/off-cli.log" 2>&1
sha256sum "$OFFBUILD03/strata" "$OFFBUILD03/worker_disabled_dispatch_test" > "$RUN03/off-binaries.sha256"
```

Save each exit status separately even when fail-fast stops the block. A successful static library build
is insufficient: the `strata` executable must link. Record OFF in CMakeCache and confirm the link graph
does not pull `strata_worker_tier` or `strata_expert_worker`. The dispatch test links the real shared
engine without worker libraries and uses an opaque worker pointer and null input/source/pool to check
fail-closed refusal before access; it performs no inference. The CLI test launches the actual OFF
engine with each of the six options, alone and with absent model/pack paths, expecting exit 2 and the
specific `requires STRATA_NATIVE_EXPERTS=ON` diagnostic. It also checks that option-shaped pack,
synthetic token-file, dump and two-argument layer-range values are consumed as values, with normal
help/version still working. The arity-only early scan covers the main and Qwen3.6 parser value options;
review that list whenever either parser adds a value-taking option. A tensor-free synthetic GGUF
header additionally selects Qwen3.6 and checks all six actual worker flags in serve mode after an option
whose arity differs between parsers. No real model files, worker connection or GPU allocation is needed
for these refusal cases.

## 2. Native-ON regressions on 03 and 05

On 03 keep the known-good ON build in place and incrementally configure it against the correction
source using its existing Ninja sm_86 toolchain/ggml cache and options, explicitly keeping native ON.
Do not pass native-OFF or compatibility-test overrides into it.

```sh
cmake -S "$SRC03" -B "$ONBUILD03" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  > "$RUN03/on-configure.log" 2>&1
timeout --signal=TERM --kill-after=15s 1800 \
  cmake --build "$ONBUILD03" --target strata worker_dispatch_test -j1 > "$RUN03/on-build.log" 2>&1
ctest --test-dir "$ONBUILD03" -V -j1 -R '^(worker_dispatch_test|worker_dispatch_host_zero_test)$' \
  > "$RUN03/on-dispatch.log" 2>&1
STRATA_VERIFY_DEVICE_PLAN=1 timeout 120 "$ONBUILD03/worker_dispatch_test" \
  > "$RUN03/on-device-plan.log" 2>&1
sha256sum "$ONBUILD03/strata" "$ONBUILD03/worker_dispatch_test" > "$RUN03/on-binaries.sha256"
```

These small GPU fixtures cover the ownership query used by both planning paths, request post/join and
native reference output. On 05 preserve its existing native-ON CPU-only Makefiles build and pinned
ggml cache. Supply absolute `SRC05`, `ONBUILD05`, `RUN05`; do not point to Geekom paths.

```sh
cmake -S "$SRC05" -B "$ONBUILD05" -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  > "$RUN05/on-configure.log" 2>&1
timeout --signal=TERM --kill-after=15s 1800 \
  cmake --build "$ONBUILD05" --target strata-expert-worker worker_local_test worker_residency_test -j1 \
  > "$RUN05/on-build.log" 2>&1
STRATA_WORKER_FIXTURE_KEEP=1 ctest --test-dir "$ONBUILD05" -V -j1 \
  -R '^(worker_local_binary_test|worker_residency_test)$' > "$RUN05/on-fixtures.log" 2>&1
sha256sum "$ONBUILD05/strata-expert-worker" "$ONBUILD05/worker_residency_test" > "$RUN05/on-binaries.sha256"
```

Use only freshly rebuilt safe fixtures. The older pre-fix binary warning remains in force. The nine
legacy/compatibility suites already have a user-attributed 16d065e receipt; their protocol/runtime
source is unchanged in this correction. Do not describe that old receipt as a new-head PASS. If these
narrow checks expose a failure, preserve it and return to the coding task before expanding validation.
HIP compilation/execution still needs its own environment and is not claimed by the CUDA results.
No SYCL validation or worker feature is claimed here.

## 3. Recover complete identities from small receipts and report

Recover full manifest/index/profile/binary hashes from the retained text/JSON/SHA receipts under
`/home/hermes/strata-val-16d065e`, `/home/hermes/strata-cuda-val-16d065e`, and the retained trial receipts
associated with `/home/zutfen/strata-res16d-trial/strata-res16d-evidence.tar.gz`.
Read only the existing small identity receipts, for example with bounded `rg` searches in the actual
receipt directories. Do not scan GGUF payloads, repeat manifest preparation, hash all shards or copy
model assets. The pasted manifest abbreviation with nonhex `t318` is unusable. If a complete trusted
identity receipt is missing, report exactly which one is missing; do not reconstruct or guess a hash.
The archive SHA-256 supplied by the user is recorded in [the attributed receipt](evidence/hermes-16d065e-receipt.md);
verify it against the retained archive receipt as available and label what was actually read.

Return the exact correction source SHA, unchanged ON cache/toolchain identities, new OFF cache options,
all binary hashes/exit codes, refusal output, each named fixture result and remaining gaps. Keep the
16d065e observations attributed, including 428.4 s versus 0.1 s CPU setup; do not present the unequal
519.5 s/146.2 s whole windows as like-for-like startup or attribute their difference causally.
`107000 + 16384 MiB` is `120.4921875 GiB`. Do not assert prior worker/tunnel liveness or renew/stop them.
Stop before merge. A full-model rerun is a later scope decision if a newly discovered runtime change
requires one; this build-boundary correction does not change resident weight execution.
