# Native CPU-worker LAN and NIC profile: Hermes handoff

**Status: source and static review only. All builds, tests, NIC commands and inference NOT RUN locally.**
No dependencies were installed and no fleet SSH, host settings changes, push, merge or deployment was
performed. Do not build or run tests on Geekom or Valinor. This handoff describes later remote work; it
does not itself authorize changes to a live worker, firewall or NIC.

## Source receipt and scope

Base: `56a4815ef1d7d5e128d6a67ef756b1789d4bfb63`, branch
`feature/cpu-rpc-latency-diagnostics`. The Geekom copy initially had a dangling linked-worktree `.git`
reference. After the user supplied the missing metadata, Git resolved the expected branch and HEAD and
reported a clean tree. Earlier commits were retained. Use the final implementation commit reported with
this handoff as `CANDIDATE`; do not use the temporary patch-preparation snapshot as history.

The changes add CPU-worker numeric binding plus exact source peers and a separate, explicitly invoked
Linux NIC tool. No inference wire, arithmetic, model placement, timer, residency or reconnect algorithm
changes. New cases cover admission before protocol dispatch, resident reconnect, allowed controls and
NIC opt-in/readback/restore. The legacy default control fixture remains in place.

Local static review: `git apply --check` and `git diff --check` completed without errors. Text comparison
against the base confirmed `listen_loopback`, `accept_one`, `connect_numeric_controlled` and
`connect_host` remain source-identical. The wire codecs and arithmetic files are absent from the diff.
Two independent focused reviewers inspected actual source and fixtures after `ed521df9`. Their static
findings led to nonblocking opt-in listeners, finite peer-policy deadlines, host/permanent NIC validation,
snapshot corruption checks and explicit recovery of an unexpected owned mode. The default loopback
helpers remain source-identical. No protocol read or session/residency dispatch precedes peer admission
in the inspected path. Normal time-based idle expiry remains in the original loop. Real NIC behavior, output
parity and backend compilation remain unverified; these source checks are not execution evidence.

User-supplied diagnostic evidence motivates this work; it was not reproduced locally:

- Matched real inference: ITR 3 to 0 improved 6.65 to 7.58 tokens/s, with identical outputs.
- At ITR 0, a temporary plain TCP relay versus SSH forwarding improved 7.59 to 8.73 tokens/s, with
  identical completed outputs. Relay repeats varied about 4.9%. This is diagnostic evidence, **not proof
  of native LAN performance**.
- The relay was dramatically slower without NODELAY on both legs. Retain client NODELAY and explicitly
  select worker `--tcp-nodelay` for the transport comparisons.
- C6-off added no benefit at ITR 0; EEE tuning was not warranted. Neither is part of this change.

## Remote source and regression gates

The commit is local only; no branch or binary was pushed or published. Before a remote runner can use
it, the parent/operator must transfer the refreshed incremental `cpu-worker-lan-nic.bundle` through the
existing authorized file-transfer path. It requires base `56a4815ef1d7d5e128d6a67ef756b1789d4bfb63` in the
remote object store. Verify the transferred bundle hash matches the sender, then verify/import it and
create a new detached candidate checkout. Do not copy a linked-worktree `.git` pointer alone. A missing
base object, unavailable transfer path or inability to allocate a separate checkout is a blocker; do
not substitute an upstream revision or assume the local-only commit is remotely fetchable.

For example, on each approved remote build host, with operator-supplied paths and final commit:

```sh
set -eu
: "${REPO:?existing remote Strata object store with the required base}"
: "${BUNDLE:?transferred incremental bundle}"
: "${SRC:?new candidate checkout path that does not exist}"
: "${CANDIDATE:?exact final commit from the source handoff}"
BASE=56a4815ef1d7d5e128d6a67ef756b1789d4bfb63
git -C "$REPO" cat-file -e "$BASE^{commit}"
git -C "$REPO" bundle verify "$BUNDLE"
git -C "$REPO" fetch "$BUNDLE" HEAD
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = "$CANDIDATE"
test ! -e "$SRC"
git -C "$REPO" worktree add --detach "$SRC" "$CANDIDATE"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
git -C "$SRC" merge-base --is-ancestor "$BASE" HEAD
test -z "$(git -C "$SRC" status --porcelain)"
git -C "$SRC" diff --check "$BASE" HEAD
```

Preserve the exact candidate SHA with the receipt. The source import writes Git objects and new source
files only. Do not run from or overwrite a live worker checkout or binary. No remote import, build,
installation, binary replacement or publishing step has been performed by this local handoff.

Use CPUworker05's retained compatible toolchain/dependency sources and Unix Makefiles generator, but a
new candidate build directory. A CMake cache is tied to its `CMAKE_HOME_DIRECTORY`; passing a new source
path to an old build cache does not retarget it. Leave retained build directories untouched. Before
configuration, the operator must supply a reviewed CMake initial-cache file (`SETTINGS05`) carrying the
retained compiler/toolchain, CPU ISA and build options, plus the retained pinned llama.cpp source path
(`GGML_SRC05`) and its revision from the prior receipt (`GGML_REV05`). Do not copy generated/internal cache
paths into that settings file. The explicit local ggml source option below avoids downloads. Missing
settings, compatible toolchain, dependency source or recorded revision is a blocker, not a reason to
install or refresh dependencies. Use the existing split: CPU worker and fixtures on CPUworker05, engine
CUDA build on host03. No build on engine02, Geekom or Valinor.

```sh
set -eu
: "${SRC:?clean candidate source}"
: "${CANDIDATE:?exact final source commit}"
: "${RETAINED_BUILD05:?existing CPUworker05 reference cache, read only}"
: "${SETTINGS05:?operator-reviewed initial cache with retained toolchain/options}"
: "${GGML_SRC05:?retained pinned llama.cpp source with ggml subdirectory}"
: "${GGML_REV05:?exact dependency revision from retained receipt}"
: "${BUILD05:?new CPUworker05 candidate build directory}"
: "${BIN05:?new CPUworker05 candidate output directory}"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
test -f "$SETTINGS05"
test -f "$GGML_SRC05/ggml/CMakeLists.txt"
test "$(git -C "$GGML_SRC05" rev-parse HEAD)" = "$GGML_REV05"
test -z "$(git -C "$GGML_SRC05" status --porcelain)"
test "$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$RETAINED_BUILD05/CMakeCache.txt")" = "Unix Makefiles"
test ! -e "$BUILD05"
test ! -e "$BIN05"
mkdir -m 700 "$BIN05"
cmake -C "$SETTINGS05" -S "$SRC" -B "$BUILD05" -G "Unix Makefiles" \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_GGML_DIR="$GGML_SRC05" -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN05"
cmake --build "$BUILD05" --parallel 1 --target strata-expert-worker worker_lan_test \
  worker_socket_test expert_worker_test expert_worker_e2e_test expert_tier_test \
  worker_local_test worker_residency_test worker_service_residency_test worker_compat_test rpc_timing_test
ctest --test-dir "$BUILD05" -V -j1 -R \
  '^(worker_lan_test|worker_socket_test|expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_local_test|worker_local_binary_test|worker_residency_test|worker_service_residency_test|worker_compat_test|rpc_timing_test)$'
python "$SRC/tests/worker/worker_nic_test.py"
python "$SRC/tests/worker/worker_control_test.py" \
  --worker "$BIN05/strata-expert-worker" --control "$SRC/tools/worker_control.py"
python "$SRC/tests/worker/worker_control_test.py" --restricted-listener \
  --worker "$BIN05/strata-expert-worker" --control "$SRC/tools/worker_control.py"
```

These fixtures use synthetic weights, loopback source aliases or a fake NIC backend. They do not prove
physical-LAN admission, real NIC support or throughput. Preserve logs and failures. The NIC fixtures
must issue no real ethtool writes. The source policy checks wildcard/DNS/CIDR refusal; `worker_lan_test`
checks empty, partial, shutdown, unload, setup and inference frames from a denied 127/8 source before
protocol handling, unchanged resident counters, fresh allowed reconnect and output parity.

Build CUDA `strata` separately on host03 after the same source import. As above, supply reviewed retained
toolchain/options as `SETTINGS03` and use a fresh build directory with the existing pinned dependency
source. Preserve CUDA architecture/compiler options from the retained receipt; no new dependency,
architecture or toolchain pin is asserted by this handoff:

```sh
set -eu
: "${SRC03:?new imported clean candidate checkout on host03}"
: "${CANDIDATE:?exact final source commit}"
: "${RETAINED_BUILD03:?existing host03 reference cache, read only}"
: "${SETTINGS03:?operator-reviewed initial cache with retained CUDA toolchain/options}"
: "${GGML_SRC03:?retained pinned llama.cpp source with ggml subdirectory}"
: "${GGML_REV03:?exact dependency revision from retained receipt}"
: "${BUILD03:?new candidate CUDA build directory on host03}"
: "${BIN03:?new candidate output directory on host03}"
test "$(git -C "$SRC03" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC03" status --porcelain)"
test -f "$SETTINGS03"
test -f "$GGML_SRC03/ggml/CMakeLists.txt"
test "$(git -C "$GGML_SRC03" rev-parse HEAD)" = "$GGML_REV03"
test -z "$(git -C "$GGML_SRC03" status --porcelain)"
test "$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$RETAINED_BUILD03/CMakeCache.txt")" = "Unix Makefiles"
test ! -e "$BUILD03"
test ! -e "$BIN03"
mkdir -m 700 "$BIN03"
cmake -C "$SETTINGS03" -S "$SRC03" -B "$BUILD03" -G "Unix Makefiles" \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DSTRATA_GGML_DIR="$GGML_SRC03" -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$BIN03"
cmake --build "$BUILD03" --parallel 1 --target strata
```

Since `wire_sock.cpp` is shared, build GPU-worker protocol consumers too where enabled and build affected
HIP/SYCL/Windows consumers on their supported
builders before requesting review. Mac and Windows NIC tuning are unsupported. Report missing backends
as NOT RUN; a CPU-only fixture does not establish backend or platform compatibility.

## Operator-controlled physical checks

Use [the operator guide](README.md) for exact command syntax. Resolve the intended physical NIC,
destination IP, source IP and port. Capture the existing interface/driver/coalesce discovery. Inspect
the firewall; obtain operator authorization for any needed rule. The code makes no firewall changes.
Use one selected local address, exact peers, and an operator-approved unused port. No wildcard bind.

First run a model-free native worker with a deliberately mismatched exact peer and attempt status from
the real engine host. It must close without an ACK while the worker remains available. Then select the
actual engine source and verify allowed status/unload/stop against only this disposable empty worker.
Capture listener/source evidence and clean up this owned worker. Do not send unload/stop to a live
resident worker or make a synthetic replay result stand in for this native service check.

LAN mode has no simultaneous 127.0.0.1 listener. For worker-local controls, add the worker's actual source
IP to the exact peer list and run `worker_control.py status --host "$WORKER_IP" --port "$WORKER_PORT"`.
Prove that selection on the disposable worker. The filtered-loopback CLI fixture proves parsing and
control compatibility, not physical-LAN source selection. Adapt the SSH arm's forwarding destination to
the actual listener; a forward to 127.0.0.1 cannot reach a worker bound only to its LAN address. Admit the
forward's actual source explicitly, and record route-specific bind/peer selections. The authorized
parent/operator must also transfer the host03 engine executable to its execution host and verify its
hash before inference; no binary is assumed published or installed by these source/build steps.

For the validated e1000e NIC, save one original snapshot with its corruption checksum. Record `rx-usecs 3` as
dynamic-conservative mode, never "3 microseconds". Explicitly apply low latency, require full readback
with `rx-usecs 0` and all other coalesce fields unchanged, and explicitly restore to the exact saved
map. A failure or uncertain write requires discovery before any retry; explicit restore can recover an
unexpected numeric owned mode only when unrelated settings and exact host/NIC identity still match.
Do not tune another driver by analogy. Only after successful apply/readback/restore and matched workload validation should an
operator add the exact identity and results reference to the test configuration. Release stays off.

## Matched real inference

Worker replacement releases its arena. Before replacing a worker, name its exact process/service,
obtain the existing operator approval, capture counters and retain the previous executable. Recreate
and verify the same CPU partition and residency. Keep unrelated services, model assets and routes
untouched. NIC changes and worker replacement must be explicit operator steps outside inference startup.

Run serial matched arms using candidate binaries, the same model, prompt set, seed, output cap, CPU/GPU
placement, partition, pool workers, residency and client/accepted NODELAY settings. Keep timing flags
off for speed arms. Use a separate warmup and at least three completed requests per arm, with repeated
arms to observe variation:

1. SSH route, original NIC snapshot setting (record whether it is 3 or already 0).
2. SSH route, explicit NIC mode 0 (isolates NIC setting when the original was 3).
3. Native LAN route, mode 0 (isolates transport against arm 2).
4. Native LAN route, exact original NIC setting (completes the route/setting comparison).
5. Repeat arm 3 and a reference arm to assess the roughly 4.9% earlier relay variation.

If the original setting is already 0, do not invent a "3" baseline or overwrite the original snapshot.
Any separately authorized mode-3 arm needs its own explicit operator setting and readback record. Do
not change mode codes through a fabricated snapshot. An optional timing-on attribution arm must be
labeled separately and must not be the uninstrumented speed result.

Require identical completed outputs and routing, zero refusal/timeout regressions, correct resident
reuse and expected counts. Compare end-to-end tokens/s and same-host client exchange distributions
between matched arms. Do not subtract cross-host clocks or sum overlapping phase times. Retain individual
results, including failures and slower arms, then restore the exact NIC snapshot and confirm readback.
Report passed, failed and NOT RUN separately, plus candidate/binary hashes, route and mode per arm, any
remaining backend gaps and whether operator restore completed. No result authorizes a default change,
merge or deployment. Keep the work focused on this comparison.
