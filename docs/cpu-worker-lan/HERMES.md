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
Unconfirmed static analysis of the new path found no protocol read or session/residency dispatch before
peer admission. Normal time-based idle expiry remains in the original loop. Real NIC behavior, output
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

On a clean remote candidate checkout, verify `git rev-parse HEAD` equals `CANDIDATE`, the base above is
an ancestor, `git status --short` is empty, and `git diff --check CANDIDATE^` is clean. Preserve the exact
candidate SHA with the receipt. Do not run from or overwrite a live worker checkout or binary.

Use CPUworker05's retained compatible dependency cache and Unix Makefiles build directory. Keep
dependencies disconnected and outputs in a new candidate-specific directory. A missing compatible cache
is a blocker; do not install dependencies or switch generators. Use the existing split: CPU worker and
fixtures on CPUworker05, engine CUDA build on host03. No build on engine02, Geekom or Valinor.

```sh
: "${SRC:?clean candidate source}"
: "${BUILD05:?retained compatible CPUworker05 build cache}"
: "${BIN05:?new CPUworker05 candidate output directory}"
test ! -e "$BIN05" && mkdir -m 700 "$BIN05"
test "$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD05/CMakeCache.txt")" = "Unix Makefiles"
cmake -S "$SRC" -B "$BUILD05" -DSTRATA_BUILD_TESTS=ON \
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

Build CUDA `strata` separately on host03 with its retained compatible cache, pinned dependencies and new
output directory as in the prior latency handoff. Since `wire_sock.cpp` is shared, build GPU-worker
protocol consumers too where enabled and build affected HIP/SYCL/Windows consumers on their supported
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

For the validated e1000e NIC, save one immutable original snapshot. Record `rx-usecs 3` as
dynamic-conservative mode, never "3 microseconds". Explicitly apply low latency, require full readback
with `rx-usecs 0` and all other coalesce fields unchanged, and explicitly restore to the exact saved
map. A failure or uncertain write requires discovery before any retry. Do not tune another driver by
analogy. Only after successful apply/readback/restore and matched workload validation should an
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
