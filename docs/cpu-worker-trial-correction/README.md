# First two-host trial correction candidate

Experimental correction on `inferswarm/two-host-trial-correction`, additive to
`17ecafca61065444b54c97a0d1b0bae4280769b0` (which includes CPU worker
`484235ff60e08181f5fbc2d0084291a8111f4e7f`). At pre-change fetch, that integration branch still matched
17ecafca; current main was `e88ee39d1d6be6e3364183af3b39b62a7ed6f211`: candidate was two commits ahead,
main had no additional commits. Work used a fresh isolated clone/session on geekom. No existing
worktree, earlier trial evidence or main branch was modified.

## Findings and correction

**Dispatcher:** source tracing establishes that the kind 0 cache row is scratch, not a CPU expert result.
With default `STRATA_DEC_BATCH` and a host plan, `expert_pool_dispatch_multi` deliberately leaves it
untouched. `RemoteExpertOpt::copy_rows`/the verifier's mapped-row copy zeros cache destinations on the
GPU before weighting. With `STRATA_DEC_BATCH=0` or device planning, the host zeros those rows. The old
fixture initializes every row to -1 and asserts zero before the GPU copy, explaining the reported
kind 0-only failures. This is a fixture-contract defect; no production routing change is justified by
that report alone. Actual CUDA execution of the corrected fixture remains pending.

The correction checks actual published cache-group destinations, token IDs and slot pointers, exactly
once per cache occurrence, with no local/worker entry in the GPU plan. It requires untouched poison or
host zeros according to the real policy, preserves source-read counts/local native-row equality and
worker-row zeros, then runs the **real CUDA row-copy mask and sum combine**. Both cache and worker
input rows are deliberately poisoned to make a missing mask visible. Expected local rows now come
from the independent native reference rather than reusing actual rows. T=1/4/8, worker omission,
double-counting and changed-token controls remain. The default-off and dead-worker checks remain.
Native cache matvecs are host-emulated; this does not establish real GPU-matvec or full-model parity.

**Worker harness:** geekom's unmodified worker test passed. Adding a deterministic 400 ms fixture delay
at the original location—after the third raw connection opened—reproduced `FAIL: setup acked` followed
by exit 139/segfault. The worker's 250 ms idle-header deadline correctly expired while the harness was
constructing fixtures. The supplied inferswarm05 logs are unavailable here, so this demonstrates the
reported failure mechanism without asserting that05's precise scheduling has been independently
measured. There is no evidence here of a lost valid SETUP_BEGIN after it reached the service.

Fixture generation now precedes connection, with that fixed delay retained. Every raw SETUP_ACK is
checked for message/status/exact 20 B length and a nonzero decoded generation; short/overflowed reads
are rejected and dependent work returns on failure. Raw replies clear stale payloads and validate the
magic. Tests independently retain 250 ms idle-header and partial-payload expiration, reject old READY on
reconnect, and require a fresh setup generation. No timeout inflation or silent retry was added.

**Transport implementation:** separate red probes established two real bound defects in the socket
helper: a peer trickling four bytes over 400 ms succeeded despite a 250 ms `recv_all` budget, and a
backpressured blocking send returned after 998 ms. The helper now shares one monotonic budget across
partial progress/EINTR and uses nonblocking I/O after readiness polling. Both red probes pass at 250 ms.
`worker_socket_test` bounds read and write completion even when progress occurs. This is a separate
transport correction, not the explanation for the slow fixture's missing ACK.

Bounds apply to each full send/read call. Frame header and payload reads have separate budgets;
blocking DNS/connect still do not honor the supplied connect timeout, and native kernel execution is
not preempted. The later experiment uses numeric loopback endpoints through owned forwards, with a
separate whole-process wall cap. Windows nonblocking/select code changed but was not built or run here.

## Executed local validation

Geekom Linux x86-64, GCC 16.2.1, CPU-only Release, pinned ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. No CUDA compiler/toolkit build was available; retained CUDA
headers alone do not provide an executed test. Loopback test execution required leaving the restricted
network sandbox; no inference host was contacted. Exact source changes are in this candidate.

- Baseline unmodified `expert_worker_test`: passed with local loopback enabled.
- Deterministic slow-fixture test before correction: failed SETUP_ACK, then segfault/exit 139.
  [Red log](evidence/slow-fixture-red.log). After correction:
  [green log](evidence/slow-fixture-green.log), unchanged 250 ms service timeout.
- Socket budget probes before correction: read 400 ms/write 998 ms, two failures.
  [Red](evidence/socket-red.log), [green](evidence/socket-green.log):250ms/250ms, zero failures.
- `ctest --test-dir build-cpu --output-on-failure -R
  '^(expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_socket_test)$'`:
  **4/4 passed**, 18.24 s. [CTest output](evidence/cpu-focused.log).
- Three fixed repetitions each of deadline and socket cases, pinned to one CPU alongside one CPU-load
  process: **6/6 passed**. Full worker and tier suites under the same contention: **2/2 passed**.
  [Recorded summary](evidence/adverse-summary.json). The reusable
  [adverse test runner](../../tools/worker_trial_adverse.py) always preserves failures and runs a fixed
  case list; it never retries a failure to obtain PASS.

Build command used locally (the dependency directory is an ordinary pinned source extraction):

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR="$PWD/.deps/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d"
cmake --build build-cpu --target strata-expert-worker expert_worker_test expert_worker_e2e_test expert_tier_test worker_socket_test -j2
```

The affected suites cover setup/upload/frame dimensions, memory-budget refusal, session/restart,
readiness, read/write deadlines, disconnect, direct/transport parity, repeated IDs, zero/mixed/all CPU
ownership and tier failure. No broad unrelated suite or CI cycle was run. Engine default-off code is
unchanged; its real dispatcher regression is retained but CUDA execution is pending.

A CPU real-dispatch seam was investigated: `expert_source.cpp` can compile against retained CUDA
headers, but its dispatcher links helper/device operations and `RemoteExpertOpt`'s allocation/copy/
combine is in a CUDA translation unit with kernels. The host-only CUDA compile probe failed. Splitting
or substituting that runtime to make a green CPU test would add scope or remove the actual mask
consumer under investigation. No syntax-only check is reported as test execution. Reuse03's already
successful CUDA build for the two-policy fixture retest in the recipe.

## Evidence and asset reconciliation

The supplied bundle was not found at `/home/zutfen/strata-trial-20261019-bundle.zip`, Downloads, this
workspace or plausible local task/temp artifact locations. Its size 2402791 B, SHA256
`9af57084a960c33c7814c62bcdc7c88b87b0df92ce3008f1e5211fa75e456e2a` and 21/21 manifest validation remain
**user-reported**. No original failing host log was available to copy; the red logs above are newly
produced local reproducers and are labeled accordingly.

The physical report says both builds passed,03's worker dispatcher failed twice with 5120 kind 0 scratch
checks, and05's worker segfaulted while e2e/tier passed. Those are supplied observations, not our own host
measurements. No full model ran here.

The 175.6 GiB six-shard set on05/01 and no bytes/pack on03 are likewise reported inventory. The trial's
“157 G free” lacks filesystem attribution. Older03 `/srv/models` sdb1 SSD (~1.8TB free) and root
(~163GiB free) measurements are user-reported history; no current03 mount was established. A local
retained census with an explicit sdb1 mount table actually names05 and must not be repurposed as03
proof. The handoff requires concise fresh `findmnt -T /srv/models` and `df -B1 / /srv/models` receipts.

The native pack can reference original Q8 expert slices, avoiding a 119.53125 GiB experts.bin duplicate;
metadata tables alone do not replace dense.bin and the original split GGUF members. A new pack is not
expert requantization. Actual dense output size, any required dense compatibility conversion and
preparation duration need the real header preflight. The recipe supplies that command, conservative
storage bound, transfer/stop-and-wait setup estimates, exact worker-complement formula, 16 GiB separate
headroom reserve and a minimal later asset-approval scope. No asset copy or preparation happened here.

## Review and remaining gates

One independent scoped review inspected the worker/socket/test/CMake correction against 17ecafca,
its retained red/green evidence and the handoff. It reported **no material code correctness findings**.
Its recipe findings were corrected: the primary cache must use a budgeted positive count (the remote
mode currently refuses primary auto), native serve requires `--spec 2` and positive `--prefill`, and
MTP-free plain decoding uses `--suffix-draft 0 --lookup-chain 0`. The recipe now uses `--prefill 64`,
fail-fast shells and the correctly spelled dense compatibility flag. The review also flagged
unexecuted CUDA/Windows paths and per-I/O rather than whole-request deadline limits. It did not
independently rerun the local tests. This review is not a hardware validation claim.

[Hermes recipe](../CPU_WORKER_RECIPE.md) is self-contained in the published candidate. First retest05's
CPU tests and03's incremental CUDA fixture build, preserving the previous logs. Then request bounded
asset preparation with exact paths/bytes. Full-model execution, output checks and deliberate worker
failure remain separately approved hardware work under InferSwarm #347. No PR, merge, broad CI,
inference-host SSH/GPU operation or asset preparation is part of this coding correction.
