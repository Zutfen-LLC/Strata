# Larger CPU-worker setup chunks

Experimental branch: `inferswarm/setup-chunks-1m` in
[Zutfen-LLC/Strata](https://github.com/Zutfen-LLC/Strata), descended directly from physically tested
`96576b32942509dfc37ae3b4e7bc0dd561fb3c26` (`inferswarm/two-host-trial-correction`).
Use the full published SHA from the handoff. This change is independent of Hermes asset preparation.
No inference-host connection, GPU execution, model operation, PR, broad CI or merge ran in this task.

## Behavior and bounds

`SETUP_ACK` already carries a data-byte cap. The worker now advertises **1,048,576 B (1 MiB)**;
no message, field, protocol version or pipelining was added. `WorkerOptions::upload_chunk_max` may
advertise a smaller positive cap, enforced both before receiving/allocating the UPLOAD payload and
when validating its chunk length. Invalid worker options refuse to listen.

The old client subtracted the 40 B upload header from that data cap. The new client uses the cap
as data bytes, accepts every cap from 1 through 1,048,576, and rejects zero/larger caps, an inexact
20 B SETUP_ACK, a zero session, and zero or overflowing u64 blob lengths before hashing/upload.
Successful UPLOAD_ACKs must be exactly 12 B and match the session and next offset. A successfully framed ST_OK ACK with invalid fields
closes the connection and clears the client session. Transport/framing errors still return false;
typed worker refusals retain the connection under the existing behavior. A failed setup clears the client's old setup state.

Each slot remains in order, bounded by its declared total length, with one whole-slot digest check
at completion. Final chunks use `min(cap, bytes_remaining)`; a 5,222,400 B expert is five chunks
at 1 MiB, the last **1,028,096 B**. Ownership, native arithmetic/reduction, readiness, failure latching,
session invalidation and the absolute nonblocking I/O budgets are unchanged. The inference request
cap remains **82,532 B including its header**, under 96 KiB, checked independently of the upload cap.

Client framing holds one bounded data payload and one encoded frame at a time: at most
1,048,616 + 1,048,632 B of logical framing buffers, plus the caller's single blob staging buffer
(5,222,400 B at the trial geometry). The worker allocates at most 1,048,616 B for a received upload
payload, besides the budgeted arena and existing native execution workspace. Allocator overhead,
vector spare capacity, socket buffers and ordinary page cache are separate; this is a source bound,
not a measured RSS claim. No upload-sized queue or per-slot client mirror was added.

## Measured local checks

Geekom, Linux x86-64, GCC 16.2.1, Release CPU-only, pinned ggml
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. All measurements below used generated native Q8_0/Q8_0
fixtures, not a giant model. Local loopback execution required escaping the restricted socket sandbox.

The benchmark sends one 5,222,400 B blob through an observing loopback relay to the real service,
using the real client. Wall time covers setup + digest/upload + READY; fixture generation, connection
establishment and inference parity checks are outside that interval. Delay 1 means one requested
1 ms sleep after receiving each UPLOAD_ACK, before forwarding it; scheduler overhead is included.
Delay 0 has the same relay, without an intentional delay. These are single observations, not stable
throughput estimates; no timing threshold is asserted.

| Advertised data cap | Actual bytes | Chunks / ACKs | Short tail | Wall, delay 0 | Wall, 1 ms per ACK |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 KiB | 5,222,400 | 80 / 80 | 45,056 | 28.272 ms | 108.525 ms |
| 512 KiB | 5,222,400 | 10 / 10 | 503,808 | 19.726 ms | 30.402 ms |
| 1 MiB | 5,222,400 | 5 / 5 | 1,028,096 | 21.894 ms | 25.198 ms |

The observer also recorded upload-frame-plus-ACK wire bytes: **5,229,120 / 5,223,240 / 5,222,820**,
respectively (setup/READY traffic excluded), and maximum received upload payloads of
65,576 / 524,328 / 1,048,616 B. Every measured upload passed whole-slot digest/readiness checks and
T=1/4/8 bitwise comparisons with the direct native CPU result. The 1 MiB choice minimizes ACKs;
the delay-0 observations do not establish a throughput advantage over 512 KiB.
[Raw measurements](evidence/setup-bench.log).

The final [four CPU suites](evidence/cpu-focused.log) passed **4/4 in 19.26 s**.
The fixed [adverse scheduling run](evidence/adverse-summary.json) passed **8/8** on one CPU alongside
one busy process: three deadline cases, three socket cases, the full worker suite and the tier suite.
Both actual compatibility pairs passed: [new client / base worker](evidence/compat-new-old.log) and
[base client / new worker](evidence/compat-old-new.log). The old client was also rebuilt directly from
`git show 96576b32942509dfc37ae3b4e7bc0dd561fb3c26:src/worker/expert_client.cpp`, with its base header,
and linked ahead of the current worker archive into the e2e fixture; the service used the retained
base build for the reverse pairing. Only the binary path changed for that reverse e2e fixture.
The base checkout stayed clean and at the exact base SHA.

Focused controls cover:

- Real negotiated uploads at all three caps, with exact observed counts, bytes and partial tails;
  bitwise native output parity at T=1/4/8 for each cap.
- Smaller advertised caps, including 1 B and legacy 64 KiB; zero/cap+1/UINT32_MAX caps;
  zero/overflowing blob sizes; short/extra SETUP_ACK; wrong session/offset/short UPLOAD_ACK.
- Header-only UPLOAD cap+1/UINT64_MAX and INFER cap+1 lengths close before a payload is supplied;
  zero chunk, overflowing offset, short declared data, cut full-cap upload deadline and session voiding.
- Existing ordering, digest, duplicate/ownership, memory-budget and restart controls; full-size
  backpressure success, incomplete write deadline, trickled read deadline and disconnect.

Compatibility is bounded to the exact base above: an actual new client against its old 64 KiB
worker, and its actual old client against the new 1 MiB worker. The latter still subtracts 40 B
and therefore sends at most 1,048,536 B per chunk. Both use the real e2e fixture (digest/readiness,
transported/direct bitwise output, counters and shutdown). This does not promise arbitrary
historical/future SEW1 versions. The smaller-cap fake peer is an additional control, not the
basis for the old/new compatibility claim.

## Estimated effect on the later upload

For N trial experts, bytes stay **N × 5,222,400**. Upload/ACK exchanges fall from **80N** to **5N**:
16 times fewer (93.75% fewer). This includes per-slot tails, rather than incorrectly rounding the
whole arena once. For the prior illustrative 21,176-slot complement (102.9945 GiB), exchanges fall
from **1,694,080 to 105,880**. An assumed 1 ms of exchange overhead contributes 28.23 min versus
1.76 min; 5 ms contributes 141.17 min versus 8.82 min. Those are arithmetic estimates, not measured
RTTs or physical-network speedups. Bytes, bandwidth costs, full-slot hashing and controller SSH
forwarding still matter. Do not increase the I/O deadline to hide stalls or infer a full-model result.

## Portable incremental host handoff (commands only; not executed here)

Run in fail-fast Bash. Obtain CANDIDATE from the final handoff, set SRC to the dedicated clean
trial checkout and BUILD to its existing absolute build directory on each host. Preserve the earlier
logs. These variables refer to host-local paths; no geekom file is needed. Refuse an unrelated checkout
or dirty source. The previously tested trial is exactly 96576b32; do not reset another checkout.

```sh
set -euo pipefail
: "${CANDIDATE:?Full 40-character SHA from the published handoff}"
: "${SRC:?Dedicated clean source checkout}"
: "${BUILD:?Existing absolute build directory}"
test "${#CANDIDATE}" -eq 40
cd "$SRC"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = 96576b32942509dfc37ae3b4e7bc0dd561fb3c26
test "$(git ls-remote origin refs/heads/inferswarm/setup-chunks-1m | cut -f1)" = "$CANDIDATE"
git fetch origin inferswarm/setup-chunks-1m
test "$(git rev-parse FETCH_HEAD)" = "$CANDIDATE"
git merge-base --is-ancestor 96576b32942509dfc37ae3b4e7bc0dd561fb3c26 "$CANDIDATE"
git switch --detach "$CANDIDATE"
```

On **inferswarm05**, preserve its successful Makefiles generator and CPU ISA/cache settings:

```sh
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF
cmake --build "$BUILD" --target strata-expert-worker expert_worker_test expert_worker_e2e_test expert_tier_test worker_socket_test -j2
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^(expert_worker_test|expert_worker_e2e_test|expert_tier_test|worker_socket_test)$'
timeout 60 "$BUILD/expert_worker_test" --setup-only
timeout 60 "$BUILD/expert_worker_test" --setup-bench
python3 tools/worker_trial_adverse.py --build "$BUILD" --out "$SRC/setup-adverse-$CANDIDATE"
```

On **inferswarm03**, preserve its existing Ninja/CUDA 13.1.115/GCC 14.2.0/pinned ggml settings.
Check its CMakeCache still targets architecture 86. This rebuilds affected dependencies incrementally;
no clean build or model is needed. Building `strata` ensures the engine links the updated client.

```sh
cmake -S "$SRC" -B "$BUILD" -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$BUILD" --target strata expert_worker_test worker_socket_test worker_dispatch_test -j2
timeout 60 "$BUILD/expert_worker_test" --setup-only
timeout 60 "$BUILD/expert_worker_test" --setup-bench
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^(worker_dispatch_test|worker_dispatch_host_zero_test|worker_socket_test)$'
STRATA_VERIFY_DEVICE_PLAN=1 timeout 120 "$BUILD/worker_dispatch_test"
```

Keep the first failure and stop dependent commands; no repeat-to-PASS. The smoke uploads only small
in-process generated fixtures on loopback. Full-model execution and host SSH are outside this coding
scope. Asset preparation retains the separately approved six-shard **188,225,033,248 B** copy and
`--compat-bf16` reference pack with no `experts.bin`; this optimization performs neither operation.
The remaining storage/placement/run constraints in [the original recipe](../CPU_WORKER_RECIPE.md) apply.

## Independent review and limits

One independent scoped review compared production code, tests, handoff and retained evidence with
96576b32. It found **no material correctness issues**. Its documentation finding was fixed: only
invalid fields in successfully framed successful ACKs take the new explicit close/clear path;
transport errors and typed refusals keep their existing behavior. The reviewer inspected evidence
and did not independently rerun the suites.

No CUDA/HIP/SYCL backend build, Windows/macOS execution, physical network benchmark or full-model
upload ran here. Earlier physical 05/03 successes at 96576b32 are supplied context, not results of
this candidate. The incremental host commands above are the remaining validation handoff; they
were not executed by this coding task. No remaining local coding/test/review blocker is known.
