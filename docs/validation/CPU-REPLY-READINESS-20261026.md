> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Reply readiness: last bounded feasibility discriminator

Issues #15/#16 remain deferred. Implementation pin:
`e280b317875064b46a16aafd542e921f9512941f`, based on `0f91ce8a`.
The real captured-data measurement on 05 is **NOT RUN** by this author.

The existing `cpu_group_numeric_diagnostic --capture-record0` gains optional
`--readiness`. Before starting children, it retains only the already verified,
selected real blobs (at most 104,448,000 bytes), runs the existing native pool
with the original per-peer masks and first-appearance expert grouping, and
compares every active output row against its independent native reference.
Selected buffers and the measurement pool are released before children start.
The original capture, peer and group checks still run unchanged afterwards.

An optional caller-owned pool trace has at most 4096 cache-line-separated
completion timestamps (262,208 bytes total). Each existing down task reads the
monotonic clock once after its writes and before its ordinary done increment.
Down start, down completion/parking and pool completion are recorded locally.
There are no new tasks, callbacks, sockets, atomics or allocations in workers.
Null is the default; it adds no diagnostic clocks or trace allocation.
The pool's task claims, arithmetic, phase and parking barriers remain in place.
The extra null check still changes compiled hot-path code; no production speed
or binary-identity claim follows.

All interpretation happens after parking. Whole-row readiness is the maximum
completion time of all intersecting tasks. Byte-prefix readiness follows the
existing ascending entry/float order, including T2 entries revisiting a job.
`first_range_ns` is any nonempty finished task; `first_prefix_ns` is the first
wire row's first range; `first_row_ns` is any complete row;
`first_wire_row_ns` is the first wire row complete. All times are relative to
the traced pool call. Raw task ranges/timestamps remain in the same output.

The serializer assumes 125,000,000 payload bytes/s, zero encode/publication/
syscall/validation cost, no extra chunk headers and unlimited immediate sender
progress. It includes the existing 16-byte frame header, 20-byte row header,
2-byte entry indices and 10,240 bytes per row. Batch starts at pool completion;
streaming optimistically starts its header at pool entry. A segment starts at
max(previous send finish, segment readiness). Whole rows use the same recurrence
with each row's completion. This models transmission only, not global ordered
reduction or a functioning protocol. Partial data would still need final
validation before publication. Peer timings are independent and must not be
summed as live request savings.

Two untraced warmups precede three baseline-then-trace pairs per peer. Every
call's rows must match. `paired_delta_ns` includes cache/order/scheduling noise;
it is not calibrated instrumentation overhead. Clock/store overhead and altered
worker completion order perturb the trace itself. This is one warm T2/layer0
record with 12 workers, compact slots and retained selected weights, not T1,
full-arena DRAM pressure, LAN, long decode or production qualification. Do not
extrapolate its modeled savings to the 11,088 live remote calls.

Local Linux/GCC 16.2.1 Release checks: `pool_tasks_test`,
`cpu_capture_micro_test`, and `cpu_worker_group_test` passed. The compact test
exercises the trace and exact row checks with synthetic weights; a deterministic
serializer case covers out-of-order completion, a task crossing experts, missing
CPU entries and repeated T2 jobs. Multi-expert trace equality and oversized-trace
refusal are covered by the pool test. Initial sandbox socket-dependent checks
failed; both passed with temporary loopback sockets allowed. The synthetic run's
paired deltas ranged from -49,385 to +27,820 ns, illustrating noise rather than
negative overhead. CUDA/HIP/SYCL, other architectures and real captured-data
execution are NOT RUN; PR #10 remains draft.

## One 05 execution, dispatched only by the parent

Use a fresh output root. Do not run either withdrawn full-subset replay, connect
to a full resident, launch GEN, reload models or change the network. Commands
use the actual Git checkout and the same retained capture/assets/ggml as #20.
No dependencies are installed. The existing compact diagnostic enforces its
180-second resource guard and only controls its own temporary loopback children.
Return the log and `peers/peer*.log`, including any failure; do not rerun to chase
better timings. Tiny/no modeled savings close this exploration; large modeled
savings remain an optimistic bound and do not authorize transport work.

```bash
set -euo pipefail
PIN=e280b317875064b46a16aafd542e921f9512941f
GIT=${PRIVATE_ARTIFACT_ED09B531}
SRC=${PRIVATE_ARTIFACT_31AC9B35}
BUILD=${PRIVATE_ARTIFACT_46823945}
OUT=${PRIVATE_ARTIFACT_2364544E}
GGML=${PRIVATE_ARTIFACT_DFD8EAFF}
CAP=${PRIVATE_ARTIFACT_573112EC}
MAN=${PRIVATE_ARTIFACT_C2D77087}
MODEL=${PRIVATE_ARTIFACT_9AA3D2B0}
test ! -e "$SRC" && test ! -e "$BUILD" && test ! -e "$OUT"
test "$(git -C "$GGML" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test -z "$(git -C "$GGML" status --porcelain)"
test "$(sha256sum "$CAP" | cut -d' ' -f1)" = 218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208
test "$(sha256sum "$MAN" | cut -d' ' -f1)" = 8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318
git -C "$GIT" fetch origin "$PIN"
mkdir -m 700 "$SRC" "$OUT"
git -C "$GIT" archive "$PIN" | tar -x -C "$SRC"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_GGML_DIR="$GGML" \
  > "$OUT/configure.log" 2>&1
cmake --build "$BUILD" --target cpu_group_numeric_diagnostic cpu_capture_micro_test \
  cpu_worker_group_test pool_tasks_test -j 2 > "$OUT/build.log" 2>&1
ctest --test-dir "$BUILD" -R '^(pool_tasks_test|cpu_capture_micro_test|cpu_worker_group_test)$' \
  --output-on-failure > "$OUT/checks.log" 2>&1
timeout --signal=TERM --kill-after=15s 180s "$BUILD/cpu_group_numeric_diagnostic" \
  --worker-a "$BUILD/strata-expert-worker" --worker-b "$BUILD/strata-expert-worker" \
  --capture-record0 "$CAP" "$MAN" "$MODEL" "$OUT/peers" --readiness \
  > "$OUT/readiness.log" 2>&1
```
