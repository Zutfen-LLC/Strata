# Exact first-boundary replay

This adds one offline mode to `gpu_capture_replay`. It changes no product code,
quantization, arithmetic, tolerance, scheduler or protocol. The parent coordinates
execution separately from the Q6 benchmark. No resident is needed.

The supplied inventory is window 0/layer 0, T2/K10/H2560: 20 CPU-backed entries,
20 distinct experts (104,448,000 bytes if all weights were recomputed). B replaces
13 entries with 13 distinct CUDA experts (67,891,200 bytes). The accepted SRA1
artifact already contains the full original-index CPU rows for historical record
0. Therefore this replay reads **no CPU expert weights** and evaluates only the
13 B-selected CUDA rows. Use the already staged diagnostic blobs where present.
Do not repeat the old 16-record survey, grouped controls or CPU stage campaign.

Historical record 0's identity against the actual trace is still **unvalidated**
until the runtime gate below passes. Availability of SRA1 or a previous subset
control does not establish that identity.

## One bounded execution

Use the clean published source. Preserve the existing CUDA build settings and old
ELFs/logs when updating the diagnostic target. Keep the engine binary unchanged.
Build only these diagnostic targets on the qualified CUDA host:

```sh
cmake --build "$CUDA_BUILD" --target gpu_capture_replay gpu_capture_replay_check -j 2
ctest --test-dir "$CUDA_BUILD" -R '^first_boundary_replay_check$' --output-on-failure
```

On the host holding the traces, use its existing NumPy interpreter. The exporter
uses the current trace reader and checks both complete sealed traces. `SRA1` must
be the already accepted 8,110,340-byte cross-build artifact, whose SHA-256 is pinned
in the exporter; it is not a newly generated substitute. The parent/operator has its
known relay path and receipt. No broad search or regeneration belongs here.

```sh
umask 077
"$PYTHON_WITH_NUMPY" "$SRC/tools/export_first_boundary.py" \
  "$A0_TRACE" "$B_TRACE" "$SRA1" "$NEW_PACKET"
sha256sum "$NEW_PACKET"
```

The new exclusive mode-0600 packet is exactly **287,220 bytes**. It contains only
the first boundary, the original prompt IDs and the full record-0 CPU rows/partial.
It is private input data, not a new worker wire format. Transfer only this packet
using the existing operator transport, and compare its printed SHA-256 after
transfer. Both complete traces and SRA1 stay on their existing host. The historical
capture, original B plans and selected cache blobs were already used by the prior
offline diagnostic. Preserve them; do not open fleet cache roots through GpuCache.

Run once on the qualified CUDA device. Supply all four original B plans in the
**original configured peer order: 01, 03a, 03b, 04**. Use the actual recorded B
backend environment; KQ split remains off. Do not enable new fusion or ISA flags.

```sh
env STRATA_KQ_SPLIT=0 timeout --signal=TERM --kill-after=5s 180s \
  "$CUDA_BUILD/gpu_capture_replay" --boundary \
  "$HISTORICAL_CAPTURE" "$NEW_PACKET" "$STAGED_BLOBS" "$NEW_OUTPUT" \
  "${B_PLANS[@]}" >"$NEW_LOG" 2>&1
```

`NEW_OUTPUT` must be absent; its parent and the log parent must already exist.
No services, model GEN, worker/resident connection, full arena, temporary access,
new tunnel or package installation is used. The mode creates only its own output
and rotating one-expert cache. The 13 blob reads validate their assignment digests.
If a blob is missing, report the exact NEEDED basename; stage only that selected
blob through existing transport, without another selection campaign.

## Gates and results

Before CUDA initialization or weight reads, require these bitwise gates:

1. Historical prompt digest and GEN length, record 0/window 0/layer 0/token slice,
   actual raw F32 inputs, expert IDs, routing weights and CPU-backed mask match.
2. Historical partials and the accepted SRA1 partial match the recorded A0 union.
3. The reused SRA1 rows reproduce A0 through the independent ordered reference
   and production CPU reducer. The same full CPU row set through the real mixed
   assembly/reducer, using B's ownership, produces identical bits.

Only then evaluate the 13 selected entries with the existing CUDA executor. Each
result must restore exactly its original entry index; no surrogate rows are used.
Replace only those entries in the retained CPU matrix, keeping all 20 original
slots and routing weights. Compare this mixed matrix through the independent
reference, production CPU reducer and production mixed assembly/reducer with
**identical rows**. Finally require its union to match recorded B bit for bit.

The real gate prints `BOUNDARY_IDENTITY`, per-entry CPU/CUDA row metrics, exact
reconstruction/control metrics and `BOUNDARY_DONE`. Output retains both dense
20-entry row matrices and both reconstructed unions (450,560 bytes total), plus
the private one-expert cache. CUDA weight residency is one 5,222,400-byte expert;
the executor uses its existing 256 MiB budget. CUDA work has a 60-second internal
budget checked between expert calls and the 180-second outer stop above.

Exit 0 means both unions and identical-row controls matched exactly. Exit 1 means
B did not reconstruct; preserve the computed rows and report that boundary without
attribution. Exit 2 means an identity, ownership, input, CUDA or reducer-control
refusal. Any SRA1 reuse/A0 mismatch stops **before CUDA**, with no fallback to the
104 MB full-selected recomputation. Do not retry, start another model request or
relax a gate automatically.

Exact reconstruction and passing controls would distinguish row arithmetic from
assembly/accumulation at this boundary. They would not prove that this difference
alone causes every later error or qualify whole-model CPU/CUDA correctness.

## Validation actually performed

Local Linux/GCC 16.2.1: two Python exporter tests passed. Synthetic inputs cover
fixed size/pinned provenance, raw input/router identity, mask overlap and wrong
SRA1 record prefix. The CPU diagnostic compiled at Release optimization and
linked the existing unchanged CPU libraries; `--boundary-check` passed. It tests
raw-F32/IDs/weights/mask/slice/partial refusals, SRA1 row/partial refusal, packet
parsing/provenance refusal, cancellation-sensitive original ordering, and both
identical-row controls with synthetic CPU and mixed row sets. One deliberately
changed row prints a mismatch that the test requires to be refused.

```sh
python -m unittest discover -s tests/core -p test_first_boundary_export.py -v
"$CPU_DIAGNOSTIC/gpu_capture_replay_check" --boundary-check
```

The CUDA-enabled diagnostic translation unit passed a host syntax check with CUDA
headers. No outstanding local test failures. Actual CUDA compilation/execution,
the real historical-input match, SRA1 reuse against live A0 and reconstruction of
live B are **not run here**. No CUDA compiler/device or private real artifacts are
available locally. Keep draft status; parent dispatches the bounded commands.


The subsequent opt-in [CPU-equivalent Q8 control](Q8-CPU-REFERENCE-CONTROL-20261026.md)
uses the same packet and replay, but explicitly targets CPU rows/A0 union. The
retained-B reconstruction target above is unchanged when that option is absent.
