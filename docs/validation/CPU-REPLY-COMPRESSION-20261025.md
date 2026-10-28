# Offline exact-row compression benchmark

## Supplied terminal result

operator reports build and focused check PASS at `74ffe391b6ec32f3e61aab6ce5386d779f0cf007`.
The benchmark took 5.91 seconds with peak RSS 76 MiB, zero worker/socket/GEN contact, all 16
independent reconstructed-record reference gates passing, and all 160 frame/codec combinations
round-tripping exact bytes across 32 peer payloads and five codecs. These are results supplied through
the parent, not execution or independent private raw-receipt auditing by this coding thread.

| Codec | Reported aggregate byte reduction | Frames meeting the fixed selection rule |
| --- | --- | --- |
| LZ4 | -0.42% (expansion) | 0 / 32 |
| Snappy | -0.04% (expansion) | 0 / 32 |
| Shuffle + LZ4 | 5.05% | 0 / 32 |
| Zstd level 1 | 7.33% | 0 / 32 |

The clarified retained summary gives these totals over the same 32 frames:

| Codec | Modeled wire time saved | Sum of per-frame encode/decode p50 costs | Sum of per-frame encode/decode p95 costs |
| --- | --- | --- | --- |
| Shuffle + LZ4 | 0.936 ms | 5.932 ms | 6.126 ms |
| Zstd level 1 | 1.355 ms | 6.259 ms | 6.481 ms |

Thus summed p50 codec cost exceeds modeled wire savings by 4.996 / 4.904 ms, respectively;
using the summed p95 costs gives 5.190 / 5.126 ms. These sums of frame statistics describe serial
costs, not measured elapsed time for a pipeline or an end-to-end percentile.

The proposed policy chooses raw throughout. The diagnostic intentionally still timed every candidate;
the measured compression costs are not a production fallback implementation. A faster link reduces
the available wire-time saving, so it does not rescue a negative serialized codec-plus-link result.
This evidence does not justify another codec sweep or production compression integration.

A persistent pipeline could hide some codec work; it was not measured by this serial benchmark.
Buffers, libraries and contexts were already retained, so process startup was not the measured cost.
For independent items, ideal pipeline throughput approaches `max(encode, compressed_link, decode)`
instead of their sum, plus startup/drain and queue costs. A single reply's latency retains dependencies.
Chunk-level streaming could overlap stages within a reply, but the current whole-block API/frame path
would need explicit chunk framing and bounded reassembly. Across requests, the next model layer/token
depends on current replies; it cannot be compressed early. Existing peer I/O threads already overlap
independent same-window work. A helper thread adds queue/ownership/cache costs; a separate process adds
IPC and possibly copies (or shared-memory lifetime synchronization), not free codec throughput.

If the sampled 5.05–7.33% reductions held over ALL supplied 1601093934 reply bytes, even zero-cost
compression would save only about 80.86–117.36 MB, or 0.647–0.939 ideal serialized 1 Gb/s seconds.
That is 4.11–5.96% of the combined request/reply bytes, not a demonstrated inference-time bound or gain.
For an illustrative 100000-byte reply, raw transmission is 800 microseconds; those fractions save
40.40–58.64 microseconds. The fixed selector's p95 codec budget would be 32.32–46.91 microseconds.
Actual frames were evaluated with their own measured costs; every codec/frame failed that margin.
Failure of the 80% selector alone does not prove all serial net savings were negative, nor identify
which stage would dominate a pipeline. The clarified totals above do establish aggregate serial
overhead, but some of that work could overlap transmission in a true chunk pipeline. Per-frame,
stage-separated costs and dependency/queue schedules are needed to compute achievable overlap;
encode-plus-decode sums alone cannot do that. The zero-cost byte-saving ceiling remains unchanged.
No pooled percentile or measured pipeline gain is available. This is a retained-data question,
not a new run; incremental row streaming is being reviewed separately.

Aggregation semantics checked in source: `raw_bytes` is the production reply payload, excluding its
16-byte outer header. `wire_bytes` adds that header and an 8-byte allowance for compressed replies.
A wire-byte aggregate should be `1 - sum(wire_bytes) / sum(raw_bytes + 16)` over the SAME 32 frames,
with each frame counted once, not multiplied by 64 or averaged by percentage. The table records the
supplied byte-reduction summary; its precise header/denominator convention was not independently
recomputed from private metrics here. The executable has no aggregate percentile: each printed p50/p95
is nearest-rank over 64 timings for ONE frame/codec. The fixed selector adds that frame's encode p95
and decode p95; this is a conservative screening statistic, not a measured end-to-end p95 or a pooled
percentile. operator clarified that its timing table used the median across 32 frame-specific p50s/p95s,
not a pooled percentile. The cost totals above instead SUM the corresponding per-frame encode and
decode statistics. Neither operation reconstructs the pooled 2048 individual timing observations.

Coverage remains one original prompt through its first-logit row, including prefill and the final
prompt token. It is not a long generated-token distribution. No resident state was changed.
The recipe below is retained evidence, not a request to repeat the run.

## Next exact wire-volume opportunity: assessment only

`src/worker/expert_wire.cpp::encode_infer` sends F32 activations. The worker's `execute` in
`src/worker/expert_worker.cpp` then calls `native_quant_act` once per active token, before handing the
packed bytes to unchanged native expert jobs. `native_fmt` derives `act_bytes` from the gate/up
dot-product activation type; `kNativeActBytes=4096` is a scratch stride, not the transferable length.
For the current Q8_0 gate/up layout, the [pinned ggml type table](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.c#L257)
selects Q8_0 activations: 2560 F32 values occupy 10240 bytes, versus 80 blocks of 34 bytes = 2720 packed
bytes. That is a 73.4375% reduction in activation bytes, before request metadata. Original IDs, masks,
weights, entry ordering, exact row replies and the GPU/helper merge need not change.

The supplied split totals are 367586984 request bytes and 1601093934 reply bytes, 1968680918 combined.
Requests are only 18.67% of those bytes. Even eliminating ALL request traffic saves at most 18.67%
of that aggregate (2.941 seconds of ideal serialized 1 Gb/s transmission). Treating every request byte
as an activation gives an optimistic packed-activation ceiling of 13.71% of the combined bytes,
about 269.95 MB / 2.160 serialized link-seconds. Actual savings are lower because headers, IDs, masks
and weights remain. These are byte/serialized-link ceilings, not live TPS or full-duplex critical-path
speedup claims; replies remain dominant.

Recommended next experiment is an offline format-equivalence check using the SAME captured inputs,
not a new quantizer or protocol implementation. Verify `NativeFmt.gu_act` and `act_bytes`, compare
engine-side and worker-side packed bytes under their actual compiled quantizers, round-trip the
explicit block representation, and feed those exact bytes into the existing native row computation.
Require bitwise rows and original ordered partials, with masks/zero-work and finite/signed-zero/edge
fixtures. Preserve the source finite-input checks before quantizing. This moves an existing lossy
quantization boundary; it is not reversible encoding of the original F32 inputs. Nominally identical
source alone does not prove cross-build/ISA/rounding equality. Endianness, block layout and scale
validation also need an explicit contract; never send C structs/padding or the full 4096-byte stride.
If those gates pass, a later opt-in capability/message could let workers consume the packed bytes
directly while preserving legacy F32 requests. That production change is not implemented or requested
here. No new tests, codec tuning, worker contact or runtime integration follows from this result note.

## Retained benchmark scope

Evidence selection only. No production codec/schema change, worker connection, new arena, GEN,
resident replacement or fleet execution by this coding thread. Consumer 1 Gb/s is the baseline.
The earlier large-row packing benchmark measured serialization, not compression.

The saved capture contains real inputs and weighted partials, not unweighted rows. This diagnostic
reuses the microfixture's independent native row computation, reading and digest-checking one expert
blob (at most 5222400 bytes) at a time from existing assets. ALL selected ordered full reductions must
bitmatch capture before codec timing. Original full-bitmap alternating ownership produces each peer's
actual production indexed-row payload; inactive peers are skipped. No zeros/random data claim stands
in for measured real compressibility.

Select the first and last available record for each layer 0/12/24/47 and T1/T2, deduplicated, at most
16 records and 32 peer payloads. Report phase/position and missing strata. This is one prompt through
its first-logit row; phase2 is the final prompt token, not a long autoregressive decode sample.
It cannot establish distributions across all generated tokens, layers or prompts.

Candidates are raw memcpy, LZ4 acceleration1, per-row four-byte-lane shuffle plus LZ4, Zstd level1,
and optional Snappy. Header/index bytes are preserved by the shuffle. Only installed public C APIs
are loaded with dlopen; no headers, sources, dependencies or installations are added to Strata.
LZ4 1.10.0 and Zstd 1.5.7 are checked at runtime. Snappy's C API has no version query: preserve the
installed package receipt and library hash. Without its explicit binding it reports SKIPPED, never
a performance failure. The parent reports the separately authorized fixture-host installation of
official distro Snappy 1.2.2 succeeded. Bind that runtime for this comparison; do not skip it.

16 warmups and 64 measured iterations per payload/candidate. Caller buffers and LZ4/Zstd contexts are
retained, with capacity gates. Library/context creation, reference computation and per-candidate
initialization/warmup time are reported separately. Actual shuffle/unshuffle/copies stay timed.
Snappy's stable C API owns per-call internal workspace allocation; its unavoidable cost stays timed,
and is not claimed allocation-free. No private C++/experimental API is used to remove that cost.
Exact byte round trips, including signed zero, precede and follow timing. The focused synthetic check
also covers maximum sample size, malformed/truncated data, sampling bounds, trusted source corruption
and independent-reference mismatch; it makes no synthetic compressibility claim.

Per-frame p50/p95 and min/max are printed for encode/decode, with raw/compressed/wire sizes and a
conservative maximum of the installed codecs' compression bounds. Add 8 hypothetical metadata bytes
for compressed replies and preserve the existing 16-byte outer header. This is an allowance, not a
new protocol. At exactly 1 Gb/s, one byte takes 8 ns: link_saved_ns = (raw_wire - compressed_wire)*8.
The fixed selection rule requires smaller wire size AND encode_p95+decode_p95 <= 0.8*link_saved_ns;
otherwise choose raw. Internal framing is already included in each library's compressed size.
Do not tune this rule after results. It is a screening rule, not a measured live TPS prediction:
TCP/Ethernet overhead, overlap, contention, stragglers, production scheduling and mixed-frame cache
effects are not modeled. Fixed candidate order and repeated warm buffers limit timing interpretation.

## Fixed execution on the CPU fixture host

Parent supplies private bindings; do not publish absolute machine paths:

| Binding | Meaning |
| --- | --- |
| `SRC`, `CPU_BUILD` | Existing clean e145df6e source and its retained Release CPU cache, read-only inputs |
| `PIN` | Exact additive candidate SHA containing this document and target |
| `NEW_SRC`, `NEW_BUILD` | New absolute worktree/build paths, both absent |
| `RUN_ROOT` | New absolute disk-backed evidence directory, absent, with an existing parent |
| `CAPTURE` | Existing 37544692-byte capture, SHA256 218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208 |
| `MANIFEST` | Existing 1575943-byte manifest, SHA256 8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318 |
| `LOCAL_ROOT` | Existing original model assets; no model staging/copy |
| `STRATA_BENCH_LZ4_SO` | Installed absolute liblz4.so.1 binding |
| `STRATA_BENCH_EXTRA_SO` | Installed absolute libzstd.so.1 binding |
| `STRATA_BENCH_SNAPPY_SO` | Installed absolute libsnappy.so binding after the approved install; unset only if unavailable |

No changes to the shared runtime checkout/cache/ELFs. The new build copies only public configuration
values and reuses the retained pinned ggml source and any existing compiler launcher/cache. No fetch
of dependencies. Build only `cpu_reply_compression_bench`; the only CTest is
`cpu_reply_compression_check`. Without explicit library bindings general CTest runs skip this test;
this handoff must observe Passed, not Skipped. No CUDA/OFF build or old fixture repeat is needed.

Fetch PIN through GitHub. Extract this one marked block mechanically, refusing an empty extraction;
do not rewrite it as a new driver. For example:

```sh
git -C "$SRC" fetch origin "$PIN"
git -C "$SRC" show "$PIN:docs/validation/CPU-REPLY-COMPRESSION-20261025.md" |
  sed -n '/^# BEGIN COMPRESSION05$/,/^# END COMPRESSION05$/p' > "$BLOCK"
test -s "$BLOCK"
bash "$BLOCK"
```

`BLOCK` is a fresh private temporary file outside RUN_ROOT. Export all bindings first. All new logs
stay writable. Use the existing timeout/process-memory guards, not a new supervisor or a tiny
inherited file-size limit: the small synthetic fixture writes real Q8 files larger than its log.

```sh
# BEGIN COMPRESSION05
set -euo pipefail
umask 077
test ! -e "$RUN_ROOT"; test ! -e "$NEW_SRC"; test ! -e "$NEW_BUILD"
test -f "$CPU_BUILD/CMakeCache.txt"
test "$(git -C "$SRC" rev-parse HEAD)" = e145df6ec79f3d57c5c48215c4f9a9d0ed1e33eb
test -z "$(git -C "$SRC" status --porcelain)"
test -f "$STRATA_BENCH_LZ4_SO"; test -f "$STRATA_BENCH_EXTRA_SO"
test -n "${STRATA_BENCH_SNAPPY_SO:-}"; test -f "$STRATA_BENCH_SNAPPY_SO"
test -x /usr/bin/time
mkdir -m 700 "$RUN_ROOT"
printf '%s  %s\n' 218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208 "$CAPTURE" \
  8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318 "$MANIFEST" | sha256sum -c - > "$RUN_ROOT/input-hashes.log"
git -C "$SRC" cat-file -e "$PIN^{commit}"
git -C "$SRC" worktree add --detach "$NEW_SRC" "$PIN" > "$RUN_ROOT/worktree.log" 2>&1
test "$(git -C "$NEW_SRC" rev-parse HEAD)" = "$PIN"
python3 -B - <<'PY'
import os,pathlib,re,subprocess
e=os.environ; cache={}
for line in (pathlib.Path(e['CPU_BUILD'])/'CMakeCache.txt').read_text().splitlines():
    m=re.match(r'([^:#]+):([^=]+)=(.*)',line)
    if m: cache[m[1]]=(m[2],m[3])
assert cache['CMAKE_HOME_DIRECTORY'][1]==e['SRC']
assert cache['CMAKE_BUILD_TYPE'][1]=='Release' and cache['STRATA_ENABLE_CUDA'][1]=='OFF'
assert cache['STRATA_BUILD_TESTS'][1]=='ON' and cache['STRATA_NATIVE_EXPERTS'][1]=='ON'
ggml=cache.get('STRATA_GGML_DIR',('PATH',''))[1] or str(pathlib.Path(e['CPU_BUILD'])/'_deps/strata_llamacpp-src')
assert (pathlib.Path(ggml)/'ggml/include/ggml.h').is_file()
assert subprocess.check_output(['git','-C',ggml,'rev-parse','HEAD'],text=True).strip()=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
keys=[k for k,(t,v) in cache.items() if t not in ('INTERNAL','STATIC') and
      (k.startswith('STRATA_') or re.fullmatch(r'CMAKE_(C|CXX)_(COMPILER|COMPILER_LAUNCHER|FLAGS(_RELEASE)?)',k) or k=='CMAKE_BUILD_TYPE')]
values={k:cache[k] for k in keys}
values.update(STRATA_GGML_DIR=('PATH',ggml),FETCHCONTENT_FULLY_DISCONNECTED=('BOOL','ON'),
              CMAKE_RUNTIME_OUTPUT_DIRECTORY=('PATH',str(pathlib.Path(e['RUN_ROOT'])/'bin')),
              CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=('PATH',str(pathlib.Path(e['RUN_ROOT'])/'bin')))
with (pathlib.Path(e['RUN_ROOT'])/'initial.cmake').open('x') as f:
    for key,(kind,value) in sorted(values.items()):
        assert ']===]' not in value
        if kind not in ('BOOL','STRING','PATH','FILEPATH'): kind='STRING'
        f.write(f'set({key} [===[{value}]===] CACHE {kind} "" FORCE)\n')
(pathlib.Path(e['RUN_ROOT'])/'generator.txt').write_text(cache['CMAKE_GENERATOR'][1])
PY
timeout --signal=TERM --kill-after=5s 90s cmake -S "$NEW_SRC" -B "$NEW_BUILD" \
  -G "$(cat "$RUN_ROOT/generator.txt")" -C "$RUN_ROOT/initial.cmake" > "$RUN_ROOT/configure.log" 2>&1
timeout --signal=TERM --kill-after=5s 600s cmake --build "$NEW_BUILD" \
  --target cpu_reply_compression_bench --parallel 4 > "$RUN_ROOT/build.log" 2>&1
sha256sum "$RUN_ROOT/bin/cpu_reply_compression_bench" "$STRATA_BENCH_LZ4_SO" "$STRATA_BENCH_EXTRA_SO" > "$RUN_ROOT/binary-libraries.sha256"
if test -n "${STRATA_BENCH_SNAPPY_SO:-}"; then sha256sum "$STRATA_BENCH_SNAPPY_SO" >> "$RUN_ROOT/binary-libraries.sha256"; fi
python3 -B - <<'PY'
import os,pathlib,re,json
e=os.environ; need=(2+8)<<30
available=int(re.search(r'^MemAvailable:\s+(\d+)',pathlib.Path('/proc/meminfo').read_text(),re.M)[1])*1024
assert available>=need
def unescape(s): return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),s)
mounts=[]
for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
    left,right=line.split(' - ',1); fields=left.split()
    if right.split()[0]=='cgroup2' and unescape(fields[3])=='/': mounts.append(pathlib.Path(unescape(fields[4])))
assert len(mounts)==1, 'full cgroup2 ancestry must be visible'
member=[s.split(':',2)[2] for s in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')]
assert len(member)==1 and member[0].startswith('/') and '..' not in pathlib.PurePosixPath(member[0]).parts
mount=mounts[0]; p=mount.joinpath(*pathlib.PurePosixPath(member[0]).parts[1:]); rows=[]
while p!=mount:
    current=int((p/'memory.current').read_text()); row=dict(path=str(p),current=current)
    for key in ('memory.max','memory.high'):
        value=(p/key).read_text().strip(); assert value=='max' or int(value)-current>=need
        row[key]=value
    rows.append(row); p=p.parent
# No memory.current is required at the actual root. Missing nonroot files fail closed.
with (pathlib.Path(e['RUN_ROOT'])/'admission.json').open('x') as f: json.dump(dict(available=available,mount=str(mount),membership=member[0],ancestors=rows),f,indent=2)
PY
( ulimit -v 2097152
  timeout --signal=TERM --kill-after=5s 190s ctest --test-dir "$NEW_BUILD" --output-on-failure \
    --no-tests=error -R '^cpu_reply_compression_check$' > "$RUN_ROOT/check.log" 2>&1 )
grep -Eq 'cpu_reply_compression_check.*Passed' "$RUN_ROOT/check.log"
! grep -Eq 'Skipped|Not Run|FAILED' "$RUN_ROOT/check.log"
test -n "${STRATA_BENCH_SNAPPY_SO:-}"
( ulimit -v 2097152
  /usr/bin/time -v -o "$RUN_ROOT/resources.log" timeout --signal=TERM --kill-after=5s 190s "$RUN_ROOT/bin/cpu_reply_compression_bench" \
    --capture "$CAPTURE" "$MANIFEST" "$LOCAL_ROOT" "$STRATA_BENCH_LZ4_SO" "$STRATA_BENCH_EXTRA_SO" \
    > "$RUN_ROOT/compression.log" 2>&1 )
test "$(wc -c < "$RUN_ROOT/compression.log")" -le 262144
grep -Eq '^CPU_REPLY_COMPRESSION_DONE ok=1 records=[0-9]+ frames=[0-9]+ codecs=5 socket_contacts=0 worker_loads=0 model_GEN=0$' "$RUN_ROOT/compression.log"
! grep -Eq '^CPU_REPLY_COMPRESSION_FAIL' "$RUN_ROOT/compression.log"
python3 -B - <<'PY'
import json,os,pathlib
r=pathlib.Path(os.environ['RUN_ROOT']); rows=[]; done=None
for line in (r/'compression.log').read_text().splitlines():
    fields=dict(part.split('=',1) for part in line.split()[1:] if '=' in part)
    if line.startswith('COMPRESSION_FRAME '): rows.append(fields)
    if line.startswith('CPU_REPLY_COMPRESSION_DONE '): assert done is None; done=fields
assert done and 1<=int(done['records'])<=16 and 1<=int(done['frames'])<=32
assert len(rows)==int(done['frames'])*int(done['codecs'])
assert len({(v['record'],v['peer'],v['codec']) for v in rows})==len(rows) and all(v['exact']=='1' for v in rows)
with (r/'metrics.json').open('x') as f: json.dump(dict(done=done,frames=rows),f,indent=2)
print('COMPRESSION05_PASS metrics='+str(len(rows))+' no_resident_contact=1')
PY
# END COMPRESSION05
```

Stop on any failed command, exact-reference mismatch, skip instead of Passed, missing source range,
resource refusal, alarm/timeout or missing/duplicate metric. Preserve all logs; no automatic rerun or
new sample/parameter tuning. Internal checks require owned swap0, RSS<=2GiB, MemAvailable>=8GiB and no
new host OOM kill; the process address-space guard is 2GiB. Per-frame retained caller scratch is<=2MiB;
sample rows<=3.125MiB plus bounded capture/manifest storage and one blob. No additional worker arena.
Expected offline work under 3 minutes, with 180-second internal ceiling; isolated compilation may
take another 1–5 minutes. Retain old runtime/cache/artifacts and the full resident unchanged.

Report sampled phases/layers/shapes, library hashes/package versions, raw versus compressed size
distribution and expansion, p50/p95 CPU costs, selected/fallback counts and estimated net savings.
Do not sum overlapped live timing or call these transport estimates measured network/TPS gains.
Build/check/benchmark are authored and **NOT RUN** on the coding desktop.

Stable API sources inspected: [LZ4 1.10.0](https://github.com/lz4/lz4/blob/v1.10.0/lib/lz4.h),
[Zstd 1.5.7](https://github.com/facebook/zstd/blob/v1.5.7/lib/zstd.h),
[Snappy C API](https://github.com/google/snappy/blob/1.2.2/snappy-c.h) and
[Snappy workspace allocation](https://github.com/google/snappy/blob/1.2.2/snappy.cc).
