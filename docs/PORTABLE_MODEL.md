# Portable local model session (opt-in)

`portable_model` runs the fleet qwen4exp Qwen3.8 Flash Next Q8_0 model through
Strata's portable layers and its in-process CPU expert pool. Input is exact
token IDs. The existing Python tokenizer can supply them. This path has no
network listener and does not invoke another model engine.

The initial implementation supports one serial text sequence, all 48 layers,
and context limits from 1 to 256 tokens. Defaults remain off. The existing
CUDA production executable and allocations are separate.

## Build and run

Use the pinned llama.cpp dependency
`3cf03257f219afbe7334045ff7c6a06ac68c627d`:

```sh
cmake -S . -B build-portable-model \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON \
  -DSTRATA_PORTABLE_QSA=ON -DSTRATA_PORTABLE_MODEL=ON \
  -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp
cmake --build build-portable-model --target portable_model portable_head_test portable_model_test -j4
```

Add `-DGGML_VULKAN=ON` to build with Vulkan. Device execution is explicit:

```sh
build-portable-model/portable_model \
  --model /path/to/model-00001-of-00006.gguf \
  --manifest /path/to/accepted/SWLOCAL1/manifest \
  --tokens 123,456,789 --context 16 --max-new 3 \
  --threads 4 --expert-workers 12 \
  --vulkan 0,1 --device-cap-mib 7680,7680 --split 32 \
  --dump-logits /existing/output/directory \
  --placements /path/to/placements.tsv
```

The IDs above are placeholders, not a retained test prompt. The named output
directory must already exist. Without `--vulkan`, execution is CPU-only.
Accelerator fallback is forbidden unless `--allow-cpu-fallback` is explicitly
supplied; the placement report records fallback reasons. Choose device caps
from the actual free memory and leave measured headroom for the driver and
other users. Each V340 die has its own 8 GiB; caps are never pooled. The CLI records
Vulkan memory snapshots before model loading and after every token; placement
TSV rows carry layer IDs. For the exact parent-dispatched real-model stages,
use [the versioned Hermes recipe](../agenthandoffs/portable-model-hermes-v4.md),
which also samples aggregate VRAM for both PCI devices during the run.

Repeated Vulkan indices are rejected before initialization. Session admission
also rejects distinct backend handles that share the same ggml device identity,
so one physical die cannot receive two independent ledgers. Equal device
names/descriptions do not imply a shared identity.

The split is contiguous: layers [0,32) on the first device, [32,48) on the
second, then the global output head on the second. `--split` can select another
fixed boundary. This default leaves fewer layers beside the large vocabulary
head. It is a correctness placement, not a measured optimal placement.

Every consumed prompt or generated token produces its full-vocabulary
little-endian F32 row as `pos-N.f32`, where position starts at zero. Teacher
forcing uses `--max-new 0`. These rows work with the existing
`tools/validation-harness/compare_logits.py`; retain its fixed limits and let
the reference owner interpret failures. Matching a few greedy tokens does
not establish full-model quality.

The first parent-dispatched real run at `ae6489205c9eca5c072610dbacbc78c8489dffda`
completed all48 layers and the head on the two V340 dies with genuine CPU experts.
Three generated IDs matched the retained CUDA prefix, but the unchanged full-vocabulary
numerical screen at position37 failed (relative logit L2 `0.0717160202`,
KL `0.0001161040`). This is execution evidence, not a quality pass. See the
[versioned result and numerical-policy audit](../agenthandoffs/portable-model-first-real-audit-v1.md)
for all metrics, resource observations, conditional native policies and the proposed
retained-output diagnostic. No rerun or optimization was performed for that audit.

## Ownership and token semantics

`PortableModelSession` owns the six-shard `GgufModel` for the session's lifetime.
It validates split headers, tensor inventory, payload extents/non-overlap,
active-layer shapes/types and metadata. Embedding, huge PLE tables and routed
expert weights remain read-only mapped. Layer and head weights have owned
host copies and scheduler backend copies. The actual QSA indexer Q/K matrices
are BF16 and are widened exactly into F32 storage with one 10 KiB row buffer;
those projection nodes request F32 accumulation. No re-quantization or native
BF16 shader support is required. Startup conversions are reported separately
as `load / BF16_TO_F32_EXACT / CPU`; scheduled projection work remains strict
Vulkan when requested.

Every token gathers a new host Q8 embedding and broadcasts it into four HC
streams. Previous-token HC residuals never become the next token's initial
hidden state. GDN/QSA/PLE histories do persist until reset. The authoritative
position advances only after all layers and the output head succeed.

A failed `step` poisons the session, including invalid inputs and failures
after earlier layers committed. Another step is refused until explicit
`reset` and replay of the desired token sequence. Reset failure also leaves
the session poisoned. Generation preflight can reject an empty/over-budget
request without changing state. Only a **generated** tokenizer EOS is consumed
and stops generation. Input and teacher-forced turn separators are fed normally, including tokenizer EOS
inside a chat prompt. The API defaults `step` to input; callers feeding a
sampled token pass `PortableTokenKind::generated`. The fleet's tokenizer EOS
is 248046, independently of its PLE predecessor-history boundary 248044. Both
metadata fields retain independent integer/range validation; PLE's existing
configuration validation and hash/history behavior are unchanged.
Context exhaustion and max-new exhaustion have separate stop reasons. Exact
logit ties select the lowest token ID. There is no MTP, batching, speculation,
or tokenizer implementation in this executable.

## Experts and retained identity

The accepted `SWLOCAL1` manifest is expert identity evidence. Its file entries
can cover only expert-containing shards, such as shards 4–6 of this model;
it is not a six-shard checksum manifest. The existing parser validates it.
The provider compares every expert tensor's shape/type, source filename/size
and absolute role ranges with the GGUF directory. Each selected expert is
checked against the retained blob digest on first use, then its verified native
view is reused by this model instance. There is no setup
scan of all expert bytes and no new full-file checksum claim.

Standard `NativeFmt` Q8 experts consist of the GGUF gate, up and down slices
in order. The provider validates row sizes, offsets and blob size against
`native_fmt`, copies first-use slices into a reusable ten-expert staging arena
for verification, then points native jobs directly into the session-owned GGUF
mappings. Repeated hits skip both copies and paired forward/reverse FNV-1a
scans. Quantization still uses `native_quant_act` and execution calls
`ExpertPool::run_split_multi_native`. It never substitutes the canonical
engine's packed weight layout, expands experts to F32, or creates another
unbounded arena. Rows are unweighted and follow route order. `PortableFfn`
alone applies router weights and reduces them.

Model files must remain immutable from open through session destruction,
including during each dispatch. The provider checks every source's size and
modification time before dispatch; a time change invalidates all verified
entries, while a size change or metadata error rejects the request. Changed
selected payloads still fail their first digest check. Intentional model edits
require reopening the session; this is not a hot-reload facility. Reset retains
verification state for unchanged files and does not repair model identity.

This differs from the former provider's digest on every dispatch. Metadata
checks detect ordinary writes, not adversarial mutation during execution,
restored timestamps, same-size replacements preserving modification time, or
filesystems that hide changes within their timestamp resolution. Read-only
mapping permissions do not make backing files immutable. Native execution
reads the mappings, not the staging snapshot whose digest was checked, so
source stability must cover copy, digest and native execution even on first use. No file lock or new
security setting is added. The explicit `--expert-dispatch-oracle` comparison
mode retains the former copy/digest-per-dispatch path and executes copied
bytes. Neither mode is a cryptographic or concurrent-mutation guarantee.

`expert_work` CLI records are cumulative per session: hits, verifications,
copied bytes, digest input bytes (two logical byte accesses each), native
batches and source invalidations. Verification keys are local layer/expert
indices bound to validated role layout and the owned model; no global cache
or F32 expert expansion exists. Cache metadata fits within the existing
32 MiB manifest/index/state allowance. Staging and host admission caps remain
unchanged. See `agenthandoffs/portable-cpu-expert-proof-v2.md` for the matched
comparison recipe. Parent/Hermes reports all 41 raw logit rows and stdout
byte-identical in the physical V340 pair at `ac85ff3`, with fewer copy/digest
bytes and unchanged native batch counts. The fixed oracle-first timing is
order-biased and does not establish a general speedup. See
`agenthandoffs/portable-cpu-expert-physical-results-v1.md` for the attributed
receipt. The failed CUDA parity screen remains failed.

## Memory and current limits

The report distinguishes unique mapped shard file bytes, anonymous host
weight copies, scheduler buffers and conservative state/workspace bounds.
The PLE factory may retain another read-only mapping of the same backing
file; that is an address-space alias, not an anonymous 54 GB table copy.
Selected expert staging is 52,224,000 bytes at H2560/FF640/top10.

Actual scheduler reservations are queried per backend, including scheduler
weight copies. Admission also includes CPU plans, metadata, committed and
candidate histories, pool scratch/stacks, output buffers and conservative
pinned-Vulkan private workspace allowances. QSA's temporary attention
scheduler uses the remaining per-layer graph allowance on each backend.
The global head includes a full F32 matrix workspace allowance derived from
its actual vocabulary size. Each new module is checked against remaining
caps before allocation, then reconciled to actual scheduler reservations and
workspace bounds. These bounds can reject a placement that a more optimized
implementation could fit.

The real-weight supervisor separately admits the sum of six unique shard stat
sizes as a worst-case file working set, plus the32GiB anonymous cap and8GiB
reserve (about215.3GiB total for this model). It checks both MemAvailable and
any finite inherited cgroup-v2 memory allowance. During loading and decoding,
it samples MemAvailable, swap counters and child VmSwap every0.2s and stops
only its own process group on an8GiB reserve breach, new system swap-out or
child swap. Historical system swap is recorded and may be nonzero. The32GiB
CLI cap alone does not bound total resident model memory. File mappings are
logical address space; actual residency changes as pages are touched. Sampled
guards and initial headroom do not guarantee that no transient OOM can occur
between observations. The supervisor never changes a cgroup or retries.

Driver pipelines/command pools and unrelated backend users are outside this
owned-allocation ledger; operator-selected caps need actual headroom. No claim
of total measured VRAM peak or optimized performance is made. GDN/QSA
recurrent state and layer residuals still move through host memory each token.

The full-model synthetic test uses sparse files with the actual layer and
vocabulary dimensions. It executes all 48 layers, configured layer-1 PLE,
a real nonzero CPU expert, full-vocabulary scalar checks, different token
embeddings, a layer-10 digest failure, poison/reset/replay, greedy generation,
EOS and context stops. It takes several GiB of RAM and about 22 seconds in
the initial local CPU run. The standalone head test also accepts `--vulkan INDEX` and checks actual
H2560/HC4/rank320 with64 output rows and strict device placement. Separate
existing tests check nonzero attention,
recurrent and PLE histories against independent references. This synthetic
result does not establish real-weight numerical quality or physical AMD
execution of the assembled model.

## Sequential resident stdio (POSIX, opt-in)

Replace `--tokens` and `--max-new` with `--stdio --max-requests 4`. Model,
backends, CPU worker pool, GPU plans and immutable expert verification views
are constructed once. The existing one-shot output and arithmetic are unchanged.
There is no network listener or concurrent execution.

After loading, stdout emits one flushed JSON line:

```json
{"ready":true,"schema":1,"load_ms":123.0,"context":256,"max_requests":4}
```

Send ASCII `GEN <max_new> <id,id,...>\n`, for example `GEN 2 123,456\n`.
This uses the native GEN/QUIT control words with the portable CLI's comma-separated
IDs; native optional generation parameters are unsupported. Each request is
independent. Unsigned decimals only, no spaces within the list, no CRLF, at most
4096 bytes before LF, nonempty prompt, IDs below the loaded vocabulary, prompt
length and max_new each at most context (1–256). Generation stops at context if
the combined work would exceed it. The entire request is validated before reset
or mutation. A malformed, oversized or incomplete-at-EOF request exits nonzero.

Every accepted GEN resets GDN recurrent/convolution state, QSA KV/pool/tail state,
PLE history, previous tokens, position and generation EOS. It retains immutable
weights, expert executors, plans, verified CPU views and lifetime counters.
Monotone local executor request sequence numbers persist. Only a generated
metadata EOS ends generation; input EOS separators remain ordinary prompt input.

Each request emits the existing per-token JSON rows and stop row, followed by a
flushed `{"done":N,...}` line (N starts at 1). It includes setup_reset_ms,
prompt_ms, decode_ms, selection_prompt_ms, selection_decode_ms, terminal_feed_ms,
actual prompt_count/output_count and selection step counts. As in one-shot timing,
the selection decode interval starts before the last prompt forward step; terminal
feed records the final selected token's forward step. These wall intervals include
the CLI's synchronous reporting overhead. load_ms is emitted only at readiness;
first-request lazy CPU verification remains in prompt/decode work.

Request deltas subtract lifetime counters: cpu_hits_delta,
cpu_verifications_delta, cpu_copied_bytes_delta, cpu_digest_bytes_delta,
source_invalidations_delta, gpu_entries_delta, cpu_entries_delta and the ordered
gpu_entries_per_device_delta array. CPU entry counts are the existing hybrid
adapter's cold entries (zero without plans). Device memory peaks and row-error
maxima in stderr remain cumulative observations, not subtractable request metrics.
No success/done receipt is emitted for a failed request. Partial token rows are
not a completed result. `--placements` describes the last successful request when
the process exits normally; it is not emitted on an execution failure.

`QUIT\n`, clean EOF between requests, or the configured request limit (1–64,
default 4) exits normally. SIGINT/SIGTERM requests cooperative cleanup: idle stdin
polls at 100 ms; loading and a running token finish their current blocking backend
operation before the signal is checked and ownership unwinds. An outer process
supervisor must enforce a hard deadline if a backend stalls. EOF mid-frame fails.
Execution, identity, allocation or output failure exits without reset/retry;
poisoned state is never reused by this mode. RAII releases model/providers before
borrowed backends. No request logits/buffers accumulate. Logit dumping and the
one-shot `--timings` file are incompatible with stdio; request metrics use stdout.

The resident CLI checks opening shard/manifest/plan sizes and modification times
before readiness, before reset, around token work and before a successful receipt.
An observed change requires exit/reopen, including CPU-only operation. This
preserves the existing immutable-file contract and first-use expert digests; it
is not a whole-file hash or a defense against writers deliberately restoring
metadata. Do not edit source files while the process is open.

`portable_resident_v1.py` supervises one loaded process for one excluded warmup
and three identical measured requests. It retains all four receipts and requires
same-backend token/row repeatability, no later CPU re-verification/copy and GPU
entries on both dies for every request. See the
[Hermes recipe](../agenthandoffs/portable-resident-hermes-v1.md). Local sparse and
fake-process tests establish lifecycle behavior, not real-model quality, AMD
performance, or cross-backend bit identity. The existing strict numerical parity
failure remains unchanged. Unequal process lifecycles are an unmeasured confounder;
this capability alone does not explain the full native/portable latency gap.


### Physical resident checkpoint on05

Hermes [reported a successful AMD resident checkpoint](../agenthandoffs/portable-resident-amd-results-v1.md)
at runtime source `45513fd885babcf81d252b73804a14d2a751a74a`: one loaded process,
four identical token/greedy-row results, four output tokens each, unchanged
480-per-die plans. Requests2–4 had zero CPU verification/copy/digest/invalidation
deltas, with positive GPU participation on both dies for every request. One-time
load was18.19s; first-use prompt/decode was74.47s/10.46s, while the three later
requests recorded prompt39.98–40.06s and decode6.39–6.48s. Selection intervals are
separate overlapping views and must not be added to those phase times.

This measures first-use versus warm behavior in that one process. It does not
isolate verification from other warming effects or compare against a matched
CUDA configuration. No full resident quality screen or cross-backend numerical
parity is established. The original strict parity failure remains unchanged.
The [NVIDIA02 recipe](../agenthandoffs/portable-resident-nvidia02-v1.md) reuses the
same runtime source and explicit local paths/PCI selection; it remains pending
parent review/dispatch. This documentation update authorizes no additional run.

### Existing CPU expert pool phase deltas

`expert_work()` exposes the pool's existing cumulative host-wall milliseconds as
pool_gu_ms, pool_quant_ms, pool_down_ms and pool_elapsed_ms. Resident done receipts
subtract before/after snapshots into `cpu_pool_gu_ms_delta`,
`cpu_pool_quant_ms_delta`, `cpu_pool_down_ms_delta`, `cpu_pool_elapsed_ms_delta`.
They accumulate across reset; reading a snapshot adds no timing or work. Existing
one-shot stdout, stderr counters, timing-file schema and arithmetic are unchanged.

The source of the first three counters is ExpertPool::ms_multi_gu/ms_multi_q/
ms_multi_down. In run_split_multi_native, gate/up wraps run_phase(5), including
worker parking, publication, host participation, drain and re-park. Quantization
covers host intermediate activation quantization. Down starts after quantization,
includes phase setup and run_phase(6), with its synchronization. These are wall
intervals, not pure arithmetic or summed worker CPU time.

pool_elapsed_ms reads the **drain** output of phase_ms(). CpuExperts exclusively
uses run_split_multi_native: its drain accumulator encloses the native call's
batch loop and three phase intervals, plus batch setup/bookkeeping. Legacy run()
wait-park/repark counters are not separately timed for this native path; their zero
values would not mean synchronization is free. The phase sum is contained in the
enclosing native elapsed interval (subject to floating subtraction/rounding).
Never add elapsed to the phases. These snapshots cover all native calls through
this provider, including CPU reference rows if --expert-check-rows is enabled.
The timed recipe leaves that diagnostic off.

Selected expert copy/digest, initial activation quantization and route/row buffer
preparation outside the native call are excluded from pool elapsed. Forward wall
time outside it also includes routing, GPU experts, dense/backend work, transfers,
merging and logging; it must not all be labeled dense compute. These are complete
request deltas, not separate prompt/decode pool breakdowns or a new profiler.
The [bounded AMD timing recipe](../agenthandoffs/portable-pool-phase-hermes-v1.md)
keeps the old480 plans and frozen case1 for parent review before dispatch.

The portable and native users of the same compiled strata_kernels_cpu/ggml-cpu
libraries in one build do not establish a separate portable-only scalar kernel.
That shared source/link relationship does not establish ISA parity between
separately deployed builds; their actual build flags and dispatch remain unverified.
