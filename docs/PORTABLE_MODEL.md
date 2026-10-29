# Portable local model session (opt-in)

`portable_model` runs the fleet qwen4exp Qwen3.8 Flash Next Q8_0 model through
Strata's portable layers and its in-process CPU expert pool. Input is exact
token IDs. The existing Python tokenizer can supply them. This path has no
network listener and does not invoke another model engine.

The initial implementation supports one serial text sequence, all 48 layers,
and context limits from 1 to 256 tokens. Defaults remain off. The existing
CUDA production executable and allocations are separate.

## Bounded prompt chunks

`--prompt-chunk-size N` opts into layer-major prompt advancement for N=2–8.
It implies prompt head elision. The first `prompt_length - 1` tokens are split
into chunks of at most N; a one-token tail uses the existing T1 FFN. Each chunk
advances ordered attention through one layer, then runs that layer's T-column
FFN and grouped CPU experts before moving to the next layer. Native CPU jobs
quantize each needed activation once and group tokens by expert ID. Resident
GPU experts still use the existing single-row graph per entry; this flag does
not add batched GPU expert kernels or independent request batching.

The last prompt token, every generated token and the terminal feed remain on
the existing T1 head path, including `--max-new 0` and context boundaries. Input
tokenizer EOS remains a separator; PLE keeps its separate EOS/image predecessor
rules. The default and standalone `--prompt-head-elision` paths stay serial.
`--dump-logits` is incompatible with either policy that omits prompt heads.

C++ callers set `PortableModelOptions::prompt_chunk_size` (zero disables it).
`generate` applies the policy automatically. `advance_chunk` is an input-only
seam for one nonempty segment of at most the configured size; arbitrary smaller
segments are allowed. It exposes no logits. A failure poisons the model until
reset/replay because earlier layers may already have advanced. Position, skipped
heads and completed model chunk counters publish only after all 48 layers
succeed. Route diagnostics are staged until then. The CLI emits no completed
rows for a failed chunk. Exhaustive T1 `step` remains available.

### Admission before allocation

For the enabled policy, construction pre-admits a conservative complete module
bound **before** allocating each layer or chunk shape. This reserves two aggregate
layer graph caps (fieldwise maxima can combine graph and unused-workspace high
waters), the existing private/state/resident-weight bounds and the additional
FFN private allowance. It then prepares T1 through N one at a time, using one
shared immutable FFN weight allocation and at most one active chunk execution.
The fieldwise memory envelope is charged before the next layer, head or resident
expert allocation. Runtime may select only one of those admitted shapes.

No caps are increased and no per-T execution cache is retained. Changing tail
size releases the previous chunk execution. The model also reserves bounded
embedding/row/route staging. `report().chunk_memory_is_shape_envelope` identifies
this policy: scheduler and workspace entries are admission envelopes rather
than claims about current allocation/RSS. CLI stderr labels scheduler values as
`scheduler_shape_envelope_bytes` and emits `prompt_chunk_admission shape_envelope=1`.
Actual weight and persistent-state allocations remain counted once. Conservative
preflight can reject a configuration that a tighter peak plan could fit; do not
raise caps or reassign experts to make a rejected configuration pass.

### Receipts and timing

Chunk token rows retain PR76 schema 2, `head_policy: "prompt_head_elision"`,
`logits_available: false` and `greedy_next: null` at skipped positions. They also
bind `chunk_policy: "layer_major_v1"` and integer `prompt_chunk_size: N`. Final
prompt and generated rows carry available predictions with the same policy.
Stop rows retain their existing fields. Fresh and resident adapters bind the
size/policy in summaries and argv, and reject mismatched, missing, duplicated,
foreign or incorrectly typed enabled references. Older references remain usable
only with chunking disabled.

For canonical CLI/`generate` segmentation with P prompt tokens, G actual generated
feeds and chunk bound N, let
C = ceil((P-1)/N), including any T1 tail, and F = C+G+1. The enabled policy exposes:

- `logical_tokens_delta = P+G`, `prompt_chunk_tokens_delta = P-1`;
- `prompt_chunks_delta = C`, `model_forward_calls_delta = F`;
- `chunk_layer_calls_delta`, `chunk_ffn_pre_calls_delta`,
  `chunk_provider_calls_delta`, `chunk_ffn_post_calls_delta` = 48C;
- `model_layer_calls_delta`, `ffn_pre_calls_delta`, `provider_calls_delta`,
  `ffn_post_calls_delta` = 48F.

These unconditional counters describe completed model work. Existing timed
`model_step_calls_delta`, GDN/QSA layer calls and expert provider calls count
actual forward/coordinator executions under this policy, not P+G T1 steps.
Legacy `prompt_steps` and `selection_prompt_steps` remain logical positions and
selection boundaries. Head counts remain G+1 executed and P-1 skipped.

Coarse model/layer/provider wall scopes remain available. Detailed GDN
subphase/readback/commit timers are not captured for chunk execution: enabled
receipts explicitly set `gdn_subphases_available: false` and omit the entire
T1-only detail group. The harness rejects that group if presented as chunk
totals. It does not substitute zeros or require legacy 109/36-per-row diagnostics.
Fresh chunk runs request a timing receipt even outside benchmark mode so their
execution counters can be checked; no pooled performance claim is added.

The assembled sparse-model fixture compares T2–8, tails, mixed segmentation,
per-token/layer route IDs, requested full-vocabulary logits, continuation, A/B/A,
late provider failure/reset, admission rejection and actual native `nt > 1`.
The existing Q8 numerical gate is fixed independently of observed results.
Real-model numerical/quality screens and Vulkan execution remain separate gates.

## Prompt head elision

`--prompt-head-elision` is default off. It runs the same serial T1 embedding,
48 layers, expert routes and state commits for every prompt token, but skips
the vocabulary head for the first `prompt_length - 1` positions. The final
prompt head still runs, even with `--max-new 0` or a prompt filling context.
Generated tokens, including the terminal feed and generated EOS, keep their
existing full head execution. Input EOS remains an input separator.

This is head elision, not batched prefill: there is no multirow matrix work,
layer-major chunking or kernel change. It is a dependency for #69, not completion
of that issue's chunked-prefill acceptance. Native `Prefill::run` advances the
first `n_prompt - 1` tokens; `src/program/generate.cpp` leaves the final prompt
token for generation logits. This option adopts only that head boundary.

C++ callers opt in through `PortableModelOptions::prompt_head_elision` for
`generate()`. `step()` always computes logits. `advance(token, error)` is the
input-only state-advance seam: it shares the entire step implementation and
omits only the head, publishing no logits. Failures poison the session and
require reset/replay, including partial layer failures. Reset retains cumulative
work counters. `heads_executed` counts head attempts, including failed heads;
`heads_skipped` counts successful advances reaching the omitted-head boundary.
These counters work without enabling timers. Head placement records are absent
on state-only steps rather than exposing the previous head's placement.

The finite-logit guard runs at every executed head under this option, followed
by the existing host output checks. **No logits check exists at skipped
positions.** Layer/state finite checks follow the selected GDN/QSA policies.
A bad head can therefore be detected later than in exhaustive diagnostics.
`--dump-logits` and head elision are incompatible and rejected before loading:
omit head elision to request all teacher-forced vocabulary rows. Default
execution, receipt fields and exhaustive dumps keep their existing behavior.

The option emits schema-2 token receipts at every consumed position:

```json
{"position":0,"token":5,"generated":false,"schema":2,"head_policy":"prompt_head_elision","logits_available":false,"greedy_next":null}
```

Only nonfinal prompt positions have `logits_available: false` and a null greedy
prediction. Final prompt and generated positions have `logits_available: true`
and a real integer prediction. Default rows retain their original four fields
with no added schema/policy. Stop receipts remain unchanged. Resident `done`
receipts add `receipt_schema: 2`, `head_policy: "prompt_head_elision"`,
`heads_executed_delta` and `heads_skipped_delta`. Completed requests must report
`generated_count + 1` executed and `prompt_count - 1` skipped heads. Fresh and
resident runs also write these per-request counts as `prompt_head_work` on
stderr. Existing head timers count only heads that actually execute; all layer
and state work counts still follow the number of consumed tokens.

The fresh-quality and resident adapters forward the option and bind
`head_policy` in summaries and reference argv. Schema-2 rows require an explicit
matching policy; missing fields, foreign versions/policies, fabricated skipped
predictions and missing final predictions are rejected. Older references remain
valid only with elision off. Same-policy reset tests compare every receipt;
cross-policy comparisons compare prompt IDs, final requested logits, generated
rows and stops without inventing intermediate predictions. Numerical and GPU
performance qualification remain separate from host tests and compilation.

## QSA attention execution reuse trial

`--qsa-reuse-attention` is default off. It retains one attention Phase per QSA
layer: a 4 MiB metadata arena, one scheduler and its backend scratch buffers.
Creation allocates the exact capacity-sized graph once without executing it.
Each token resets the scheduler identities before rebuilding the exact active
shape in that arena, reapplies every placement pin, and uses the existing
allocator's retained buffer high water. There is no per-length graph cache.
Reset clears sequence state but keeps execution storage. Leave the flag off
for the legacy token-local attention Phase.

The existing fresh-quality and resident adapters accept this same flag and record
its Boolean policy. Resident suite references must match that policy in both the
fresh summary and every case argv, with exactly one flag when enabled. Missing,
duplicate or mismatched enabled references are rejected. Older references with
no QSA policy remain usable only with the default-off mode.

This follows the bounded ownership in native `SessionState` (`session_init` in
`src/core/session.cpp` and `sycl/src/core/session.cpp`) and the active native
Verifier's retained arena/execution (`src/core/verify.cpp`). It does not copy
native fused math or its graph-capture API. At the pinned GGML revision
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, Vulkan exposes no `graph_plan_*`
callbacks. GGML's scheduler reset retains allocator buffers; exact shape or
node-count changes can replan offsets. Its allocator replaces backing buffers
only when the required chunk size grows. CPU plan workspace is priced at its
retained high water, even for strict Vulkan execution. Backend-private Vulkan
workspace can still grow on first execution of a newly required shape and is
covered by the existing conservative allowance. This is scheduler/scratch reuse,
not a claim of persistent compiled Vulkan commands or zero host allocation.

With QSA production off, active FP16 KV and F32 pooled keys come from host
candidate vectors and upload their full active prefix each step. The optional
QSA production policy below replaces those inputs with device state views. The per-head F32 attention
workaround, exact BF16 indexer widening, RoPE positions, moving spare, finite
checks, stable selection ties and cutoff at 2051 cells are unchanged. No padding
or masked maximum-length softmax is introduced. In eager diagnostic mode, layer failures keep committed
state unchanged; scratch may be overwritten and is rebuilt on retry. A partial
model failure retains the existing session poison/reset policy.

`graph_bytes()` includes retained attention buffers and the admitted CPU plan
high water in this trial. `memory_usage()` reports the actual scheduler buffers
in host/device graph fields, including after a failed candidate and reset.
The remaining layer graph allowance is retained device extra rather than
`device_token_temporary_bound`; it is no longer credited as shared token-local
scratch across QSA layers. Host state/candidate staging and the 4 MiB arena stay
within the existing state/metadata bounds. Backend-private workspace, shared
matrix conversion and driver exclusions keep their existing accounting. The
same caps apply; capacity priming can reject at creation where the legacy path
would reject only on reaching a larger active shape.

The QSA CPU/Vulkan fixtures exercise legacy and reuse modes, positions 0–8,
pool boundaries, callback failures/retries, capacity, reset A/B/A, actual buffer
high water and exact same-backend state/results. BF16 and GGUF fixtures cover
both policies. Independent selection fixtures cross 2050/2051/2052 and pooled
2055/2056 boundaries, ties and all 24 heads, with a second scheduler/arena sweep
checking stable backing buffers. These synthetic gates do not establish model
quality, device admission on other cards or a speed improvement.

## QSA device state policy

`--qsa-production` is default off and requires `--qsa-reuse-attention`, a selected
accelerator and CPU fallback off. It is independent of GDN production. Either
production flag enables the existing integer logits guard before head output is
published. Standalone QSA layers use `PortableQsaOptions::production = true`.

This follows native per-layer `QsaState` ownership and the active Verifier's
KV/indexer appends. Each portable layer retains capacity-sized FP16 K/V, F32
pooled keys, an immutable spare, a four-row raw indexer tail, cell mapping and
tail bias on the selected backend. Exact active views feed the existing score,
selection and 24 separate per-head F32 attention products. It preserves the
pinned Vulkan head workaround, BF16 widening, RoPE, selection ties and cutoff.
No padding or attention arithmetic change is introduced.

Current-token projections still download for finite checks and CPU FP16
conversion, preserving rounding. Each step uploads one new K and V cell and one
raw indexer key. Pool completion uses the existing pool graph directly over the
device tail, copies its output to the completed pooled row, moves a device copy
of the immutable spare, then clears the tail. Token zero uses the existing
four-repeated-key add/scale calculation before keeping just raw row zero. Only
the current incomplete block's bias changes; mapping initializes at reset.
Normal production does not upload historical KV/pool prefixes or download
scores, selection masks or state mirrors. Current attention/residual readbacks
and host experts remain in this bounded slice.

Raw state at capacity C is `2048*C + 512*(floor(C/4)+1) + 512 + 2048 + 8*C`
bytes: 562176 bytes per layer at C256 before backend alignment. Admission charges
the actual aligned state buffer in `device_state`, separately from retained
scheduler buffers and the unchanged workspace allowance. Before constructing
each production QSA layer, model preflight also checks that backend's exact
aligned estimate from the same tensor declarations. This estimate reserves no
ledger bytes; the actual allocation is charged once after construction. It removes only the
scheduled inputs replaced by external views. Caps and context limits are
unchanged. Reset retains these allocations.

Production traces and ordinary host-history getters refuse inspection rather
than returning stale mirrors. `inspect_state()` explicitly downloads full raw
capacity for diagnostics; after a failure it can include uncommitted appends.
It is never called by normal generation. A failed production step poisons the
layer and model session. Position, route and residual publish only on success;
retry requires reset. Reset clears state and initializes mapping, and clears
poison only after successful initialization. Eager diagnostics retain their
existing observers, full state checks and rollback behavior by default.

Both validation adapters forward the new flag and record `qsa_policy` as
`device_production` or `eager_diagnostic`. Resident references must match the
summary policy and contain exactly one production flag when enabled in every
case argv. Missing, duplicate, malformed or mismatched production references
are rejected. Older references are accepted only with production off.

Synthetic fixtures compare diagnostic and production state/results on the same
backend, including positions 0/1/3/4/7/8, pool completion, unused capacity,
reset A/B/A, append failure and poison/recovery. A local allocation-failure test
checks failed reset initialization. Separate external-KV selection fixtures
retain the 2051 cutoff/tie and two-cell head regression checks. Compilation and
CPU checks do not establish Vulkan hardware correctness or model quality.

## GDN execution policy

The default is `PortableGdnPolicy::eager_diagnostic`. It keeps the existing
committed/candidate host state mirrors, intermediate finite checks, trace
values and rejected-candidate rollback. `--gdn-resident-state` alone retains
this policy. Native CUDA/HIP/SYCL defaults are unchanged.

For qualification, add `--gdn-production --gdn-resident-state` to strict Vulkan
execution. The API equivalent is `PortableModelOptions::gdn_production = true`;
standalone layers select `PortableGdnPolicy::device_production` in their options.
Production requires a selected backend, resident state and CPU fallback off.
The first slice of [issue #66](https://github.com/Zutfen-LLC/Strata/issues/66)
is implemented in [PR #72](https://github.com/Zutfen-LLC/Strata/pull/72).
Parent-dispatched qualification at `66fbe1727b45115d3e4e9a6c03b9f156c823435d`
passed focused Vulkan fixtures and bounded same-backend request comparisons.
Operator evidence and environment details are retained privately. This qualifies
the exercised opt-in path, not a default change, all devices/configurations or
full CUDA/Vulkan product parity. PR #72 remains subject to final review and merge.
The [roadmap in PR #71](https://github.com/Zutfen-LLC/Strata/pull/71) records the
historical baseline; QSA ownership and the other parity slices remain separate.

Production reuses the resident allocation and cached GDN graph. It removes the
recurrent, convolution and PLE-history host banks and their per-token downloads.
It retains separate device commit copies after FFN, rather than adopting native
CUDA's in-place commit. The pinned Vulkan copy API implements these as device
buffer copies. The small attention residual and existing FFN host seam still
transfer and validate results. PLE table gathers and token predecessor tracking
stay on the host; PLE convolution history stays on the selected backend.

`recurrent_state()`, `convolution_history()` and `ple_convolution_history()`
throw in production, even before the first token. Traces require diagnostics;
a production step requesting one fails and poisons the layer. No getter does
hidden I/O. `token_history()`, `tokens()` and `last_residual()` describe the last
successful layer step in both policies, including while poisoned. No snapshot
API is provided in this slice.

Any failed production step poisons the layer; a model step failure poisons the
session, including when earlier layers already committed. No public position,
route or logits are published for a failed model step. A caller must explicitly
reset and replay token IDs. Reset marks state poisoned before work and clears
poison only after all clearing succeeds; a recoverable reset failure stays
poisoned. Reset does not repair invalid weights or a failed device. Backend
process aborts cannot be recovered by this API.

Production adds an integer logits guard after scheduled head completion and before
logits download or token selection. It follows native Verifier's exponent-bit
classification: 16 blocks inspect the raw F32 words with
`(bits & 0x7f800000u) == 0x7f800000u` and reduce with integer OR. There are no
floating operations in the guard, including when reading signaling NaNs.
All 16 status words are overwritten each invocation. The bridge first uploads
a non-success sentinel to every word, then dispatches on the existing Vulkan
backend queue, waits for completion and reads 64 status bytes. Only all-zero
status succeeds. Original logits and final host finite checks remain unchanged.
The CPU reference uses the same raw exponent-bit classification.

The narrow Vulkan bridge is compiled into a build-tree copy of the exact
pinned backend translation unit. The external checkout is not edited. Private
source hashes are checked at configure time; incompatible sources fail clearly.
It reuses backend synchronization, descriptor and buffer utilities, owns its
pipeline and aligned status allocation, and requires the borrowed serial backend
to outlive it. It adds no generic custom-op API or global compiler changes.
The former SUB/SUM guard and its vocabulary-sized subtraction scratch are removed.

`portable_head_test --vulkan INDEX` tests raw finite extrema, signed zero,
subnormals, both infinities and signed quiet/signaling NaNs at multiple lengths,
tails and view offsets. It also injects F32 output weights and Q8 output scales
into the actual scheduled head. Guarded failures must return the distinct
`nonfinite logits (integer status guard)` error before logits download. Repeated
bad/finite cases exercise the same guard with both head encodings. No-argument
head tests use CPU and cannot qualify Vulkan. Parent-dispatched Vulkan
qualification covered these fixtures and the exercised production request path.
Same-backend emitted rows and stops matched their references; resident counters
and lifecycle checks passed. This does not establish cross-backend tensor parity,
broader device admission or completion of the remaining product features.

Admission retains actual device GDN state, candidate graph buffers, weight
placement, metadata and conservative backend-copy staging allowances. It
subtracts only removed host mirror banks. The head reports the guard's actual
aligned status buffer as device state, host metadata/status scratch separately,
and an additional 1 MiB each of host/device allowance for pipeline, descriptor
and driver resources whose exact allocation size is not exposed by Vulkan.
These are conservative admission allowances, not measured driver consumption.
The guard is reused across tokens and destroyed before its borrowed backend.

With timing capture, each production GDN step records one explicit candidate
read/finite check (residual), versus three in diagnostics or four with PLE.
Across 36 GDN layers these are 36 production versus 109 diagnostic reads/scans
per consumed token. The resident harness validates exact policy-bound counts.
Production requires all counters and exactly 36 commits per consumed token.
Commit counts remain one per successful layer. These scopes are explicit API
calls rather than driver traffic measurements. Guard work remains inside head
timing, and is not included in the GDN candidate counters.

The quality and resident harnesses forward `--gdn-production`, record the policy
and bind it in each reference command. Suite references must be generated fresh
with the same binary, placement and policy; diagnostic references are rejected
for production resident timing. Keep same-backend output comparisons, cross-
backend numerical screens and task accuracy separate.

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
use [the versioned operator recipe](../agenthandoffs/portable-model-hermes-v4.md),
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

The completed frozen twelve-case answer screen scored native CUDA **10/12**
and portable AMD/Vulkan **10/12**, with identical output-token sequences on all
twelve cases and the same two misses. Both runtime arms exited successfully;
the unchanged scorer returned **inconclusive-baseline** because native missed.
Numerical parity remains failed. See the [completed screen receipt and limits](../agenthandoffs/portable-quality-results-v1.md).

## Ownership and token semantics

`PortableModelSession` owns the six-shard `GgufModel` for the session's lifetime.
It validates split headers, tensor inventory, payload extents/non-overlap,
active-layer shapes/types and metadata. Embedding, huge PLE tables and routed
expert weights remain read-only mapped. By default, layer/head dense weights
have owned host storage and scheduler backend copies. `--gdn-resident-weights`
retains GDN attention/PLE weights on the requested backend;
`--dense-resident-weights` extends backend ownership to QSA, FFN and head weights.
CPU execution retains CPU storage. The ledger separates persistent host/device
weight allocations from scheduler copies; the flags do not relocate mapped CPU
experts or PLE tables. The actual QSA indexer Q/K matrices are BF16 and are
widened exactly before loading the selected weight backend, using one 10 KiB
row buffer; those projection nodes request F32 accumulation. No re-quantization or native
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

Provider finite-value and per-slot validation keeps diagnostic construction on
failure paths. Every element is still checked in the same order, including
masked routing slots, with the same error text and output-clearing behavior.
This also applies to the existing Vulkan expert boundary validator. Source
identity, digest, ownership and memory checks remain active. No switch is needed;
valid numerical results and execution order are unchanged.

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
comparison recipe. Parent/operator reports all 41 raw logit rows and stdout
byte-identical in the physical V340 pair at `ac85ff3`, with fewer copy/digest
bytes and unchanged native batch counts. The fixed oracle-first timing is
order-biased and does not establish a general speedup. See
`agenthandoffs/portable-cpu-expert-physical-results-v1.md` for the attributed
receipt. The failed CUDA parity screen remains failed.

## Memory and current limits

The report distinguishes unique mapped shard file bytes, persistent host/device
weight allocations, scheduler buffers and policy-specific state/workspace bounds.
The PLE factory may retain another read-only mapping of the same backing
file; that is an address-space alias, not an anonymous 54 GB table copy.
Selected expert staging is 52,224,000 bytes at H2560/FF640/top10.

Actual scheduler reservations are queried per backend, including scheduler
weight copies where required by placement. Admission also includes CPU plans,
metadata, policy-specific histories, pool scratch/stacks, output buffers and
conservative pinned-Vulkan private workspace allowances. QSA's temporary attention
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

The ledger includes the stated guard resource allowances, not a complete query
of driver pipeline/command-pool allocations. Unrelated backend users remain
outside it; operator-selected caps need actual headroom. The ledger alone does
not establish measured VRAM peaks or performance. Eager diagnostic GDN downloads
full candidate recurrence/history mirrors, including with resident state enabled.
Production GDN retains those tensors on device and transfers the residual needed
by the current FFN seam. QSA still stages active host KV/pool histories pending
its separate ownership work; layer residual exchange and final logits readout remain.

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
[operator recipe](../agenthandoffs/portable-resident-hermes-v1.md). Local sparse and
fake-process tests establish lifecycle behavior, not real-model quality, AMD
performance, or cross-backend bit identity. The existing strict numerical parity
failure remains unchanged. Unequal process lifecycles are an unmeasured confounder;
this capability alone does not explain the full native/portable latency gap.


### Physical resident checkpoint validation host

operator [reported a successful AMD resident checkpoint](../agenthandoffs/portable-resident-amd-results-v1.md)
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
The [validation host recipe](../agenthandoffs/portable-resident-nvidia02-v1.md) reuses the
same runtime source and explicit local paths/PCI selection. operator subsequently
[reported validation host PASS](../agenthandoffs/portable-resident-nvidia02-results-v1.md):
one process completed all four matching token/greedy-row results, GPU entries
173/239 per request, and zero verification/copy/digest/invalidation deltas on
requests2–4. All six focused CTests and six harness tests passed; exit0/reaped,
no child swap/new swap-out, baseline9MiB/1MiB VRAM and no owned process remained.
This is same-backend lifecycle success, not cross-backend parity or performance
evidence. No clocks, models or network services were changed. This documentation
update authorizes no additional run and leaves PR49 runtime/pinned AMD execution
unchanged.


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

### Coarse expert provider and executor boundaries

The [physical pool-timing receipt](../agenthandoffs/portable-pool-phase-amd-results-v1.md)
reports about1.55s of CPU pool elapsed in warm requests with about46.6s of
prompt/decode wall. The remainder includes other CPU/preparation/GPU/transfer and
model/backend/reporting work; it is not identified as GPU compute.

`--stdio --expert-timings` explicitly enables two coarse steady-clock boundaries.
Without --expert-timings, the new scopes read no clock and emit no new fields;
existing defaults, pool counters, one-shot outputs and timing-file schema remain
unchanged. --expert-timings requires --stdio in the CLI. The model API equivalent
is PortableModelOptions::expert_timings, default false.

Whole-provider timing starts at entry to the model's expert-row callback and
ends on return/unwind. It includes optional route logging, provider source/resource
checks, request preparation, CPU expert copy/digest/pool calls, local GPU expert
calls, reply validation and row scatter. Routing/activation preparation before
callback invocation and weighted merging after callback return remain outside.
The cumulative PortableExpertWork provider_ms/provider_calls survive reset.
Done receipts report expert_provider_ms_delta and expert_provider_calls_delta.
There is one provider callback per layer per consumed token.

Each local device's infer_rows_ms/infer_rows_calls measures only its existing
executor infer_rows invocation and return/unwind, excluding adapter checks and
request preparation outside that call. It includes the executor's host work,
transfers and existing backend waits; it is **not pure GPU execution time**. No
additional synchronization is inserted. Done receipts report ordered device
arrays gpu_infer_rows_ms_delta and gpu_infer_rows_calls_delta, with the same
ordering as the expert-device report. Devices with no participating call record
zero. Loading/prepare_source is excluded.

All these counters are cumulative, not maxima. Failed attempts are captured on
unwind but produce no successful done receipt; copied session device reports may
be stale after a failed step and must not be used as final failure measurements.
Resource sampling remains independent. Enabled scope/call counters accumulate
across reset; reading them adds no timing or work.

Provider elapsed encloses its CPU-pool and GPU-executor intervals. In the current
serial adapter, device calls run sequentially; their time remains inside provider
time. Never add provider time to pool/executor time. Existing pool enclosing time
also overlaps its own phases. Whole-request deltas do not split prompt/decode
pool/provider time. Time outside provider includes model/backend work, other CPU
work, preparation, transfers and reporting; it must not all be called dense/GPU
compute. See the [bounded four-request recipe](../agenthandoffs/portable-expert-boundary-hermes-v1.md).
Shared compiled libraries within one build establish no portable-only scalar
kernel; separately deployed build ISA parity remains unverified.

### Coarse model phase totals

With `--stdio --expert-timings`, model_work() exposes cumulative model-step,
GDN-layer, QSA-layer and output-head host-wall milliseconds and call counts.
The same opt-in controls all new scopes; defaults take no new clock readings.
Successful done receipts add model_step_ms_delta, gdn_layer_ms_delta,
qsa_layer_ms_delta, output_head_ms_delta and model_residual_ms_delta, plus their
step/layer/head call-count deltas. Counters survive reset; snapshots are read-only.
No per-op instrumentation, new GPU synchronization or execution change is added.

A layer interval encloses the existing whole Layer::step call, including residual
copy, validation, transfers, backend waits, preparation, FFN and expert-provider
calls. It is not attention-only or pure GPU time. A head interval encloses the
existing output-head call and its validation. Model-step elapsed encloses these
serial layer/head intervals and the rest of PortableModelSession::step, including
embedding gathering/broadcast, placement/report rebuilding, state bookkeeping and
normal return cleanup. Reset/load and the CLI consume wrapper remain outside.

The residual field is **model_step minus GDN minus QSA minus output_head** from
matched request deltas. It covers model work outside those intervals, not an
additional enclosing interval. The request's prompt+decode wall minus model-step
elapsed also includes CLI source checks, device-memory queries, expert-work logs,
output/greedy selection and other wrapper work; it is separate from model residual.
Report rounding/subtraction tolerance rather than silently clamping differences.

GDN/QSA totals each include their provider calls. gdn_provider_ms_delta and
qsa_provider_ms_delta capture differences in the existing provider accumulator
around each matched successful layer call, adding no new clock boundaries.
Subtract only the matching family provider interval to derive a non-provider
family remainder. Their sum reconciles to expert_provider_ms_delta for successful
requests. Do not add layer totals to provider/pool/executor times or add enclosing
model-step to its phases. Provider/pool/executor nesting remains as documented.
Current complete tokens call36 GDN layers,12 QSA layers and one output head.
Failed attempts may leave partial totals; no successful done receipt is emitted,
and the normal fail-closed/poison contract is unchanged.

GDN refinement under the same opt-in adds three host-wall scopes:

- `gdn_attention_exchange_ms_delta`: optional PLE gather, input writes and
  policy-required state/history transfers, attention scheduler call, candidate
  reads and finite checks. Explicit trace reads, when a direct API caller requests them, are included too.
- `gdn_attention_compute_ms_delta`: only the existing attention scheduler
  graph-compute call, nested inside attention/exchange. It includes existing
  backend waits and is not GPU kernel time or a measurement of PCIe traffic.
- `gdn_ffn_ms_delta`: reset, complete FFN step (including shared expert, provider,
  host merge and pre/post graphs) and residual copy.

Each has a corresponding `_calls_delta` field. All three counts match the GDN
layer count on a successful request. `gdn_outer_residual_ms_delta` equals GDN
layer minus attention/exchange minus FFN. `gdn_attention_surrounding_ms_delta`
equals attention/exchange minus compute; it includes transfers, finite checks,
PLE and host work outside that scheduler call. `gdn_ffn_non_provider_ms_delta`
equals full FFN minus the matched GDN provider delta. These are nested scopes and
subtractions, never extra disjoint intervals to add to their parents. The harness
checks finite nonnegative values, matching counts, containment and reconciliation.
Direct GDN callers opt in by passing a caller-owned `PortableGdnWork` accumulator;
null reads no clock. Failed calls can leave partial attempted-call counters.

Static source inspection identifies candidates to measure, not demonstrated costs:
QSA constructs/schedules a token-specific attention Phase and stages active
histories; eager diagnostic GDN reads full candidate state/history, while
production GDN retains those tensors on device and exchanges the FFN residual.
The head retains its main graph, checks integer status in production, then reads
logits. The model rebuilds a placement vector after each successful token; the
CLI queries device memory and writes per-token diagnostics. Memory-admission account()
and module memory_usage() calls are in construction, not the main token loop.
No existing coarse counters covered these model totals. This patch neither
removes nor attributes those candidate costs. Non-provider time can include other
CPU/preparation, GPU/transfer and reporting work; it is not all dense/GPU compute.
Operator receipts and machine-specific run instructions are retained privately.
