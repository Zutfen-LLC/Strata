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
use [the versioned Hermes recipe](../agenthandoffs/portable-model-hermes-v3.md),
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

## Ownership and token semantics

`PortableModelSession` owns the six-shard `GgufModel` for the session's lifetime.
It validates split headers, tensor inventory, payload extents/non-overlap,
active-layer shapes/types and metadata. Embedding, huge PLE tables and routed
expert weights remain read-only mapped. Layer and head weights have owned
host copies and scheduler backend copies.

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
checked against the retained blob digest before execution. There is no setup
scan of all expert bytes and no new full-file checksum claim.

Standard `NativeFmt` Q8 experts consist of the GGUF gate, up and down slices
in order. The provider validates row sizes, offsets and blob size against
`native_fmt`, copies the selected slices into a reusable ten-expert staging
arena, quantizes with `native_quant_act`, and calls
`ExpertPool::run_split_multi_native`. It never substitutes the canonical
engine's packed weight layout, expands experts to F32, or creates another
unbounded arena. Rows are unweighted and follow route order. `PortableFfn`
alone applies router weights and reduces them.

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
