# Serial Vulkan memory admission correction

**Physical480-per-die admission and execution subsequently passed.** See
[the result](portable-local-residency-480-results-v1.md). Local-only statements
below describe the original coding checkpoint.

The physical16-expert smoke revealed a real admission problem, not an allocation
failure: the old ledger charged about5.1 GB of extra workspace per die and refused
both retained larger plans. No larger run, plan edit or cap increase occurred.
The observed1.2 GiB usage is not substituted for a bound.

## Cause and safe lifetime split

The model executes layers serially. QSA constructs `Phase attention` on the stack
in `PortableQsaLayer::step`; it remains live through the FFN/expert callback and
is destroyed before the next layer. Fixed schedulers and expert tensors remain
live throughout. Therefore persistent bytes **sum**, while only one QSA temporary
scheduler allowance per physical backend can be live. That temporary peak is
still **added** to resident expert tensors and workspace, not shared with them.

The previous QSA `graph_budget - fixed_scheduler_buffers` mixed fixed private
workspace with the temporary. The corrected guaranteed temporary is precisely
`graph_budget - fixed_graphs`; `fixed_graphs - fixed_scheduler_buffers` stays in
private accounting. No whole `device_extra_bound` maximum is used.

A second source-grounded correction separates only full-matrix conversion scratch
for verified GGML Vulkan backends. At the retained GGML pin
3cf03257f219afbe7334045ff7c6a06ac68c627d, one `ggml_backend_vk_context` owns one
`prealloc_x`. Matrix conversion raises its required capacity by maximum; buffer
growth synchronizes, destroys the previous buffer and allocates the replacement.
Graphs/layers borrowing the same backend do not each own a full matrix-conversion
buffer. Portable model backends are unique per physical die, and local expert
execution borrows those same backends serially.

Only the existing full-F32-image allowance is max-reduced across these verified
Vulkan contexts. Y/input conversions, split-K, radix/prefix, metadata, other
private allowances and retained scheduler buffers keep their additive bounds.
Unverified/non-Vulkan backends keep the previous additive conversion accounting.
The head's full-F32 conversion allowance is retained; it is not replaced by a
sampled measurement or an assertion that the quantized fast path always applies.

Pinned source anchors (local retained snapshot, path provenance as before):

- ggml-vulkan.cpp: backend initialization around5182, matrix X sizing/max around
  6100–6146, synchronized replacement around11714–11754, backend cleanup12435.
- ggml-vulkan.cpp: buffer free around12495 calls ggml_vk_destroy_buffer;
  ggml-vulkan-buffers.cpp:232 releases the owning buffer reference.
- portable_qsa.cpp: fixed_graphs construction139–151 and fixed phases;
  stack attention438 onward, FFN callback before return; memory_usage split.
- portable_model.cpp: serial layer loop and per-device account; borrowed expert
  allocation is prepared only after dense/head admission.

The ledger now charges per die:

```text
sum(retained scheduler bytes)
+ sum(other private allowances)
+ max(verified Vulkan matrix-conversion capacity)
+ max(QSA token-temporary scheduler capacity)
+ resident expert tensor cap
+ separate resident runtime allowance
```

CLI die receipts expose `backend_private_bound`, `shared_conversion_bound`,
`token_temporary_bound` and the total. Caps remain7680 MiB/die, expert runtime
allowance128 MiB/die, free reserve512 MiB/die; host admission is unchanged.

## Supported larger count from the supplied smoke

Original smoke fields from parent/operator: each16-expert plan owns84008960 padded
tensor bytes. Thus480 experts need2520268800 bytes and a rounded2404 MiB cap;
1152 need6048645120 bytes and5769 MiB. No extrapolation from raw weights alone.

Even **ignoring all QSA temporary-sharing savings**, removing only repeated
matrix-conversion charges gives these conservative upper bounds:

| Bytes | Die0 | Die1 |
| --- | ---: | ---: |
| Dense upper bound | 2545020672 | 4441534336 |
| Plus480 tensor cap and128 MiB runtime | 5200015104 | 7096528768 |
| Unchanged die cap | 8053063680 | 8053063680 |

Derivation uses supplied retained scheduler bytes1256031232/1301101056 and old
workspace bounds5105806080/5111756160. For32/16 layers subtract their repeated
12.5 MiB FFN conversion images; for24/12 GDN layers subtract100 MiB each; for8/4
QSA layers subtract120 MiB each. Add one conversion peak per die:120 MiB on die0,
the unchanged2542796800-byte head bound on die1. On die1, removing its old head
charge and adding the same head peak cancel. QSA temporary max-sharing can only
reduce these upper bounds further. Other private allowances are untouched.

1152 is still ruled out on die1 by its retained scheduler plus conservative head
conversion bound plus the5769 MiB tensor cap and128 MiB runtime, before other
allowances. Select the existing **480-per-die pair**, not an arbitrary smaller
placement, and retain1152 for later evidence-driven work. Live free-memory and
selected-plan identity checks still apply; these are admission bounds, not a
physical allocation/performance result.

## Local evidence and limits

An extracted original production account() repro refused eight serial256 MiB
phase allowances under512 MiB (`RED: ... separate die0 admission cap exceeded`).
The new production ledger regression checks persistent/private sums, serial
peaks, later peak growth, independent dies, resident overlap, transactional cap
refusal and overflow. A real QSA-layer test checks the exact fixed_graphs split,
conserves the one-layer total on an unverified backend, and exercises the expert
callback while the attention scope is live. Existing QSA/model and CLI
stdout/logit differentials remain selected checks. Raw local receipts are in
untracked validation/accounting-*.log. No debug instrumentation is in source.

This C++ correction needs a fresh isolated build and binary SHA; do not reuse the
source05601aba binary for the larger run. No physical validation of the corrected
ledger has occurred in this task. No fleet SSH, model/clock operation or merge.
See [the updated larger handoff](portable-local-residency-large-v1.md).

Final local checks on this correction: fresh target rebuilds in both retained
CPU and Vulkan Release configurations; portable_qsa_test,
portable_qsa_memory_test, portable_memory_admission_test,
portable_timing_cli_test and portable_model_test all passed (5/5 each). CPU total
44.38 s; Vulkan-linked total46.24 s. The Vulkan-linked tests use the CPU fixture
path, not physical GPU execution. All32 Python controls passed. Recipe shell
syntax and diff checks passed. CUDA/HIP/SYCL builds and physical corrected-ledger
execution remain unrun; preserve draft status for independent review/dispatch.
