# Native T1 QSA scores and selection component

This standalone opt-in component follows `Verifier::record_window()` →
`idx_select` → `qsa_block_scores()` / `qsa_block_topk()` in
`src/kernels/cuda/qsa_select.cu`. It does not yet replace model QSA attention or
selection. Existing defaults, scheduler and placement remain unchanged.

The supported native dispatch is the default Ampere single-query
`block_scores_multi_body` and `block_topk_reg_kernel<33>`, with unset alternative
native dispatch controls. The CUDA probe calls those unchanged public launchers;
it refuses other physical architectures or explicit dispatch overrides.

The score uses the native float4 product association (one rounded multiply,
three FMAs), 32-lane XOR reduction, four ordered per-head ReLU additions and
finite `1e9` incomplete-tail bias. The selector retains the native 1024-thread
weighted four-pass radix algorithm, lowest-cell tie allocation and ascending
selected IDs. The independent CPU/offline selection oracle sorts cells rather
than duplicating radix selection. NaN scores sort lowest; signed zero is
normalized; infinities retain the native defensive ordering. These selector-only
cases do not expand the score arithmetic domain beyond finite normal/zero.
There is no fitted epsilon or intrinsic allowance: local arithmetic and integer
policy checks require exact words.

The bounded geometry is one query, 128 indexer dimensions, four query heads,
block size four and up to 4096 cells. Native `Verifier` score stride is
`max_cells/4 + 2`, including padding. Capacity is **2051**, from native
`qsa_selection_width`, not 2048. Through 2051 cells the native selection is the
identity branch. The competitive branch is tested synthetically at 2052, 2053
and 4096 cells; a short-context model test cannot qualify it.

Vulkan only substitutes buffer bindings, submissions and shared-memory logical
warp shuffles. It requires 1024 workgroup invocations and **37004 bytes** shared
storage (native histogram/scan storage plus 4096 bytes for logical warp shuffle
emulation). Unsupported devices refuse before submission; the component does
not switch algorithms. Its tensor allocation is capped at 2 MiB and pipeline
metadata at 128 KiB. Before/input tensors and candidate outputs must be disjoint.
Invalid geometry, aliasing and unsupported modes refuse without paid work or
candidate mutation. Completed dispatches are counted, reset/repeat and
refusal/retry are checked, and caller state remains outside this standalone API.

Eight locally generated fixtures cover zero, dense, cancellation, ties,
incomplete tail, NaN/Inf/zero ordering, the identity boundary and finite tail
bias. No model weights or captured activations are transferred. The probes
check canaries and immutable inputs, exact independent source-order scores and
cell-policy IDs; the offline checker also rejects common wrong backend outputs.
Module tests validate strict FMA, every basic operation's `NoContraction`, helper
transformations and mutation refusal. The actual producer's pre-backend output
contract is tested from an unrelated working directory.

Local Vulkan compilation and host tests are not physical qualification. Native
CUDA compilation and native/Vulkan execution are required next. No model,
window scheduling, full QSA, quality or performance parity is claimed.
