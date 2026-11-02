# Native QSA score/selection model seam

Default-disabled `--native-qsa-score-select-t1` replaces only ordinary eager T1
indexer scoring/selection. Both arms explicitly use
`--qsa-score-select-banks PREFIX`, native indexer append/pooling and its matched
resident role. The earlier HC/GDN/QSA norm methods remain fixed. Unsupported
production/chunk/fallback/alternative native dispatch modes refuse. Existing
indexer scope keeps model context at most 256; competitive selection above 2051
cells is **not** exercised by this model diagnostic.

At the actual completed projection/append boundary, the component reads the
current native indexer candidate pooled rows and persistent dead/spare, actual
normalized projected indexer query and native step tuple. Score transport uses
native `capacity/4+2` stride; append state remains `capacity/4+1`. The original
multi-score/register-radix methods produce scores and ascending IDs. IDs are
mapped exactly to the existing attention graph's 0/1 mask input. That conversion
is buffer plumbing; QSA attention math and output projection are unchanged and
remain separate native-method gaps.

The score/ID bank is detached from persistent state. Failed score, selector,
attention or FFN work cannot publish native indexer/KV/host state. Existing
FFN-success commit, reset and retry rules remain. Completed native dispatches
are counted even when downstream work fails. First-layer positions 0–7 retain
current input/output and actual consumer links; failure-only receipts exclude
uncompleted outputs. They do not prove every layer's arithmetic or whole-window
state authority.

Both arms allocate the same complete score banks and two pipeline/status owners.
At capacity 256 the logical leaves total 42788 bytes before backend alignment,
plus both component owners. Admission estimates and then checks actual allocation;
1 MiB host allowance per ordinary QSA layer covers bounded mirrors/metadata.
The original planner and per-arm actual ledger decide placement/fit; caps,
reserves and CPU miss behavior do not change. No old placement is assumed.

Local real Vulkan/model build, CPU actual producer/attention-consumer checks,
score/selector and FFN failure noncommit, reset/retry, paid counters, graph/host
budgets and actual CLI mode/refusal tests pass. The default-path CPU full model
test also passes. Device primitive behavior is mocked in the CPU transaction
test and is not hardware evidence. The model handoff first requires the frozen
standalone component report, then uses the existing fresh planning/same-image
control/on workflow. No local fleet/model invocation occurred. No full native
QSA/window, quality or performance parity or default enablement is claimed.

## Bounded physical result

Hermes reports the PR131 workflow passes at
`64a6044752dfc4443f3823f857bd9c043500d855`: build and 27 CTests, fresh planning
READY/QUIT with no GEN, then two reset-separated requests in each same-image
control/on arm. Every process exits/reaps, the controller exits zero, no active
children remain and no sealed STOP is present. This is the reported existing
recipe result, not an additional local hardware run. Context 256 exercises
identity selection only. No full native attention/window parity is established.
