# GDN attention weight residency trial

`--gdn-resident-weights` is an experimental, default-off placement option. It
uploads GDN attention weights once into the layer's selected backend instead
of streaming CPU-backed weights through scheduler splits on every step. CPU
execution retains CPU weight allocation. FFN weights, recurrent-state exchange,
expert plans and numerical acceptance limits are unchanged.

The startup ledger reports actual aligned persistent weights separately from
scheduler bytes. Both count against the existing per-device cap, alongside
workspace and resident expert budgets. The option can increase retained memory
because scheduler copies previously reused storage. Admission refusal does not
reduce expert plans or relax caps or free reserves. Placement can also change
scheduler splitting and fusion, so compare the existing numerical references
before drawing performance conclusions. No speedup or full-model fit is implied.

The CLI accepts the flag with the existing portable model options. Omit it to
retain the original placement. Check startup `actual_weight_bytes`,
`actual_scheduler_bytes`, `admitted_bytes` and `cap_bytes` separately for each
physical backend. Backend fallback remains governed by `--allow-cpu-fallback`.

The existing `portable_gdn_vulkan_test` and `portable_ple_vulkan_test` exercise
both placements against their CPU/scalar references, including reset and
rejected-callback rollback. Hardware execution is required to validate Vulkan
numerics; CPU-only test success does not qualify Vulkan residency.
