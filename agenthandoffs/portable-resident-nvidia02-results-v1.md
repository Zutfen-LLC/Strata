# NVIDIA02 resident sequential-request result

Hermes reported **PASS** for the bounded NVIDIA02 resident checkpoint at runtime
source `45513fd885babcf81d252b73804a14d2a751a74a`, binary SHA256
`34b1e8ac188b39fcf511faf94999ef1896aaaad5654ff6fb4bd5c07526e70597`.
[Primary Hermes report](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793186598876119).
This documentation task read that report; it did not run fleet commands.
All six sequential focused CTests and all six harness tests passed.

## Lifecycle and residency

One supervised model process completed all four requests, then exited with code0
and was reaped. All four token and greedy rows matched. Each request recorded
173/239 GPU entries on Vulkan0/Vulkan1. Requests2–4 each recorded zero CPU
verification, copy, digest and source-invalidation deltas. The same-backend
sequence therefore reused the verified immutable expert views and resident plans
without reloading between requests.

Vulkan0 mapped to PCI0000:02:00.0 and Vulkan1 to0000:03:00.0, the existing
RTX3060Ti/RTX3060 selection. The unchanged480-per-device plans and caps were used;
no new placement/ranking was introduced. The [reviewed recipe](portable-resident-nvidia02-v1.md)
records the exact source/input identities and explicit local02 model/manifest paths.

## Resources, cleanup and retained evidence

Resource guards passed: no child swap or new swap-out was observed. VRAM returned
to its9MiB/1MiB baseline, and no owned process remained. At the report time neither
the build/tests nor the four-request process was active. The prior PR47 binary
hash also verified unchanged.

Evidence remains under `/home/hermes/nvidia-resident-45513fd8`. Preserve source,
binary, input-plan/token identities, tests, four request receipts, resource/exit/
cleanup records and the prior qualified artifacts. The report does not supply
per-stage latency or peak-memory values; none are invented here.

No duplicate inference launch, model-weight transfer, clock change, native rerun,
network service or additional hardware run occurred for this result/documentation
update. This is a successful same-NVIDIA-backend lifecycle/reuse check. It does
not establish cross-backend numerical parity, full resident quality, matched
native performance or a general speedup. Original strict parity remains FAILED.
The [AMD resident result](portable-resident-amd-results-v1.md) is a separate
observation, not a bit-identity or performance comparator.

This documentation update changes no runtime code or pinned source. PR49's
separate CPU pool timing exposure and its pinned AMD execution remain unchanged.
The completed NVIDIA checkpoint authorizes no new experiment.
