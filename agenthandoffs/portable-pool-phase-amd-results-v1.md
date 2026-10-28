# AMD05 existing CPU pool phase result

Hermes reported completed receipts for source
`f8deab2d97baf8534b4737d23b401d9de85c7f4d`, binary SHA256
`09d982bb1c8ad83093233ec2b60bc804aa6ea0d4f03b2e113fee60ddd5c96298`.
[Primary report](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793187295876749).
This documentation task read that report; it did not run fleet commands.

All four token/greedy rows were identical. Each request recorded170/235 GPU
entries and13515 CPU entries with the retained480 plans. All three measured
requests had zero verification/copy/digest/invalidation deltas. All pool fields
were present, finite/nonnegative, and phase sums were below enclosing elapsed.

All table values are MEASURED milliseconds, as reported by Hermes:

| Request | Prompt wall | Decode wall | Pool gate/up | Pool quant | Pool down | Pool enclosing elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warmup | 75381.442 | 10216.139 | 1240.462 | 7.734 | 575.898 | 1824.528 |
| Measured1 | 40156.802 | 6448.293 | 1074.781 | 7.115 | 467.153 | 1549.480 |
| Measured2 | 40138.943 | 6458.418 | 1073.068 | 7.137 | 467.582 | 1548.223 |
| Measured3 | 40041.137 | 6430.655 | 1075.009 | 7.011 | 466.908 | 1549.346 |

The enclosing pool interval overlaps its gate/up, quant and down phases; do not
add it to them. Gate/up and down include pool synchronization. Warm request
prompt/decode wall is about46.6s, while enclosing native pool elapsed is about1.55s.
Most time therefore falls outside those native pool calls in this run. That time
includes other CPU/preparation work, GPU experts, model/backend, transfers,
merging and reporting; it is not evidence that all of it is GPU compute or dense
compute. These observations do not establish a matched CUDA performance verdict
or isolate individual causes of warming.

Both dies had active1500MHz core/945MHz memory samples; maximum sensors were
50°C/62°C. Exit was0, child reaped, no owned model or guard remained, and both
dies returned to auto, idle, at8339456B VRAM baseline. Receipts remain under
`/srv/models/inferswarm-artifacts/portable-pool-f8deab2d`, notably
`run/summary.json` and `checks/final-verification.json`. Preserve raw receipts and
prior binaries/evidence. No full resident quality/parity or ISA-parity claim follows.

The [next bounded recipe](portable-expert-boundary-hermes-v1.md) adds only opt-in
whole-provider and per-device executor host-call timing, to locate time inside
versus outside the provider. It remains pending parent review/Hermes dispatch;
this result authorizes no extra run or optimization. The separately pinned
head-admission physical job and NVIDIA45513fd8 run are unchanged.
