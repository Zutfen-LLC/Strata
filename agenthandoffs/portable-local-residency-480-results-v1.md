# Physical480-per-die local Vulkan result

Hermes completed the resident token and one frozen answer case on05 using C++
source `6d433cedc2435ea2aea7038ef62cef49a00a32b9`, binary SHA256
`560507fa0b87a31206e03f172796fb3e60aae47ff5916aa45bd0de534bddae1b`.
All five focused CTests passed sequentially before the physical run.
[Primary result](https://zutfen.slack.com/archives/C0C5MK91H3L/p1793165190231519).
This coding task read that report; it did not SSH or run fleet inference.

## Admission and actual expert execution

Both unchanged retained plans loaded480 experts per die. The corrected component
sum gate passed under the unchanged per-die cap8053063680 bytes:

| Bytes | Die0 | Die1 |
| --- | ---: | ---: |
| Admitted total | 4652268128 | 6861780064 |
| Actual resident tensor buffers | 2520268800 | 2520268800 |
| Tensor cap | 2520776704 | 2520776704 |

The one-token control recorded9 Vulkan entries /9 CPU-reference row checks on
die0 and6/6 on die1. Row max-absolute /relative-L2 diagnostics were
1.696e-4 /0.0011651 and5.960e-8 /1.247e-7, respectively. Report these observed
errors; they are not a newly asserted tolerance acceptance or full-model parity.

The source/lifetime accounting correction therefore passed this physical
admission/execution check without enlarging caps, reducing reserves or substituting
sampled usage for admission. This does not establish that every possible plan,
backend or future workload is qualified.1152 per die remains excluded under the
retained conservative head bound; no such run occurred.

## Exactly one frozen case

Frozen case1 completed exactly once, passed unchanged gold, and generated four
tokens ending in EOS. Cumulative GPU expert entries were170/235 on the two dies;
cold CPU entries were13515. CPU row-reference diagnostics were off for this timed
request. Routing weights and original-order unweighted-row merge were unchanged.

| Phase | Seconds |
| --- | ---: |
| Setup | 18.086 |
| Full-consumption prompt,25 steps | 74.720 |
| Full-consumption decode,4 steps | 10.223 |
| Selection prompt | 72.748 |
| Selection decode | 9.890 |
| Final generated-token feed | 2.304 |

This is one unmatched sample in a different expert placement from earlier runs.
It is not a steady-state TPS measurement, controlled speedup, or CUDA/Vulkan
backend verdict. The selection phase includes the final prompt forward step in
decode; the portable terminal feed remains separate. Original full-model
logit-parity gates remain FAILED. Correct case1 is a narrow answer check.

## Resources, clocks and cleanup

Token-run sampled peak VRAM was3789426688/3834265600 bytes. Swap use and swap-out
delta remained zero. Both dies had nonzero-busy samples at1500/945 MHz, with
maximum sampled temperature55 C. Both controls were restored to auto, VRAM
returned to the8339456-byte baseline per die, and no owned model/guard remained.
Both retained plan hashes and the earlier16-per-die binary hash still matched.

Artifacts remain under
`/srv/models/inferswarm-artifacts/portable-local-residency-large-v1`. Hermes
reported662937942 bytes total, including source72960636, build512201715,
checks49259 and evidence864892 bytes; the listed subtrees are not a complete
breakdown of that total. Keep the raw receipts and existing qualified artifacts.
No deletion or regeneration of old model/cache payloads occurred here.

## Quality follow-up status

The primary result predates quality dispatch: no12-case result was completed at
that checkpoint. Parent subsequently released the unchanged frozen12-case suite
with these same plans and no native rerun. That dispatch explicitly authorized a
65-minute finite high-profile guard, adapter limit3600 s and outer limit3650 s;
existing temperature, sensor-failure, owned cleanup and auto-restoration rules
remain unchanged. This is a specific parent-authorized extension for that screen,
not a general relaxation of previous30-minute recipe limits.

No quality score or cleanup result for that follow-up is claimed yet. Retain
partial STOP on failure; no automatic retries, altered gold or tolerance changes.
