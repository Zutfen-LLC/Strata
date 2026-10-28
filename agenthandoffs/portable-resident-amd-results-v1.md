> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# validation host resident sequential-request result

operator reported a physical resident checkpoint PASS at runtime source
`45513fd885babcf81d252b73804a14d2a751a74a`, binary SHA256
`c891f38e24654d2f3c6dfdd7f6a21c01304a3d9c8a2445e3d0b933d8aa533b8b`.
[Primary operator report](PRIVATE_RECEIPT).
This documentation task read that report; it did not run fleet commands.
Configure/build at parallelism2, all six sequential focused CTests and all six
harness tests passed on05; no packages were installed.

## Lifecycle, work and timings

One loaded model child completed one excluded warmup and three measured identical
requests, then exited cleanly after QUIT. All four token/greedy-row results were
identical and each request generated four output tokens. The unchanged retained
480-per-die plans were used; no placement ranking or cap changed.

| Request | CPU verifications | CPU hits | Copied bytes | Digest bytes | GPU entries die0 / die1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 4476 | 9039 | 23375462400 | 23375462400 | 170 / 235 |
| Warm1 | 0 | 13515 | 0 | 0 | 170 / 235 |
| Warm2 | 0 | 13515 | 0 | 0 | 170 / 235 |
| Warm3 | 0 | 13515 | 0 | 0 | 170 / 235 |

Requests2–4 each also recorded zero source-invalidation deltas. Verified immutable
CPU expert views and GPU plans remained resident across request resets; repeated
requests did not reload the model or redo already completed CPU expert verification.

One-time load was **18190.24ms**, separate from every request below. All values
are MEASURED milliseconds from the reported per-request receipts.

| Request | Setup/reset | Prompt | Decode | Selection prompt | Selection decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 18.51 | 74466.64 | 10456.31 | 72456.68 | 10177.34 |
| Warm1 | 22.66 | 39977.96 | 6476.94 | 38374.88 | 6466.37 |
| Warm2 | 21.31 | 40030.87 | 6422.47 | 38431.56 | 6417.71 |
| Warm3 | 19.72 | 40059.00 | 6388.57 | 38462.53 | 6389.61 |

Selection intervals overlap prompt/decode work and use the existing final-prompt
selection boundary; do not add the columns together. Keep warmup out of measured
request aggregates while retaining its first-use evidence. The eliminated copy/
digest work is directly observed; this checkpoint does not isolate it from all
other first-use/page/backend warming effects. These samples do not establish a
general performance verdict or justify comparison with an unequal CUDA setup.

## Resources and cleanup

operator retained1118 samples across the process. Peak VRAM was
3794292736/3839123456 bytes on dies0/1. Minimum effective host availability was
261151698944 bytes and peak child RSS was39627325440 bytes. Child swap and new
swap-out remained zero. Both dies had nonzero-busy samples at1500/945MHz;
maximum sampled temperature was60°C.

Both clock controls returned to auto, VRAM returned to8339456 bytes per die,
and no owned model or guard process remained. The evidence root is
`${PRIVATE_ARTIFACT_561FB951}`, reported total
627700234 bytes. Preserve the raw request/resource/clock/cleanup receipts,
source/build/binary identity and old artifacts; do not replace them with rounded
values from this summary.

## Scope and next checkpoint

This is a successful bounded physical lifecycle/reuse observation for one repeated
frozen case, not a full resident quality screen. Original strict numerical parity
remains FAILED; cross-backend bit identity is not claimed. No new quality score,
matched CUDA comparison or generalized speedup is established. Observation,
acceptance and execution authority remain distinct.

The [validation host variant](portable-resident-validation host-v1.md) uses the same pinned runtime
source and the existing explicit model-root/manifest/device/PCI options. It is
published for parent review and operator dispatch; no NVIDIA resident run or new
hardware authorization is claimed by this documentation update.
