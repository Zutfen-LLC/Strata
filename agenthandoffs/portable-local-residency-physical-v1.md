> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Physical local expert-residency smoke result

operator completed the bounded discovery and resident one-token runs on 05.
[Primary receipt](PRIVATE_RECEIPT).
C++ source 05601aba4960da84b8556df57844c4bd3dce1f6b, Python source
0ff4923fb9dd53aec0ad247c6492c4e47cb3c90a, binary SHA256
`7d8718ac7dfa9897b248fb208f72117f3c731c765ed10697e0c70e2bc0d59ae4`.
Four focused CTests passed sequentially before physical execution.

Each die had 16 assigned experts total, 16 actual Vulkan expert entries and 16 CPU
reference row checks. Total dispatch was 32 GPU entries and 448 cold CPU entries.
Vulkan0 row max absolute error was 1.19e-7, relative L2 1.30e-7; Vulkan1 values
were 4.77e-7 and 1.33e-7. These are diagnostics, not a new cross-backend tolerance
acceptance or full-model parity result. No larger placement or quality suite ran.

| Sampled resource | Die0 | Die1 |
| --- | ---: | ---: |
| Peak used VRAM bytes | 1351016448 | 1395855360 |
| Minimum free VRAM bytes | 7222140928 | 7177302016 |
| Restored baseline VRAM bytes | 8339456 | 8339456 |

Minimum effective host availability was 261550075904 bytes; sampled process peak
RSS was 12782338048 bytes. Swap use and swap-out delta remained zero. Maximum
sampled temperature was 48 C. Both controls were high during the guard, restored
to auto and independently read back. Samples showed 1500/945 MHz, but the sampler
caught nonzero busy only on die0: active-clock coverage on die1 is not established.
Both model children were reaped; no owned model/guard remained. Prior qualified
binary identity remained unchanged.

Evidence remains at
`${PRIVATE_ARTIFACT_E375CE52}`, including
plan hashes, full row metrics, compact resource receipts and clock logs. This
coding task read the linked report; it did not SSH hosts or independently fetch
private raw files. No runtime throughput conclusion follows from this one-token
functional proof. The original full-model logit-parity gates remain FAILED.

Next steps are in [the larger-placement handoff](portable-local-residency-large-v1.md).
