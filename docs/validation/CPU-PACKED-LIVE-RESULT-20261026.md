# Issue #14: terminal live comparison — PASS

Keep #14: the bounded comparison showed a useful improvement. The completed comparison used the existing runner for one original short prompt, raw then packed, with identical model/settings and the same split residents on the established 1 Gb/s setup.

| Measure | Raw | Packed |
| --- | ---: | ---: |
| Prompt seconds | 4.5332 | 3.8928 |
| Decode seconds | 38.8994 | 34.3038 |
| Request wall seconds | 43.434 | 38.198 |
| Decode tokens/s (213 tokens) | 5.48 | 6.21 |
| Request bytes, both peers (MB as reported) | 247.53 | 67.90 |

Both arms matched the original reference: 213 tokens, 102724 entries, and bitwise-equal first logits. Request wall fell 5.236 seconds, about 12.1%. Request bytes fell 72.6%; this is not total-wire reduction. Replies were unchanged at 509.9 MB and 543.0 MB for the two peers.

Both active peers demonstrably used packed requests, with zero fallbacks or failures:

| Peer | Packed requests | Raw request bytes | Packed request bytes | Reply bytes, unchanged |
| --- | ---: | ---: | ---: | ---: |
| Local | 11083 | 123729266 | 33918090 | 509897520 |
| Remote | 11088 | 123801696 | 33981440 | 542999844 |

Group packing took 120.18 ms across 11136 calls and 12000 quantized rows, included in wall time; row reduction took 219.2 ms. These are coordinator/group-dispatch counters, not worker compute time.

Restoration completed: the full raw-capable resident was restored, complete and forever idle, with one allocation and zero owned swap. The remote subset stopped cleanly and its owned processes were reaped. Recovery was staged but unused.

These are supplied operator execution results, not new execution or an independent raw-log audit by the documentation author. One raw-then-packed sample has no noise band; it establishes no broad performance guarantee or production qualification.

Preparation and harness overhead were disproportionate. Future research should use the cheapest decisive experiment and existing tools. The user has requested automatic packed selection for qualified compatible peers, with raw fallback and an explicit opt-out. That default-selection change belongs to a fresh focused task; it is not implemented by this result update. Current code remains unchanged. This result session is complete: no new tests, code, experiment, merge or deployment.
