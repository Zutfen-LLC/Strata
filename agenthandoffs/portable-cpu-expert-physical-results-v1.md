> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# Physical V340 portable expert reuse receipt — v1

Parent/operator reports PASS for the single bounded oracle/optimized pair on05
at exact source `ac85ff3fdf0834c13437b69b1568d8704e9b0717`.
Source: [operator receipt, October28,2026](PRIVATE_RECEIPT).
This coding task read that receipt through Slack; it did not access05,
launch inference, or independently inspect the host-local raw artifacts.

The same binary and frozen 38-token input were used for both arms under the
v2 recipe and existing supervisor. Build and all 11 selected tests passed.
The exact recipe comparator passed: all 41 raw F32 logit rows and stdout
were byte-identical, and both generated `[71093,12305,198]`.
This establishes portable-versus-portable output preservation for this pair.
It does not qualify CUDA parity; the previous native-CUDA screen remains failed.

| Observed work | Oracle | Optimized |
| --- | ---: | ---: |
| Verification hits | 0 | 14,069 |
| Expert verifications | 19,680 | 5,611 |
| Native batches | 1,968 | 1,968 |
| Source invalidations | 0 | 0 |

The optimized arm removed **73,473,945,600 copied bytes** and the same
number of digest input bytes. Paired forward/reverse FNV-1a scans therefore
removed **146,947,891,200 logical byte accesses**. These work counts do not
attribute elapsed time to copies, digest scans or native pool execution.

| Time observation | Oracle, first | Optimized, second |
| --- | ---: | ---: |
| `/usr/bin/time` elapsed | 234.01s | 122.21s |
| Supervisor interval | 233.84s | 122.04s |

The fixed oracle-first order can give the optimized arm a warmer page cache.
One pair does not establish a general speedup or isolate timing attribution.
Historical05 timings are not a matched baseline. No timing ratio or new
performance threshold is used for acceptance.

Sampled child RSS high-water was 37.688GiB oracle and 37.695GiB optimized.
Sampled aggregate VRAM peaks were approximately 1.188GiB on device0 and
1.229GiB on device1 in both arms. Sampled child and system swap stayed zero.
These samples are not guaranteed transient peaks or measurements of exclusively
owned device allocations.

Source snapshots and binary hashes matched before and after. Six-shard
identity/size/mtime snapshots matched. Metadata invariance does not prove
adversarial immutability: native execution reads mappings rather than the
previously verified staging snapshot, and sources must remain unchanged
across copy, digest and native execution. No concurrency or security guarantee
is added by this receipt.

No pair-owned processes remained; both GPU VRAM readings returned approximately
to their pre-run baseline. The existing `nvtop` process was left untouched.
Source, build and evidence remain on05's secondary2TB filesystem:

- Root: `${PRIVATE_ARTIFACT_643493BE}`
- Pair: `${PRIVATE_ARTIFACT_4CF25D31}`
- Retained source: 85,897,216 bytes.
- Retained build: 542,232,576 bytes.
- Retained pair: 84,750,336 bytes.

No additional run is requested or dispatched by this documentation update.
PR43 remains draft and unmerged. Prior evidence, models and the separate
numerical investigation are preserved. Clean CUDA/HIP/SYCL compilation was
not supplied by this receipt.
