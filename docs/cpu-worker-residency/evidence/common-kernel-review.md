# Common CPU-kernel boundary: independent static review

Reviewed the working correction from `79f4da0e09b4f7ff0e331809316cd2fc18684b51` after the user's
reported native-OFF link failure. Two independent reviewers used the code-review skill's spec and
standards/quality axes. They audited all common CPU/core/program consumers and native-only dependencies,
not just modified lines, plus top-level and SYCL source lists, ISA flags and separate Metal configuration.
The final handoff/documentation diff was reviewed separately after transport recovered.

The user prohibits Geekom configure, builds, tests, binaries, benchmarks, model loads and host operations.
That instruction overrides repository backend-build requirements for this candidate. No reviewer or
coding action executed those checks here. Successful linking and runtime behavior cannot be established
by this review.

## Standards and quality

No material source defect, documented-standard breach or material heuristic smell was found in the
common-kernel relocation. Q2/activation and BF16 implementations are real math, listed once per build,
with prior source ISA flags. Native ggml implementations remain conditional. Existing OFF error/abort
paths resolve native-format callers. BF16 function bodies and reduction order are visibly unchanged.
A stale header comment naming the old BF16 source file was corrected.

**P2 handoff finding:** the first resource helper read root `memory.max` unconditionally. A genuine
cgroup-v2 hierarchy root legitimately lacks that non-root interface, so a normal bare host could fail
admission before validation. The correction checks the verified v2 mount/root separately, while still
checking an exposed mount-root cap, every non-root ancestor and MemAvailable. Missing interfaces on
unknown/non-root layouts still stop execution. The helper was inspected, not run.

## Spec

The whole-boundary audit found no additional unguarded native-only IQ/K-quant/NVFP4/trellis dependency
from common consumers. Existing native admission and MTP ggml guards remain fail-closed. Native-ON math
is intended to remain unchanged; source identity and flags do not prove compiled numerical parity.
CUDA/shared HIP/gfx906 and CPU-only use the corrected target; SYCL's separate CPU list is synchronized.
Metal remains a separate ARM/stub build. No new backend/older-CPU runtime support is inferred.

The spec reviewer independently found the same **P2 cgroup-root handoff issue**, resolved as described
above. Evidence/status documents distinguish 79f4da0 OFF FAIL, independent 05 fixture PASS and earlier
16d065e full-model PASS; all are attributed user reports. The new correction remains runtime-unvalidated.
The handoff preserves failure receipts/cache identities, permits verified incremental reuse, and gates
all 05 execution on complete 03 OFF and ON success. Unchanged safe binaries do not require forced rebuilds.

## Outcome and limits

Both independent final follow-ups found the resource-helper correction resolved, with no remaining
material finding on either axis. Required remote gates are actual OFF engine/dispatch links, synthetic BF16/Q2
checks and CLI refusal, then focused ON and 05 checks in order. No full-model reload is required by this
diff. HIP/SYCL/Windows/older-CPU compilation/runtime and full-model numerical parity remain unverified.
