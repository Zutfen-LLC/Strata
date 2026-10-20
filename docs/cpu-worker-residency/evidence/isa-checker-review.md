# TQ ISA-checker registration: independent static review

Reviewed the working correction from `7992bb0d293126ed0738a9b73d6c3fec7b744e78`. Two independent
reviewers used spec and standards/quality axes. The user authorized only edits, lightweight inspection
and static review on Geekom: neither reviewer nor this coding round ran configure, builds, Python tests,
test binaries, benchmarks, models, system inspection or inferswarm operations.

## Spec

No material checker or regression-source defect found. The production diff adds only `tq_avx512.cpp`
to `WIDE_OBJECTS` and `tq_rows_avx512` to `WIDE_PUBLIC`. `INIT_FREE_OBJECTS` still derives from the wide
object list. No broad function/namespace exemption, error suppression, shared CPUID or runtime/kernel
change was introduced.

The deterministic fixtures drive the real `run_elf()` parser with mocked nm/objdump text. Positive
anonymous-helper and public-wrapper cases need both registrations. Weak entry/helper/clones, unrelated
public helpers and the entry in another object remain negative cases. EVEX and VEX initializer cases
exercise rejection; the VEX-only case must reach the initializer rule independently of wide-op detection.
A direct decorated-name policy case covers the PE entry/weak decision, not the full PE parser/runtime.

**P2 handoff:** mandatory historical ON/OFF library hashes could block reuse because the preceding recipe
recorded executable hashes but did not require archive digests. The correction permits a fallback only
when archive digests are absent and retained source/cache/toolchain, BASE-identical compiled inputs and
prior executable hashes are verified. A bounded serial Ninja dry-run must prove no work. Scheduled work
stops the run; current archive hashes are recorded as new provenance, not fabricated historical hashes.

**P3 handoff:** an `if rg` skip guard could treat matcher absence/read errors as clean because errexit is
suppressed in the condition. The correction admits required tools explicitly and requires matcher status
1 (no match); status 0 or any error stops the handoff.

## Standards and quality

The independent quality reviewer found the checker diff and regression paths narrow and coherent, with
no material source finding. It independently identified the same archive-provenance and skip-guard
handoff issues. The implementation follows the existing unittest policy-test style and keeps rejection
behavior intact. The attributed receipt separates complete OFF PASS, partial ON PASS and the ISA failure,
with device-plan/05 unrun; no overall PASS or locally inspected evidence is claimed.

## Outcome and limits

Both independent narrow follow-ups confirmed the two handoff corrections resolved. No material
finding remains open on either axis. The [Hermes continuation](../ISA_CHECKER_HERMES.md) runs regressions and archive scans
remotely, then the remaining device-plan/05 gates sequentially, preserving safe outputs and accepting
canonical HTTPS as well as SSH. No test or corrected scan result is established statically. The new
head remains runtime-unvalidated; HIP/SYCL/Windows and full-model numerical parity remain unclaimed.
