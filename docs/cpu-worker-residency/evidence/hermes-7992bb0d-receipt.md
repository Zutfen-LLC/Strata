# Attributed Hermes validation receipt for 7992bb0d

Revision: **7992bb0d293126ed0738a9b73d6c3fec7b744e78**. These are user-provided Hermes report facts;
this coding task did not access the hosts, inspect the retained files or independently verify hashes.
This revision is **not an overall PASS**. The subsequent checker-only correction is runtime-unvalidated.

## Reported passed: complete native-OFF stage on 03

- Actual `strata` and `worker_disabled_dispatch_test` executable links passed.
- Four synthetic router/Q2 checks passed: normal and forced-AVX variants for each.
- All 50 CLI checks passed, including unavailable-worker refusal and argument-consumption cases.
- OFF ISA scan reported zero violations. Common objects appeared once, with no worker/ggml dependencies.

## Reported passed, then failed: native-ON stage on 03

Configure/build passed. Four synthetic math checks and two worker dispatch checks passed.
The first failure was `isa_guard_check`, CTest RC **8**: `tq_avx512.cpp.o` was classified as wide ISA
outside allowed translation units, with **606 instructions**, first **vmovq**.
The subsequent device-plan check and **all 05 checks did not run**. No overall success is inferred.

The user supplied an independent read-only inspection: `tq_avx512.cpp` and `tq_cpu.cpp` were byte-identical
on main and candidate; dispatch checks `tq_isa_ok`/`cpu_avx512_ok`; the AVX512 translation unit has no
namespace-scope dynamic initialization and its helpers are anonymous. This coding task also inspected
the source/diff statically. The missing checker registrations predate the candidate. The correction
adds only the intended object and public entry name to the checker, retaining initializer, weak/COMDAT
and unexpected-public-symbol rejection. Kernel and shared CPUID code are unchanged.

## Reported retained evidence

- 03: `/home/hermes/strata-val-7992bb0d-03-r2`.
- Controller: `/home/zutfen/strata-val-7992bb0d-controller/REPORT.md`.
- Reported archive SHA-256:
  `01094f70846ba3c829ff2925e20eddcadc0b46ec607f09cfe35f6f502d417e4a`.

No full binary/cache hashes were included in this message; recover them from the existing small
receipts before reusing outputs. Do not fabricate them or hash model assets. This archive is distinct
from the earlier full-model trial archive. Preserve the [79f4da0 link failure](hermes-79f4da0-failure.md),
its independent 05 results and the [16d065e full-model PASS](hermes-16d065e-receipt.md) separately.
The latest report adds no HIP/SYCL/Windows or full-model numerical-parity qualification.
