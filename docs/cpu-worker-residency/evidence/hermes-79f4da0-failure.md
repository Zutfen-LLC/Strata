# Attributed Hermes validation receipt for 79f4da0

Revision: **79f4da0e09b4f7ff0e331809316cd2fc18684b51**. This records the user's Hermes report,
not locally inspected logs or independently executed results. No host was accessed by this coding task.
The preceding [16d065e full-model/fixture PASS](hermes-16d065e-receipt.md) remains separate.

## Reported failed: native-OFF CUDA link on 03

Configure passed. The actual build failed after **7 minutes 31 seconds**, at step **97/98**, linking
`worker_disabled_dispatch_test`. The link line correctly omitted `strata_worker_tier` and
`strata_expert_worker`; the worker-specific boundary was clean. Common CPU dependencies were still missing:

- `RouterLookahead` in always-compiled `expert_source.cpp` referenced `bf16_rows_dot_multi` and
  `bf16_rows_dot_multi_avx1`.
- Always-compiled `expert_layout.cpp` referenced `act_quant_q8_1_avx1`,
  `q2_0_gguf_rows_multi_avx1` and `q2_0_gguf_rows_multi_avx1_legacy`.
- Static source inspection places the BF16 AVX2 definition in `kq_avx2.cpp`, the BF16 AVX1 definition
  in `kq_avx1.cpp`, and the Q2/activation definitions in `q2_avx1.cpp`. All three translation units
  were listed only inside `STRATA_NATIVE_EXPERTS`.

`strata` itself never linked. OFF dispatch and CLI tests **did not run**. Native-ON checks on 03 were
blocked and **did not run**. Hermes made no source edits or retries. Verdict: **79f4da0 native-OFF FAIL**,
not a runtime-unvalidated success or evidence of unavailable-option refusal.

## Reported passed: independent 05 fixtures

Two named tests actually ran: `worker_local_binary_test` **0.44 s**, `worker_residency_test` **16.70 s**.
The build correctly rebuilt nothing: worker targets were unchanged from the safe, validated 16d065e
lineage, and the exact worker hash matched the earlier receipt. No complete binary hash was supplied
in this message, so none is invented here. This was not a forbidden unsafe pre-fix binary run.
Unchanged safe code does not need forced recompilation merely to change mtimes.

These checks ran independently despite the earlier 03 failure. Retain them as evidence; they do not
change the OFF failure verdict. The next handoff explicitly stops all later execution at the first
failure and gates 05 on completion of the entire 03 stage.

## Reported identities and retained locations

- Recovered complete manifest SHA-256:
  `8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318`.
- Evidence archive SHA-256 was freshly recomputed by Hermes and matched:
  `fd623801b5a99bc4ca1396f7a7b0271dec4ad493f2782c6e331851810c0e5189`.
- On both 03 and 05, receipts: `/home/hermes/strata-native-off-79f4da0`.
- Failed 03 build: `/home/hermes/strata-trial-17ecafc/build-native-off`.
- Controller summary: `/home/zutfen/strata-native-off-79f4da0-summary`, containing `REPORT.md`,
  failure log, identities and fixtures.

The full manifest value supersedes the earlier invalid abbreviated spelling. These are user-supplied
facts; this coding task did not inspect or hash these files. Preserve all first-failure logs, cache and
toolchain receipts. The subsequent common-kernel correction still requires its own build/link and
focused runtime checks. HIP, SYCL, Windows and older-CPU runtime behavior remain unvalidated.
