# Validation and review boundary

This candidate updates PR #2 additively from bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5,
based on main 34a59d83a51bdb3f0299d65306740bd357818715. It retains the original
selective assets/cache, SGW1 CUDA executor and fixed ownership integration.
It does not import experiment branch ancestry or diagnostic runtime hooks.

## Source attribution

| Retained change | Source revision | Consolidation boundary |
| --- | --- | --- |
| Selective remote CUDA assets/cache, protocol and ownership | bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5 | Original PR #2 source/tests retained |
| Contiguous bounded GPU frame send and socket fixture | 3dc8363bd0794de6e8ea283eba46bc43d5a84da9 | gpu_wire.cpp and gpu_frame_test.cpp copied exactly; only their CMake target added |
| Explicit no-expiry CPU service residency | 98b5791d6042d3b25eac0bb5c5f6e372f78166a5, introduced at 0a54d0db28f884d731ccbc6ada55c62f3f976651 | Wire sentinel, worker option/lifetime/log quota and real-worker fixture copied exactly, including corrected service_setup helper |
| CPU-first post with failure drain | 98b5791d6042d3b25eac0bb5c5f6e372f78166a5 | Submission hunk/header and focused fixtures retained; CUDA failure assertion now uses existing device combine/readback; success finish/merge unchanged |

Capture/replay, first-layer trace nodes, diagnostic environment/CLI switches,
fleet controllers, role bindings, approval ledgers and campaign packets are
excluded. Original PR #2 campaign instructions/review packet are removed from the
candidate tree, but remain in historical commits and experiment branches. No
kernel, quantizer, placement, transport route or security configuration changes.

## Owner-supplied executed evidence

These results are supplied reports, not runs performed or private files inspected
by the consolidation author. Exact commands and resource limits for the historical
CPU-first run are in the immutable [98b5791d procedure](https://github.com/Zutfen-LLC/Strata/blob/98b5791d6042d3b25eac0bb5c5f6e372f78166a5/docs/network-gpu-worker/experiments/CPU-FIRST-REMOTE-OVERLAP-V1-20261022.md).

- At 3dc8363b, the CPU-only real codec/socket correctness fixture and three-arm
  loopback comparison passed on the test fleet. Split median RTT was about 44 ms
  for control/T1/T2; contiguous was .067/.328/.528 ms. T8 was comparable
  (2.018/2.089 ms). This measures loopback, not a relay or model speedup.
- At 0a54d0db, focused combined gates passed after the fixture-only helper was
  renamed service_setup. The later source contains that correction. A fresh CPU
  arena loaded once, then reconnected in .004 s and remained independently owned.
  This supports the exercised no-expiry mode, not every lifecycle/platform case.
- At 98b5791d, the standalone four-case submission fixture passed on 05, the real
  CPU/CUDA dispatcher/merge fixture passed on 03, and four launcher/report mocks
  passed on 05. The candidate engine was built on 03; retained accepted 04 GPU and
  05 CPU worker binaries were not replaced. Seven model requests emitted 53 tokens.
  Same-mode output IDs, fixed-input logits and reference likelihoods were unchanged
  byte-for-byte versus the prior distributed engine. Report bundle SHA-256:
  `3c42c9cb95a687ee0c774d8a09ab6e7ae8abd087e978fd4f40512775153f1aec`.
- That separate-run five-task smoke workload used the retained resident CPU arena,
  full remote cache, primary/helper RTX 3060s and remote RTX 3090. G batch time was
  49.89 s versus prior G 95.24 s and B 65.66 s: 47.6% and 24.0% lower respectively,
  or about 31.6% higher batch workrate than B. Decode was 3.33 tok/s versus B 2.53
  and prior G 1.78. All five answers matched B: four strict passes and the same
  substantively correct fenced-code format failure. This demonstrates a benefit
  on that workload, not general quality, numerical parity or scaling.

## Failed and unrun checks

Historical cross-mode centered relative-L1 screening remains NUMERICAL_FAIL
(about .0352/.0500 against .03); KL probes passed. No threshold is relaxed.
CPU/CUDA row arithmetic and reduction grouping reproduced the earliest saved seed
difference, not the full-logit discrepancy. Low KL and matching task answers do
not establish universal equivalence or quantify quality loss. The original failed
fixture compile and later diagnostic failures remain in experiment history.

This consolidation has only static source-equivalence, dependency and failure-path
review plus git whitespace/readback checks. No desktop scripts, builds, tests,
models or fleet operations were run. Exact-head configuration/build/fixtures and
native-OFF/backend compilation have NOT RUN. Historical runtime results apply to
the revisions above; removing disabled diagnostic code is not executed validation
of this head. Review caught a retained fixture call to an excluded diagnostic
accessor. Its no-partial-commit assertion now runs the existing device combine
against a zero destination and reads back the result; no runtime accessor was
added. This corrected assertion has NOT RUN. Existing hosted title CI does not
compile the engine. Keep PR #2
draft until maintainers assess these targeted validation gaps and numerical limits.
No new model campaign, service operation, merge or deployment is part of this PR
preparation. The intentionally resident CPU service and remote cache are untouched.
