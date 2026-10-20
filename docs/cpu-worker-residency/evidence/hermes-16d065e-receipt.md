# Attributed Hermes validation receipt for 16d065e

Tested revision: **16d065e23211f2d422c56a3497a8be200c1e1d4d**.
This receipt records the user's reported Hermes results supplied to the coding task. Codex did not run
these commands, connect to either host, inspect the archive or independently verify its contents/hash.
The later native-OFF correction is a different revision and remains runtime-unvalidated.

## Reported passed: CPU fixtures and compatibility on inferswarm05

Receipt directory: `/home/hermes/strata-val-16d065e`. Fresh candidate and exact-base binaries were used,
with ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`, serial `-j1` and `OMP_NUM_THREADS=2`.
The report says no unsafe pre-fix residency binary was executed.

- `worker_residency_test`: 49 PASS, zero failures, 16.64 seconds; failed-child cleanup and partial-line
  parser regressions ran first.
- All nine focused suites passed: `worker_local_test`, `worker_local_binary_test`, `expert_worker_test`,
  `expert_worker_e2e_test`, `expert_tier_test`, `worker_socket_test`, `worker_compat_test`,
  `worker_legacy_client_test`, `worker_current_client_old_worker_test`.
- Actual base-built worker and legacy source client were exercised. Fixture reuse was 20,889,600 B,
  with zero new loaded/verified/network weight bytes, one arena allocation and fresh sessions after
  an eleven-second gap. Mismatch, expiry, unload, controls, slow telemetry and termination passed.

## Reported passed: CUDA fixtures on inferswarm03

Receipt directory: `/home/hermes/strata-cuda-val-16d065e`. The existing Ninja sm_86 CUDA cache was
preserved. The incremental build completed 37 steps serially. Five CTest checks passed:
`worker_dispatch_test`, `worker_dispatch_host_zero_test`, `worker_local_test`, `worker_local_binary_test`
and `worker_residency_test`. `STRATA_VERIFY_DEVICE_PLAN=1` reported all checks passed.
The report records no cgroup cap and MemAvailable of at least 14.4 GiB.

## Reported passed: one full-model cold/warm engine restart trial

Fixed ownership was CUDA0 1,056 experts, CUDA1 2,263 experts, and CPU 21,257 blobs: **111,012,556,800 B
(103.39 GiB)**. Reported CPU setup was **428.4 s cold and 0.1 s on warm reconnect**, after engine exit
and an eleven-second idle gap. Session ids changed. Loaded/verified/network counters were unchanged
on reconnect; reused bytes increased by 111,012,556,800 B, cache_hits rose from zero to one, and
arena_allocations stayed at one.

Cold A equalled warm A1 and warm A2 at all 128 generated tokens. The three warm requests reported
4.5–4.6 tokens/s, with both GPUs and the CPU participating. Minimum host MemAvailable was 21.3 GiB.
Engine exits were zero and both GPUs returned to 1 MiB. These are the user's observations from this
one trial, not an independent full-model numerical-parity or sustained-load qualification.

Reported evidence archive: `/home/zutfen/strata-res16d-trial/strata-res16d-evidence.tar.gz`.
User-supplied SHA-256: `fd623801b5a99bc4ca1396f7a7b0271dec4ad493f2782c6e331851810c0e5189`.
This supplied value is recorded without an independent archive read here.

## Interpretation and missing identity details

- The 519.5 s cold window contained one request; the 146.2 s warm window contained three. Those are
  not like-for-like startup timings. Do not attribute their roughly 373 s difference causally.
  The direct CPU setup comparison is 428.4 s versus 0.1 s.
- The memory budget arithmetic is `107000 + 16384 MiB = 120.4921875 GiB`, not 123.4 GiB.
- A pasted manifest abbreviation contained nonhex `t318`; it is not a valid SHA-256. No complete
  manifest, binary or profile hash is invented in this document. Hermes must recover the complete
  values from existing small identity receipts, not guess them or hash the entire model again.
- The worker and tunnels had finite residency/process limits. Their current liveness is unknown.
  This task does not renew, query, unload, stop or otherwise operate those services.
- Native-OFF builds were not covered by this receipt. HIP has not been validated; SYCL uses separate
  migrated sources and this receipt does not establish that it has the worker feature.

The earlier coding-task [failed](prior-residency-failed.log) and
[interrupted](prior-suite-interrupted.log) runs remain preserved. Their static corrections and the
warning against executing unsafe pre-fix binaries remain in [validation.md](validation.md).
