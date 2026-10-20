# Validation status by revision

**79f4da0e09b4f7ff0e331809316cd2fc18684b51: user-reported native-OFF CUDA link FAILED.**
Configure passed; the 7m31s build failed at step 97/98 linking `worker_disabled_dispatch_test`.
`strata` never linked, OFF dispatch/CLI tests did not run, and ON checks on 03 were blocked.
The separate 05 fixture passes and recovered full identities are preserved in the
[attributed failure receipt](hermes-79f4da0-failure.md). They do not change this verdict.
The later common-kernel boundary correction is runtime-unvalidated; see its
[static dependency audit](../NATIVE_OFF_AUDIT.md) and [independent reviews](common-kernel-review.md).

**16d065e23211f2d422c56a3497a8be200c1e1d4d: user-reported Hermes runtime validation passed.**
The later [attributed receipt](hermes-16d065e-receipt.md) records CPU/CUDA fixtures, compatibility and one
full-model cold/warm restart trial. This coding task did not independently execute or inspect that run.
The first native-OFF correction then failed at 79f4da0 as recorded above. The subsequent common-kernel
correction remains runtime-unvalidated until its [sequential remote handoff](../NATIVE_OFF_HERMES.md) passes.

At the original coding handoff, 16d065e had not been built or run. The failed/interrupted local history
below is preserved; it is not erased by the later user-run receipt. The user prohibits builds, tests,
test binaries, benchmarks and model loads on Geekom. Only code/document edits, lightweight repository
inspection and independent static reviews are permitted. Static review is not runtime validation.
This coding task has not connected to or operated either inferswarm host or dispatched broad CI.

## Retained earlier coding-task evidence, not a passing run

The earlier CPU-only configure/build used GCC 16.2.1 on Geekom and the retained ggml source cache at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Those binaries predate 16d065e and the correction. Restricted
socket attempts could not create sockets; permitted loopback runs were subsequently used. Several
legacy suites passed before interruption, but the combined run did not finish.

The [interrupted suite log](prior-suite-interrupted.log) ends at worker_residency_test. The separate
[failed residency log](prior-residency-failed.log) completed in 12.55 s with CTest RC 8 and
`FAIL exception: connect failed, cancelled or timed out`. Earlier assertions in that FAILED run showed
20,889,600 B reused, zero loaded/verified/network-weight deltas, one arena allocation, native reference
A/B/A checks, mismatch refusal, unload and expiry. These are partial observations, not a passing suite
or validation of 16d065e or the correction. There is no full-model numerical parity claim.

## Static findings from the retained failure

The test reused `worker-30000.log` for sequential child processes. The parent could see an old listening
line before the new child truncated that file, then connect to the departed process's port. This is a
credible static explanation for the connection failure after the expiry subtest, not a reproduced cause.
The harness now creates a unique empty log with mkstemp before fork. Every worker instance has a
separate log; staged, unbuffered test output and optional `STRATA_WORKER_FIXTURE_KEEP=1` preserve evidence.

More seriously, the old new-test destructor sent SIGTERM, called wait_exit(), and on a false return sent
SIGKILL to the stored PID. wait_exit() reaped failed/signal exits and set that PID to -1 before returning
false. Consequently that path could call kill(-1, SIGKILL), affecting other processes permitted to the
user. A stale-port constructor race could reach this early-child-exit cleanup. This is a credible
explanation for the desktop disruption; the retained logs do not prove the actual signal path ran.
The source now gates every child signal on pid > 0, separates reap/exit status from ownership, and checks
ownership again after waiting. A failed child clears ownership and destruction cannot signal it again.
The new harness_cleanup regression exercises a reaped unsuccessful child before other lifecycle cases.
Review also corrected partial-line startup parsing/constructor cleanup, and idle expiry while telemetry
replies are blocked. Successful inference reply deadlines remain distinct from idle expiry.
Do not execute any pre-fix worker_residency_test binary. Hermes must compile the published source first.

## Original validation handoff and later user receipt

The original [Hermes handoff](../HERMES.md) gated model activity on serial synthetic checks. The user
later reported those checks, CUDA fixtures and one cold/warm trial passed for 16d065e; the
[attributed receipt](hermes-16d065e-receipt.md) records scope, numbers, archive identity and caveats.
The native-OFF correction now needs its own [narrow handoff](../NATIVE_OFF_HERMES.md), not an automatic
repeat of the full model. No correction-head build or execution has been performed on Geekom.

## Independent static reviews

[Independent static review record](review.md) contains both reports and source dispositions.
No material finding remained open in those original static reviews. At that coding handoff runtime
checks were pending; the later 16d065e receipt is recorded separately. The correction needs fresh
checks. The correction's [independent review and dispositions](native-off-review.md) are recorded
separately. Static review does not establish compilation or execution correctness.
