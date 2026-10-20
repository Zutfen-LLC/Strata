# Candidate status: NOT RUNTIME VALIDATED

The final source has not been built or executed. The user prohibited local builds, tests, test binaries,
benchmarks and model loads on Geekom after desktop/session disruptions. Only code/document edits,
lightweight repository inspection and static independent reviews are permitted. Static review is not
runtime validation. No inferswarm05/03 connection, GPU execution, full-model test or broad CI dispatch
was performed by this coding task.

## Retained earlier evidence, not a candidate PASS

The earlier CPU-only configure/build used GCC 16.2.1 on Geekom and the retained ggml source cache at
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Those binaries predate the final source changes. Restricted
socket attempts could not create sockets; permitted loopback runs were subsequently used. Several
legacy suites passed before interruption, but the combined run did not finish.

The [interrupted suite log](prior-suite-interrupted.log) ends at worker_residency_test. The separate
[failed residency log](prior-residency-failed.log) completed in 12.55 s with CTest RC 8 and
`FAIL exception: connect failed, cancelled or timed out`. Earlier assertions in that FAILED run showed
20,889,600 B reused, zero loaded/verified/network-weight deltas, one arena allocation, native reference
A/B/A checks, mismatch refusal, unload and expiry. These are partial observations, not a passing suite
or validation of this final candidate. There is no full-model numerical parity claim.

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
Do not execute any pre-fix worker_residency_test binary. Hermes must compile the published source first.

## Validation to run on inferswarm05

[Hermes handoff](../HERMES.md) gates all model activity on conservative serial CPU build and focused
synthetic tests, including harness cleanup, lifecycle, direct native parity, controls and exact-base
old/new wire compatibility. Run those commands on inferswarm05, not Geekom. Stop on the first failure;
retain unique child logs, source/binary identity, process exits and the corrected connection-stage output.

## Independent static reviews

Pending. These will report source findings and dispositions, never a runtime PASS.
