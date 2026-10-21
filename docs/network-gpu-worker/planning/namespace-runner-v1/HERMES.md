# Proposed no-model runner window — NOT AUTHORIZED / NOT EXECUTED

After separate explicit approval, use ONLY this pinned runner on ROLE_CPU_C.
Keep PR2 draft/unmerged at bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5 and main at
34a59d83a51bdb3f0299d65306740bd357818715. No operations on other workload hosts.

Set ARTIFACT_COMMIT and MANIFEST_SHA to the full author publication receipt.
Privately bind FILES_ROOT (new delivery directory), EVIDENCE_ROOT (new private
output directory), PYTHON (the approved existing interpreter) and PYTHON_SHA
(its exact binary SHA-256). No uploads or whole repository clone are required.

Controller retrieval/verification/launch allowance30s; runner on-target budget
210s, including60s cleanup/sealing reserve. Total240s. Record controller-local
retrieval/launch start/end, bytes, hashes and exit status as they occur. Do not
generate a Python admission script, custom supervisor, selector or deadline
manager. If trusted existing transport/tooling cannot perform this bounded
retrieval and invocation, report that blocker; do not invent another launcher.

Fetch namespace-runner-v1/SHA256SUMS.runner-v1 and only its listed files from:
https://raw.githubusercontent.com/Zutfen-LLC/Strata/ARTIFACT_COMMIT/docs/network-gpu-worker/planning/
Preserve planning-relative paths under FILES_ROOT. Limit transfer to256KiB;
externally verify manifest SHA and every listed file before invocation, using
existing hash tools. No install, dependency/model/source download or environment,
network, security, cgroup, permission, service or swap change. Preserve all old
evidence. Stop on protected-ref drift, byte mismatch or elapsed retrieval bound.

Next minimal approved subset should be synthetic-only because no trusted
administrator receipt exists. Invoke ONE reviewed command; replace only private
paths and approved hashes:

```sh
PYTHON -B -I FILES_ROOT/namespace-runner-v1/runner.py --mode synthetic-only --evidence-root EVIDENCE_ROOT --manifest-sha256 MANIFEST_SHA --interpreter-sha256 PYTHON_SHA
```

The runner verifies artifacts/interpreter and owns the test child; it runs
25 new fixtures followed by unchanged44 in that single bounded child. It must
not ask for receipt, PID1, namespace, cgroup, PSI or subreaper admission for this
mode. Required direct-child pidfd/proc/interpreter/evidence protection failure
still blocks. Do not replace the fixed selection or bootstrap.

Require exit0 and exact ordered markers:
NAMESPACE_RUNNER_FIXTURES_PASS_V1 tests=25 live_signals=0 workloads=0
SUPERVISOR_INTEGRATION_FIXTURES_PASS_V1 tests=44 live_signals=0 workloads=0
NAMESPACE_RUNNER_FINAL_V1 status=PASS_SYNTHETIC_ONLY

Require REPORT.json confirms both counts/statuses, actual admission/stage start
and end observations, owned process identity, complete true-parent reaping,
live NOT_REQUESTED; verify SEALED.json and SHA256SUMS. No surviving owned child,
no incomplete/error/skip/output-cap/cleanup failure. The unchanged26 prior cases
retain their user-reported PASS and are not repeated for this delivery.

Alternative ONLY if separately approved: --mode synthetic-then-live. With no
receipt/pins, expect synthetic25+44 PASS (if they actually pass), live BLOCKED,
exit3, no live child/three-case marker. This is an expected trust refusal, not an
overall live PASS. Do not supply a fabricated receipt to make it green.

Actual live invocation requires a genuinely independently approved administrator
receipt/source with current host/boot/initial namespace binding. The approver,
not the receipt itself, must supply the receipt's exact digest, verified source
ID, expected host ID and exact machine-id file digest. README.md defines that
human verification boundary; a hash/string alone does not authenticate a source.
No collector, sudo, permission or trust changes are authorized by this prompt.

If that prerequisite is later genuinely supplied AND a live subset is separately
approved, the same runner command adds these explicit options (no new script):
--mode synthetic-then-live --receipt RECEIPT_PATH
--approved-receipt-sha256 APPROVED_RECEIPT_SHA --approved-source-id APPROVED_SOURCE
--expected-host-id APPROVED_HOST --expected-machine-id-sha256 APPROVED_MACHINE_SHA
Keep the same210/240s envelope. Refuse if the full live stage cannot fit before
the reserved tail. Do not reset clocks or extend closeout.

Only then may the unchanged three harmless-child cases run, preserving original
limits/guards:128MiB/10CPU per process, at most three probe processes plus runner,
strict fixed-baseline64MiB swap decline, host/cgroup/event coverage, no pressure,
no GPU/model/worker/service/socket listener. Require the three case markers and
SUPERVISOR_OWNED_PROBE_PASS_V1 cases=3 models=0 gpus=0 services=0
plus NAMESPACE_RUNNER_FINAL_V1 status=PASS_SYNTHETIC_AND_LIVE and exit0/sealed
evidence/complete cleanup. No trusted receipt means NO live invocation.

Stop at first failure/blocker or after the single prescribed attempt. No repair,
rerun, hidden retry, weakened provenance or resource adjustment. Return the new
bounded evidence/report and STOP; no post-deadline archive/copy exception.
Keep mappings/raw logs private. No full-model retry, production transport,
residency activation, merge or use of earlier expired approvals.
