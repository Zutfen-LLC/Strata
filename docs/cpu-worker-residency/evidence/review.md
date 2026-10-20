# Independent static spec and quality review

This is the historical review for 16d065e before runtime validation. The later
[user-attributed Hermes receipt](hermes-16d065e-receipt.md) supersedes the old validation status,
not the preserved source findings. A native-OFF correction has its own static review and runtime gate.

Reviewed base `5260f2b7c8166021d86f57e4eb80fc44b050e570` to draft `bcc5792`, then the source/staging
corrections below. Two independent reviewers inspected the diff and relevant source/docs. Neither
reviewer edited files or ran builds, tests, binaries, benchmarks, models or host connections.
The user expressly prohibited those actions on Geekom, overriding the repository's backend-build
requirements for this unvalidated candidate. At that review, **NOT RUNTIME VALIDATED** was the candidate status.

## Standards and quality

The reviewer used AGENTS.md, CONTRIBUTING.md and component documentation, plus the code-review skill's
heuristic smell baseline. Initial report, lightly cleaned:

1. **P2 — Constructor exceptions can orphan a worker.** Startup `getline` can return a partial final
   line; `stoi` can throw before construction finishes, so the destructor never runs. Parse complete
   lines with a bounded nonthrowing parser and clean up child ownership on constructor exceptions.
2. **P2 — Published evidence links are missing their files.** validation.md linked two logs absent
   from Git. Include the sanitized retained logs, as CONTRIBUTING.md requires actual evidence.
3. **P2 — Compatibility wording overstates validation.** README called the base worker “tested”
   although current cross-version source was not executed. Distinguish source inspection from pending
   runtime validation, as CONTRIBUTING.md requires.

The reviewer found the positive, unreaped PID guards intact, with no remaining negative/group signal
path in that harness. Unique pre-fork logs removed the earlier stale-log race. No additional material
production defect or baseline smell was reported.

Follow-up found **one additional P2 test observation race**: the nonreading telemetry peer may keep
its connection after the arena expires, so a second client's status query can time out even though
weights were freed. Check the expiry log while that peer is still connected, then close it before
querying counters. This is an observation-path race, not evidence of failed arena expiry.

Resolution: startup now waits for newline, parses with bounded `from_chars`, rejects partial/invalid
ports, cleans up any positive owned child on constructor exceptions, and disables copying. Both logs
are explicitly tracked. Compatibility wording now says static inspection with validation pending on
05. The slow-reader case checks `reason=expiry` before disconnect, then closes the socket before querying
resident/expiry counters. The final narrow static re-review reported: “No remaining material issue found
in this source-only re-review. Runtime validation remains pending on inferswarm05.”

## Spec

The reviewer used the full user persistence requirements and amended no-local-validation boundary.
Initial report, lightly cleaned:

1. **P2 — Idle expiry can be delayed by a blocked reply.** The user requires bounded residency and
   observable bounded cleanup. Send cancellation checked only stop, so unread STATUS replies could
   retain expired weights until the independently configured I/O deadline. Include expiry in applicable
   sends or define an enforced bounded exception; add a slow-reader fixture.
2. **P2 — Retained evidence absent from candidate.** The unresolved failure was linked but its logs
   were not tracked. Include them.
3. **P3 — Compatibility wording overstates validation.** Describe exact-base behavior as source
   inspection pending the 05 gate, consistent with NOT RUNTIME VALIDATED.

The reviewer otherwise found complete manifest/partition matching, fresh session/READY, immutable reuse,
legacy defaults and the fixture-first operational gate aligned with the spec.

Resolution: telemetry/setup/refusal sends now cancel on idle expiry. Successful inference replies keep
their existing active I/O deadline; successful reply completion starts idle age. The new slow-reader
case is included as unrun source. Evidence and compatibility wording were corrected as above. Static
follow-up found all three findings resolved and no new material spec issue. It also inspected the
harness parser and child guards. It expressly did not establish compilation, runtime correctness,
compatibility or real-host residency/GPU behavior.

## Review outcome

Quality: four actionable findings across initial/follow-up review, all resolved in source. Spec: three
findings, all resolved in source. No material finding remains open in either static review. This is
not a runtime PASS; at that coding handoff all final-source runtime checks were pending. The later
user-reported 16d065e results are recorded in the attributed receipt, not inferred from static review.
