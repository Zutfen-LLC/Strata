# Supervisor source handoff v1 (sanitized historical source)

Sanitized source handoff for integrating the validated replacement resource
monitor (artifact `a3b706deae4101e90592bed58701df20f9bd4da2`, fixture set
`RESOURCE_MONITOR_FIXTURES_PASS_V1 tests=26`) into the original bf51f09e
full-model trial supervisor. Implementation remains
`bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`; PR #2 stays draft/unmerged; main
stays `34a59d83a51bdb3f0299d65306740bd357818715`. This branch adds only this
directory and authorizes nothing beyond static integration review.

## What these files are

Complete renderings of the four private supervisor scripts that produced the
failed B0 run (the run stopped at 01:38:55Z by the ROLE_CPU_C monitor's
NEW-SWAP guard, engine rc=1 downstream of the owned worker's TERM). Each file
is the FULL original with host identity removed — no functions were dropped and
nothing was rewritten to "work". They are sanitized historical source: NOT
byte-identical originals and NOT runnable deployment instructions. Every
intentional textual edit is tagged `# sanitization:` inline; everything else —
logic, thresholds, units, ordering, error paths and the original defects — is
preserved verbatim, including the strict
`[ "$SF" -lt $(( SWAP_BASE - 65536 )) ]` admission-baseline swap test and the
original cgroup-guard behavior (`|| echo 0` fallbacks included).

| Public file | Original role | Original lines | Complete? |
| --- | --- | ---: | --- |
| [driver05-ROLE_CPU_C.sh](driver05-ROLE_CPU_C.sh) | CPU-worker driver: admission, launch, 2 s monitor loop, resource stop, final state | 1-113 | complete file |
| [driver03-ROLE_ENGINE_A.sh](driver03-ROLE_ENGINE_A.sh) | Engine driver: B0/B1/G0/G1 phases, admission re-checks, per-phase sampler, gates | 1-151 | complete file |
| [driver04-ROLE_GPU_B.sh](driver04-ROLE_GPU_B.sh) | GPU-worker driver: launch/stop subcommands, plan-pin, admission, sampler, TERM/escalate stop | 1-117 | complete file |
| [orchestrate-ROLE_CONTROLLER.sh](orchestrate-ROLE_CONTROLLER.sh) | Controller orchestrator: relay bridge, launches, phase polling, abort cleanup | 1-133 | complete file |

See [SOURCE-MAP.md](SOURCE-MAP.md) for the per-function original line ranges
and [INTEGRATION-INTERFACE.md](INTEGRATION-INTERFACE.md) for the Codex
integration entry points and the required baseline/deadline interface.

## Provenance and custody (summary)

Originals are retained privately in the controller evidence directory against
evidence bundle sha256
`c068f03406ef02e151ed71b342ae57e12f29737cbf3dd2b4d5422f430798b0f4`
(verified equal to the expected bundle hash before extraction of these
copies). Exact hashes of the four originals are recorded only in the private
transformation record; publishing them here would bind the sanitized text to
the private run and is intentionally not done. The sanitized files were
produced by hand review of each original; the transformation record mapping
public files to original hashes and private paths stays off-GitHub.

## Sanitization policy (what was removed vs omitted)

Removed identifiers (tagged inline, replaced by README role variables or
explicitly dropped):
- Private absolute paths: run/evidence dirs, executables, model shard/pack,
  profile, manifests, cache dir, control-tool path, source checkouts.
- Loopback port numbers and the concrete ssh relay route (hosts, transport
  session names, usernames, controller forward ports).
- The GPU plan integrity pin (`EXPECTED_GPU_PLAN_A_SHA` value) and the private
  CPU-worker build-receipt id in identity.txt — integrity pins double as
  placement proof and are private per the planning README.
- Original protocol filenames (`STOP-03`, `G0-GATE`, `G1-GATE`, `PHASE-VERDICT`,
  `STATUS`, pid/epoch/rc receipts) are retained unchanged: they are the
  controller-driver file contract itself, and rewriting them would be a
  substantive logic edit. Their numeric suffixes are internal labels with no
  public host binding.

Substantive omissions (NOT sanitization; these private artifacts are simply
not published): raw run logs, samples/violations TSVs, admission/identity
receipts, the evidence tarball, the controller-side analyzer and vocab probe,
the private worker control tool, and all engine-side C++ source. The engine
side is represented only by its CLI contract in driver03 (the audit's static
finding that session negotiation — SETUP_BEGIN — triggers arena load remains a
private-audit statement, not source published here).

Unresolved dependencies / missing provenance (explicit):
- `WORKER_CONTROL_CPU_C` (worker_control.py) is referenced but its source is
  private; its verified surface is `status|unload|stop` only.
- The historical admission receipts exist as WALL-CLOCK artifacts only
  (admission.txt, worker-launch.epoch, T0.txt). No monotonic-clock receipt was
  ever taken; see INTEGRATION-INTERFACE.md for what the replacement monitor
  requires instead. No historical namespace-identity or ownership proofs were
  produced by the original run beyond /proc exe/cmdline/start-time echoes; the
  replacement's pidfd/namespace receipts are new instrumentation, not
  historical facts, and none are fabricated here.

## Integrity

`SHA256SUMS.supervisor-source-v1` binds this README, the source map, the
integration interface and the four sanitized scripts. The handoff reports that
file's own SHA-256 and size separately (a file cannot contain its own hash).
Publication/retrieval of these files authorizes no execution: no shell, no
syntax checks, no builds, tests, model operations, worker launches, remote
host operations or trial retry. Integration and any runtime validation remain
separately approved steps.
