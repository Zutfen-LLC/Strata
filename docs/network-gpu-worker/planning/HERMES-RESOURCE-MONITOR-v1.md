# Future synthetic resource-monitor validation: NOT EXECUTED

This is a proposal requiring separate explicit user approval. It authorizes no present execution, live monitoring, worker/model launch or full-model retry. Implementation is still bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5; PR #2 draft/unmerged. Use the exact immutable artifact commit and SHA256SUMS.resource-monitor-v1 supplied with publication, never a moving branch. See [amendment](RESOURCE-MONITOR-AMENDMENT-v1.md) for provenance, limitations and the missing driver integration gate.

Target ROLE_CPU_C only, using existing trusted transport/authentication, existing Python3.9+ standard library and existing shell tools; ROLE_CONTROLLER only retrieves/collects small text artifacts. ROLE_EDITOR may not execute Python, syntax checks, tests, scripts, binaries, builds, benchmarks or model operations. No other target, environment setup, installation, host probing, live proc/cgroup reads, process termination, services, ports, models, source/dependency downloads or repository clone. Missing prerequisites block rather than expanding scope. Frozen placement and all original resource/time caps remain unchanged; the old model-trial approval does not carry forward.

New private identifiers: ARTIFACT_COMMIT=exact published artifact SHA; ARTIFACT_DIR=new private controller directory; FIXTURE_DIR=new private ROLE_CPU_C directory; FIXTURE_PYTHON=existing Python3.9+ interpreter; VALIDATION_LOG=new private controller evidence. Parent supplies their actual bindings privately. All ROLE_* meanings follow README. Do not publish operational bindings, execution output or private receipts.

One global monotonic240-second wall including retrieval/transport/cleanup. Stop new work at225 seconds; finish by240. Clip nested operations to remaining time. Input transfer is only the ten named text files below, <=64 KiB/file and <=256 KiB total; later private text evidence is bounded separately at16 MiB, so total input+evidence<=16.25 MiB. No raw production logs or model assets. Existing authentication only; no credential setup. Fetch each file through authenticated GitHub Contents API raw media type at ARTIFACT_COMMIT; per-request cap10 seconds plus5 kill grace. Stop at first identity/hash/size/transport failure; retain failure evidence; no retry just to obtain PASS.

Allowed retrieval names under docs/network-gpu-worker/planning/:

```text
README.md
resource_telemetry.py
resource_guard.py
resource_telemetry_test.py
resource_guard_test.py
resource_monitor_fixtures.py
RESOURCE-MONITOR-AMENDMENT-v1.md
HERMES-RESOURCE-MONITOR-v1.md
RESOURCE-MONITOR-STATIC-REVIEW-v1.md
SHA256SUMS.resource-monitor-v1
```

Fetch each individually, bounding saved bytes (for example pipe raw response through head -c65536 with pipefail), into an exclusively created private directory. API template: `gh api --method GET --header 'Accept: application/vnd.github.raw+json' "repos/Zutfen-LLC/Strata/contents/docs/network-gpu-worker/planning/$name?ref=$ARTIFACT_COMMIT"`. Check every downloaded byte/size against the immutable publication handoff and trusted checksum-file SHA, then `sha256sum --check --strict SHA256SUMS.resource-monitor-v1`. That file covers the other nine files. Total allowed text stays<=256 KiB. No unverified source may run. Transfer this verified fixed set to the new FIXTURE_DIR through the existing approved transport and verify it again there. Never overwrite earlier evidence. Do not fetch other repository sources.

Execute exactly these synthetic targets serially, one interpreter process, no parallel suite or discovery:

```text
resource_telemetry_test.TelemetryTests
resource_guard_test.GuardTests
```

The supplied runner selects both targets and traps live operations. Run only the following after approval and both-side verification. FIXTURE_PYTHON and FIXTURE_DIR are private absolute paths; a new log belongs to the bounded validation evidence. No Python code in this prompt is to be executed on ROLE_EDITOR.

```sh
(
  ulimit -v 262144
  ulimit -t 30
  ulimit -f 16384
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  timeout --signal=TERM --kill-after=5s 55 "$FIXTURE_PYTHON" -B -I -c \
    'import sys; sys.path.insert(0,sys.argv[1]); import resource_monitor_fixtures as f; raise SystemExit(f.main())' \
    "$FIXTURE_DIR"
)
```

Hard test wall<=60 seconds including kill grace, virtual-memory cap256 MiB, CPU time30 seconds, output<=16 MiB. Controller/supervisor tracks only the owned interpreter PID/start and must not signal unrelated processes. No actual process termination or production side effect occurs INSIDE fixtures; timeout is only the external bound on that owned test process. Tests use fake metrics/mounts/processes/pidfds and live-operation traps. Python -B prevents bytecode files. No compileall/py_compile or extra syntax/build/test commands are needed.

Acceptance: RC0, no skips, exactly26 tests, explicit marker `RESOURCE_MONITOR_FIXTURES_PASS_V1 tests=26 live_signals=0 workloads=0`, no FAIL marker, no unexpected-operation assertion, and no time/resource cap crossed. Require coverage of byte/page units, unavailable/malformed vs zero, strict64MiB boundary, cumulative vs interval/baseline semantics, original-baseline/deadline preservation at attach, no diagnostic-policy substitution, PID reuse, namespace proof BEFORE pin/discovery, mount/root/ancestor resolution, unlimited+event checks, required missing coverage, changed groups/limits and fault-isolated owned stopping/descriptor closes. Capture complete stdout/stderr, RC, interpreter identity/version from existing provenance, exact artifact/file hashes and resource/cleanup receipts. Mark NOT RUN if prerequisite/approval missing; failed assertions are FAILED, never relabelled as an environment PASS.

Keep evidence<=16 MiB total. Collect only text receipts through existing transport; preserve prior evidence. Confirm only the owned test process exited; no services/forwards/GPU allocations/arenas/cache were created. Retain the small new source/evidence directories without deletion of older assets. Report then STOP. A synthetic PASS proves these fixtures only. The actual driver integration, live pidfd/proc/cgroup behavior, independent watchdog, admission release gate and outer resource guards remain unvalidated. Require sanitized integration source, independent review and new runtime approval before broader validation; require another explicit full-model approval before any retry. Do not recompute placement or extend workload/time/memory allowances.
