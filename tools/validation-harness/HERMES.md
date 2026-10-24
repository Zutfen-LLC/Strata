# Focused Hermes validation: stable runtime harness

Use the immutable PR head supplied with this handoff as `CANDIDATE`. Base is merged main
`aca40713de68c59d8659dea5973a6dcbf4bf1b6c`; the sanitized source witness is
`27392d03bb75bf423e9e41a5db3ffe51b3b2b902`, whose parent is that same main. Only the clean witness
was fetched. No superseded leaked source is needed. This is separate from PR #6 and changes no engine.

All executable checks are **NOT RUN on Geekom/Valinor**. Source inventory/canonical-byte comparison,
static AST/JSON parsing, source review and staged-diff inspection are the local checks. Preserve that
distinction. The current twelve-worker resident on the inference host is outside this validation.
Do not contact, query, stop, unload or repin it. No model, GEN campaign, C++ build, package installation,
network permission or host setting change is requested.

On the existing authorized Hermes controller, use a clean candidate source and the retained Python
runtime. No private model assets or credentials are required. Verify the full source SHA, preserve
source status and rerun only the corrected reference case once under the existing owned timeout/output
guard:

```sh
set -euo pipefail
: "${SRC:?clean candidate checkout on Hermes}"
: "${CANDIDATE:?full immutable candidate SHA}"
test "$(git -C "$SRC" rev-parse HEAD)" = "$CANDIDATE"
test -z "$(git -C "$SRC" status --porcelain)"
(
  cd "$SRC/tools/validation-harness/tests"
  timeout --signal=TERM --kill-after=5s 120s \
    python3 -B -m unittest -v \
      test_harness.HarnessTests.test_reference_binding_schema_and_indices_checked_before_start
)
```

Supplied Hermes result for `4916b4a44450bac589a63e6ef7ab5c3d96c2978a`: controller Python 3.13.5,
20 tests in 7.825 seconds, exit 1, 19 passed and one error. The reference fixture retained the key
`missing` after its first subcase, so `ValueError('missing reference asset')` prevented the broken-JSON
and empty-request-reference subcases from running. Other lifecycle/staging/reaping/uncertain-STOP and
TIME_WAIT cases passed, with clean cleanup reported. These are results supplied by Hermes through
the parent task; Codex did not execute the run or retrieve the private receipts. Preserve that first
failure on Hermes. The additive correction binds the registered `reference` asset for both remaining
subcases and checks that neither launches a process or creates an output root. The production guard
is unchanged. This corrected case is pending remote validation.

Use a new result directory and bounded captured stdout/stderr. The suite uses stdlib temporary files,
loopback socket pairs, mock control exchanges and tiny Python children. Fake engine/monitor scripts
exercise the actual broker/guardian's stdin ownership, startup cleanup, nonzero exit and reap paths;
there is no GPU query or native/model execution. Preserve the first failure and exit status; do not
repeat to select a pass. Report duration, test count, source SHA and raw synthetic logs. The focused
rerun must reach all three rejection paths; it does not replace the original full-suite result.

Review the example config and the config-only variation case: ordered entries change config
values without editing helpers, and base pins remain unchanged. Missing paths,
imports, reference bindings, wrong endpoints, stale helpers and existing output roots must all fail
before owner launch. The STOP timeout case must retain an uncertain control outcome even after the
actual owned child exits; no second exchange may occur. Reaping must be checked with the retained
handle/`waitpid`, not just an absent exe link. Live-listener and TIME_WAIT cases must distinguish each
other. Startup and child failure cases must not leave a child or report a successful owner.

There is no automatic second stage against live hardware. After synthetic PASS, return one flat
receipt with passed/failed/NOT RUN, exact candidate and blockers. Any later real arm needs a separate
explicit private configuration and operator authorization. Use one staging operation on its execution
host, preserve pins after transfer, choose retain/stop explicitly, and leave the current resident model
alone until that later action is authorized. Stop before merge or deployment.
