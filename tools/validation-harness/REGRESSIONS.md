# Focused retained-harness regressions

The source witness is sanitized `27392d03`, not exact private runtime bytes. Its completed W13/W14
driver waited the full 420 seconds on its own unreaped guardian. Its separate Popen proposal was
explicitly unexercised. Hermes reports that the synthetic suite at `4916b4a4` ran 20 tests on controller
Python 3.13.5: 19 passed, one reference-fixture error, 7.825 seconds, exit 1, clean cleanup. The fixture
failed to select its registered asset after checking a missing binding; the later JSON/schema paths
were not reached. Hermes reports that the fixture-only correction at `64383360` passed its focused
rerun: one test, 0.095 seconds, exit 0, all three subcases reached. A subsequent static review found
the pre-listener startup-abort gap below; that additive lifecycle correction is pending the new
focused selection in [HERMES.md](HERMES.md).
These are supplied Hermes results, not locally executed checks. All executable desktop checks remain
**NOT RUN**; no historical model run validates these edits.

Later live evidence supplied by the parent identified a harness affinity error: the worker had four
loading lanes, but its caller and three transient loader helpers were all restricted to CPU0 while
the inference pool was idle. The launcher had applied the singleton compute caller mask before SETUP.
This is an affinity inheritance defect in the harness, not a failure to create product loader lanes.
The phase correction and new synthetic cases remain **NOT RUN** locally.

| Observed wrapper problem / static defect | Correction and synthetic case |
| --- | --- |
| Own guardian becomes zombie; bare `/proc/PID` remains | Retain Popen, wait/reap, save status before absence/probe; lost-ACK test asserts `waitpid` reports no child |
| Relative executable prevents startup before state.json | Validate bindings/imports before launch; early-owner-exit case detects/reaps actual child without state |
| Missing/stale staged helper | Required helper inventory plus hash/import verification before any Popen |
| Cleanup assertion races owner/monitor exit | Reap owner first, wait on recorded identities; failure propagates rather than reporting clean completion |
| Timed-out worker STOP executes later | Block controls for all open engine sessions; persist uncertain intent, never retry, reconcile via retained Popen |
| Port-free check precedes owned resident stop | Stop/reap before archive/probe/relaunch; ordering case forbids pre-stop bind |
| Bind probe mistakes TIME_WAIT for live listener | Same SO_REUSEADDR as real worker; paired live-listener/TIME_WAIT socket case |
| Handwritten hashes or overwritten upload pins | One new-root staging operation hashes source/destination bytes; repeated staging cannot overwrite pins |
| Startup identity/monitor exception bypasses cleanup | Cleanup guard and signal handler precede first child launch; injected monitor-start failure checks child reaping |
| Static finding: pre-listener timeout cleanup requires a listener and masks startup failure | Abort the just-launched owned guardian, reap before child absence checks, preserve the timeout; real no-listener child case plus attached/STOP-intent refusal |
| Engine/worker nonzero exit hidden by owner status 0 | Save child status and return failure from owner; fake engine/worker cases preserve exit code 7 |
| Config variations require editing executable driver | Ordered config entries change counts/masks/requests/assets; derived configs leave pinned base bytes unchanged |
| Reference/dependency mismatch discovered after requests | Validate reference indices, manifest/control endpoint and actual monitor environment before launch |
| Caller pinned before SETUP makes loader helpers inherit one CPU | Use configured caller/pool CPU union for setup; after owned idle READY, restore final masks before GEN; real child/thread inheritance and phase-boundary cases |

No malformed-source corpus, unrelated backend suite, model campaign or new supervisor is added.
