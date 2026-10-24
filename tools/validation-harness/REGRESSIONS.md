# Focused retained-harness regressions

The source witness is sanitized `27392d03`, not exact private runtime bytes. Its completed W13/W14
driver waited the full 420 seconds on its own unreaped guardian. Its separate Popen proposal was
explicitly unexercised. The corrections below remain **NOT RUN** until Hermes executes the synthetic
suite; no historical run is validation of these edits.

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
| Engine/worker nonzero exit hidden by owner status 0 | Save child status and return failure from owner; fake engine/worker cases preserve exit code 7 |
| Config variations require editing executable driver | Ordered config entries change counts/masks/requests/assets; derived configs leave pinned base bytes unchanged |
| Reference/dependency mismatch discovered after requests | Validate reference indices, manifest/control endpoint and actual monitor environment before launch |

No malformed-source corpus, unrelated backend suite, model campaign or new supervisor is added.
