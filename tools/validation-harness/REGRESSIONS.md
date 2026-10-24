# Harness regressions found during W13/W14 (2026-10-24)

Each case below is a concrete launcher defect hit in this series. Fixes live
in the recommended `w13w14/w13w14.py`; the as-run code is preserved verbatim
under `w13w14/as-run/`. Fixes marked NOT YET EXERCISED have not been run on
live hardware — the completed W13/W14/R12 campaign used the as-run code.

## 1. Stop-wait polled bare /proc/PID and burned the full deadline on the driver's own zombie child — FIXED (NOT YET EXERCISED)

The worker guardian (`worker_owner.py`) is a `subprocess.Popen` child of the
driver. When the worker exits, the guardian exits too — but as the driver's
own child it becomes a zombie that `/proc/<pid>` keeps reporting until the
driver reaps it. The as-run stop loop

```python
while time.monotonic() < deadline and any(pathlib.Path('/proc', str(v['pid'])).exists() ...):
```

therefore never cleared and every worker transition spun to its full 420 s
deadline even though the worker exited within seconds. This was initially
misread as "graceful ~7-minute arena release" — the arena-release rebound
happens in seconds; the wait loop was the whole delay. (Observed twice:
W13→W14 and W14→R12 transitions.)

The fix reaps the child properly: each launch retains the guardian's Popen
handle (`GUARDIAN_POPEN`), and `stop_owned_worker` calls
`guardian_popen.wait(timeout=420)` first — detecting exit AND reaping the
zombie, retaining the exit status — then polls the remaining identities with
an exe-existence check (`_alive`), which reads zombies as gone. Merely
switching the poll to `/proc/<pid>/exe` would have avoided the delay without
reaping the child; the Popen.wait form is the complete fix.

## 2. Port-free preflight ran while the old resident still owned the listener — FIXED (exercised in the completed run)

`preflight()` asserted the worker port bindable before the authorized stop of
the old receipt-bound resident, aborting with `OSError(98)` even though the
listener legitimately existed. Fixed by deferring the port check to the stop
path (which proves release with its own bind) when the old resident is alive.

## 3. Hand-copied sha256 pins — FIXED (exercised)

Two aborted launches came from transcription errors in 64-hex-char pins
retyped into the driver source. Pins must be derived mechanically from the
staged bytes (e.g. a `pins.json` written by hashing the files) and patched
into the driver by script, never hand-copied. The published drivers keep the
pin-map pattern (`HASHES`/`pins.json`) for exactly this reason.

## 4. TIME_WAIT on the worker control port after a stop — FIXED (exercised)

The stop path's release probe bound without `SO_REUSEADDR`; worker-side-closed
status connections leave TIME_WAIT entries on 127.0.0.1 for up to 60 s, so the
immediate re-bind raised `Address already in use`. Fixed with `SO_REUSEADDR`
plus a bounded 90 s retry, mirroring the real worker listener.

## 5. As-run driver known gaps (documented, not fixed here)

- Engine-cleanup loop in the abort handler references arm sockets that only
  exist while an owner lives; harmless (guarded by existence checks).
- Report line for the resident status command prints a helper path hint that
  may not exist (`tools/worker_control.py`); the actual control script is in
  the campaign root.
- The 420 s stop/deadline constants are host-class assumptions; scale with
  arena size if reused elsewhere.
