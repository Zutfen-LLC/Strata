# W13/W14-style resident worker-count comparison

`as-run/w13w14.py` is the exact driver executed on the campaign host
(2026-10-24, verdict: 13 workers level with the 12-worker reference, 14 a
regression; restored 12; sanitized paths only, no logic changes).

`w13w14.py` in this directory is the RECOMMENDED next-run version: identical
except `stop_owned_worker`/`stop_old_resident` retain each launch's guardian
`subprocess.Popen` handle and reap it with `wait(timeout=...)`, retaining the
exit status. The as-run version polled bare `/proc/<pid>` existence, which
never clears for the driver's own zombie children, so every worker
transition burned its full 420 s stop deadline despite the worker exiting
within seconds. See REGRESSIONS.md case 1.

`config.template.json` documents every binding field with placeholders; the
as-run config carried live absolute paths, GPU UUIDs and port — substitute
your own, then pin exact sha256s in `pins.json`-style preflight checks
(derive pins mechanically from staged bytes, never hand-copy them: see
REGRESSIONS.md case 3).
