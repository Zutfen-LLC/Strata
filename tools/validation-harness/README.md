# Retained runtime validation harness

This is the small config-driven extraction of the existing Hermes runtime harness. It keeps the
engine stdin broker, worker guardian, independent resource monitor, control codec and retained
GEN/DONE parsing. One ordered config replaces per-experiment runtime wrappers; there is no automatic
warmup, repeat, performance-based winner, restore arm or arm retry. Build/link drivers remain outside
this runtime change. No engine, loader or inference algorithm changes are included; PR #6 is separate.

Source witness: clean sanitized commit
[`27392d03bb75bf423e9e41a5db3ffe51b3b2b902`](https://github.com/Zutfen-LLC/Strata/tree/27392d03bb75bf423e9e41a5db3ffe51b3b2b902/tools/validation-harness).
All 22 supplied files were compared byte-for-byte to that commit before editing. Its role manifest
omits `w13w14/README.md` and contains no content hashes; the extra README was verified too. This is
sanitized source, not the exact private bytes. The superseded publication was not fetched or used.
The as-run driver and its unexercised Popen proposal remain available at that immutable witness;
historical campaign scripts, handwritten binary hashes and frequency-setting code are not imported
into this branch. The new corrections are **NOT RUN** locally and are not claimed as tested Hermes code.

## Files and boundaries

| File | Retained role / focused change |
| --- | --- |
| `harness/campaign.py` | Identity checks, broker RPC, native result parsing and session sequence; owner handles, idle guard and stop/reap correction |
| `harness/owner.py` | Engine broker and bounded drains; config binding and cleanup covering startup failures |
| `harness/worker_owner.py` | Worker guardian; same cleanup boundary and failure propagation |
| `harness/monitor.py` | Existing independent pidfd/resource monitor; configured environment and worker root |
| `harness/worker_control.py` | Existing SEW1 control codec, unchanged |
| `harness/run.py` | Ordered load/warm entries extracted from W13/W14 and PR7, with config-only counts/masks/requests/assets |
| `harness/bundle.py` | One exclusive staging operation; mechanical source/destination hashes and complete preflight before owners start |

Linux and Python 3.9+ are required. The real monitor requires the existing `nvidia-smi` command; no
packages, privileges or network access are installed. The harness runs locally on the selected runtime
host, reached through the operator's existing authorized mechanism. It adds no SSH or fleet supervisor.

## Private configuration and staging

Copy `config.example.json` into a private location and fill every placeholder. Hosts, absolute binary
and model paths, environment, expected arena bytes, worker/loader counts, affinity masks, request
indices, output root and final/failure retain-or-stop choices belong there. `@engine`, `@worker` and
`@manifest` bind the mechanically staged assets; they become absolute argv paths. Model shards stay
in place; only explicitly named asset files are copied. Do not list full model shards as staged assets.

On the execution host, after the separate operator approval for the selected experiment:

```sh
python3 -B /path/to/checkout/tools/validation-harness/harness/bundle.py --config /private/config.json
python3 -B /private/new-run-root/run.py --run-root /private/new-run-root
```

Use the printed staging root for the second command. Staging starts no processes. It validates source
imports, absolute paths, manifests/endpoints, finite deadlines, masks, reference files and all arm
selections, then copies helpers and assets once and hashes the actual bytes. Existing output roots
are refused. `pins.json` and base config are never overwritten; every arm gets its own immutable config
file. Verification precedes each entry, so an absent/stale helper fails before owner launch. Upload
input assets through the existing approved transfer path first; do not upload over the completed bundle.

An entry with `restart: true` stops/reaps the worker launched by this driver, archives its receipts,
starts the configured worker, and runs zero GEN requests through READY/QUIT. Warm entries use
`restart: false` with explicit retained request indices. Reused worker binary/count/loader settings
must match. Put any desired final restore load explicitly in the list. There are no hidden arms.
Expected placement log markers and optional reference tokens/worker entries are checked unchanged.

Worker caller/pool masks are applied to owned tasks before load. Engine-main masks are applied when
READY is received, before GEN; they do not establish engine startup placement. Use existing engine
argv placement options where startup placement matters. Receipts record actual masks and topology.
Request bytes and order come from the pinned request file; there is no retokenization.

To reuse an existing resident, set `attach_root` to its receipt root, `attached_worker_idle: true`,
all entries to reuse, and both retain policies. The operator must first establish exclusive idle use
of that worker. The harness checks recorded engine identities without sending status to discover idle
state. It will not stop or replace an attached guardian whose Popen handle it does not own. It never
signals an arbitrary process or PID group inferred from a stale receipt.

## Cleanup and evidence

Any engine owner, including an idle READY owner, blocks worker status/STOP/unload until its session
is reaped and child/monitor absence is checked. The retained owned-listener check precedes controls.
STOP records an intent before sending; timeout means uncertain, not failed-to-execute. It is not
reissued. The retained guardian `Popen.wait()` reaps the child and preserves its return code before
absence assertions and the listener probe. A pending stop can be reconciled through that same handle;
an unreaped/uncertain result blocks replacement. `SO_REUSEADDR` matches the worker and avoids treating
TIME_WAIT as a live listener. No port-free assertion runs ahead of an existing owned resident's stop.

If a newly launched worker fails startup before a listener is available, `failure_worker: "stop"`
terminates its retained guardian handle and uses that guardian's existing child cleanup. The driver
waits/reaps before checking recorded identities, without a control exchange. A separate startup-abort
receipt retains the original failure and cleanup outcome. Attached residents and workers with a STOP
intent are excluded; uncertain cleanup is recorded without replacing the original failure or retrying.
Missing or incomplete identity receipts leave child absence uncertain even after guardian reaping.

The original monitor/resource limits and 32 MiB stdout/stderr caps remain. Individual reads are
bounded to 64 KiB; startup exceptions enter owned cleanup, and child failure cannot become a successful
owner exit. Cleanup failures leave explicit uncertain receipts and block worker control. Retaining a
worker keeps its guardian and monitor alive. This is a harness process hierarchy, not a new service.

Keep the private config, pins, per-entry configs, owner exit/stop receipts, outputs, failures and
summary together. Do not commit private bindings, generated caches or raw campaign logs. Inspect the
exact intended staged file list and diff before publication. See [regressions](REGRESSIONS.md) and
[focused Hermes validation](HERMES.md). No desktop build, test or inference is authorized by this file.
