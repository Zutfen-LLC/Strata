# Fixed CPU peer bindings

Start with the working private local `config.example.json` contract. Keep its engine/model paths,
GPU visibility, environment, GEN lines, masks and deadlines. This is an opt-in extension of the same
ordered `run.py`; there is no extra campaign script. All example identifiers below are placeholders.

Add `capture_first_logits: true` to capture the existing engine diagnostic at a unique arm path.
Do not supply a global `STRATA_DUMP_FIRST_LOGITS`. Keep `worker_bytes` equal to the full CPU complement
and an explicit primary `--memory-budget-mib` large enough for the local arms. Add:

```json
{
  "worker_peers": [
    {"name": "local", "local": true, "endpoint": "127.0.0.1", "port": 27431},
    {
      "name": "second", "endpoint": "__APPROVED_WORKER_IPV4__", "port": 27431,
      "control_argv": ["__ABS_EXISTING_AUTHORIZED_TRANSPORT__", "__PRIVATE_TRANSPORT_ARGUMENTS__"],
      "config": {
        "host": "__ACTUAL_SECOND_HOSTNAME__", "run_root": "__NEW_ABS_SECOND_OUTPUT_ROOT__",
        "assets": {"worker": "__ABS_SECOND_WORKER_BINARY__", "manifest": "__ABS_SECOND_MANIFEST__"},
        "worker_argv": ["@worker", "--bind-ip", "__APPROVED_WORKER_IPV4__", "--port", "27431",
          "--allow-peer-ip", "__APPROVED_ENGINE_SOURCE_IPV4__", "--workers", "12",
          "--local-load-threads", "4", "--memory-budget-mib", "__SECOND_BUDGET_MIB__",
          "--deadline-ms", "10000", "--local-setup-ms", "600000",
          "--local-root", "__ABS_SECOND_LOCAL_MODEL_ROOT__", "--local-manifest", "@manifest", "--residency-forever"],
        "worker_host": "__APPROVED_WORKER_IPV4__", "port": 27431,
        "workers": 12, "loader_threads": 4, "worker_bytes": 58067865600,
        "caller_mask": [0], "pool_masks": [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12]],
        "worker_env": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "OMP_NUM_THREADS": "2"},
        "monitor_env": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        "monitor_gpu": false, "worker_start_seconds": 120, "shutdown_seconds": 420, "cleanup_seconds": 90
      }
    }
  ]
}
```

Replace example CPU masks and all fields from the verified runtime bindings; the numbers above are
syntax examples, not a topology prescription. The transport must run the *staged* second-host
`worker_owner.py /absolute/staged/config.json --rpc` with stdin/stdout preserved. It must use existing
authorized access, no prompts, terminal, banners or stdout mixing. If that route does not already
exist, stop before worker launch and report the missing binding; this feature grants no access.

On each selected host, use the same source checkout and private config. Stage once per host:

```sh
# Second host, existing authorized execution path; starts no worker.
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$PRIVATE_CONFIG" --worker second
# Engine host, after the second bundle is available; starts no worker.
python3 -B "$SRC/tools/validation-harness/harness/bundle.py" --config "$PRIVATE_CONFIG"
# Engine host, only after prerequisite checks and named owned-worker handoff approval.
python3 -B "$ENGINE_RUN_ROOT/run.py" --run-root "$ENGINE_RUN_ROOT"
```

Second-host paths are checked on that host. Before START, the engine controller compares helper
hashes, worker/manifest hashes, normalized argv (including budget), environment, masks and deadlines
against its pinned config. By default these hashes bind to controller assets `worker` and `manifest`;
optional `asset_bindings: {"worker": "worker_second", "manifest": "manifest"}` names another staged
controller asset, for a deliberately different worker build. Source/destination hashes are mechanical;
never transcribe them by hand or upload over a staged bundle.

Each existing arm adds ordered `cpu_peers` and matching `peer_bytes`. The first peer is always local.
The selected bytes must sum to top-level `worker_bytes` and fit explicit budgets. Counts and compute
masks stay identical across arms. A remote peer has one contiguous lifetime, beginning at a zero-GEN
restart entry; the final arm is local-only. This first trial does not adopt an unrelated remote resident.

| Entry | restart | requests | cpu_peers | peer_bytes |
| --- | --- | --- | --- | --- |
| A0-load | true | [] | [local] | [116135731200] |
| A0 | false | [0,1,2] | [local] | [116135731200] |
| B-load | true | [] | [local,second] | [58067865600,58067865600] |
| B | false | [0,1,2] | [local,second] | [58067865600,58067865600] |
| A1-load | true | [] | [local] | [116135731200] |
| A1 | false | [0,1,2] | [local] | [116135731200] |

All entries explicitly carry the existing workers/loaders/masks/placement markers.
By default, the base engine argv keeps one `--remote-cpu-experts` binding for preflight; arm selection replaces it
with two `--remote-cpu-worker` bindings only for B. One-peer arms retain the legacy option. To test
one-member group execution explicitly, set top-level `cpu_route: "group"` and replace that one argv
flag with `--remote-cpu-worker`, keeping the exact control endpoint and trusted manifest. Omission
or `cpu_route: "legacy"` requires the legacy flag. Conflicting, duplicate or mismatched bindings are
refused before staging. The explicit choice survives single-peer argv construction; two-peer arms
still use group mode. A warm single-endpoint config needs no `worker_peers` fields. Keep the same
engine/worker assets across arms to isolate placement. Retain policy applies to the final local worker;
the second subset is stopped/reaped before A1. Do not use `attach_root` for this restart sequence.

Per-host identities, phase masks, status/accounting, transport exit and child-reap acknowledgements
are retained under the main output root's `peers/<name>` and the second host's own bundle. Controls
originate from the engine host only while all engine owners are reaped. During an idle READY engine
session, only owned-TID affinity commands run on the second guardian, after the local session receipt
check; no STATUS/STOP/unload is queued. SETUP uses the configured CPU union, then final masks apply
before GEN on both hosts. A timed-out STOP is attempted once and reconciled through that guardian.
Failure abort uses existing owned-child cleanup and requires a bounded reaping acknowledgement;
transport exit without that acknowledgement is explicitly uncertain and blocks replacement.
