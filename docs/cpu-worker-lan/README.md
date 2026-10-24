# CPU-worker LAN and NIC controls

These controls are independent. Release defaults remain loopback binding and NIC tuning off. No worker,
engine, installer or inference startup invokes the NIC tool. Model arithmetic, placement, SEW1 framing,
residency and reconnect handling are unchanged by this feature. Verification is pending; see
[the Hermes handoff](HERMES.md). All new tests are **NOT RUN locally**.

## Native LAN worker

Choose one local numeric IPv4 address and each exact allowed client source. For example, with addresses
selected by the operator (these variables are not supplied defaults):

```sh
: "${WORKER_IP:?selected local LAN address}"
: "${ENGINE_IP:?permitted engine source address}"
: "${WORKER_PORT:?operator-approved port}"
./strata-expert-worker --port "$WORKER_PORT" --bind-ip "$WORKER_IP" \
  --allow-peer-ip "$ENGINE_IP" --tcp-nodelay
```

Add the existing local-source, residency, budget and pool settings from the approved configuration.
Repeat `--allow-peer-ip` for another approved engine or operator control source. An explicit list has no
implicit localhost exception. Default loopback callers retain their previous behavior when no list is
supplied. The one-connection service remains serial; an operator control request waits for the active
engine connection to close, as before.

The allowlist is checked before protocol reads, TCP_NODELAY setup and protocol dispatch. A rejected peer
cannot send SHUTDOWN, UNLOAD or setup frames into the state machine. Normal idle expiry still follows
elapsed time; rejected traffic does not renew the residency timer. No frame format or status payload was
added. Existing numeric client addresses and `worker_control.py --host` work with either route.

This is a trusted-LAN mode. Source filtering does not authenticate a host or process, and traffic is
unencrypted. The operator must review the firewall and restrict the selected destination/interface/port
to the intended sources before opening the listener. The tool neither creates nor changes firewall
rules. Do not use this service directly on an untrusted network. Use the existing SSH-forward route when
authentication or encryption is needed. An allowed process can unload weights or stop the service.

Existing client TCP_NODELAY remains in place. Accepted CPU sockets still require `--tcp-nodelay`; choose
it explicitly for matched LAN latency runs. GPU-worker accepted socket defaults remain unchanged.

## Operator NIC profile (Linux)

Use `python tools/worker_nic.py discover --interface IFACE` to read driver, firmware, kernel, PCI identity
and coalescing settings. Discovery reports support or actionable reasons for refusal. It never changes
settings. Missing `ethtool` is reported; this tool does not install it or escalate privileges.

The first supported mapping is e1000e with an original `rx-usecs` value of 0 or 3. On this driver, 3 is
dynamic-conservative interrupt moderation, **not a literal 3 microsecond delay**. The low-latency profile
requests mode 0, disabling moderation. Other drivers and other original modes are refused rather than
given an assumed mapping or a restore value that may not round-trip. The mode mapping is supported by
the [Linux e1000e documentation](https://docs.kernel.org/networking/device_drivers/ethernet/intel/e1000e.html)
and its [ethtool getter/setter](https://github.com/torvalds/linux/blob/master/drivers/net/ethernet/intel/e1000e/ethtool.c).
Disabling moderation can increase interrupt load; assess it on the intended workload.

The operator runs each action explicitly, using their normal privilege mechanism for apply and restore:

```sh
python tools/worker_nic.py snapshot --interface IFACE --snapshot /operator/path/nic-original.json
python tools/worker_nic.py apply --interface IFACE --snapshot /operator/path/nic-original.json
# The command above defaults to off and only reads settings.
python tools/worker_nic.py apply --interface IFACE --snapshot /operator/path/nic-original.json --profile low-latency
python tools/worker_nic.py discover --interface IFACE
python tools/worker_nic.py restore --interface IFACE --snapshot /operator/path/nic-original.json
```

Snapshots are created exclusively with mode 0600 and flushed to disk; an existing path is never
overwritten. Apply requires a matching snapshot from the same boot and exact interface, driver and
hardware identity. It changes only `rx-usecs`, then reads back the entire coalescing setting map. Restore
uses the saved original mode and verifies equality with the original map. Applying or restoring an
already matching map makes no write. Neither action overwrites unrelated setting drift. Serialize
operator actions and keep the snapshot until restore succeeds. A reboot or driver change requires
inspection and a new snapshot, not blind reuse of the old file.

A failed write or readback reports an uncertain outcome and preserves the original snapshot. Read back
with `discover` before retrying. If only the owned mode changed, explicitly restore with that snapshot.
If other settings drifted, resolve them with the operator; the tool refuses to overwrite them. It does
not silently roll back, restart links, reload drivers or claim an unsuccessful restore succeeded.
EEE, C-states, clocks, offloads, routes and security settings are outside this profile.

## Test configuration

Copy `packaging/worker-nic-test.example.json` to an operator-owned file. Its profile defaults to
`low-latency`, but its validated hardware list is deliberately empty. After remote apply/readback/restore
and matched inference validation, add a record with `identity` equal to the full discovered `identity`
object and `evidence` naming the retained results. Do not put a partial identity, a driver wildcard or
unverified hardware in the list. Driver, firmware or kernel changes require a new validation record.

```sh
python tools/worker_nic.py apply --interface IFACE --snapshot /operator/path/nic-original.json \
  --test-config /operator/path/validated-worker-nic.json
```

The explicit test configuration selects low latency only on an exact supported match. Normal use
defaults off. The test configuration is never loaded implicitly. This feature supplies no enabled boot
service or persistence hook; after reboot, the operator must inspect and opt in again.
