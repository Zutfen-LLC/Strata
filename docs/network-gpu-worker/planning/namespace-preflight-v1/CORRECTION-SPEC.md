# Namespace preflight v1 — findings and correction specification

Status: FINDINGS + SPEC for a reviewed implementation. Nothing here has been
executed against any host. Codex (or another reviewed coding agent) implements
this once; each future Hermes validation run must run the published preflight,
never invent its own launcher.

## 1. Exact cause

- Failing operation: `os.readlink('/proc/1/exe')` inside the admission
  launcher generated ad hoc by the validation controller (the run's
  `python3 -c` preflight), statement 6 of the embedded script.
- Result: `PermissionError: [Errno 13] Permission denied`, RC 1, JSON snapshot
  never emitted, both validation stages never invoked (BLOCKED at admission).
- Source classification: the readlink is NOT in any published a7f54f3 file.
  The published code's namespace handling is `verify_initial_namespaces()`
  (resource_telemetry.py), which only compares *supplied* device/inode pairs
  against `/proc/self/ns/*` and an owned process's `ns/*` via a proc dirfd.
  It never reads PID 1 and has no executable-identification step at all.
- Intent of the failed line: contribute to "host/init namespace provenance"
  for the run report. It conflated three separable concerns: (a) executable
  identification of PID 1, (b) process ownership, (c) namespace provenance.
  Even if it had succeeded, recognizing PID 1's executable would NOT prove the
  supplied pairs identify the initial host namespaces.
- The permission denial itself is expected: since kernel 4.11 (hidepid/hardened
  procfs), readlink of `/proc/<pid>/exe` requires ptrace-may-attach access
  (or CAP_SYS_PTRACE); a plain unprivileged account is denied on foreign-root
  processes. This is a launcher-authoring defect, not a host misconfiguration.

## 2. Correctly separated concerns

1. Executable identification (of owned processes only): pinned via
   `Identity(exe_device, exe_inode)` from `os.stat("exe", dir_fd=proc_fd)` in
   the published code. Applies to owned children, never to PID 1.
2. Process ownership: uid match + pidfd + proc dirfd + starttime pin, exactly
   as `OwnedProcess.pin` already does. Unprivileged. Unchanged.
3. Namespace provenance: TWO distinct questions that must not be merged:
   a. self-consistency — the supplied pairs match `/proc/self/ns/*` and the
      owned process's `ns/*` (published verifier; unprivileged; unchanged);
   b. initial-host binding — whether the pairs are THE initial host
      namespaces. This is NOT derivable from (a), not from PID 1's comm,
      not from an SSH connection, not from a systemd exe name, and not from
      absence of container markers. It requires an externally trusted input.

## 3. Available provenance (observed, read-only, current boot)

One narrowly scoped observation on ROLE_CPU_C (2026-10-21, existing account,
command and full result recorded locally; sanitized result below):

- `os.stat('/proc/self/ns/{pid,mnt,cgroup}')` — readable, values observed.
- `/proc/1/comm` — readable, value `systemd` (explicitly NOT trust evidence).
- `/proc/sys/kernel/random/boot_id` — readable, value observed (binds the
  receipt to the current boot).
- `os.stat('/proc/1/ns/*')` and `os.readlink('/proc/1/exe')` — EACCES (13)
  for this account.

Conclusion: on this host and account, self-vs-PID-1 namespace equality is
NOT establishable unprivileged. Every path that would let the account itself
assert "these pairs are the initial host namespaces" is denied. Therefore the
initial-host binding must come from an administrator-supplied receipt, or the
trust model must change (see §7).

## 4. Required trusted input (administrator receipt)

Smallest sufficient receipt (per host, per boot):

```json
{
  "role": "ROLE_CPU_C",
  "host_identity": "<admin-asserted host identity, e.g. serial or admin host key fingerprint>",
  "boot_id": "<value of /proc/sys/kernel/random/boot_id at receipt time>",
  "issued_utc": "<RFC3339>",
  "namespaces": {"pid": [<dev>, <ino>], "mnt": [<dev>, <ino>], "cgroup": [<dev>, <ino>]},
  "method": "<how admin established initial-host binding, e.g. root shell stat of /proc/1/ns on the host console>",
  "signature_or_channel": "<out-of-band admin attestation>"
}
```

Freshness/binding checks (all mandatory):
- `boot_id` in receipt == live `/proc/sys/kernel/random/boot_id` (receipt dies
  with the boot; re-issue after every reboot).
- Each pair == live `os.stat('/proc/self/ns/<kind>')` — this simultaneously
  proves the account itself runs in the receipted namespaces.
- Exact schema `{pid, mnt, cgroup}`, pairs of non-negative ints; anything
  else is malformed-data, not a soft pass.
- `issued_utc` not in the future; consumer records receipt hash in evidence.

## 5. Corrected preflight contract (deterministic; implement once)

Input: the admin receipt (§4) plus an explicit per-run deadline.

Phases, in order, fail-closed:

P0 unprivileged self facts — uid, `/proc/self/ns/{pid,mnt,cgroup,user}` stat,
pidfd_open(self), subreaper prctl, cgroup files, meminfo/vmstat/PSI, owned-
conflict scan (uid-owned cmdline match). Any OSError → record and continue
only if the item is optional (marked below); core items are mandatory.

P1 receipt validation (§4 checks). Outcomes:
- receipt missing / not supplied → `ADMISSION-BLOCKED: receipt-required`;
- EACCES/EPERM on any self read → `ADMISSION-BLOCKED: permission-denied:<path>`;
- boot_id mismatch → `ADMISSION-BLOCKED: receipt-stale-reboot`;
- pair mismatch → `ADMISSION-BLOCKED: namespace-mismatch:<kind>`;
- schema violation → `ADMISSION-BLOCKED: receipt-malformed`.
`ENOENT`/`ENOSYS` on optional feature probes (NS_GET ioctls, psi files) →
`unsupported:<feature>` recorded as NOT ESTABLISHED, never faked as pass,
and never fatal unless a recipe phase requires that feature.

P2 admission snapshot — emit the full JSON snapshot with exact admission
start/end timestamps (see §6) and the receipt hash; only then may any stage
invoke the published runtime/live_probe entry points with the receipted pairs
as `namespaces`.

Split prerequisites:
- Synthetic suite (44 cases): requires P0 + process-ownership facts only.
  Host-only namespace provenance MUST NOT gate synthetic tests.
- Live probes (3 cases): additionally require P1 success (receipt). No
  receipt → synthetic may still run; live probes stay BLOCKED, explicitly.

## 6. Timing contract

- One monotonic deadline covers the whole admission + stages + cleanup:
  `deadline = time.monotonic() + BUDGET` fixed before any subprocess.
- Admission start timestamp: `mono` captured immediately before launching the
  preflight subprocess (the gap the failed run disclosed); admission end
  captured at snapshot emission; both persisted in evidence.
- Every subprocess gets `min(remaining_deadline, per_stage_cap)`; each poll
  loop checks the global deadline; cleanup is INSIDE the deadline with a
  reserved tail: seal evidence at `deadline - 60s` hard, no new stage may
  start after `deadline - 120s`. If sealing would overrun, write the
  truncated-blocked report with real timestamps, never reconstructed ones.
- The prior run's 245.96 s against a 240 s window remains an OVERRUN, is not
  relabeled compliant, and is the reason for the reserved sealing tail.

## 7. Design question for review (not an operational workaround)

For THIS owned-child probe, the runtime only ever needs: pairs that are
(i) the account's own current namespaces and (ii) identical across self and
every owned child (the published verifier's actual guarantee). The
initial-host claim is strictly stronger than what the probe consumes. Options:
A. Keep initial-host requirement; require the admin receipt (§4) for live.
B. Weaken to "self-consistent owned-namespace receipt" (no admin input; the
   probe's guarantees are unchanged for owned children, but the report must
   then say `namespaces: owned-self-consistent` instead of `initial-host`).
Option B is a code/design change requiring review; it must not be applied as
an operational workaround to admit a run. Spec default: A.

## 8. Non-goals / preserved guards

No change to verify_namespaces/OwnedProcess/guards/rlimits/64 MiB cap; no
rerun of either stage here; no execution of the corrected preflight in this
handoff; failed-run evidence retained unmodified (REPORT SHA-256
f20ce2ed7218de78303ffea491705277ce58d9a1ccb0a6e3146611a79d3818d3).
