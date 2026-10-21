# Deterministic no-model runner v1 — NOT EXECUTED

This replaces the controller-generated admission launcher with one reviewed
runner and fixed child entry points. Possessing these files grants no execution
authority. No Python, syntax check, fixture, binary, build, live sampling or
remote host operation was executed by the author or reviewers.

Base input is `64cdafa99f95ac34712c956f5ba104dd542f1c3c`, parent
`a7f54f3ade61d8b712489d28f689bf92cd141aa5`. The three supplied input digests and
13999-byte directory total were verified before inspection. Historical handoff,
old supervisor/monitor sources and original budget remain byte-identical.

The user reports the earlier attempt BLOCKED at admission: permission denied
reading PID1's executable in an unpublished launcher; zero of44 synthetic or3
live cases ran. Evidence closeout245.96s exceeded the240s limit. Preserve that
procedural overrun and missing admission subprocess timestamps; it was not a
product/SSH failure or test PASS. Report digest:
`f20ce2ed7218de78303ffea491705277ce58d9a1ccb0a6e3146611a79d3818d3`.
The private evidence was not inspected here. The reported26-case PASS applies
only to unchanged a3b706d, not the44/3 or new25 cases.

## One entry, two modes

Invoke runner.py with explicit artifact/interpreter pins and a new private
evidence root. Hermes must not generate another admission script, selector,
supervisor, wrapper or deadline manager. Fetch and externally hash-check the
pinned delivery before invoking it; a program cannot authenticate its own
bootstrap against a maliciously replaced copy.

`synthetic-only` runs one fixed interpreter child: the25 new runner fixtures,
then the unchanged44 integration fixtures, with separate begin/PASS markers and
counts. It requires only safe new evidence paths, pinned artifact/interpreter
identities, a consistent local proc PID view and direct-child pidfd ownership.
It does not read namespace pairs, cgroup state, PSI or PID1; does not require
an administrator receipt; and does not configure a subreaper. Parent-death
prctl is used only on the newly forked child to prevent an orphan if its parent
dies. Required ownership/provenance/resource failures still block execution.

`synthetic-then-live` performs the same synthetic invocation first. Only its
success permits live admission. Missing or untrusted receipt inputs yield
synthetic PASS and live BLOCKED separately, exit3, with no live child invocation.
Authentic external inputs plus successful live facts/guard coverage permit one
unchanged three-case live_probe.py invocation. No fallback weakens initial-host
provenance to owned namespace self-consistency. Neither mode adds runtime
authority; each must be the explicitly approved subset.

## Trust input boundary

There is currently NO trusted administrator receipt. Live remains BLOCKED until
one genuinely exists. RECEIPT-TEMPLATE.json is deliberately invalid, not an
attestation, and cannot establish trust by being populated by the runner account.

An approver must independently verify the source/channel and administrator's
initial-host namespace collection, then supply these four private invocation
pins through the approved handoff, separately from the receipt:

- approved-receipt-sha256: SHA-256 of exact approved receipt bytes;
- approved-source-id: identifier of the independently verified source/channel;
- expected-host-id: the independently approved host identity;
- expected-machine-id-sha256: SHA-256 of exact live host machine-id file bytes,
  including any newline, bound by that verified administrator/source to the host.

Do not derive approval pins by copying claims from an untrusted receipt. A hash
detects byte changes; it does not authenticate a person, channel or host. The
runner cannot automate that human/admin source verification with current inputs.
If it has not occurred, omit the pins and keep live BLOCKED. No credentials,
signing service, privileged collector, permission change or installation is
introduced. Machine-id binding is a check within the approved host/channel trust
model, not cryptographic hardware attestation or protection against hostile
cloned hosts. Such authenticity cannot be inferred from this runner.

Receipt schema is exact: version1 integer(not boolean), role ROLE_CPU_C, approved
host/source/machine digest, current boot UUID, UTC issuance timestamp, descriptive
method and exactly pid/mnt/cgroup namespace pairs. Pairs contain exactly two
nonnegative integers, rejecting booleans. Duplicate JSON keys, nonfinite values,
excess nesting, malformed fields, extra keys, digest mismatch and nonregular or
oversize input block admission. Receipt reads are limited to16KiB with no final
symlink/FIFO/device and detect size/identity/mtime/ctime change during reading.

Issuance must be YYYY-MM-DDTHH:MM:SSZ, not in the future, no older than24h.
There is no clock-skew waiver; operator clock trust is an admission prerequisite.
The live machine-id digest must match the separately approved pin and receipt;
the live boot UUID must match the receipt; live self pairs must match the
administrator's initial-host receipt. Reboot or pair/host mismatch blocks.
Only then does unchanged live code evaluate actual applicable cgroup/headroom
and event coverage. Required unavailable protection blocks, never zero. Optional
diagnostic observations are not converted into a new stop policy.

Permission errors preserve a structured observation with operation, errno and
bounded private path where available. Missing, malformed, stale, wrong-host and
untrusted outcomes are distinct. An admission start/end snapshot is persisted
even when live admission is blocked; completed synthetic results remain intact.

## Ownership and corrected historical claims

No required admission reads PID1's executable or namespaces. Recognizing its
executable name would not prove initial-host provenance. The historical kernel
version/attach-mode explanation is not relied on: Linux proc exe access uses
PTRACE_MODE_READ_FSCREDS. The observations establish only the reported denied
paths/account; no exhaustive exclusion of all unprivileged trust methods is
claimed.

The old OwnedProcess.pin checks supplied identity, pidfd/proc dirfd/start time,
namespace pairs and parent relationships; it has NO explicit UID equality check.
New owned_child.py does explicitly check its newly forked child's proc directory
UID, parent, PID/start and interpreter inode. It proves wait ownership before
releasing exec. No arbitrary existing process is adopted or signalled. Synthetic
ownership does not use the old initial-host verifier. A normal terminal proc
read race is harvested through the held pidfd; a completed identity mismatch
remains a failure. Cleanup permits held-pidfd stop of an unexpected owned exec.
All signals use that child instance's pidfd, never groups, ports or numeric PID
fallback. Unreleased children receive gate EOF if pinning fails. TERM gets2s,
then KILL and up to3s actual parent wait/reap. Cleanup failure cannot be PASS.

## Executable time/resource envelope

Proposed total is unchanged240s: at most30s controller retrieval/hash checking
and launch, then210s on-target measured from runner module entry. Most imports,
argument parsing, evidence setup, admission, subprocesses and closeout consume
that fixed target clock. Interpreter startup before module entry belongs to the
controller's30s allowance, with controller-local actual start/end receipts.
If retrieval/startup does not fit30s, do not launch the runner. Do not compare
raw controller/target monotonic epochs.

Target work stops at entry+150s, reserving60s for cleanup/sealing until entry+210s.
Synthetic admission has10s; live admission5s, both clipped/refused against that
work boundary. Each stage is60s plus5s owned cleanup. A stage starts only if its
whole65s fits before the reserve. Worst stated allowances10+65+5+65=145s leave5s
for target setup/transition; unusually slow setup refuses subsequent work rather
than extending the envelope. Late live admission cannot consume the closeout
reserve. Actual stages/admissions have local mono/wall start/end observations;
no timestamps are recreated. Alarms are not swallowed as malformed observations.

Runner soft address space128MiB; synthetic interpreter256MiB,30CPU-seconds;
stdout+stderr combined8MiB. Live probe keeps its original128MiB/10CPU per
process,45s internal bound, exact strict64MiB fixed-baseline swap guard and
host/cgroup/event protection. Maximum4 processes: runner plus live guardian,
monitor and harmless child; aggregate address-space ceilings512MiB. No product,
GPU, network, model, service, pressure or swap manipulation is possible through
the fixed entry arguments. No production threshold changes are made.

Evidence is restricted to a new private directory,64 entries,32MiB aggregate
payload and8MiB per file. Logs and stage results are written incrementally.
Closeout hashes only this bounded owned tree; no archival scans, evidence bundle
copies or old directory sweeps. REPORT.json, SHA256SUMS and SEALED.json plus the
final marker are required. Missing final seal/marker or incomplete cleanup means
INCOMPLETE, never PASS. No post-deadline bookkeeping exception exists. Kernel
uninterruptible I/O or host failure cannot be made wall-clock bounded by Python;
such failure is incomplete evidence, not an extended budget or a success claim.

Exit0 means requested subset PASS; exit3 means BLOCKED; exit2 means failed or
incomplete. Reports distinguish runner25/integration44/live3 results and preserve
earlier success when a later stage is blocked/failed. Stop after the prescribed
attempt, without repair/retry. See HERMES.md for the proposed invocation.
