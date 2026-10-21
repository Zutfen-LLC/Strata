"""New Linux monitor replacement, revision 1; diagnostic collection only.

NOT EXECUTED. No model, network, subprocess launch or numeric-PID signalling.
Private operational bindings and observed output must never be published.
"""
from dataclasses import asdict, dataclass
import errno
import math
import os
from pathlib import Path
import re
import select
import signal

READ_CAP = 262144
MAX_PROCESSES = 16
MAX_TASKS = 256
MAX_CHILD_SCANS = 2
STATUS_FIELDS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap")
HOST_FIELDS = ("MemAvailable", "SwapTotal", "SwapFree")


class CoverageError(ValueError):
    """A required observation or ownership proof cannot be evaluated."""


class ProcessExited(CoverageError):
    """A held pidfd confirms exit; never inferred from an unreadable entry."""


@dataclass(frozen=True)
class Observation:
    value: object = None
    status: str = "unavailable"
    reason: str = "missing"


def ok(value):
    return Observation(value, "ok", "")


def malformed(reason):
    return Observation(None, "malformed", reason)


def required(observation, label):
    if observation.status != "ok":
        raise CoverageError(label + ":" + observation.status + ":" + observation.reason)
    return observation.value


def read_text(path, dir_fd=None):
    """Bound reads, including zero-sized procfs files; reject final symlinks."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=dir_fd)
        try:
            chunks, total = [], 0
            while total <= READ_CAP:
                chunk = os.read(fd, min(16384, READ_CAP + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            if total > READ_CAP:
                return malformed("read-cap")
            return ok(b"".join(chunks).decode("ascii"))
        finally:
            os.close(fd)
    except UnicodeError:
        return malformed("encoding")
    except OSError as exc:
        return Observation(None, "unavailable", errno.errorcode.get(exc.errno, "IO"))


def kib_fields(text, keys):
    if text.status != "ok":
        return {key: text for key in keys}
    found = {key: [] for key in keys}
    for line in text.value.splitlines():
        key, sep, value = line.partition(":")
        if sep and key in found:
            found[key].append(value.strip())
    result = {}
    for key, rows in found.items():
        if not rows:
            result[key] = Observation()
        elif len(rows) != 1 or not re.fullmatch(r"[0-9]+\s+kB", rows[0]):
            result[key] = malformed("expected-one-kB-field")
        else:
            result[key] = ok(int(rows[0].split()[0]) * 1024)
    return result


def counters(text, keys):
    if text.status != "ok":
        return {key: text for key in keys}
    found = {key: [] for key in keys}
    for line in text.value.splitlines():
        parts = line.split()
        if parts and parts[0] in found:
            found[parts[0]].append(parts[1:])
    result = {}
    for key, rows in found.items():
        if not rows:
            result[key] = Observation()
        elif len(rows) != 1 or len(rows[0]) != 1 or not rows[0][0].isdigit():
            result[key] = malformed("expected-one-unsigned-counter")
        else:
            result[key] = ok(int(rows[0][0]))
    return result


def psi(text):
    if text.status != "ok":
        return text
    result = {}
    try:
        for line in text.value.splitlines():
            fields = line.split()
            if not fields or fields[0] not in ("some", "full") or fields[0] in result:
                raise ValueError()
            values = dict(part.split("=", 1) for part in fields[1:])
            if len(fields) != 5 or set(values) != {"avg10", "avg60", "avg300", "total"}:
                raise ValueError()
            row = {name: float(values[name]) for name in ("avg10", "avg60", "avg300")}
            if not all(math.isfinite(v) and 0 <= v <= 100 for v in row.values()):
                raise ValueError()
            if not values["total"].isdigit():
                raise ValueError()
            row["total_us"] = int(values["total"])
            result[fields[0]] = row
        if "some" not in result:
            raise ValueError()
        # CPU full may legitimately be absent on older supported kernels.
        result.setdefault("full", asdict(Observation(None, "unsupported", "not-exported")))
        return ok(result)
    except (ValueError, TypeError):
        return malformed("psi-schema")


def counter_delta(current, baseline):
    if current.status != "ok":
        return current
    if baseline.status != "ok":
        return Observation(None, "unavailable", "baseline-unavailable")
    if current.value < baseline.value:
        return malformed("counter-decreased")
    return ok(current.value - baseline.value)


class DiagnosticBaselines:
    """First-observation anchor and previous sample; never silently re-anchor."""
    def __init__(self, sample):
        self.initial = sample["vmstat_pages"]
        self.previous = self.initial

    def annotate(self, sample):
        current = sample["vmstat_pages"]
        sample["vmstat_since_observation_start_pages"] = {
            k: counter_delta(current[k], self.initial[k]) for k in self.initial}
        sample["vmstat_interval_pages"] = {
            k: counter_delta(current[k], self.previous[k]) for k in self.initial}
        self.previous = current


@dataclass(frozen=True)
class Identity:
    pid: int
    start_ticks: int
    exe_device: int
    exe_inode: int


def parse_stat(text):
    """comm may contain spaces and ')'; starttime is field 22."""
    raw = required(text, "process-stat")
    try:
        left, right = raw.index("("), raw.rindex(")")
        pid = int(raw[:left].strip())
        rest = raw[right + 1:].split()
        ppid, start = int(rest[1]), int(rest[19])
        if pid <= 0 or ppid < 0 or start < 0 or len(rest[0]) != 1:
            raise ValueError()
        return pid, ppid, start
    except (ValueError, IndexError):
        raise CoverageError("malformed-process-stat") from None


def verify_initial_namespaces(expected, proc_fd=None):
    if not isinstance(expected, dict) or set(expected) != {"cgroup", "mnt", "pid"}:
        raise CoverageError("initial-host-namespace-receipt-required")
    for kind in ("cgroup", "mnt", "pid"):
        wanted = expected[kind]
        if not isinstance(wanted, (tuple, list)) or len(wanted) != 2 or \
                any(type(v) is not int or v < 0 for v in wanted):
            raise CoverageError("namespace-receipt-schema")
        own = os.stat("/proc/self/ns/" + kind)
        if (own.st_dev, own.st_ino) != tuple(wanted):
            raise CoverageError("initial-host-namespace-mismatch:" + kind)
        if proc_fd is not None:
            other = os.stat("ns/" + kind, dir_fd=proc_fd)
            if (other.st_dev, other.st_ino) != tuple(wanted):
                raise CoverageError("owned-process-namespace-mismatch:" + kind)


class OwnedProcess:
    """Both proc directory FD and pidfd refer to the pinned process instance."""
    def __init__(self, identity, proc_fd, pid_fd, namespace_receipt=None):
        self.identity, self.proc_fd, self.pid_fd = identity, proc_fd, pid_fd
        self.namespace_receipt = namespace_receipt

    @classmethod
    def pin(cls, pid, expected=None, parent=None, namespace_receipt=None):
        if not callable(getattr(os, "pidfd_open", None)) or \
                not callable(getattr(signal, "pidfd_send_signal", None)):
            raise CoverageError("pidfd-required")
        if parent is not None:
            namespace_receipt = parent.namespace_receipt
        elif not isinstance(expected, Identity) or expected.pid != pid:
            raise CoverageError("supervisor-root-identity-receipt-required")
        # Must precede pidfd/proc pinning and child enumeration: both numeric
        # PID views must name the same initial host process namespace.
        verify_initial_namespaces(namespace_receipt)
        if pid <= 1:
            raise CoverageError("unsafe-pid")
        try:
            pid_fd = os.pidfd_open(pid, 0)
        except ProcessLookupError:
            if parent is None:
                raise
            parent.verify()
            return None  # No process instance exists at this lookup; signal nothing.
        proc_fd = None
        retained = False
        try:
            if select.select([pid_fd], [], [], 0)[0]:
                raise ProcessExited("discovered-process-exited")
            proc_fd = os.open("/proc/" + str(pid), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            stat = read_text("stat", proc_fd)
            if stat.status == "unavailable" and stat.reason in ("ENOENT", "ESRCH") and \
                    select.select([pid_fd], [], [], 0)[0]:
                raise ProcessExited("discovered-process-exited-before-stat")
            actual_pid, ppid, start = parse_stat(stat)
            if actual_pid != pid:
                raise CoverageError("process-identity-mismatch")
            if parent is not None:
                parent.verify()
                if ppid != parent.identity.pid:
                    raise CoverageError("child-parent-mismatch")
            exe = os.stat("exe", dir_fd=proc_fd)
            identity = Identity(actual_pid, start, exe.st_dev, exe.st_ino)
            verify_initial_namespaces(namespace_receipt, proc_fd)
            item = cls(identity, proc_fd, pid_fd, namespace_receipt)
            if actual_pid != pid or (expected is not None and identity != expected):
                raise CoverageError("process-identity-mismatch")
            item.verify()
            retained = True
            return item
        except (ProcessExited, OSError) as exc:
            terminal = isinstance(exc, ProcessExited) or (exc.errno in (errno.ENOENT, errno.ESRCH) and
                exc.filename in ("/proc/" + str(pid), "exe", "ns/cgroup", "ns/mnt", "ns/pid"))
            if parent is not None and terminal and select.select([pid_fd], [], [], 0)[0]:
                parent.verify()
                return None  # Proven exit, not permission/identity/schema failure.
            raise
        finally:
            if not retained:
                try:
                    if proc_fd is not None:
                        os.close(proc_fd)
                finally:
                    os.close(pid_fd)

    def alive(self):
        if self.pid_fd is None:
            raise CoverageError("owned-process-handle-closed")
        return not select.select([self.pid_fd], [], [], 0)[0]

    def verify(self, executable=True):
        if not self.alive():
            raise ProcessExited("owned-process-exited")
        stat = read_text("stat", self.proc_fd)
        if stat.status == "unavailable" and stat.reason in ("ENOENT", "ESRCH") and not self.alive():
            raise ProcessExited("owned-process-exited-during-stat")
        pid, _, start = parse_stat(stat)
        if (pid, start) != (self.identity.pid, self.identity.start_ticks):
            raise CoverageError("owned-process-identity-changed")
        if executable:
            try:
                exe = os.stat("exe", dir_fd=self.proc_fd)
            except OSError as exc:
                if exc.errno in (errno.ENOENT, errno.ESRCH) and not self.alive():
                    raise ProcessExited("owned-process-exited-during-exe") from None
                raise
            if (exe.st_dev, exe.st_ino) != (self.identity.exe_device, self.identity.exe_inode):
                raise CoverageError("owned-process-executable-changed")
        if not self.alive():
            raise ProcessExited("owned-process-exited")

    def read(self, name):
        self.verify()
        value = read_text(name, self.proc_fd)
        try:
            self.verify()
        except ProcessExited:
            if value.status != "ok" and not (value.status == "unavailable" and
                                               value.reason in ("ENOENT", "ESRCH")):
                required(value, "process-read:" + name)  # Preserve genuine read failures.
            raise
        return value

    def status(self):
        # All reads use held FDs, not a reused numeric /proc/PID pathname.
        values = kib_fields(self.read("status"), STATUS_FIELDS)
        return {"identity": asdict(self.identity), "memory_bytes": values}

    def children(self):
        base = "/proc/" + str(self.identity.pid) + "/task"
        path, scan, before, after, vanished = base, 0, (), (), []
        try:
            self.verify()
            task_fd = os.open("task", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                              dir_fd=self.proc_fd)
            try:
                def tasks():
                    names = []
                    # scandir(fd) duplicates a descriptor with a shared directory
                    # offset. Each pass needs a fresh open description at offset 0.
                    scan_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                                      dir_fd=task_fd)
                    try:
                        with os.scandir(scan_fd) as entries:
                            for entry in entries:
                                names.append(entry.name)
                                if len(names) > MAX_TASKS:
                                    raise CoverageError("task-enumeration-cap")
                    finally:
                        os.close(scan_fd)
                    if any(not re.fullmatch(r"[0-9]{1,10}", n) or int(n) <= 0 for n in names) or \
                            len(set(names)) != len(names):
                        raise CoverageError("task-enumeration-schema")
                    if str(self.identity.pid) not in names:
                        raise CoverageError("required-leader-task-absent")
                    return tuple(sorted(names, key=int))

                children = set()
                for scan in range(1, MAX_CHILD_SCANS + 1):
                    path, after, vanished = base, (), []
                    before = tasks()
                    for task in before:
                        path = base + "/" + task + "/children"
                        value = read_text(task + "/children", task_fd)
                        if value.status == "unavailable" and value.reason == "ENOENT":
                            vanished.append(task)
                            continue  # Prove disappearance below; never fabricate [].
                        raw = required(value, "children")
                        if any(not re.fullmatch(r"[0-9]{1,10}", p) or int(p) <= 1 for p in raw.split()):
                            raise CoverageError("children-schema")
                        children.update(map(int, raw.split()))
                        if len(children) > MAX_PROCESSES:
                            raise CoverageError("children-cap")
                    self.verify()
                    path = base
                    after = tasks()
                    self.verify()
                    still_present = set(vanished).intersection(after)
                    if still_present:
                        path = base + "/" + min(still_present, key=int) + "/children"
                        raise CoverageError("stable-required-entry-absence:ENOENT")
                    # Always make the bounded second pass, including surviving
                    # tasks already read: an exiting thread can reparent children
                    # onto them. Neither task nor child sets must converge.
                    # Children born after this finite observation are discovered
                    # by later samples/adoption; this is not an atomic snapshot.
                return sorted(children)
            finally:
                os.close(task_fd)
        except (CoverageError, OSError) as exc:
            terminal = isinstance(exc, ProcessExited) or (isinstance(exc, OSError) and
                       exc.errno in (errno.ENOENT, errno.ESRCH) and not self.alive())
            detail = str(exc)
            if isinstance(exc, OSError):
                detail = errno.errorcode.get(exc.errno, "IO") + ":" + detail
            error = ProcessExited if terminal else CoverageError
            raise error("owned-children:pid=" + str(self.identity.pid) +
                                ":start_ticks=" + str(self.identity.start_ticks) +
                                ":exe_device=" + str(self.identity.exe_device) +
                                ":exe_inode=" + str(self.identity.exe_inode) +
                                ":path=" + path + ":scan=" + str(scan) + ":cause=" + detail +
                                ":tasks_before=" + repr(before) + ":tasks_after=" + repr(after) +
                                ":vanished=" + repr(vanished)) from None

    def send(self, sig):
        # Never fall back to os.kill(pid), kill-by-port or process-group signals.
        if not self.alive():
            return "already-exited"
        # An unexpected exec still belongs to this pidfd/start-time instance;
        # reject its samples but permit safe cleanup of the owned task.
        self.verify(executable=False)
        signal.pidfd_send_signal(self.pid_fd, sig, None, 0)
        return "sent"

    def close(self):
        errors = []
        for name in ("proc_fd", "pid_fd"):
            fd = getattr(self, name)
            setattr(self, name, None)  # Invalidate before close; never retry a reused FD.
            if fd is not None:
                try:
                    os.close(fd)
                except OSError as exc:
                    errors.append(exc)
        if errors:
            raise OSError("owned-descriptor-close-failed") from errors[0]


class OwnedTree:
    def __init__(self, root):
        self.root = root
        self.items = {root.identity.pid: root}

    def retire_exited(self):
        for pid, item in list(self.items.items()):
            if item is not self.root and not item.alive():
                del self.items[pid]
                item.close()

    def refresh(self):
        verify_initial_namespaces(self.root.namespace_receipt, self.root.proc_fd)
        self.root.verify()
        self.retire_exited()
        queue = [p for p in self.items.values() if p.alive()]
        for parent in queue:
            if self.items.get(parent.identity.pid) is not parent:
                continue  # A prior iteration retired this confirmed exited handle.
            try:
                children = parent.children()
            except ProcessExited:
                if parent is self.root or parent.alive():
                    raise
                continue
            for pid in children:
                if pid in self.items:
                    try:
                        self.items[pid].verify()
                        continue
                    except ProcessExited:
                        if self.items[pid] is self.root or self.items[pid].alive():
                            raise
                        self.retire_exited()
                try:
                    child = OwnedProcess.pin(pid, parent=parent)
                except (CoverageError, OSError) as exc:
                    raise CoverageError("owned-child-pin:pid=" + str(pid) + ":parent=" +
                                        repr(parent.identity) + ":" + type(exc).__name__ + ":" + str(exc)) from None
                if child is None:
                    continue
                self.retire_exited()
                if len(self.items) >= MAX_PROCESSES:
                    child.close()
                    raise CoverageError("owned-process-cap")
                self.items[pid] = child
                queue.append(child)
        self.root.verify()
        return [p for p in self.items.values() if p.alive()]

    def close(self):
        errors = []
        for item in self.items.values():
            try:
                item.close()
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise OSError("owned-tree-close-failed") from errors[0]


def host_sample(proc_root=Path("/proc")):
    return {
        "host_bytes": kib_fields(read_text(proc_root / "meminfo"), HOST_FIELDS),
        "vmstat_pages": counters(read_text(proc_root / "vmstat"), ("pswpin", "pswpout")),
        "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
        "psi": {kind: psi(read_text(proc_root / "pressure" / kind))
                for kind in ("cpu", "memory", "io")},
    }
