"""Host-local independent watchdog, parent reaper and monitored exec gate.

Static implementation only. Import is inert. Production launch needs a new
approved window/configuration; the future probe uses no product executable.
"""
from dataclasses import asdict
import json
import os
import select
import signal
import socket
import time

from resource_guard import (CoverageError, GuardStop, Journal, cgroup_snapshot,
                            membership, verify_namespaces)
from resource_telemetry import (DiagnosticBaselines, Identity, OwnedProcess, OwnedTree,
                                host_sample, verify_initial_namespaces)
from linux_owned import identity, parent_death, pin_child, promote_exec, subreaper
from policy import ReleaseGate, RoleGuard, Stamp


def collect(tree, namespaces):
    items = tree.refresh()
    sample = host_sample()
    sample["processes"] = [item.status() for item in items]
    sample["worker_memory_bytes"] = next(p["memory_bytes"] for p in sample["processes"]
                                         if p["identity"]["pid"] == tree.root.identity.pid)
    sample["cgroup"] = cgroup_snapshot(tree.root, namespaces)
    for item in items:
        verify_namespaces(item, namespaces)
        if str(membership(item.read("cgroup"))) != sample["cgroup"]["membership"]:
            raise CoverageError("owned-descendant-outside-covered-group")
    return sample


def packet(sock, value):
    raw = json.dumps(value, allow_nan=False).encode("ascii")
    if len(raw) > 4096 or sock.send(raw) != len(raw):
        raise CoverageError("bounded-control-packet")


def receive(sock):
    raw, _, flags, _ = sock.recvmsg(4097)
    if not raw:
        raise CoverageError("control-eof")
    if len(raw) > 4096 or flags & socket.MSG_TRUNC:
        raise CoverageError("control-packet-cap")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise CoverageError("control-packet-schema")
    return value


def monitor_child(root, namespaces, guard, outer, phase, journal_path, channel, key):
    tree, journal = OwnedTree(root), Journal(journal_path)
    baseline = None
    sequence = 0
    try:
        while True:
            started = time.monotonic()
            sample = collect(tree, namespaces)
            extra = outer.sample()
            if baseline is None:
                baseline = DiagnosticBaselines(sample)
            baseline.annotate(sample)
            journal.emit("INTEGRATION_RESOURCE_SAMPLE_V1", key=key, sequence=sequence,
                         mono=started, sample=sample, outer=extra)
            guard.check(key[1], sample, admission=sequence == 0)
            outer.check(extra, admission=False, phase=phase)
            if time.monotonic() - started >= 2:
                raise CoverageError("resource-sample-overrun")
            packet(channel, {"kind": "COVERED", "key": key, "sequence": sequence, "mono": started})
            sequence += 1
            time.sleep(max(0, started + 2 - time.monotonic()))
    except BaseException as exc:
        try:
            journal.emit("INTEGRATION_MONITOR_REFUSED_V1", key=key,
                         reason=type(exc).__name__ + ":" + str(exc))
            packet(channel, {"kind": "REFUSED", "key": key, "reason": type(exc).__name__})
        finally:
            try:
                journal.close()
            finally:
                tree.close()


def child_exec(gate, parent, argv, env, input_fd, out_fd, err_fd):
    parent_death(parent)
    if os.read(gate, 1) != b"G":
        os._exit(125)
    os.close(gate)
    os.dup2(input_fd, 0)
    os.dup2(out_fd, 1)
    os.dup2(err_fd, 2)
    # All control descriptors are CLOEXEC; a workload never keeps lease FDs.
    os.execve(argv[0], argv, {**os.environ, **env})


class Job:
    """The watchdog is the actual parent of BOTH workload and monitor."""
    def __init__(self, key, argv, env, stdin_fd, namespaces, guard, outer, deadline,
                 directory, *, monitor_behavior=None):
        self.key, self.namespaces = tuple(key), namespaces
        if monitor_behavior is not None and self.key[1] != "ROLE_PROBE":
            raise CoverageError("test-hook-in-production-role")
        self.items, self.reaped = {}, {}
        self.children = set()
        self.root = self.monitor = None
        self.channel = None
        self.exec_gate = None
        self.deadline = deadline
        self.last_sequence = -1
        self.gate = ReleaseGate(*self.key, time.monotonic(), deadline)
        self.directory = directory
        subreaper()
        verify_initial_namespaces(namespaces)
        try:
            self._start(argv, env, stdin_fd, guard, outer, monitor_behavior)
        except BaseException:
            self.stop("partial-launch")
            raise

    def _start(self, argv, env, stdin_fd, guard, outer, monitor_behavior):
        gate_r, gate_w = os.pipe2(os.O_CLOEXEC)
        self.exec_gate = gate_w
        out = os.open(self.directory + "/workload.out", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        err = os.open(self.directory + "/workload.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        parent = os.getpid()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(gate_w)
                child_exec(gate_r, parent, argv, env, stdin_fd, out, err)
            except BaseException:
                os._exit(126)
        self.children.add(pid)
        os.close(gate_r)
        os.close(out)
        os.close(err)
        self.root = promote_exec(pin_child(pid, self.namespaces), argv[0])
        self.items[pid] = self.root
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.channel = parent_socket
        monitor_r, monitor_w = os.pipe2(os.O_CLOEXEC)
        pid = os.fork()
        if pid == 0:
            try:
                parent_socket.close()
                os.close(monitor_w)
                os.close(self.exec_gate)
                parent_death(parent)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                if os.read(monitor_r, 1) != b"M":
                    os._exit(125)
                os.close(monitor_r)
                if monitor_behavior is not None:
                    monitor_behavior(child_socket)
                else:
                    monitor_child(self.root, self.namespaces, guard, outer, self.key[2],
                                  self.directory + "/resources.jsonl", child_socket, self.key)
            finally:
                os._exit(0)
        self.children.add(pid)
        os.close(monitor_r)
        child_socket.close()
        try:
            self.monitor = pin_child(pid, self.namespaces)
            self.items[pid] = self.monitor
            os.write(monitor_w, b"M")
        finally:
            os.close(monitor_w)
        self.gate.watchdog = True  # Parent-owned pidfds and wait ownership now proven.

    def poll(self):
        now = time.monotonic()
        self.gate.check(now)
        if self.channel in select.select([self.channel], [], [], 0)[0]:
            message = receive(self.channel)
            seq, at = message.get("sequence"), message.get("mono")
            if message.get("kind") != "COVERED" or tuple(message.get("key", ())) != self.key or \
                    type(seq) is not int or seq != self.last_sequence + 1 or \
                    type(at) not in (int, float) or not now - 4 <= at <= now:
                raise CoverageError("monitor-refused-stale-or-malformed")
            self.last_sequence = seq
            self.gate.covered_at = at
            if not self.gate.released:
                self.gate.release(self.key, now)
                os.write(self.exec_gate, b"G")
                os.close(self.exec_gate)
                self.exec_gate = None
        if not self.monitor.alive():
            raise CoverageError("monitor-exited")
        if not self.root.alive():
            return self.reap(self.root.identity.pid)
        return None

    def reap(self, pid):
        if pid in self.reaped:
            return self.reaped[pid]
        # This method is only called on direct/adopted children of this watchdog.
        got, status = os.waitpid(pid, os.WNOHANG)
        if got == 0:
            return None
        self.reaped[got] = os.waitstatus_to_exitcode(status)
        return self.reaped[got]

    def stop(self, reason, *, grace=15):
        if self.exec_gate is not None:
            os.close(self.exec_gate)
            self.exec_gate = None
        errors = []
        def live(item):
            try:
                return item.alive()
            except BaseException as exc:
                errors.append(type(exc).__name__)
                return True
        # Discover bounded descendants before stopping their parents. Failure is
        # reported incomplete; no arbitrary PID or group kill fills the gap.
        for item in list(self.items.values()):
            try:
                if live(item):
                    tree = OwnedTree(item)
                    for child in tree.refresh():
                        if child.identity.pid not in self.items:
                            self.items[child.identity.pid] = child
                        elif child is not item:
                            child.close()
            except BaseException as exc:
                errors.append(type(exc).__name__)
        for item in self.items.values():
            try:
                item.send(signal.SIGTERM)
            except BaseException as exc:
                errors.append(type(exc).__name__)
        end = min(self.deadline, time.monotonic() + max(0, min(grace, 15)))
        while time.monotonic() < end:
            if all(not live(item) for item in self.items.values()):
                break
            time.sleep(0.05)
        for item in self.items.values():
            try:
                if live(item):
                    item.send(signal.SIGKILL)
            except BaseException as exc:
                errors.append(type(exc).__name__)
        # Reap direct/adopted children only; negative waitpid is NOT a signal.
        reap_end = min(self.deadline, time.monotonic() + 1)
        while time.monotonic() < reap_end:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid:
                self.reaped[pid] = os.waitstatus_to_exitcode(status)
            elif all(pid in self.reaped for pid in self.children | set(self.items)):
                break
            else:
                time.sleep(0.02)
        result = {"reason": reason, "reaped": self.reaped.copy(), "errors": errors,
                  "complete": not errors and all(pid in self.reaped for pid in self.children | set(self.items))}
        for item in self.items.values():
            try:
                item.close()
            except OSError:
                result["complete"] = False
        if self.channel is not None:
            self.channel.close()
        return result


class Session:
    """One persistent host clock/baseline across all its serial jobs."""
    def __init__(self, run_id, role, domain, namespaces, outer):
        self.run_id, self.role, self.domain = run_id, role, domain
        self.namespaces, self.outer = namespaces, outer
        self.admission = self.baseline = self.guard = None

    def admit(self):
        verify_initial_namespaces(self.namespaces)
        if self.admission is None:
            self.admission = Stamp(self.domain, time.monotonic(), time.time())
        own = OwnedProcess.pin(os.getpid(), expected=identity(os.getpid(), self.namespaces),
                               namespace_receipt=self.namespaces)
        try:
            # Before each job, the agent has no live owned job children.
            current = collect(OwnedTree(own), self.namespaces)
        finally:
            own.close()
        if self.baseline is None:
            self.baseline = current
            self.guard = RoleGuard(self.role, self.baseline)
        self.guard.check(self.role, current, admission=True)
        return self.admission
