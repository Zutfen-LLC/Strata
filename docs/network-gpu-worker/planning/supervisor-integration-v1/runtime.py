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
            if not root.alive():
                packet(channel, {"kind": "EXITED", "key": key})
                return
            try:
                sample = collect(tree, namespaces)
            except BaseException:
                if root.alive():
                    raise
                # Terminal sampling is distinct from a completed sample that
                # violated a guard. A preceding REFUSED is never rewritten.
                packet(channel, {"kind": "EXITED", "key": key})
                return
            extra = outer.sample()
            if baseline is None:
                baseline = DiagnosticBaselines(sample)
            baseline.annotate(sample)
            journal.emit("INTEGRATION_RESOURCE_SAMPLE_V1", key=key, sequence=sequence,
                         mono=started, sample=sample, outer=extra)
            guard.check(key[1], sample, admission=sequence == 0)
            if sequence == 0 and key[1] == "ROLE_ENGINE_A":
                # The workload is still behind the exec gate. Preserve the
                # strict local floor at initial coverage as well as admission.
                outer.check(extra, admission=True, phase=phase)
            diagnostic = outer.check(extra, admission=False, phase=phase)
            if diagnostic is not None:
                journal.emit("LOCAL_GPU_RESERVE_DIAGNOSTIC_V1", key=key, sequence=sequence,
                             mono=started, **diagnostic)
                if any(diagnostic["first_warning"]):
                    # At most one warning per selected local GPU per phase;
                    # later samples retain minima and every crossing count.
                    journal.emit("GPU_RESERVE_WARNING_V1", key=key, sequence=sequence,
                                 mono=started, **diagnostic)
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
    os.execve(argv[0], argv, env)  # Exact approved environment; inherit no overrides.


class PartialLaunch(CoverageError):
    def __init__(self, cause, cleanup):
        super().__init__("partial-launch:" + type(cause).__name__ + ":" + str(cause))
        self.cleanup = cleanup


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
        self.transient_fds = set()
        self.fd_errors = []
        self.transient_sockets = []
        self.stop_result = None
        self.deadline = deadline
        self.last_sequence = -1
        self.gate = ReleaseGate(*self.key, time.monotonic(), deadline)
        self.directory = directory
        subreaper()
        verify_initial_namespaces(namespaces)
        try:
            self._start(argv, env, stdin_fd, guard, outer, monitor_behavior)
        except BaseException as exc:
            cleanup = self.stop("partial-launch")
            raise PartialLaunch(exc, cleanup) from exc

    def _start(self, argv, env, stdin_fd, guard, outer, monitor_behavior):
        gate_r, gate_w = os.pipe2(os.O_CLOEXEC)
        self.transient_fds.update((gate_r, gate_w))
        self.exec_gate = gate_w
        out = os.open(self.directory + "/workload.out", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.transient_fds.add(out)
        err = os.open(self.directory + "/workload.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.transient_fds.add(err)
        parent = os.getpid()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(gate_w)
                child_exec(gate_r, parent, argv, env, stdin_fd, out, err)
            except BaseException:
                os._exit(126)
        self.children.add(pid)
        self.close_fd(gate_r)
        self.close_fd(out)
        self.close_fd(err)
        self.root = promote_exec(pin_child(pid, self.namespaces), argv[0])
        self.items[pid] = self.root
        parent_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.transient_sockets.extend((parent_socket, child_socket))
        self.channel = parent_socket
        monitor_r, monitor_w = os.pipe2(os.O_CLOEXEC)
        self.transient_fds.update((monitor_r, monitor_w))
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
        self.close_fd(monitor_r)
        child_socket.close()
        self.transient_sockets.remove(child_socket)
        try:
            self.monitor = pin_child(pid, self.namespaces)
            self.items[pid] = self.monitor
            os.write(monitor_w, b"M")
        finally:
            self.close_fd(monitor_w)
        self.gate.watchdog = True  # Parent-owned pidfds and wait ownership now proven.

    def close_fd(self, fd):
        self.transient_fds.discard(fd)  # Invalidate before close; no reused-FD retry.
        if fd == self.exec_gate:
            self.exec_gate = None
        try:
            os.close(fd)
        except OSError as exc:
            self.fd_errors.append(type(exc).__name__)
            raise

    def poll(self):
        now = time.monotonic()
        terminal = False
        # Drain preceding guard failures before accepting terminal process exit.
        for _ in range(32):
            if self.channel not in select.select([self.channel], [], [], 0)[0]:
                break
            message = receive(self.channel)
            if message.get("kind") == "EXITED" and tuple(message.get("key", ())) == self.key:
                terminal = True
                break
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
                self.close_fd(self.exec_gate)
        if not self.root.alive():
            return self.reap(self.root.identity.pid)
        if terminal:
            raise CoverageError("false-terminal-monitor-observation")
        self.gate.check(now)
        if not self.monitor.alive():
            raise CoverageError("monitor-exited")
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

    def adopt(self):
        # This dedicated host guardian creates only this Job's children. Once
        # their parent exits, subreaper adoption is proven by waitid in pin_child.
        own = OwnedProcess.pin(os.getpid(), expected=identity(os.getpid(), self.namespaces),
                               namespace_receipt=self.namespaces)
        try:
            for pid in own.children():
                if pid not in self.items and pid not in self.reaped:
                    if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                        continue  # Exited child will be genuinely waitpid-reaped below.
                    if len(self.items) >= 16:
                        raise CoverageError("owned-adopted-process-cap")
                    item = pin_child(pid, self.namespaces)
                    self.items[pid] = item
        finally:
            own.close()

    def stop(self, reason, *, grace=15):
        if self.stop_result is not None:
            return self.stop_result
        errors = list(self.fd_errors)
        for fd in list(self.transient_fds):
            try:
                self.close_fd(fd)
            except OSError as exc:
                errors.append(type(exc).__name__)
        def live(item):
            try:
                return item.alive()
            except BaseException as exc:
                errors.append(type(exc).__name__)
                return True
        try:
            self.adopt()
        except BaseException as exc:
            errors.append(type(exc).__name__)
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
        end = min(self.deadline - 1, time.monotonic() + max(0, min(grace, 14)))
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
                self.adopt()
                for item in self.items.values():
                    if live(item):
                        item.send(signal.SIGKILL)
            except BaseException as exc:
                errors.append(type(exc).__name__)
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            except OSError as exc:
                errors.append(type(exc).__name__)
                break
            if pid:
                self.reaped[pid] = os.waitstatus_to_exitcode(status)
            elif all(pid in self.reaped for pid in self.children) and \
                    all(not live(item) for item in self.items.values()):
                break
            else:
                time.sleep(0.02)
        exited = [pid for pid, item in self.items.items() if not live(item)]
        result = {"reason": reason, "reaped": self.reaped.copy(), "errors": errors,
                  "exited_descendants_not_reaped_here": [pid for pid in exited if pid not in self.reaped],
                  "complete": not errors and all(pid in self.reaped for pid in self.children) and
                  len(exited) == len(self.items)}
        for item in self.items.values():
            try:
                item.close()
            except OSError:
                result["complete"] = False
                errors.append("owned-descriptor-close-failed")
        for sock in self.transient_sockets:
            try:
                sock.close()
            except OSError:
                result["complete"] = False
                errors.append("owned-socket-close-failed")
        self.transient_sockets.clear()
        self.stop_result = result
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
