"""Direct fork-child ownership for the runner; NO namespace/cgroup prerequisite.

Only reviewed interpreter stages are accepted by the caller. No group/numeric
PID signalling, no subreaper setup, no adoption of existing processes.
"""
import os
import select
import signal
import time

from resource_telemetry import parse_stat, read_text
from linux_owned import parent_death


class OwnershipError(ValueError):
    pass


class Child:
    def __init__(self, argv, executable_identity):
        self.pid = self.pidfd = self.procfd = self.gate = None
        self.fds, self.streams = set(), {}
        self.reaped = False
        self.rc = None
        self.identity = None
        self.executable_identity = executable_identity
        self.started = time.monotonic()
        self.cleanup = None
        try:
            gate_r, gate_w = os.pipe2(os.O_CLOEXEC)
            self.fds.update((gate_r, gate_w))
            self.gate = gate_w
            out_r, out_w = os.pipe2(os.O_CLOEXEC)
            self.fds.update((out_r, out_w))
            err_r, err_w = os.pipe2(os.O_CLOEXEC)
            self.fds.update((err_r, err_w))
            parent = os.getpid()
            self.pid = os.fork()
            if self.pid == 0:
                try:
                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    signal.signal(signal.SIGINT, signal.SIG_DFL)
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    parent_death(parent)
                    os.close(gate_w)
                    if os.read(gate_r, 1) != b"G":
                        os._exit(125)
                    null = os.open(os.devnull, os.O_RDONLY)
                    os.dup2(null, 0)
                    os.dup2(out_w, 1)
                    os.dup2(err_w, 2)
                    os.execve(argv[0], argv, {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"})
                except BaseException:
                    os._exit(126)
            self.close_fd(gate_r)
            self.close_fd(out_w)
            self.close_fd(err_w)
            # We are the unreaping direct parent. A recycled foreign PID cannot
            # replace this child before our own wait. Namespace trust is not used.
            os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            self.pidfd = os.pidfd_open(self.pid, 0)
            self.procfd = os.open("/proc/" + str(self.pid), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            actual, ppid, ticks = parse_stat(read_text("stat", self.procfd))
            if actual != self.pid or ppid != parent or os.fstat(self.procfd).st_uid != os.getuid():
                raise OwnershipError("direct-child-identity")
            self.identity = (actual, ticks)
            self.verify()
            self.streams = {out_r: "stdout", err_r: "stderr"}
            for fd in self.streams:
                os.set_blocking(fd, False)
            if os.write(self.gate, b"G") != 1:
                raise OwnershipError("short-exec-release")
            self.close_fd(self.gate)
        except BaseException as exc:
            result = self.stop(time.monotonic() + 5)
            error = OwnershipError("partial-child-launch:" + type(exc).__name__)
            error.cleanup = result
            raise error from exc

    def close_fd(self, fd):
        self.fds.discard(fd)
        if fd == self.gate:
            self.gate = None
        os.close(fd)

    def alive(self):
        if self.pidfd is None:
            raise OwnershipError("missing-pidfd")
        return not select.select([self.pidfd], [], [], 0)[0]

    def verify(self):
        actual, parent, ticks = parse_stat(read_text("stat", self.procfd))
        exe = os.stat("exe", dir_fd=self.procfd)
        if (actual, ticks) != self.identity or parent != os.getpid() or \
                (exe.st_dev, exe.st_ino) != self.executable_identity:
            raise OwnershipError("child-pid-start-parent-or-executable-changed")

    def poll(self):
        if self.reaped:
            return self.rc
        if self.pidfd is not None and self.alive():
            try:
                self.verify()
            except OwnershipError:
                raise  # A completed live identity mismatch is never terminal success.
            except (OSError, ValueError):
                if not self.alive():
                    return self.harvest()
                raise
            return None
        return self.harvest()

    def harvest(self):
        if self.reaped:
            return self.rc
        got, status = os.waitpid(self.pid, os.WNOHANG)
        if got:
            self.reaped, self.rc = True, os.waitstatus_to_exitcode(status)
        return self.rc

    def stop(self, deadline):
        if self.cleanup is not None:
            return self.cleanup
        errors = []
        if self.gate is not None:
            try:
                self.close_fd(self.gate)  # Unreleased child gets EOF, never exec.
            except OSError as exc:
                errors.append(type(exc).__name__)
        def send(sig):
            if self.pidfd is not None:
                try:
                    signal.pidfd_send_signal(self.pidfd, sig, None, 0)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    errors.append(type(exc).__name__)
        try:
            if self.pid is not None and self.pid > 0 and not self.reaped:
                send(signal.SIGTERM)
                term_end = min(deadline - 3, time.monotonic() + 2)
                try:
                    while time.monotonic() < term_end and self.harvest() is None:
                        time.sleep(.02)
                except (OSError, ValueError) as exc:
                    errors.append(type(exc).__name__)
                if not self.reaped:
                    send(signal.SIGKILL)
                while time.monotonic() < deadline and not self.reaped:
                    self.harvest()
                    if not self.reaped:
                        time.sleep(.02)
        except (OSError, ValueError) as exc:
            errors.append(type(exc).__name__ + ":" + str(exc))
        finally:
            for fd in list(self.fds):
                try:
                    self.close_fd(fd)
                except OSError as exc:
                    errors.append(type(exc).__name__)
            for name in ("procfd", "pidfd"):
                fd = getattr(self, name)
                setattr(self, name, None)
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError as exc:
                        errors.append(type(exc).__name__)
        self.cleanup = {"pid": self.pid, "identity": self.identity, "reaped": self.reaped,
                        "rc": self.rc, "errors": errors,
                        "complete": not errors and (self.pid is None or self.reaped)}
        return self.cleanup
