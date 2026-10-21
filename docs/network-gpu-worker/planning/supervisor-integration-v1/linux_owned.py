"""Linux owned-child primitives; no operation occurs at import."""
import ctypes
import errno
import os
import signal

from resource_telemetry import (CoverageError, Identity, OwnedProcess, parse_stat,
                                read_text, verify_initial_namespaces)


def prctl(option, value):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "required-owned-process-prctl")


def parent_death(expected_parent):
    prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG, local child only.
    if os.getppid() != expected_parent:
        os._exit(125)


def subreaper():
    prctl(36, 1)  # PR_SET_CHILD_SUBREAPER; only this owned watchdog.


def identity(pid, namespaces):
    verify_initial_namespaces(namespaces)
    fd = os.open("/proc/" + str(pid), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        actual, _, ticks = parse_stat(read_text("stat", fd))
        exe = os.stat("exe", dir_fd=fd)
        if actual != pid:
            raise CoverageError("pid-view-mismatch")
        return Identity(pid, ticks, exe.st_dev, exe.st_ino)
    finally:
        os.close(fd)


def pin_child(pid, namespaces):
    # Caller MUST be the unreaping parent; waitid proves that relationship.
    os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    return OwnedProcess.pin(pid, expected=identity(pid, namespaces), namespace_receipt=namespaces)


class ExecProcess(OwnedProcess):
    """One explicit transition from blocked Python child to pinned target inode."""
    def allow_exec(self, executable):
        st = os.stat(executable)
        self.target_exe = (st.st_dev, st.st_ino)
        self.transitioned = False

    def verify(self, executable=True):
        super().verify(executable=False)
        if not executable:
            return
        try:
            st = os.stat("exe", dir_fd=self.proc_fd)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESRCH):
                self.reconcile_missing("exe")
            raise
        actual = (st.st_dev, st.st_ino)
        before = (self.identity.exe_device, self.identity.exe_inode)
        if actual == self.target_exe:
            self.identity = Identity(self.identity.pid, self.identity.start_ticks, *actual)
            self.transitioned = True
        elif self.transitioned or actual != before:
            raise CoverageError("unexpected-exec-in-owned-child")


def promote_exec(item, executable):
    item.__class__ = ExecProcess
    item.allow_exec(executable)
    return item
