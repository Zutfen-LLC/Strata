"""Focused synthetic regression only. Authored, NOT RUN by the author.

Direct reviewed entry; no proc access, process launch, signal or socket action
is permitted in fixtures. Import is inert. Live experiment authority is separate.
"""
from contextlib import contextmanager
from pathlib import Path
import errno
import os
import resource
import signal
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import agent
from resource_guard import CoverageError
from resource_telemetry import Observation

HEADER = "sl local_address rem_address st queues timer retrans uid timeout inode\n"
ROW = "0: 0100007F:3039 00000000:0000 0A 0:0 00:0 0 1000 0 123 1\n"


@contextmanager
def snapshot(links, table=HEADER + ROW):
    process = Mock(proc_fd=17)
    process.read.return_value = Observation(table, "ok", "")
    def target(name, **kwargs):
        value = links[name]
        if isinstance(value, BaseException):
            raise value
        return value
    with patch("agent.os.open", return_value=50), patch("agent.os.close") as close, \
            patch("agent.os.scandir") as entries, patch("agent.os.readlink", side_effect=target):
        entries.return_value.__enter__.return_value = iter(SimpleNamespace(name=name) for name in links)
        yield process, entries, close


class ListenerTests(unittest.TestCase):
    def test_closed_unrelated_fd_preserves_verified_listener(self):
        with snapshot({"3": FileNotFoundError(errno.ENOENT, "closed", "3"), "4": "socket:[123]"}) as (process, _, close):
            self.assertTrue(agent.owned_listener(process, 12345))
            self.assertGreaterEqual(process.verify.call_count, 4)
        close.assert_called_once_with(50)

    def test_vanished_socket_is_not_positive_ownership(self):
        with snapshot({"3": FileNotFoundError(errno.ENOENT, "closed", "3")}) as (process, _, close):
            self.assertFalse(agent.owned_listener(process, 12345))
            self.assertGreaterEqual(process.verify.call_count, 3)
        close.assert_called_once_with(50)

    def test_whole_process_loss_is_not_ignored(self):
        with snapshot({"3": FileNotFoundError(errno.ENOENT, "closed", "3")}) as (process, _, close):
            process.verify.side_effect = [None, CoverageError("owned-process-exited")]
            with self.assertRaises(CoverageError):
                agent.owned_listener(process, 12345)
            process.read.assert_not_called()
        close.assert_called_once_with(50)

    def test_permission_denied_stays_fail_closed(self):
        with snapshot({"3": PermissionError(errno.EACCES, "denied", "3")}) as (process, _, close):
            with self.assertRaises(PermissionError):
                agent.owned_listener(process, 12345)
            process.read.assert_not_called()
        close.assert_called_once_with(50)

    def test_other_errno_stays_fail_closed(self):
        for error in (errno.EBADF, errno.EIO, errno.ENOTDIR):
            with self.subTest(errno=error), snapshot({"3": OSError(error, "fixture")}) as (process, _, close):
                with self.assertRaises(OSError):
                    agent.owned_listener(process, 12345)
                process.read.assert_not_called()
            close.assert_called_once_with(50)

    def test_table_failure_and_malformed_row_stay_fail_closed(self):
        for value in (Observation(None, "unavailable", "EACCES"),
                      Observation(HEADER + "malformed\n", "ok", "")):
            with self.subTest(value=value), snapshot({"3": "socket:[123]"}) as (process, _, close):
                process.read.return_value = value
                with self.assertRaises(CoverageError):
                    agent.owned_listener(process, 12345)
            close.assert_called_once_with(50)

    def test_enumeration_failure_and_bound_stay_fail_closed(self):
        with snapshot({"3": "socket:[123]"}) as (process, entries, close):
            entries.side_effect = FileNotFoundError(errno.ENOENT, "process-gone")
            with self.assertRaises(FileNotFoundError):
                agent.owned_listener(process, 12345)
        close.assert_called_once_with(50)
        with snapshot({str(i): "socket:[123]" for i in range(257)}) as (process, _, close):
            with self.assertRaisesRegex(CoverageError, "owned-fd-cap"):
                agent.owned_listener(process, 12345)
        close.assert_called_once_with(50)

    def test_loopback_port_and_inode_must_match_exactly(self):
        for row in (ROW.replace("0100007F", "00000000"),
                    ROW.replace("3039", "303A"), ROW.replace("123 1", "124 1"),
                    ROW.replace(" 0A ", " 01 ")):
            with self.subTest(row=row), snapshot({"3": "socket:[123]"}, HEADER + row) as (process, _, close):
                self.assertFalse(agent.owned_listener(process, 12345))
            close.assert_called_once_with(50)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1048576, 128 * 1048576))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    def expired(*_):
        # unittest deliberately propagates KeyboardInterrupt instead of catching
        # it as one failed case and continuing after a one-shot alarm.
        raise KeyboardInterrupt("focused-fixture-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ListenerTests)
    forbidden = AssertionError("live operation forbidden in focused fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.readlink", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("os.kill", side_effect=forbidden), patch("os.pidfd_open", side_effect=forbidden), \
            patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 8:
        raise SystemExit(1)
    print("OWNED_LISTENER_FIXTURES_PASS_V1 tests=8 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
