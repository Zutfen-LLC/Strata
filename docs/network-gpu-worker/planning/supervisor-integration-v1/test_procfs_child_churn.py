"""PROCFS-CHILD-CHURN-V1 focused mocked entry; authored NOT RUN here."""
from contextlib import contextmanager
from pathlib import Path
import resource
import signal
import sys
from unittest import TestCase, TextTestRunner, defaultTestLoader
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import commands
import resource_guard as g
import resource_telemetry as t
import runtime
from test_procfs_task_race import enumeration, process

MIB = 1048576


def stat(pid=51, ppid=41, start=901):
    return t.ok(str(pid) + " (fixture) " + " ".join(["S", str(ppid)] + ["0"] * 17 + [str(start)]))


@contextmanager
def pin_fixture(observation=None, exe_error=None):
    with patch.object(t, "verify_initial_namespaces"), \
            patch.object(t.os, "pidfd_open", return_value=301), \
            patch.object(t.os, "open", return_value=302), \
            patch.object(t.os, "close") as closed, \
            patch.object(t, "read_text", return_value=stat() if observation is None else observation) as read, \
            patch.object(t.os, "stat", return_value=Mock(st_dev=1, st_ino=3), side_effect=exe_error), \
            patch.object(t.select, "select", return_value=([], [], [])) as selected:
        yield closed, read, selected


class ChurnTests(TestCase):
    def test_continuously_changing_child_maps_do_not_require_quiescence(self):
        values = [("41/children", t.ok("51")), ("41/children", t.ok("52"))]
        with enumeration([("41",)] * 4, values) as (read, scanned):
            self.assertEqual(process().children(), [51, 52])
        self.assertEqual(read.call_count, 2)
        self.assertEqual(scanned.call_count, 4)
        # The prior two-identical-map rule would keep scanning and refuse churn.

    def test_surviving_children_and_grandchildren_are_discovered(self):
        root, child, grandchild = process(), process(), process()
        child.identity, grandchild.identity = t.Identity(52, 902, 1, 3), t.Identity(62, 903, 1, 4)
        root.children, child.children, grandchild.children = \
            Mock(return_value=[51, 52]), Mock(return_value=[62]), Mock(return_value=[])
        tree = t.OwnedTree(root)
        with patch.object(t, "verify_initial_namespaces"), \
                patch.object(t.OwnedProcess, "pin", side_effect=[None, child, grandchild]) as pin:
            self.assertEqual(tree.refresh(), [root, child, grandchild])
        self.assertEqual([c.args[0] for c in pin.call_args_list], [51, 52, 62])
        self.assertIs(pin.call_args_list[-1].kwargs["parent"], child)

    def test_child_gone_before_pidfd_open_creates_no_identity_or_signal(self):
        root = process()
        with patch.object(t, "verify_initial_namespaces"), \
                patch.object(t.os, "pidfd_open", side_effect=ProcessLookupError(3, "gone")), \
                patch.object(t.os, "open") as opened:
            self.assertIsNone(t.OwnedProcess.pin(51, parent=root))
        opened.assert_not_called()
        root.verify.assert_called_once()

    def test_confirmed_exit_during_pin_closes_both_handles(self):
        root = process()
        with pin_fixture(exe_error=FileNotFoundError(2, "gone", "exe")) as (closed, _, selected):
            selected.side_effect = [([], [], []), ([301], [], [])]
            self.assertIsNone(t.OwnedProcess.pin(51, parent=root))
        self.assertEqual([c.args[0] for c in closed.call_args_list], [302, 301])
        with pin_fixture() as (closed, read, selected):
            selected.return_value = ([301], [], [])
            self.assertIsNone(t.OwnedProcess.pin(51, parent=root))
        read.assert_not_called()
        closed.assert_called_once_with(301)

    def test_live_missing_entries_permissions_schema_and_reuse_still_fail(self):
        cases = [
            (stat(), FileNotFoundError(2, "live exe missing", "exe")),
            (stat(), PermissionError(13, "denied", "exe")),
            (t.Observation(None, "unavailable", "EACCES"), None),
            (t.ok("malformed"), None), (stat(pid=99), None), (stat(ppid=99), None),
        ]
        for observation, error in cases:
            with self.subTest(observation=observation, error=error), \
                    pin_fixture(observation, error) as (closed, _, _), \
                    self.assertRaises((t.CoverageError, OSError)):
                t.OwnedProcess.pin(51, parent=process())
            self.assertEqual([c.args[0] for c in closed.call_args_list], [302, 301])
        with pin_fixture() as (closed, read, _), \
                self.assertRaisesRegex(t.CoverageError, "identity-changed"):
            read.side_effect = [stat(start=901), stat(start=902)]
            t.OwnedProcess.pin(51, parent=process())
        self.assertEqual([c.args[0] for c in closed.call_args_list], [302, 301])

    def test_exited_tracking_is_retired_before_pid_slot_reuse(self):
        root, old, new = process(), process(), process()
        old.identity, new.identity = t.Identity(51, 901, 1, 3), t.Identity(51, 999, 1, 4)
        old.alive.return_value = False
        old.close, new.close = Mock(), Mock()
        root.children, new.children = Mock(return_value=[51]), Mock(return_value=[])
        tree = t.OwnedTree(root)
        tree.items[51] = old
        with patch.object(t, "verify_initial_namespaces"), \
                patch.object(t.OwnedProcess, "pin", return_value=new) as pin:
            self.assertEqual(tree.refresh(), [root, new])
        old.close.assert_called_once()
        pin.assert_called_once_with(51, parent=root)
        self.assertEqual(tree.items[51].identity.start_ticks, 999)
        # The old exited pidfd never signals the new numeric-PID occupant.
        with patch.object(t.signal, "pidfd_send_signal") as sent:
            self.assertEqual(old.send(signal.SIGTERM), "already-exited")
        sent.assert_not_called()

    def test_collection_skips_only_confirmed_nonroot_exit(self):
        root, child = process(), process()
        tree = Mock(root=root)
        tree.refresh.return_value = [root, child]
        row = {"identity": {"pid": 41}, "memory_bytes": {}}
        with patch("runtime.host_sample", return_value={}), \
                patch("runtime.cgroup_snapshot", return_value={"membership": "/fixture"}), \
                patch("runtime.capture_process", side_effect=[row, t.ProcessExited("gone")]):
            self.assertEqual(runtime.collect(tree, {})["processes"], [row])
        for failures in ([t.ProcessExited("root gone")],
                         [row, t.CoverageError("owned-process-identity-changed")],
                         [row, PermissionError(13, "denied")]):
            with self.subTest(failures=failures), patch("runtime.host_sample", return_value={}), \
                    patch("runtime.cgroup_snapshot", return_value={"membership": "/fixture"}), \
                    patch("runtime.capture_process", side_effect=failures), \
                    self.assertRaises((t.CoverageError, OSError)):
                runtime.collect(tree, {})

    def test_namespace_exit_is_distinct_from_required_monitor_read_failure(self):
        child = process()
        child.status, child.read = Mock(return_value={}), Mock(return_value=t.ok("0::/fixture\n"))
        child.alive.return_value = False
        receipt = {kind: (1, 2) for kind in ("cgroup", "mnt", "pid")}
        with patch.object(g, "namespace_identity", side_effect=FileNotFoundError(2, "gone", "ns/cgroup")), \
                self.assertRaises(t.ProcessExited):
            g.capture_process(child, receipt, "/fixture")
        for outcomes in ([PermissionError(13, "denied", "ns/cgroup")],
                         [(1, 2), FileNotFoundError(2, "required self view", "/proc/self/ns/cgroup")]):
            with self.subTest(outcomes=outcomes), \
                    patch.object(g, "namespace_identity", side_effect=outcomes), \
                    self.assertRaises(OSError):
                g.capture_process(child, receipt, "/fixture")

    def test_exit_does_not_mask_real_read_failure(self):
        for value in (t.Observation(None, "unavailable", "EACCES"), t.malformed("encoding")):
            child = process()
            child.verify.side_effect = [None, t.ProcessExited("gone")]
            with self.subTest(value=value), patch.object(t, "read_text", return_value=value), \
                    self.assertRaises(t.CoverageError) as error:
                child.read("status")
            self.assertNotIsInstance(error.exception, t.ProcessExited)
        child = process()
        child.verify.side_effect = [None, t.ProcessExited("gone")]
        with patch.object(t, "read_text", return_value=t.Observation(None, "unavailable", "ENOENT")), \
                self.assertRaises(t.ProcessExited):
            child.read("status")

    def test_adoption_reconciles_wait_and_pin_exit_without_foreign_signals(self):
        for wait_result in (None, Mock(), ChildProcessError()):
            job, own = object.__new__(runtime.Job), process()
            job.namespaces, job.items, job.reaped = {}, {}, {}
            own.children, own.close = Mock(return_value=[51]), Mock()
            with self.subTest(wait_result=wait_result), patch("runtime.identity", return_value=own.identity), \
                    patch.object(t.OwnedProcess, "pin", side_effect=[own, None]) as pin, \
                    patch("runtime.os.waitid", return_value=wait_result if not isinstance(wait_result, Exception) else None,
                          side_effect=wait_result if isinstance(wait_result, Exception) else None):
                job.adopt()
            self.assertEqual(job.items, {})
            self.assertEqual(pin.call_count, 2 if wait_result is None else 1)
            own.close.assert_called_once()
        job, own, old, new = object.__new__(runtime.Job), process(), process(), process()
        old.alive.return_value = False
        old.close, own.close = Mock(), Mock()
        new.identity = t.Identity(51, 999, 1, 4)
        own.children = Mock(return_value=[51])
        job.namespaces, job.items, job.reaped = {}, {51: old}, {51: 0}
        with patch("runtime.identity", return_value=own.identity), \
                patch.object(t.OwnedProcess, "pin", side_effect=[own, new]), \
                patch("runtime.os.waitid", return_value=None), \
                self.assertRaisesRegex(t.CoverageError, "owned-adopted-pid-reuse"):
            job.adopt()
        self.assertIs(job.items[51], new)  # Proven owned replacement remains cleanup-tracked.
        self.assertNotIn(51, job.reaped)
        old.close.assert_called_once()

    def test_termination_remains_bounded_and_pidfd_owned(self):
        job = object.__new__(runtime.Job)
        job.stop_result, job.fd_errors, job.transient_fds = None, [], set()
        job.exec_gate, job.transient_sockets, job.deadline = None, [], 10
        job.adopt = Mock()
        job.items = {pid: Mock(identity=t.Identity(pid, pid, 1, pid)) for pid in (41, 42)}
        job.children, job.reaped = {41, 42}, {}
        for item in job.items.values():
            item.alive.return_value = True
            item.send.side_effect = lambda sig, p=item: setattr(p.alive, "return_value", sig != signal.SIGKILL)
        times = iter(i / 10 for i in range(100))
        with patch("runtime.time.monotonic", side_effect=lambda: next(times)), patch("runtime.time.sleep"), \
                patch("runtime.OwnedTree", side_effect=lambda p: Mock(items={p.identity.pid: p}, refresh=lambda: [p])), \
                patch("runtime.os.waitpid", side_effect=[(41, 0), (42, 0), ChildProcessError()]):
            result = job.stop("fixture", grace=0)
        self.assertTrue(result["complete"])
        for item in job.items.values():
            self.assertEqual([c.args[0] for c in item.send.call_args_list], [signal.SIGTERM, signal.SIGKILL])
            item.close.assert_called_once()
        self.assertIs(job.stop("again"), result)

    def test_smoke_and_full_streams_preserve_new_cumulative_ceiling(self):
        smoke, full = b"GEN 64 9419\nQUIT\n", b"GEN 64 9419\nGEN 64 14773,6353\nGEN 64 9419\nQUIT\n"
        self.assertEqual([commands.requests(p) for p in ("B0", "B1", "G0", "G1")],
                         [smoke, full, smoke, full])
        count = sum(commands.requests(p).count(b"GEN 64 ") for p in ("B0", "B1", "G0", "G1"))
        self.assertEqual(count, 8)
        self.assertEqual((2 + count, 128 + count * 64), (10, 640))
        self.assertLessEqual(128 + count * 64, 704)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    def expired(*_):
        raise KeyboardInterrupt("focused-child-churn-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = defaultTestLoader.loadTestsFromTestCase(ChurnTests)
    forbidden = AssertionError("live operation forbidden in child churn fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.stat", side_effect=forbidden), patch("os.readlink", side_effect=forbidden), \
            patch("os.fork", side_effect=forbidden), patch("os.waitid", side_effect=forbidden), \
            patch("os.waitpid", side_effect=forbidden), patch("os.kill", side_effect=forbidden), \
            patch("os.pidfd_open", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 12:
        raise SystemExit(1)
    print("PROCFS_CHILD_CHURN_FIXTURES_PASS_V1 tests=12 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
