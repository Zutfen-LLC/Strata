"""PROCFS-TASK-RACE-V1 mocked regression entry; authored NOT RUN here.

Exercises the real children()/OwnedTree path. No live procfs or workloads.
"""
from contextlib import contextmanager
from pathlib import Path
import resource
import signal
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import commands
import resource_telemetry as t

MIB = 1048576
MISSING = t.Observation(None, "unavailable", "ENOENT")


def process():
    item = t.OwnedProcess(t.Identity(41, 900, 1, 2), 101, 102, {})
    item.verify = Mock()
    item.alive = Mock(return_value=True)
    return item


@contextmanager
def enumeration(snapshots, observations):
    snapshots, observations = iter(snapshots), iter(observations)
    scan_fds = []
    def open_directory(path, flags, *, dir_fd):
        if (path, dir_fd) == ("task", 101):
            return 201
        if (path, dir_fd) != (".", 201):
            raise AssertionError("unexpected directory open")
        fd = 202 + len(scan_fds)
        scan_fds.append(fd)
        return fd
    def scan(fd):
        if not scan_fds or fd != scan_fds[-1]:
            raise AssertionError("fresh directory offset required")
        result = Mock()
        result.__enter__ = Mock(return_value=iter(SimpleNamespace(name=n) for n in next(snapshots)))
        result.__exit__ = Mock(return_value=False)
        return result
    def read(path, fd):
        expected, value = next(observations)
        if (path, fd) != (expected, 201):
            raise AssertionError("unexpected task read: " + repr((path, fd, expected)))
        return value
    with patch.object(t.os, "open", side_effect=open_directory) as opened, \
            patch.object(t.os, "scandir", side_effect=scan) as scanned, \
            patch.object(t.os, "close") as closed, \
            patch.object(t, "read_text", side_effect=read) as reads:
        yield reads, scanned
    assert [(c.args[0], c.kwargs["dir_fd"]) for c in opened.call_args_list] == \
        [("task", 101)] + [(".", 201)] * len(scan_fds)
    assert all(c.args[1] == t.os.O_RDONLY | t.os.O_DIRECTORY | t.os.O_CLOEXEC
               for c in opened.call_args_list)
    assert [c.args[0] for c in closed.call_args_list] == scan_fds + [201]


class ProcfsRaceTests(unittest.TestCase):
    def test_disappearance_rescans_prior_task_and_discovers_descendants(self):
        root = process()
        child, grandchild = Mock(), Mock()
        child.identity = t.Identity(51, 901, 1, 3)
        grandchild.identity = t.Identity(61, 902, 1, 4)
        child.children.return_value, grandchild.children.return_value = [61], []
        child.alive.return_value = grandchild.alive.return_value = True
        # Task 42 dies after 41 was read. Its child reparents to 41, so
        # merely skipping the failed read would silently lose child 51.
        snapshots = [("41", "42"), ("41",)] + [("41",)] * 4
        values = [("41/children", t.ok("")), ("42/children", MISSING),
                  ("41/children", t.ok("51")), ("41/children", t.ok("51"))]
        tree = t.OwnedTree(root)
        with enumeration(snapshots, values) as (reads, _), \
                patch.object(t, "verify_initial_namespaces"), \
                patch.object(t.OwnedProcess, "pin", side_effect=[child, grandchild]) as pin:
            self.assertEqual(tree.refresh(), [root, child, grandchild])
        self.assertEqual([c.args[0] for c in reads.call_args_list].count("41/children"), 3)
        self.assertEqual([(c.args[0], c.kwargs["parent"]) for c in pin.call_args_list],
                         [(51, root), (61, child)])
        self.assertGreaterEqual(root.verify.call_count, 3)

    def test_successful_reads_survive_task_exit_and_reparenting(self):
        snapshots = [("41", "42"), ("41",)] + [("41",)] * 4
        values = [("41/children", t.ok("")), ("42/children", t.ok("51")),
                  ("41/children", t.ok("52")), ("41/children", t.ok("52"))]
        with enumeration(snapshots, values):
            self.assertEqual(process().children(), [51, 52])

    def test_new_tasks_and_changed_child_mapping_need_reconciliation(self):
        cases = [
            ([("41",), ("41", "42")] + [("41", "42")] * 4,
             [("41/children", t.ok(""))] +
             [("41/children", t.ok("")), ("42/children", t.ok("51"))] * 2),
            ([("41", "42")] * 6,
             [("41/children", t.ok("")), ("42/children", t.ok("51")),
              ("41/children", t.ok("51")), ("42/children", t.ok("")),
              ("41/children", t.ok("51")), ("42/children", t.ok(""))]),
        ]
        for snapshots, values in cases:
            with self.subTest(snapshots=snapshots), enumeration(snapshots, values):
                self.assertEqual(process().children(), [51])

    def test_still_present_task_missing_children_is_fatal(self):
        with enumeration([("41",)] * 2, [("41/children", MISSING)]), \
                self.assertRaisesRegex(t.CoverageError, "stable-required-entry-absence") as error:
            process().children()
        self.assertIn("/proc/41/task/41/children", str(error.exception))
        self.assertIn("start_ticks=900", str(error.exception))
        self.assertIn("ENOENT", str(error.exception))

    def test_permissions_and_malformed_data_are_never_skipped(self):
        for value in (t.Observation(None, "unavailable", "EACCES"),
                      t.Observation(None, "unavailable", "ESRCH"), t.malformed("encoding"),
                      t.ok("oops"), t.ok("1"), t.ok("-51"), t.ok("5" * 5000)):
            with self.subTest(value=value), enumeration([("41",)], [("41/children", value)]), \
                    self.assertRaises(t.CoverageError) as error:
                process().children()
            self.assertIn("/proc/41/task/41/children", str(error.exception))
            self.assertIn("start_ticks=900", str(error.exception))
        with patch.object(t.os, "open", side_effect=PermissionError(13, "denied")), \
                self.assertRaisesRegex(t.CoverageError, "EACCES.*denied"):
            process().children()

    def test_process_loss_or_identity_reuse_during_race_is_fatal(self):
        for reason in ("owned-process-exited", "owned-process-identity-changed",
                       "owned-process-executable-changed"):
            root = process()
            root.verify.side_effect = [None, t.CoverageError(reason)]
            with self.subTest(reason=reason), \
                    enumeration([("41", "42")],
                                [("41/children", t.ok("")), ("42/children", MISSING)]), \
                    self.assertRaisesRegex(t.CoverageError, reason):
                root.children()

    def test_unresolved_churn_is_bounded_and_fails_coverage(self):
        snapshots, values = [], []
        for index in range(t.MAX_CHILD_SCANS):
            task = str(42 + index)
            snapshots.extend([("41", task), ("41",)])
            values.extend([("41/children", t.ok("")), (task + "/children", MISSING)])
        with enumeration(snapshots, values) as (reads, scanned), \
                self.assertRaisesRegex(t.CoverageError, "unresolved-task-children-coverage"):
            process().children()
        self.assertEqual(reads.call_count, t.MAX_CHILD_SCANS * 2)
        self.assertEqual(scanned.call_count, t.MAX_CHILD_SCANS * 2)

    def test_required_task_directory_schema_and_caps_still_fail(self):
        for tasks in ((), ("oops",), ("0",), ("42",), tuple(map(str, range(1, t.MAX_TASKS + 2)))):
            with self.subTest(tasks=tasks), enumeration([tasks], []), \
                    self.assertRaises(t.CoverageError):
                process().children()
        too_many = " ".join(map(str, range(51, 52 + t.MAX_PROCESSES)))
        with enumeration([("41",)], [("41/children", t.ok(too_many))]), \
                self.assertRaisesRegex(t.CoverageError, "children-cap"):
            process().children()
        with patch.object(t.os, "open", side_effect=FileNotFoundError(2, "task missing")), \
                self.assertRaisesRegex(t.CoverageError, "ENOENT.*task missing"):
            process().children()

    def test_phase_specific_streams_preserve_comparison_and_usage_bounds(self):
        self.assertEqual(commands.requests("B0"), b"GEN 64 9419\nQUIT\n")
        full = b"GEN 64 9419\nGEN 64 14773,6353\nGEN 64 9419\nQUIT\n"
        class Bindings(dict):
            def __missing__(self, key):
                return "/generic/" + key
        self.assertEqual(commands.requests("G0"), commands.requests("B0"))
        for phase in ("B1", "G1"):
            self.assertEqual(commands.requests(phase), full)
        self.assertEqual(sum(commands.requests(p).count(b"GEN 64 ")
                             for p in ("B0", "B1", "G0", "G1")), 8)
        self.assertLessEqual(2 + 8, 11)
        self.assertEqual(128 + 8 * 64, 640)
        for phase, cap in (("B0", 1200), ("B1", 600), ("G0", 1800), ("G1", 900)):
            _, env, actual_cap = commands.product("ROLE_ENGINE_A", phase, Bindings())
            self.assertEqual(env["STRATA_DUMP_FIRST_LOGITS"], "/generic/RUN_A/" + phase + ".first-logits")
            self.assertEqual(actual_cap, cap)
        with self.assertRaises(t.CoverageError):
            commands.requests("CPU")


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    def expired(*_):
        raise KeyboardInterrupt("focused-procfs-race-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProcfsRaceTests)
    forbidden = AssertionError("live operation forbidden in procfs race fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.readlink", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("os.kill", side_effect=forbidden), patch("os.pidfd_open", side_effect=forbidden), \
            patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 9:
        raise SystemExit(1)
    print("PROCFS_TASK_RACE_FIXTURES_PASS_V1 tests=9 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
