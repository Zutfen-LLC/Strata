"""PERIODIC-OBSERVATION-V1 focused mocked entry; authored NOT RUN here."""
from pathlib import Path
import resource
import signal
import sys
from types import SimpleNamespace
from unittest import TestCase, TextTestRunner, defaultTestLoader
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import agent
import commands
import controller
import resource_telemetry as t
from policy import GuardStop, bounded_job_deadline, ReleaseGate
from test_procfs_task_race import enumeration, process, ProcfsRaceTests
from test_procfs_child_churn import ChurnTests

MIB = 1048576


class ObservationTests(TestCase):
    def test_continuous_thread_and_child_creation_never_requires_equal_sets(self):
        snapshots = [("41", "42"), ("41", "43"), ("41", "44"), ("41", "45")]
        values = [("41/children", t.ok("51")), ("42/children", t.ok("52")),
                  ("41/children", t.ok("53")), ("44/children", t.ok("54"))]
        root = process()
        with enumeration(snapshots, values) as (read, scanned):
            self.assertEqual(root.children(), [51, 52, 53, 54])
        self.assertEqual((read.call_count, scanned.call_count), (4, 4))
        self.assertEqual(t.MAX_CHILD_SCANS, 2)

    def test_complete_passes_with_different_thread_sets_match_latest_failure(self):
        snapshots = [("41", "42")] * 2 + [("41", "43")] * 2
        values = [("41/children", t.ok("51")), ("42/children", t.ok("")),
                  ("41/children", t.ok("52")), ("43/children", t.ok("53"))]
        with enumeration(snapshots, values):
            self.assertEqual(process().children(), [51, 52, 53])

    def test_disappearing_threads_reparent_children_amid_continuing_churn(self):
        snapshots = [("41", "42"), ("41", "43"), ("41", "43"), ("41", "44")]
        missing = t.Observation(None, "unavailable", "ENOENT")
        values = [("41/children", t.ok("")), ("42/children", missing),
                  ("41/children", t.ok("51")), ("43/children", missing)]
        root, survivor = process(), process()
        survivor.identity = t.Identity(51, 901, 1, 3)
        survivor.children = Mock(return_value=[])
        with enumeration(snapshots, values), patch.object(t, "verify_initial_namespaces"), \
                patch.object(t.OwnedProcess, "pin", return_value=survivor) as pin:
            self.assertEqual(t.OwnedTree(root).refresh(), [root, survivor])
        pin.assert_called_once_with(51, parent=root)

    def test_real_read_and_identity_requirements_remain_fatal(self):
        ProcfsRaceTests.test_still_present_task_missing_children_is_fatal(self)
        ProcfsRaceTests.test_permissions_and_malformed_data_are_never_skipped(self)
        ChurnTests.test_namespace_exit_is_distinct_from_required_monitor_read_failure(self)
        ChurnTests.test_exit_does_not_mask_real_read_failure(self)

    # Reuse focused protection cases, not their historical campaign entrypoints.
    test_live_pin_failures = ChurnTests.test_live_missing_entries_permissions_schema_and_reuse_still_fail
    test_safe_retired_pid_reuse = ChurnTests.test_exited_tracking_is_retired_before_pid_slot_reuse
    test_required_root_and_terminal_descendants = ChurnTests.test_collection_skips_only_confirmed_nonroot_exit
    test_adoption_lifecycle = ChurnTests.test_adoption_reconciles_wait_and_pin_exit_without_foreign_signals
    test_bounded_owned_termination = ChurnTests.test_termination_remains_bounded_and_pidfd_owned

    def test_effective_deadlines_and_fresh_leases_only_shorten_limits(self):
        self.assertEqual(bounded_job_deadline(100, 2000, 400), 400)
        self.assertEqual(bounded_job_deadline(100, 300, 400), 300)
        for limit in (100, 115):
            with self.subTest(limit=limit), self.assertRaises(GuardStop):
                bounded_job_deadline(100, 2000, limit)
        self.assertGreater(bounded_job_deadline(100, 2000, 115.01), 115)
        gate = ReleaseGate("r", "ROLE_ENGINE_A", "B1", 100, 400)
        job = SimpleNamespace(deadline=400, gate=gate)
        agent.clip_job_limit(job, 300)
        self.assertEqual((job.deadline, gate.deadline), (300, 300))
        agent.clip_job_limit(job, 500)
        self.assertEqual((job.deadline, gate.deadline), (300, 300))
        agent.clip_job_limit(job, 116)
        self.assertEqual((job.deadline, gate.deadline, gate.ready_until), (116, 116, 101))
        with self.assertRaises(GuardStop):
            gate.check(101)

    def test_controller_launches_clipped_gpu_phase_and_waits_within_reserve(self):
        peers = {r: Mock(role=r, covered=True, closed=False, final_seen=True)
                 for r in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B")}
        ctl = controller.Controller(peers, clock=lambda: 100, start=-4300)
        # Overall deadline 1100: 950 seconds of phase lifetime remain after
        # 45 global reserve + 5 challenge transit, less than G0's 1900 maximum.
        self.assertEqual(ctl.phase_cap(1900), 950)
        self.assertEqual(ctl.phase_cap(600), 600)
        ctl.gap = Mock()
        ctl.wait = Mock(return_value={"rc": 0, "cleanup": {"complete": True}, "reason": "finished"})
        self.assertTrue(ctl.run()["complete"])
        self.assertIn(("START",), [c.args for c in peers["ROLE_GPU_B"].queue.call_args_list])
        self.assertEqual([c.kwargs["phase"] for c in peers["ROLE_GPU_B"].queue.call_args_list
                          if c.args == ("START",)], ["G0", "G1"])
        waits = {c.args[2]: c.args[3] for c in ctl.wait.call_args_list
                 if c.args[:2] == ("ROLE_ENGINE_A", "DONE")}
        self.assertEqual(waits, {"B0": 950, "B1": 600, "G0": 950, "G1": 900})
        ctl.deadline = 165
        with self.assertRaises(GuardStop):
            ctl.phase_cap(1900)  # No positive interval after 45+5+15 seconds.

    def test_two_prompt_comparison_and_usage_do_not_invent_repeated_a(self):
        smoke, comparison = b"GEN 64 9419\nQUIT\n", b"GEN 64 9419\nGEN 64 14773,6353\nQUIT\n"
        streams = [commands.requests(p) for p in ("B0", "B1", "G0", "G1")]
        self.assertEqual(streams, [smoke, comparison, smoke, comparison])
        self.assertEqual([s.count(b"GEN 64 ") for s in streams], [1, 2, 1, 2])
        self.assertEqual((4 + 6, 256 + 6 * 64), (10, 640))
        self.assertLessEqual(640, 704)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    def expired(*_):
        raise KeyboardInterrupt("focused-periodic-observation-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = defaultTestLoader.loadTestsFromTestCase(ObservationTests)
    forbidden = AssertionError("live operation forbidden in periodic observation fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.stat", side_effect=forbidden), patch("os.readlink", side_effect=forbidden), \
            patch("os.fork", side_effect=forbidden), patch("os.waitid", side_effect=forbidden), \
            patch("os.waitpid", side_effect=forbidden), patch("os.kill", side_effect=forbidden), \
            patch("os.pidfd_open", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 12:
        raise SystemExit(1)
    print("PERIODIC_OBSERVATION_FIXTURES_PASS_V1 tests=12 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
