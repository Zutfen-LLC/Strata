"""GPU-RESERVE-WARNING-V1 focused mocked fixtures; authored NOT RUN here.

No real management query, proc read, product, process, socket or external signal.
"""
from pathlib import Path
import resource
import signal
import sys
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import controller
import runtime
from outer import OuterGuards
from policy import GIB, MIB
from resource_guard import CoverageError, GuardStop


def local():
    return OuterGuards("ROLE_ENGINE_A", "/generic/tool", ("primary", "helper"),
                       selection="primary,helper")


def monitored(outer, count, *, collect_error=None, guard_error=None):
    root, journal, host_guard = Mock(), Mock(), Mock()
    if collect_error is None and guard_error is None and count:
        root.alive.side_effect = [True] * count + [False]
    else:
        root.alive.return_value = True
    host_guard.check.side_effect = guard_error
    with patch("runtime.OwnedTree"), patch("runtime.Journal", return_value=journal), \
            patch("runtime.DiagnosticBaselines"), patch("runtime.collect", side_effect=collect_error), \
            patch("runtime.packet") as packet, patch("runtime.time.monotonic", return_value=0), \
            patch("runtime.time.sleep"):
        runtime.monitor_child(root, {}, host_guard, outer, "B0", "/generic/journal", Mock(),
                              ("run", "ROLE_ENGINE_A", "B0"))
    return journal, packet, host_guard


class WarningTests(unittest.TestCase):
    def test_local_runtime_crossings_continue_with_minima_and_entry_counts(self):
        guard = local()
        samples = ([1152 * MIB, 512 * MIB], [1114 * MIB, 432 * MIB],
                   [1100 * MIB, 430 * MIB], [1152 * MIB, 512 * MIB], [0, 0])
        observations = [guard.check({"gpu_free_bytes": list(free)}) for free in samples]
        self.assertEqual(observations[0]["crossings"], [0, 0])
        self.assertEqual(observations[1]["first_warning"], [True, True])
        self.assertEqual(observations[2]["first_warning"], [False, False])
        self.assertEqual(observations[-1]["crossings"], [2, 2])
        self.assertEqual(observations[-1]["minimum_sampled_free_bytes"], [0, 0])
        self.assertEqual(observations[-1]["first_warning"], [False, False])
        self.assertEqual(observations[-1]["policy"], "GPU-RESERVE-WARNING-V1")
        # A new, strictly admitted phase starts independent diagnostics only.
        guard.check({"gpu_free_bytes": [1200 * MIB, 600 * MIB]}, admission=True)
        self.assertEqual(guard.check({"gpu_free_bytes": [1114 * MIB, 432 * MIB]})["crossings"], [1, 1])

    def test_startup_admission_boundaries_are_not_relaxed(self):
        guard = local()
        self.assertIsNone(guard.check({"gpu_free_bytes": [1152 * MIB, 512 * MIB]}, admission=True))
        for free in ([1152 * MIB - 1, 512 * MIB], [1152 * MIB, 512 * MIB - 1]):
            with self.subTest(free=free), self.assertRaisesRegex(GuardStop, "gpu-reserve-floor"):
                guard.check({"gpu_free_bytes": free}, admission=True)

    def test_monitor_records_diagnostics_without_warning_spam_or_refusal(self):
        guard = local()
        samples = ([1200, 600], [1114, 600], [1110, 432], [1200, 600], [1114, 432])
        guard.sample = Mock(side_effect=[{"gpu_free_bytes": [v * MIB for v in pair]} for pair in samples])
        journal, packet, host_guard = monitored(guard, len(samples))
        diagnostics = [c.kwargs for c in journal.emit.call_args_list if c.args[0] == "LOCAL_GPU_RESERVE_DIAGNOSTIC_V1"]
        warnings = [c.kwargs for c in journal.emit.call_args_list if c.args[0] == "GPU_RESERVE_WARNING_V1"]
        self.assertEqual(len(diagnostics), len(samples))
        self.assertEqual(len(warnings), 2)
        self.assertEqual(diagnostics[-1]["minimum_sampled_free_bytes"], [1110 * MIB, 432 * MIB])
        self.assertEqual(diagnostics[-1]["crossings"], [2, 2])
        self.assertEqual(host_guard.check.call_count, len(samples))
        self.assertEqual([c.args[1]["kind"] for c in packet.call_args_list],
                         ["COVERED"] * len(samples) + ["EXITED"])

    def test_initial_pre_exec_monitor_sample_keeps_strict_admission(self):
        for free in ([1114 * MIB, 600 * MIB], [1200 * MIB, 432 * MIB]):
            with self.subTest(free=free):
                guard = local()
                guard.sample = Mock(return_value={"gpu_free_bytes": free})
                journal, packet, _ = monitored(guard, 0)
                self.assertEqual([c.args[1]["kind"] for c in packet.call_args_list], ["REFUSED"])
                self.assertIn("gpu-reserve-floor", journal.emit.call_args.kwargs["reason"])

    def test_missing_or_malformed_gpu_observations_still_refuse(self):
        for values in ({}, {"gpu_free_bytes": None}, {"gpu_free_bytes": [1]},
                       {"gpu_free_bytes": [-1, 0]}, {"gpu_free_bytes": [True, 0]},
                       {"gpu_free_bytes": [0.0, 0]}):
            with self.subTest(values=values), self.assertRaises(CoverageError):
                local().check(values)

    def test_required_identity_and_host_guard_failures_refuse_coverage(self):
        with self.assertRaises(CoverageError):
            OuterGuards("ROLE_ENGINE_A", device_ids=("primary", "helper"), selection="helper,primary")
        for failure in ("gpu-management-identity", "gpu-management-failed"):
            with self.subTest(failure=failure):
                guard = local()
                guard.sample = Mock(side_effect=CoverageError(failure))
                journal, packet, _ = monitored(guard, 0)
                self.assertEqual(packet.call_args.args[1]["kind"], "REFUSED")
                self.assertIn(failure, journal.emit.call_args.kwargs["reason"])
        for error in (CoverageError("owned-pid-reused"), GuardStop("role-host-headroom")):
            with self.subTest(error=error):
                guard = local()
                guard.sample = Mock(return_value={"gpu_free_bytes": [1114 * MIB, 432 * MIB]})
                journal, packet, _ = monitored(guard, 0,
                    collect_error=error if isinstance(error, CoverageError) and not isinstance(error, GuardStop) else None,
                    guard_error=error if isinstance(error, GuardStop) else None)
                self.assertEqual(packet.call_args.args[1]["kind"], "REFUSED")
                self.assertIn(str(error), journal.emit.call_args.kwargs["reason"])

    def test_remote_reserve_disk_and_fixed_plan_fit_still_stop(self):
        guard = OuterGuards("ROLE_GPU_B", device_ids=("remote",), selection="remote")
        guard.check({"gpu_free_bytes": [4 * GIB], "disk_free_bytes": 4 * GIB})
        for free, disk in ((4 * GIB - 1, 4 * GIB), (4 * GIB, 4 * GIB - 1)):
            with self.subTest(free=free, disk=disk), self.assertRaises(GuardStop):
                guard.check({"gpu_free_bytes": [free], "disk_free_bytes": disk})
        need = 3289 * 5222400 + 4096 * MIB + 256 * MIB + 1598720
        guard.check({"gpu_free_bytes": [need], "disk_free_bytes": 23890755584}, admission=True, phase="G0")
        with self.assertRaisesRegex(GuardStop, "fixed-gpu-plan-does-not-fit"):
            guard.check({"gpu_free_bytes": [need - 1], "disk_free_bytes": 23890755584}, admission=True, phase="G0")
        with self.assertRaisesRegex(GuardStop, "cache-filesystem-headroom"):
            guard.check({"gpu_free_bytes": [need], "disk_free_bytes": 23890755583}, admission=True, phase="G0")

    def test_cleanup_still_requires_the_original_baseline(self):
        guard = local()
        guard.baseline = [12000 * MIB, 12000 * MIB]
        guard.sample = Mock(side_effect=[{"gpu_free_bytes": [1114 * MIB, 432 * MIB]},
                                         {"gpu_free_bytes": list(guard.baseline)}])
        self.assertFalse(guard.returned())
        self.assertTrue(guard.returned())
        self.assertEqual(guard.baseline, [12000 * MIB, 12000 * MIB])

    def test_existing_product_exit_and_phase_failure_propagate(self):
        job = object.__new__(runtime.Job)
        job.channel, job.root = Mock(), Mock()
        job.root.alive.return_value = False
        job.root.identity.pid = 123
        job.children = {123}
        job.reap = Mock(return_value=17)
        with patch("runtime.select.select", return_value=([], [], [])):
            rc = job.poll()
        self.assertEqual(rc, 17)
        peers = {role: Mock(role=role, closed=False, final_seen=True)
                 for role in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B")}
        ctl = controller.Controller(peers, clock=lambda: 0, start=0)
        ctl.wait = Mock(side_effect=[{}, {"rc": rc, "cleanup": {"complete": True}}])
        with self.assertRaisesRegex(GuardStop, "phase-failed-no-dependent-launch"):
            ctl.run()
        peers["ROLE_ENGINE_A"].queue.assert_called_once_with("START", phase="B0")
        peers["ROLE_GPU_B"].queue.assert_not_called()
        for peer in peers.values():
            peer.close.assert_called_once()


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    def expired(*_):
        raise KeyboardInterrupt("focused-gpu-warning-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WarningTests)
    forbidden = AssertionError("live operation forbidden in reserve fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.readlink", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("os.kill", side_effect=forbidden), patch("os.pidfd_open", side_effect=forbidden), \
            patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 9:
        raise SystemExit(1)
    print("GPU_RESERVE_WARNING_FIXTURES_PASS_V1 tests=9 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
