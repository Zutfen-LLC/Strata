"""GPU-COMPLETION-V1 focused mocked entry; authored NOT RUN locally."""
import copy
from pathlib import Path
import resource
import signal
import sys
from unittest import TestCase, TextTestRunner, defaultTestLoader
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import commands
import controller
import gpu_completion as c
from outer import OuterGuards
from policy import CoverageError, GuardStop, TrialOrder, bounded_job_deadline

MIB = 1048576


def baseline():
    def artifact(name):
        return {"path": "/retained/" + name, "sha256": "a" * 64}
    return {"archive_sha256": c.ARCHIVE_SHA, "implementation": c.IMPLEMENTATION,
            "plan_sha256": c.PLAN_SHA,
            "B0": {"rc": 0, "done_tokens": [64], "output": artifact("B0.out"),
                   "logits": [artifact("B0.first-logits.0")]},
            "B1": {"rc": 0, "done_tokens": [64, 64], "output": artifact("B1.out"),
                   "logits": [artifact("B1.first-logits.0"), artifact("B1.first-logits.1")]}}


def peers():
    return {role: Mock(role=role, covered=True, closed=False, final_seen=True)
            for role in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B")}


class Bindings(dict):
    def __missing__(self, key):
        return "/existing/" + key


class CompletionTests(TestCase):
    def test_controller_runs_only_gpu_phases_with_fresh_cpu_and_prior_baseline(self):
        owned = peers()
        with patch("controller.verified_file") as verified:
            ctl = controller.Controller(owned, gpu_completion=True, baseline=baseline(),
                                        clock=lambda: 0, start=0)
        self.assertEqual(verified.call_count, 5)
        ctl.gap = Mock()
        ctl.wait = Mock(return_value={"rc": 0, "cleanup": {"complete": True}, "reason": "finished"})
        result = ctl.run()
        self.assertEqual(result["completed_phases"], ["G0", "G1"])
        self.assertEqual(result["baseline_archive_sha256"], c.ARCHIVE_SHA)
        self.assertEqual((result["historical_active_s"], result["window_deadline_mono"]), (4889.5, 2700))
        for role in ("ROLE_ENGINE_A", "ROLE_GPU_B"):
            starts = [x.kwargs for x in owned[role].queue.call_args_list if x.args == ("START",)]
            self.assertEqual(starts, [{"phase": "G0", "gpu_completion": True},
                                      {"phase": "G1", "gpu_completion": True}])
        self.assertEqual([x.kwargs["phase"] for x in owned["ROLE_CPU_C"].queue.call_args_list
                          if x.args == ("START",)], ["CPU"])
        ctl.gap.assert_called_once()
        self.assertEqual(c.PHASES, ("G0", "G1"))

    def test_selective_order_never_fabricates_baseline_or_releases_on_failure(self):
        order = TrialOrder(c.PHASES)
        for phase, cpu, gpu in (("B0", True, True), ("G1", True, True),
                                ("G0", False, True), ("G0", True, False)):
            with self.subTest(phase=phase, cpu=cpu, gpu=gpu), self.assertRaises(CoverageError):
                order.permit(phase, cpu, gpu)
        order.permit("G0", True, True)
        with self.assertRaises(GuardStop):
            order.finish("G0", 17, True)
        with self.assertRaises(CoverageError):
            order.permit("G1", True, True)
        with self.assertRaises(CoverageError):
            TrialOrder(("G1",))

    def test_baseline_requires_successful_exact_streams_and_real_artifact_hashes(self):
        receipt, verify = baseline(), Mock()
        self.assertIs(c.validate_baseline(receipt, verify), receipt)
        self.assertEqual(verify.call_count, 5)
        self.assertEqual([x.args[0] for x in verify.call_args_list],
                         ["/retained/B0.out", "/retained/B0.first-logits.0", "/retained/B1.out",
                          "/retained/B1.first-logits.0", "/retained/B1.first-logits.1"])
        for change in ("archive", "product", "plan", "incomplete", "logit", "path"):
            bad = copy.deepcopy(receipt)
            if change == "archive": bad["archive_sha256"] = "b" * 64
            if change == "product": bad["implementation"] = "wrong"
            if change == "plan": bad["plan_sha256"] = "wrong"
            if change == "incomplete": bad["B1"]["done_tokens"] = [64]
            if change == "logit": bad["B1"]["logits"].pop()
            if change == "path": bad["B1"]["output"]["path"] = "relative"
            with self.subTest(change=change), self.assertRaises(CoverageError):
                c.validate_baseline(bad, Mock())
        with self.assertRaisesRegex(CoverageError, "artifact-mismatch"):
            c.validate_baseline(receipt, Mock(side_effect=CoverageError("artifact-mismatch")))

    def test_setup_lifetimes_and_cpu_idle_cannot_undercut_new_setup_maximum(self):
        b = Bindings()
        cpu, _, absolute = commands.product("ROLE_CPU_C", "CPU", b, gpu_completion=True)
        self.assertEqual(cpu[cpu.index("--memory-budget-mib") + 1], "106428")
        self.assertEqual(cpu[cpu.index("--residency-ms") + 1], "2700000")
        self.assertEqual(absolute, 7200)
        for phase in c.PHASES:
            engine, _, ec = commands.product("ROLE_ENGINE_A", phase, b, gpu_completion=True)
            gpu, _, gc = commands.product("ROLE_GPU_B", phase, b, gpu_completion=True)
            self.assertEqual(engine[engine.index("--remote-gpu-setup-ms") + 1], "1800000")
            self.assertEqual(gpu[gpu.index("--setup-ms") + 1], "1800000")
            self.assertEqual((ec, gc), (2700, 2700))
            self.assertGreater(min(ec, gc) - 15, 600 + c.SETUP_MS / 1000)
            self.assertEqual(gpu[gpu.index("--reserve-mib") + 1], "4096")
            self.assertEqual(gpu[gpu.index("--cache-budget-mib") + 1], "18432")
        with self.assertRaises(CoverageError):
            commands.product("ROLE_ENGINE_A", "B0", b, gpu_completion=True)
        old, _, _ = commands.product("ROLE_GPU_B", "G0", b)
        self.assertEqual(old[old.index("--setup-ms") + 1], "1200000")

    def test_resume_preserves_existing_cache_and_refuses_missing_or_symlink_roots(self):
        guard = OuterGuards("ROLE_GPU_B", device_ids=("remote",), selection="remote", cache="/retained/cache")
        with patch("outer.os.mkdir") as mkdir, patch("outer.Path.is_dir", return_value=True), \
                patch("outer.Path.is_symlink", return_value=False):
            guard.admit_cache("G0", resume=True)
            guard.admit_cache("G1", resume=True)
        mkdir.assert_not_called()
        for exists, link in ((False, False), (True, True)):
            with patch("outer.Path.is_dir", return_value=exists), \
                    patch("outer.Path.is_symlink", return_value=link), self.assertRaises(CoverageError):
                guard.admit_cache("G0", resume=True)
        with patch("outer.os.mkdir") as mkdir:
            guard.admit_cache("G0")
        mkdir.assert_called_once_with(guard.cache, 0o700)

    def test_resumed_cache_equation_accepts_valid_delta_or_full_fallback(self):
        for reused in (0, 3000 * 5222400, c.ASSIGNED_BYTES):
            missing = c.ASSIGNED_BYTES - reused
            result = c.cache_accounting("G0", reused=reused, transferred=missing, persisted=missing,
                source_read=missing, cached=c.ASSIGNED_BYTES, assigned=c.ASSIGNED_BYTES)
            self.assertEqual(result["cache_state"], "resumed")
        result = c.cache_accounting("G1", reused=c.ASSIGNED_BYTES, transferred=0, persisted=0,
            source_read=0, cached=c.ASSIGNED_BYTES, assigned=c.ASSIGNED_BYTES)
        self.assertEqual(result["new_assigned_payload_bytes"], 0)
        valid = dict(reused=3000 * 5222400, transferred=289 * 5222400, persisted=289 * 5222400,
                     source_read=289 * 5222400, cached=c.ASSIGNED_BYTES, assigned=c.ASSIGNED_BYTES)
        for field, value in (("reused", True), ("transferred", -1), ("source_read", 0),
                             ("persisted", 0), ("cached", 0), ("assigned", c.ASSIGNED_BYTES + 1)):
            with self.subTest(field=field), self.assertRaises(CoverageError):
                c.cache_accounting("G0", **{**valid, field: value})
        with self.assertRaises(CoverageError):
            c.cache_accounting("G0", **{**valid, "reused": 16011879040,
                "transferred": c.ASSIGNED_BYTES - 16011879040})
        with self.assertRaises(CoverageError):
            c.cache_accounting("G1", **valid)

    def test_new_window_clips_every_owner_without_resetting_historical_usage(self):
        with patch("controller.verified_file"):
            ctl = controller.Controller(peers(), gpu_completion=True, baseline=baseline(),
                                        clock=lambda: 200, start=100)
        self.assertEqual(ctl.deadline, 2800)
        self.assertEqual(ctl.phase_cap(2700), 2550)
        for original in (200 + 2700, 100 + 7200):
            self.assertEqual(bounded_job_deadline(200, original, 2750), 2750)
        self.assertEqual(c.HISTORICAL_ACTIVE_S, 4889.5)
        with patch("controller.verified_file"), self.assertRaises(CoverageError):
            controller.Controller(peers(), gpu_completion=True, baseline=baseline(), retain=True)

    def test_only_three_remaining_requests_are_rendered(self):
        streams = [commands.requests(p) for p in c.PHASES]
        self.assertEqual(streams, [b"GEN 64 9419\nQUIT\n", b"GEN 64 9419\nGEN 64 14773,6353\nQUIT\n"])
        count = sum(s.count(b"GEN 64 ") for s in streams)
        self.assertEqual((8 + count, 512 + count * 64), (11, 704))


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    def expired(*_):
        raise KeyboardInterrupt("focused-gpu-completion-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = defaultTestLoader.loadTestsFromTestCase(CompletionTests)
    forbidden = AssertionError("live operation forbidden in GPU completion fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.stat", side_effect=forbidden), patch("os.readlink", side_effect=forbidden), \
            patch("os.fork", side_effect=forbidden), patch("os.waitid", side_effect=forbidden), \
            patch("os.waitpid", side_effect=forbidden), patch("os.kill", side_effect=forbidden), \
            patch("os.pidfd_open", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 8:
        raise SystemExit(1)
    print("GPU_COMPLETION_FIXTURES_PASS_V1 tests=8 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
