"""Focused identity/reporting mocks; authored NOT RUN. No live workloads."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import tempfile
from unittest import TestCase, TextTestRunner, defaultTestLoader
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cpu_first_overlap as overlap
import paired_profile as profile
import paired_g_continuation as gpu
import paired_tasks as tasks
from test_paired_tasks import plan
ROOT = None


def identity():
    return {"diagnostic": overlap.TAG, "diagnostic_revision": "c" * 40, "candidate_source_revision": "c" * 40,
            "candidate_base_revision": overlap.BASE, "retained_engine_sha256": "a" * 64,
            "candidate_engine_sha256": "b" * 64, "binary_sha256": "b" * 64, "role": "ROLE_ENGINE_A"}


class Cases(TestCase):
    def test_explicit_candidate_and_accepted_pins(self):
        overlap.candidate_identity(identity())
        for key, value in (("binary_sha256", "d" * 64), ("candidate_engine_sha256", "a" * 64),
                           ("candidate_source_revision", "d" * 40), ("candidate_base_revision", "d" * 40)):
            with self.assertRaises(ValueError):
                overlap.candidate_identity({**identity(), key: value})
        missing = identity()
        missing.pop("diagnostic_revision"); missing.pop("candidate_source_revision")
        with self.assertRaises(ValueError):
            overlap.candidate_identity(missing)
        with patch.object(tasks, "read_json", return_value={**identity(), "role": "ROLE_GPU_B"}), \
                self.assertRaisesRegex(ValueError, "engine config"):
            overlap.report(plan(), "/pack", "/B", "/old", "/oldG", "/oldGconfig", "/newG", "/wrong-role")

    def test_accepted_g_binding_is_not_replaced_by_candidate_hash(self):
        cfg = {"diagnostic": gpu.TAG, "diagnostic_revision": overlap.BASE, "role": "ROLE_ENGINE_A",
               "plan_sha256": gpu.PLAN_SHA, "binary_sha256": "a" * 64}
        with patch.object(tasks, "read_json", return_value=cfg), self.assertRaisesRegex(ValueError, "accepted G"):
            overlap.prior_g(plan(), "/unused", "/old", "b" * 64)

    def test_candidate_g_only_schedule_retains_resource_flags(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "plan")
            path.write_text(json.dumps(plan()))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            cfg = {**identity(), "supervisor_commit": profile.SUPERVISOR, "approved_window_s": 1500,
                   "execution_approved": True, "plan": str(path), "plan_sha256": sha, "run_root": "/private/new"}
            with patch.object(gpu, "PLAN_SHA", sha):
                agent, _ = profile.configure(ROOT, cfg)
            import gpu_completion as completion
            self.assertEqual(tuple(completion.ENGINE_CAPS), ("G0", "G1"))
            self.assertEqual(sum(agent.requests(p).count(b"GEN ") for p in gpu.PHASES), 7)
            with self.assertRaises(ValueError):
                agent.requests("B0")
            class Bindings(dict):
                def __missing__(self, key):
                    return "/private/" + key
            argv, env, cap = agent.product("ROLE_ENGINE_A", "G0", Bindings(), gpu_completion=True)
            self.assertEqual((env["STRATA_REQUIRE_CPU_RESIDENT_HIT"], cap), ("1", 1500))
            self.assertNotIn("STRATA_DUMP_FIRST_LOGITS", env)
            self.assertNotIn("--dump-routing", argv)
            self.assertIn("--remote-gpu-experts", argv)
            with self.assertRaisesRegex(ValueError, "never start a CPU"):
                agent.product("ROLE_CPU_C", "CPU", {}, gpu_completion=True)

    def test_same_mode_drift_is_reported_without_threshold_change_or_replay(self):
        with tempfile.TemporaryDirectory() as root:
            old, new = Path(root, "old"), Path(root, "new")
            old.mkdir(); new.mkdir()
            for i in range(2):
                for directory in (old, new):
                    (directory / ("G1.first-logits." + str(i))).write_bytes(b"finite fixture stand-in")
            cfg = Path(root, "candidate.json")
            cfg.write_text(json.dumps({**identity(), "plan_sha256": gpu.PLAN_SHA}))
            rows = {"tasks": [{"tokens": [1]}], "probes": [{"tokens": [2]}], "likelihood": [{"answer_mean_nll": 1}]}
            def evaluated(*_):
                return {"modes": {"G": rows}, "retained_baseline": {"engine_binary_sha256": "a" * 64},
                        "historical_numerical_verdict": "NUMERICAL_FAIL", "common_input_logits": [{"within_unchanged_0.03_limits": False}]}
            with patch.object(overlap, "prior_g", return_value=rows), patch.object(gpu, "continuation_report", side_effect=evaluated), \
                    patch.object(tasks, "make_plan", side_effect=AssertionError("no retokenization")):
                result = overlap.report(plan(), "/pack", "/B", "/old-config", old, "/old-G-config", new, cfg)
                self.assertEqual(result["scheduling_verdict"], "NO_OBSERVED_DRIFT")
                (new / "G1.first-logits.0").write_bytes(b"changed")
                result = overlap.report(plan(), "/pack", "/B", "/old-config", old, "/old-G-config", new, cfg)
            self.assertEqual(result["scheduling_verdict"], "SCHEDULING_REGRESSION")
            self.assertEqual(result["historical_numerical_verdict"], "NUMERICAL_FAIL")
            self.assertEqual((result["cumulative_started_request_ceiling"], result["cumulative_generated_token_ceiling"]), (38, 1103))
            self.assertFalse(result["common_input_logits"][0]["within_unchanged_0.03_limits"])


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-root", required=True)
    ROOT = str(Path(parser.parse_args().supervisor_root).resolve(strict=True))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1048576, 256 * 1048576))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    signal.alarm(10)
    forbidden = AssertionError("no process/socket/fleet/model work in focused fixture")
    with patch("subprocess.Popen", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("socket.socket", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(defaultTestLoader.loadTestsFromTestCase(Cases))
    if not result.wasSuccessful() or result.skipped or result.testsRun != 4:
        raise SystemExit(1)
    print("CPU_FIRST_PROFILE_V1 tests=4 workloads=0")


if __name__ == "__main__":
    main()
