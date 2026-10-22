"""Focused CPU-only mocks. Authored NOT RUN; never launches a model or service."""
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
import paired_tasks as tasks
import paired_profile as profile
ROOT = None


def plan():
    return {"diagnostic": tasks.TAG,
            "tasks": [{"name": row[0], "ids": [10, 20, 30], "max_new": 32} for row in tasks.TASKS],
            "probes": [{"name": tasks.TASKS[i][0], "ids": [10, 20, 30], "max_new": 1, "answer_start": 2}
                       for i in tasks.PROBES]}


def done(n=1, prompt=3, finish="stop", prompt_ms=100, decode_ms=200):
    return f"DONE {n} {prompt} {prompt_ms} {decode_ms} {finish} 0 0 0 3 3 0 0 0.0 3 1 chunk=0 experts_streamed=0 worker_entries=9 worker_req_wall=20.0"


class Cases(TestCase):
    def test_fixed_request_accounting_and_prompt_bound(self):
        p = plan()
        tasks.validate_plan(p)
        streams = [tasks.requests(p, phase) for phase in tasks.PHASES]
        self.assertEqual(sum(s.count(b"GEN ") for s in streams), 14)
        self.assertEqual(sum(sum(r["max_new"] for r in tasks.phase_rows(p, phase)) for phase in tasks.PHASES), 324)
        self.assertTrue(all(s.endswith(b"QUIT\n") and len(s) <= 4096 for s in streams))
        p["tasks"][0]["ids"] *= 22
        with self.assertRaisesRegex(ValueError, "64 tokens"):
            tasks.validate_plan(p)

    def test_public_checks_are_strict_and_do_not_execute_code(self):
        for name, _, answer in tasks.TASKS:
            self.assertTrue(tasks.check_answer(name, answer))
            self.assertFalse(tasks.check_answer(name, "Explanation: " + answer))
        self.assertTrue(tasks.check_answer("extraction", '{"score":12,"name":"Jon"}'))
        self.assertFalse(tasks.check_answer("code", "import os\nos._exit(0)"))
        self.assertFalse(tasks.check_answer("code", "def is_even(n): return n%2==1"))

    def test_protocol_units_caps_and_real_failures(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root, "out")
            out.write_text("READY\nT 16\n" + done() + "\n")
            row = tasks.protocol(out, plan()["tasks"][:1])[0]
            self.assertEqual((row["engine_request_ms"], row["decode_tokens_per_s"], row["worker_exchange_ms"]), (300, 5, 20))
            for content in ("READY\nT 16\n", "READY\nERR failed\n", "READY\nT 16\n" + done(finish="nonfinite") + "\n",
                            "READY\nT 16\n" + done(n=2) + "\n"):
                out.write_text(content)
                with self.assertRaises(ValueError):
                    tasks.protocol(out, plan()["tasks"][:1])

    def test_common_input_likelihood_targets_and_completeness(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root, "lp")
            content = "0\t20\t-2.0\t1\t-1.0\t0\t-8\t-2\n1\t30\t-3.0\t1\t-1.0\t0\t-8\t-3\n"
            out.write_text(content * 2)
            self.assertEqual([r["answer_mean_nll"] for r in tasks.likelihood(out, plan()["probes"])], [3, 3])
            out.write_text(content.replace("1\t30", "1\t31") * 2)
            with self.assertRaisesRegex(ValueError, "identical supplied"):
                tasks.likelihood(out, plan()["probes"])

    def test_clock_handles_fragmented_output_and_associates_done(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root, "workload.out")
            out.write_bytes(b"REA")
            clock = profile.OutputClock(root)
            with patch.object(profile.time, "monotonic", side_effect=[10, 11]):
                clock.sample()
                with open(out, "ab") as target:
                    target.write(("DY\nT 16\n" + done() + "\n").encode())
                clock.sample()
            rows = tasks.protocol(out, plan()["tasks"][:1])
            summary = tasks.output_clock(Path(root, "output-clock.json"), rows)
            self.assertEqual(summary["ready_to_last_done_s"], 1)
            with open(out, "ab") as target:
                target.write(b"x" * 1048577)
            with self.assertRaisesRegex(ValueError, "bounded stable"):
                clock.sample()

    def test_numeric_comparison_does_not_change_historical_verdict(self):
        from array import array
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root, "a"), Path(root, "b")
            values = array("f", [float(i % 7) for i in range(248320)])
            a.write_bytes(values.tobytes())
            b.write_bytes(values.tobytes())
            result = tasks.compare_logits(a, b)
            self.assertEqual(result["centered_rel_l1"], 0)
            self.assertAlmostEqual(result["kl_b_to_g_nats"], 0)
            values[0] = float("nan")
            b.write_bytes(values.tobytes())
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                tasks.compare_logits(a, b)

    def test_profile_reuses_guarded_jobs_and_never_starts_cpu(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "plan.json")
            raw = json.dumps(plan()).encode()
            path.write_bytes(raw)
            cfg = {"diagnostic": tasks.TAG, "supervisor_commit": profile.SUPERVISOR,
                   "approved_window_s": 1500, "execution_approved": True, "diagnostic_revision": "a" * 40,
                   "plan": str(path), "plan_sha256": hashlib.sha256(raw).hexdigest(), "run_root": "/private/new"}
            agent, _ = profile.configure(ROOT, cfg)
            import gpu_completion as completion
            from policy import TrialOrder
            self.assertEqual(TrialOrder(tasks.PHASES).phases, tasks.PHASES)
            self.assertEqual(tuple(completion.ENGINE_CAPS), tasks.PHASES)
            self.assertEqual(completion.GPU_CAPS, {"G0": 1500})
            with self.assertRaisesRegex(ValueError, "never start a CPU"):
                agent.product("ROLE_CPU_C", "CPU", {}, gpu_completion=True)
            class Bindings(dict):
                def __missing__(self, key):
                    return "/private/" + key
            for phase in tasks.PHASES:
                argv, env, cap = agent.product("ROLE_ENGINE_A", phase, Bindings(), gpu_completion=True)
                self.assertEqual(cap, 1500)
                self.assertEqual(env["STRATA_REQUIRE_CPU_RESIDENT_HIT"], "1")
                self.assertNotIn("--dump-routing", argv)
                self.assertFalse(any("TRACE" in k or "CAPTURE" in k for k in env))
                self.assertEqual("STRATA_LOGPOS" in env, phase.endswith("1"))
                self.assertEqual("STRATA_DUMP_FIRST_LOGITS" in env, phase.endswith("1"))
                self.assertEqual("--remote-gpu-experts" in argv, phase.startswith("G"))
                self.assertEqual(agent.requests(phase), tasks.requests(plan(), phase))
            old = profile._configured
            profile._configured = False
            try:
                with self.assertRaisesRegex(ValueError, "approval"):
                    profile.configure(ROOT, {**cfg, "execution_approved": False})
            finally:
                profile._configured = old


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-root", required=True)
    ROOT = parser.parse_args().supervisor_root
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1048576, 256 * 1048576))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    signal.alarm(20)
    forbidden = AssertionError("no model/service/socket/process workload in focused fixture")
    with patch("subprocess.Popen", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("socket.socket", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(defaultTestLoader.loadTestsFromTestCase(Cases))
    if not result.wasSuccessful() or result.skipped or result.testsRun != 7:
        raise SystemExit(1)
    print("PAIRED_TASKS_PASS_V1 tests=7 workloads=0")


if __name__ == "__main__":
    main()
