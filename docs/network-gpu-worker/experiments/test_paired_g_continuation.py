"""Focused deterministic continuation mocks; authored NOT RUN, no workloads."""
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import tempfile
from unittest import TestCase, TextTestRunner, defaultTestLoader
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_g_continuation as gpu
import paired_profile as profile
import paired_tasks as tasks
from test_paired_tasks import plan, done
ROOT = None


class Cases(TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [ROOT, str(Path(ROOT).parent)]
        import controller
        cls.controller = controller
        if not getattr(controller, "PEER_DIAGNOSTICS_V1", False):
            raise AssertionError("actual composed supervisor peer-diagnostics patch required")

    def peer(self, path):
        process = Mock(returncode=None)
        process.stdout.fileno.return_value = 11
        process.stdin.fileno.return_value = 12
        process.stderr.fileno.return_value = 13
        with patch.object(self.controller.subprocess, "Popen", return_value=process), \
                patch.object(self.controller, "child_pidfd", return_value=99), patch.object(self.controller.os, "set_blocking"):
            return self.controller.Peer("ROLE_CPU_C", "fixture", ["never-executed"], diagnostic_path=str(path))

    def test_failed_final_is_saved_before_refusal_and_close_records_exit(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "peer.json")
            peer = self.peer(path)
            packet = {"kind": "FINAL", "role": peer.role, "run_id": "fixture", "reason": "ValueError:actual observer cause",
                      "cleanup": {"complete": True, "resident_service_untouched": True}}
            def read(fd, _):
                if fd == 13:
                    raise BlockingIOError()
                return (json.dumps(packet) + "\n").encode()
            with patch.object(self.controller.os, "read", side_effect=read), self.assertRaisesRegex(Exception, "required-peer-failed"):
                peer.pump(100)
            self.assertTrue(peer.final_seen)
            self.assertEqual(peer.events, [packet])
            self.assertEqual(tasks.read_json(path)["final_packet"], packet)
            def finish(process, fd, *, before_close):
                self.assertEqual(fd, 99)
                before_close()
                process.returncode = -15
            with patch.object(self.controller, "finish_process", side_effect=finish), \
                    patch.object(self.controller.os, "read", side_effect=[b"stderr detail\n", b""]):
                peer.close()
            result = tasks.read_json(path)
            self.assertEqual(result["exit_status"], -15)
            self.assertEqual(result["stderr_tail"], "stderr detail\n")
            self.assertEqual(result["final_packet"]["reason"], packet["reason"])

    def test_stale_identity_is_not_saved_as_valid_final(self):
        with tempfile.TemporaryDirectory() as root:
            peer = self.peer(Path(root, "peer.json"))
            packet = {"kind": "FINAL", "role": peer.role, "run_id": "wrong", "reason": "finished", "cleanup": {"complete": True}}
            with patch.object(peer, "drain_stderr"), patch.object(self.controller.os, "read", return_value=(json.dumps(packet) + "\n").encode()), \
                    self.assertRaisesRegex(Exception, "stale-peer-identity"):
                peer.pump(100)
            self.assertIsNone(peer.terminal_packet)
            self.assertFalse(peer.final_seen)

    def test_stderr_is_bounded_and_does_not_refresh_protocol_progress(self):
        with tempfile.TemporaryDirectory() as root:
            peer = self.peer(Path(root, "peer.json"))
            with patch.object(self.controller.os, "read", side_effect=[b"a" * 4096] * 16 + [b"last", b""]):
                peer.drain_stderr()
                peer.drain_stderr()
            self.assertEqual(len(peer.stderr_tail), 65536)
            self.assertEqual(peer.stderr_bytes, 65540)
            self.assertTrue(peer.stderr_tail.endswith(b"last"))
            peer.save_diagnostics()
            self.assertTrue(tasks.read_json(peer.diagnostic_path)["stderr_tail_truncated"])
            peer.stderr_eof, peer.last_progress = False, 0
            controller = self.controller.Controller.__new__(self.controller.Controller)
            controller.peers, controller.clock, controller.deadline, controller.success_at = {peer.role: peer}, lambda: 100, 200, None
            with patch.object(self.controller.select, "select", return_value=([peer.process.stderr], [], [])), \
                    patch.object(peer, "drain_stderr") as drained, self.assertRaisesRegex(Exception, "required-peer-stalled"):
                controller.pump()
            drained.assert_called_once()
            self.assertEqual(peer.last_progress, 0)

    def baseline(self, root):
        p = plan()
        cfg = Path(root, "engine.json")
        cfg.write_text(json.dumps({"diagnostic": tasks.TAG, "diagnostic_revision": gpu.BASELINE_REVISION,
                                 "role": "ROLE_ENGINE_A", "plan_sha256": gpu.PLAN_SHA, "binary_sha256": "b" * 64}))
        for phase in ("B0", "B1"):
            Path(root, phase).mkdir()
        rows, lines = [], ["READY"]
        for n in (10, 10, 10, 10, 11):
            line = done(n=n)
            lines += ["T 3"] * n + [line]
            rows.append(line)
        Path(root, "B0/workload.out").write_text("\n".join(lines) + "\n")
        Path(root, "B1/workload.out").write_text("READY\n" + ("T 3\n" + done(finish="length") + "\n") * 2)
        Path(root, "B0/output-clock.json").write_text(json.dumps({"scope": "fixture", "events":
            [{"line": line, "observed_mono": i} for i, line in enumerate(["READY", *rows])]}))
        Path(root, "B1.logpos").write_text(("0 20 -2 1 -1 0 -8 -2\n1 30 -3 1 -1 0 -8 -3\n") * 2)
        values = array("f", [0]) * 248320
        values[3] = 1
        for i in range(2):
            Path(root, "B1.first-logits." + str(i)).write_bytes(values.tobytes())
        return p, cfg

    def test_retained_b_completeness_and_plan_binding(self):
        with tempfile.TemporaryDirectory() as root:
            p, cfg = self.baseline(root)
            result = gpu.check_baseline(p, root, cfg)
            self.assertEqual((result["started_requests"], result["generated_tokens"]), (7, 53))
            value = tasks.read_json(cfg)
            value["plan_sha256"] = "wrong"
            cfg.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "config/plan"):
                gpu.check_baseline(p, root, cfg)
            saved = Path(root, "plan")
            saved.write_text(json.dumps(p))
            with patch.object(gpu, "PLAN_SHA", hashlib.sha256(saved.read_bytes()).hexdigest()):
                self.assertEqual(gpu.retained_plan(saved), p)
                saved.write_text(" " + saved.read_text())
                with self.assertRaisesRegex(ValueError, "exact retained"):
                    gpu.retained_plan(saved)

    def test_g_only_requests_caps_and_no_cpu_or_b_start(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, "plan")
            path.write_text(json.dumps(plan()))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            cfg = {"diagnostic": gpu.TAG, "supervisor_commit": profile.SUPERVISOR, "approved_window_s": 1500,
                   "execution_approved": True, "diagnostic_revision": "c" * 40, "plan": str(path),
                   "plan_sha256": sha, "run_root": "/private/new"}
            with patch.object(gpu, "PLAN_SHA", sha):
                agent, _ = profile.configure(ROOT, cfg)
            import gpu_completion as completion
            self.assertEqual(tuple(completion.ENGINE_CAPS), ("G0", "G1"))
            self.assertEqual(completion.GPU_CAPS, {"G0": 1500})
            self.assertEqual(sum(agent.requests(p).count(b"GEN ") for p in gpu.PHASES), 7)
            self.assertEqual(sum(sum(r["max_new"] for r in tasks.phase_rows(plan(), p)) for p in gpu.PHASES), 162)
            with self.assertRaises(ValueError):
                agent.requests("B0")
            with self.assertRaisesRegex(ValueError, "never start a CPU"):
                agent.product("ROLE_CPU_C", "CPU", {}, gpu_completion=True)

    def test_report_reuses_evaluator_without_retokenizing(self):
        with tempfile.TemporaryDirectory() as root:
            p = plan()
            for name in tasks.METADATA:
                Path(root, name).write_bytes(b"fixture")
            p["tokenizer_metadata_sha256"] = {name: hashlib.sha256(b"fixture").hexdigest() for name in tasks.METADATA}
            with patch.object(gpu, "check_baseline", return_value={"complete": True}), \
                    patch.object(tasks, "report", return_value={"modes": {}}) as evaluator, \
                    patch.object(tasks, "make_plan", side_effect=AssertionError("retokenization forbidden")):
                result = gpu.continuation_report(p, root, "/B", "/G", "/old-config")
            evaluator.assert_called_once_with(p, root, "/B", "/G")
            self.assertTrue(result["separate_run_comparison"])
            self.assertEqual((result["request_cap"], result["generated_token_cap"]), (7, 162))
            self.assertEqual(result["cumulative_generated_token_ceiling"], 1050)

    def test_one_clock_receipt_survives_transport_start_failure(self):
        with tempfile.TemporaryDirectory() as root:
            evidence = Path(root, "new")
            cfg = {"diagnostic": gpu.TAG, "controller_evidence": str(evidence), "plan": "unused",
                   "retained_b": {"root": "/B", "engine_config": "/old"}, "retained_engine_sha256": "b" * 64,
                   "peers": {r: [] for r in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B")}, "forwards": [["never-executed"]]}
            with patch.object(gpu, "retained_plan", return_value=plan()), \
                    patch.object(gpu, "check_baseline", return_value={"engine_binary_sha256": "b" * 64}), \
                    patch.object(self.controller, "Forward", side_effect=ValueError("fixture transport failure")), \
                    patch.object(profile.time, "monotonic", side_effect=[10, 20]), self.assertRaisesRegex(ValueError, "fixture transport"):
                profile.launch(self.controller, cfg)
            receipt = tasks.read_json(evidence / "execution-clock.json")
            self.assertEqual((receipt["start_mono"], receipt["deadline_mono"], receipt["elapsed_s"]), (10, 1510, 10))
            self.assertEqual(receipt["status"], "FAILED")


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-root", required=True)
    ROOT = str(Path(parser.parse_args().supervisor_root).resolve(strict=True))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1048576, 256 * 1048576))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    signal.alarm(20)
    forbidden = AssertionError("no process/socket/fleet/model work in focused fixture")
    with patch("subprocess.Popen", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("socket.socket", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(defaultTestLoader.loadTestsFromTestCase(Cases))
    if not result.wasSuccessful() or result.skipped or result.testsRun != 7:
        raise SystemExit(1)
    print("PAIRED_G_CONTINUATION_PASS_V1 tests=7 workloads=0")


if __name__ == "__main__":
    main()
