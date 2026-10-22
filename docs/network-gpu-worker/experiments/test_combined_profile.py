"""Focused mocked profile/service checks; no products/transports. Authored NOT RUN."""
import argparse
import errno
import resource
import signal
import tempfile
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import TestCase, TextTestRunner, defaultTestLoader
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import combined_diagnostic as profile

ROOT = None
CONFIG = {"diagnostic": profile.TAG, "supervisor_commit": profile.SUPERVISOR,
          "diagnostic_revision": "a" * 40, "approved_window_s": 1500,
          "execution_approved": True, "run_root": "/private/new-run",
          "role": "ROLE_CPU_C", "service_residency": True, "retention_approved": True}


class ProfileTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent, cls.controller = profile.configure(ROOT, CONFIG)

    def test_streams_and_separate_new_allowance(self):
        self.assertEqual(self.agent.requests("G0"), b"GEN 64 9419\nGEN 64 14773,6353\nQUIT\n")
        self.assertEqual(self.agent.requests("G1"), b"GEN 1 9419\nQUIT\n")
        self.assertEqual((2 + 1, 64 + 64 + 1), (3, 129))

    def test_caps_are_bound_before_start_with_cleanup_reserve(self):
        import gpu_completion as completion
        self.assertEqual((completion.PHASES, completion.WINDOW_S, completion.HISTORICAL_ACTIVE_S),
                         (("G0", "G1"), 1500, 6651.9))
        with patch.object(completion, "validate_baseline", return_value={"archive_sha256": "x"}):
            ctl = self.controller.Controller({}, gpu_completion=True, baseline={}, start=0, clock=lambda: 0)
        self.assertEqual((ctl.deadline, ctl.phase_cap(1500)), (1500, 1450))
        self.assertEqual(ctl.phases, ("G0", "G1"))

    def test_capture_is_absent_from_performance_and_only_explicit_in_capture_pass(self):
        class Bindings(dict):
            def __missing__(self, key):
                return "/private/" + key
        for role in ("ROLE_ENGINE_A", "ROLE_GPU_B"):
            argv, env, cap = self.agent.product(role, "G0", Bindings(), gpu_completion=True)
            self.assertFalse(any("CAPTURE" in key for key in env))
            self.assertEqual(cap, 1500)
            with patch("os.mkdir") as mkdir:
                _, captured, _ = self.agent.product(role, "G1", Bindings(), gpu_completion=True)
            self.assertEqual(sum("CAPTURE" in key for key in captured), 1)
            mkdir.assert_called_once_with("/private/new-run/capture", 0o700)
        cpu, env, _ = self.agent.product("ROLE_CPU_C", "CPU", Bindings(), gpu_completion=True)
        self.assertIn("--residency-forever", cpu)
        self.assertNotIn("--residency-ms", cpu)
        self.assertFalse(any("CAPTURE" in key for key in env))

    def test_new_execution_permission_and_one_initialization_are_required(self):
        with self.assertRaisesRegex(ValueError, "already initialized"):
            profile.configure(ROOT, CONFIG)
        with patch.object(profile, "_configured", False), self.assertRaisesRegex(ValueError, "approval/config"):
            profile.configure(ROOT, {**CONFIG, "execution_approved": False})

    def test_replay_refuses_wrong_role_or_no_safe_interval_before_ownership(self):
        with self.assertRaisesRegex(ValueError, "bounded 05 replay"):
            profile.replay(self.agent, {"role": "ROLE_ENGINE_A"}, 60)
        with self.assertRaisesRegex(ValueError, "bounded 05 replay"):
            profile.replay(self.agent, {"role": "ROLE_CPU_C"}, 20)

    def test_replay_preflight_cannot_renew_window(self):
        import runtime
        import outer
        import linux_owned
        import resource_guard
        config = {"role": "ROLE_CPU_C", "run_root": "/private/replay", "run_id": "fixture",
                  "namespaces": {}, "replay_argv": ["/bin/replay", "/capture", "/plan", "/source", "/manifest"],
                  "binary_sha256": "b" * 64}
        with patch.object(linux_owned, "parent_death"), patch.object(profile.os, "mkdir"), \
                patch.object(profile.signal, "signal"), patch.object(outer, "OuterGuards"), \
                patch.object(resource_guard, "Journal"), patch.object(runtime, "Session"), \
                patch.object(runtime, "Job") as job, patch.object(self.agent, "verified_file"), \
                patch.object(profile.time, "monotonic", return_value=141):
            with self.assertRaisesRegex(ValueError, "preflight consumed"):
                profile.replay(self.agent, config, 60, entered=100)
        job.assert_not_called()
        import policy
        self.assertEqual(policy.POLICIES["ROLE_REPLAY_C"].rss, 512 * policy.MIB)
        self.assertEqual(policy.POLICIES["ROLE_CPU_C"].rss, 108024 * policy.MIB)

    def test_service_grant_requires_full_idle_no_expiry_receipt(self):
        import resident_cpu
        from policy import Stamp
        from resource_telemetry import Identity
        ident = Identity(123,456,7,8)
        session = SimpleNamespace(admission=Stamp("fixture",1,2), baseline={}, namespaces={})
        with tempfile.TemporaryDirectory() as root:
            job = SimpleNamespace(root=SimpleNamespace(verify=Mock(),identity=ident),
                                  monitor=SimpleNamespace(identity=ident,verify=Mock()), directory=root,
                                  gate=SimpleNamespace(deadline=1500,check=Mock()), deadline=1500)
            cfg = {"run_root": root, "run_id": "fixture", "bindings": {"CPU_PORT_C": "7431"}}
            status = {"state": 0, "session": 0, "resident_bytes": resident_cpu.ARENA,
                      "idle_remaining_ms": 1800000, "complete": True}
            with patch.object(resident_cpu, "cache_status", return_value=status):
                with self.assertRaisesRegex(resident_cpu.CoverageError, "no-expiry"):
                    resident_cpu.commit(cfg,job,session)
            self.assertEqual(job.deadline,1500)
            with patch.object(resident_cpu, "identity", return_value=ident):
                resident_cpu.grant(cfg,job,session)
            self.assertTrue(math.isinf(job.deadline) and math.isinf(job.gate.deadline))
            status["idle_remaining_ms"] = resident_cpu.NO_EXPIRY
            with patch.object(resident_cpu, "cache_status", return_value=status), \
                    patch.object(resident_cpu, "identity", return_value=ident):
                receipt = resident_cpu.commit(cfg,job,session)
            self.assertEqual(receipt["lifetime"], "service-owned-no-idle-expiry")
            self.assertTrue(math.isinf(job.deadline) and math.isinf(job.gate.deadline))
            self.assertEqual(json.loads(Path(root,"resident.json").read_text())["worker"]["start_ticks"],456)

    def cpu_disconnect(self, disconnect):
        import resident_cpu
        import gpu_completion as completion
        from resource_guard import CoverageError, GuardStop
        channel = Mock()
        job = Mock(key=("fixture","ROLE_CPU_C","CPU"))
        job.gate.released = True
        job.poll.side_effect = [None, GuardStop("fixture-pressure")]
        job.stop.return_value = {"complete": True,"reaped":{},"errors":[]}
        session = Mock(domain="fixture")
        session.admission.mono = 1
        challenge = {}
        def packet(_, message):
            if message["kind"] == "CHALLENGE":
                challenge.update(message)
            if message["kind"] == "LISTENING_COVERED" and disconnect == "send":
                raise BrokenPipeError(errno.EPIPE,"fixture controller closed")
            if message["kind"] == "LISTENING_COVERED" and disconnect == "permission":
                raise PermissionError(errno.EACCES,"fixture real permission fault")
        def receive(_):
            if not challenge.get("used"):
                challenge["used"] = True
                return {"run_id":"fixture","sequence":0,"nonce":challenge["nonce"],
                        "remaining_s":600,"action":"START","phase":"CPU","gpu_completion":True}
            raise CoverageError("control-eof")
        class Bindings(dict):
            def __missing__(self,key): return "/private/"+key
        bindings = Bindings(CPU_PORT_C="7431")
        cfg = {**CONFIG,"run_id":"fixture","gpu_completion":True,"implementation":completion.IMPLEMENTATION,
               "bindings":bindings,"binary_sha256":"b"*64,"namespaces":{}}
        with patch.object(self.agent.os,"setsid"), patch.object(self.agent.signal,"signal"), \
                patch.object(self.agent.os,"mkdir"), patch.object(self.agent.os,"pipe2",return_value=(101,102)), \
                patch.object(self.agent.os,"close"), patch.object(self.agent,"Journal") as journal, \
                patch.object(self.agent,"OuterGuards"), patch.object(self.agent,"Session",return_value=session), \
                patch.object(self.agent,"Job",return_value=job), patch.object(self.agent,"verified_file"), \
                patch.object(self.agent,"owned_listener",return_value=True), patch.object(self.agent,"packet",side_effect=packet), \
                patch.object(self.agent,"receive",side_effect=receive), \
                patch.object(self.agent.select,"select",return_value=([channel],[],[])), \
                patch.object(self.agent.time,"monotonic",return_value=100), \
                patch.object(resident_cpu,"grant",return_value={"lifetime":"service-owned-no-idle-expiry"}) as grant:
            self.agent.serve(cfg,channel)
        grant.assert_called_once()
        job.stop.assert_called_once_with("agent-exit")
        markers = [call.args[0] for call in journal.return_value.emit.call_args_list]
        if disconnect == "permission":
            self.assertNotIn("SERVICE_CONTROLLER_DETACHED_V1",markers)
            self.assertIn("fixture real permission fault",journal.return_value.emit.call_args_list[-1].kwargs["reason"])
        else:
            self.assertIn("SERVICE_CONTROLLER_DETACHED_V1",markers)
            self.assertEqual(journal.return_value.emit.call_args_list[-1].kwargs["reason"],"GuardStop:fixture-pressure")

    def test_controller_disconnect_detaches_but_resource_and_permission_faults_clean(self):
        for disconnect in ("receive-eof","send","permission"):
            with self.subTest(disconnect=disconnect):
                self.cpu_disconnect(disconnect)

    def test_resident_journal_rotates_with_bounded_storage(self):
        from resident_cpu import ResidentJournal
        with tempfile.TemporaryDirectory() as root, patch.object(ResidentJournal, "limit", 100):
            path = str(Path(root,"resources.jsonl"))
            journal = ResidentJournal(path)
            for seq in range(8):
                journal.emit("sample",sequence=seq)
            journal.close()
            self.assertLessEqual(Path(path).stat().st_size,100)
            self.assertLessEqual(Path(path+".previous").stat().st_size,100)
            self.assertEqual(json.loads(Path(path).read_text().splitlines()[-1])["sequence"],7)


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--supervisor-root", required=True)
    ROOT = ap.parse_args().supervisor_root
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1048576, 128 * 1048576))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    signal.alarm(10)
    forbidden = AssertionError("live operation forbidden in profile fixture")
    with patch("subprocess.Popen", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("socket.socket", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(defaultTestLoader.loadTestsFromTestCase(ProfileTests))
    if not result.wasSuccessful() or result.skipped or result.testsRun != 9:
        raise SystemExit(1)
    print("GPU_COMBINED_PROFILE_PASS_V1 tests=9 workloads=0", flush=True)


if __name__ == "__main__":
    main()
