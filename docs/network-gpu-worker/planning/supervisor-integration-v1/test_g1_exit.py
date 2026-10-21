"""G1-EXIT-V1 focused deterministic mocks; authored NOT RUN locally."""
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
import gpu_completion as c
import resource_guard as g
import resource_telemetry as t
import runtime
from test_gpu_completion import baseline as old_baseline, peers, Bindings

MIB = 1048576
KEY = ("r", "ROLE_ENGINE_A", "G1")
NAMESPACES = {kind: (1, 2) for kind in ("cgroup", "mnt", "pid")}


def stat(start=900):
    return t.ok("41 (fixture) " + " ".join(["S", "7"] + ["0"] * 17 + [str(start)]))


def process():
    return t.OwnedProcess(t.Identity(41, 900, 1, 2), 101, 102, NAMESPACES)


def baseline():
    receipt = old_baseline()
    receipt["g0_archive_sha256"] = c.G0_ARCHIVE_SHA
    receipt["G0"] = {"rc": 0, "done_tokens": [64], "product_complete": True,
                     "supervisor_status": "FAILED",
                     "output": {"path": "/retained/G0.out", "sha256": "b" * 64},
                     "logits": [{"path": "/retained/G0.first-logits.0", "sha256": "b" * 64}]}
    return receipt


def job():
    owned = object.__new__(runtime.Job)
    owned.key, owned.root, owned.channel = KEY, process(), Mock()
    owned.root.alive, owned.root.close, owned.root.send = Mock(return_value=True), Mock(), Mock()
    owned.children, owned.reaped, owned.deadline = {41}, {}, 10
    owned.items = {41: owned.root}
    owned.stop_result, owned.fd_errors, owned.transient_fds = None, [], set()
    owned.transient_sockets, owned.exec_gate, owned.namespaces = [], None, NAMESPACES
    return owned


def pending_packet():
    return {"kind": "EXIT_PENDING", "key": KEY, "pid": 41, "start_ticks": 900}


class G1ExitTests(TestCase):
    def test_namespace_teardown_waits_for_held_pidfd_across_shared_readers(self):
        for initial_reader in (True, False):
            owned = process()
            def namespace(path, **_):
                if path == "ns/cgroup":
                    raise FileNotFoundError(2, "teardown", path)
                return SimpleNamespace(st_dev=1, st_ino=2)
            with self.subTest(initial_reader=initial_reader), \
                    patch.object(t.os, "stat", side_effect=namespace), \
                    patch.object(t, "read_text", return_value=stat()), \
                    patch.object(t.select, "select", side_effect=[([], [], []), ([102], [], [])]) as selected:
                self.assertTrue(owned.alive())  # Namespace can disappear before notification.
                with self.assertRaises(t.ProcessExited):
                    if initial_reader:
                        t.verify_initial_namespaces(NAMESPACES, 101, process=owned)
                    else:
                        g.verify_namespaces(owned, NAMESPACES)
            self.assertEqual(selected.call_args.args[-1], 0.1)

    def test_persistent_absence_never_proves_exit_and_real_identity_faults_survive(self):
        with patch.object(t, "read_text", return_value=stat()), \
                patch.object(t.select, "select", return_value=([], [], [])) as selected, \
                self.assertRaises(t.ExitPending):
            process().reconcile_missing("ns/cgroup")
        self.assertEqual([x.args[-1] for x in selected.call_args_list], [0.1, 0])
        for observation in (stat(901), t.ok("malformed"), t.Observation(None, "unavailable", "EACCES")):
            with self.subTest(observation=observation), patch.object(t, "read_text", return_value=observation), \
                    patch.object(t.select, "select") as selected, self.assertRaises(t.CoverageError) as error:
                process().reconcile_missing("ns/cgroup")
            self.assertNotIsInstance(error.exception, t.ExitPending)
            selected.assert_not_called()

    def test_terminal_hint_delivers_only_actual_guardian_rc_after_notification_race(self):
        owned, journal = process(), Mock()
        owned.alive = Mock(return_value=True)
        with patch("runtime.OwnedTree") as tree, patch("runtime.Journal", return_value=journal), \
                patch("runtime.collect", side_effect=t.ExitPending(owned.identity, "/proc/41/ns/cgroup")), \
                patch("runtime.packet") as packet, patch("runtime.time.monotonic", return_value=0):
            runtime.monitor_child(owned, NAMESPACES, Mock(), Mock(), "G1", "/new/log", Mock(), KEY)
        self.assertEqual(packet.call_args.args[1], pending_packet())
        self.assertTrue(all(x.args[0] != "INTEGRATION_MONITOR_REFUSED_V1" for x in journal.emit.call_args_list))
        tree.return_value.close.assert_called_once()
        for rc in (0, 17):
            owned_job = job()
            with self.subTest(rc=rc), patch("runtime.receive", return_value=pending_packet()), \
                    patch("runtime.select.select", side_effect=[([owned_job.channel], [], []), ([], [], [])]), \
                    patch("runtime.time.monotonic", return_value=0), \
                    patch("runtime.os.waitpid", side_effect=[(0, 0), (41, rc << 8)]) as waited:
                self.assertEqual(owned_job.poll(), rc)
            self.assertEqual(owned_job.reaped, {41: rc})
            self.assertEqual(waited.call_count, 2)

    def test_unconfirmed_terminal_or_replacement_identity_cannot_finish(self):
        owned_job = job()
        with patch("runtime.receive", return_value=pending_packet()), \
                patch("runtime.select.select", side_effect=[([owned_job.channel], [], []), ([], [], [])]) as selected, \
                patch("runtime.time.monotonic", return_value=0), \
                patch("runtime.os.waitpid", return_value=(0, 0)), \
                self.assertRaisesRegex(t.CoverageError, "terminal-wait-unconfirmed"):
            owned_job.poll()
        self.assertEqual(selected.call_args.args[-1], 0.1)
        self.assertEqual(owned_job.reaped, {})
        with patch("runtime.receive", return_value={**pending_packet(), "start_ticks": 901}), \
                patch("runtime.select.select", return_value=([owned_job.channel], [], [])), \
                patch("runtime.time.monotonic", return_value=0), patch("runtime.os.waitpid") as waited, \
                self.assertRaises(t.CoverageError):
            owned_job.poll()
        waited.assert_not_called()

    def test_permission_schema_and_preceding_refusal_are_not_rewritten_as_exit(self):
        for error in (PermissionError(13, "denied", "ns/cgroup"), t.CoverageError("namespace-mismatch")):
            owned = process()
            owned.alive = Mock(side_effect=[True, False])
            with self.subTest(error=error), patch("runtime.OwnedTree"), patch("runtime.Journal"), \
                    patch("runtime.collect", side_effect=error), patch("runtime.packet") as packet, \
                    patch("runtime.time.monotonic", return_value=0):
                runtime.monitor_child(owned, NAMESPACES, Mock(), Mock(), "G1", "/new/log", Mock(), KEY)
            self.assertEqual(packet.call_args.args[1]["kind"], "REFUSED")
        owned = process()
        owned.verify = Mock(side_effect=[None, t.ExitPending(owned.identity, "/proc/41/exe")])
        with patch.object(t, "read_text", return_value=t.Observation(None, "unavailable", "EACCES")), \
                self.assertRaises(t.CoverageError) as error:
            owned.read("status")
        self.assertNotIsInstance(error.exception, t.ExitPending)
        owned_job = job()
        owned_job.root.alive.return_value = False
        with patch("runtime.receive", return_value={"kind": "REFUSED", "key": KEY}), \
                patch("runtime.select.select", return_value=([owned_job.channel], [], [])), \
                patch("runtime.time.monotonic", return_value=0), patch("runtime.os.waitpid") as waited, \
                self.assertRaises(t.CoverageError):
            owned_job.poll()
        waited.assert_not_called()

    def test_cleanup_uses_real_reap_before_signalling_and_adoption_reconciles_exit(self):
        owned_job = job()
        owned_job.adopt = Mock()
        tree = Mock(items={41: owned_job.root})
        tree.refresh.side_effect = t.ExitPending(owned_job.root.identity, "/proc/41/ns/cgroup")
        with patch("runtime.OwnedTree", return_value=tree), \
                patch("runtime.time.monotonic", return_value=0), patch("runtime.time.sleep"), \
                patch("runtime.os.waitpid", side_effect=[(41, 0), ChildProcessError()]):
            result = owned_job.stop("terminal", grace=0)
        self.assertTrue(result["complete"])
        owned_job.root.send.assert_not_called()
        self.assertEqual(result["reaped"], {41: 0})
        owned_job = job()
        owned_job.root.verify = Mock(side_effect=t.ExitPending(owned_job.root.identity, "/proc/41/exe"))
        guardian = Mock(identity=t.Identity(77, 777, 1, 7))
        guardian.children.return_value = [41]
        with patch("runtime.identity", return_value=guardian.identity), \
                patch.object(t.OwnedProcess, "pin", return_value=guardian), \
                patch("runtime.os.waitid", return_value=SimpleNamespace(si_pid=41)), \
                patch("runtime.os.waitpid", return_value=(41, 0)):
            owned_job.adopt()
        self.assertEqual(owned_job.reaped, {41: 0})

    def test_serve_reaches_completion_module_without_retain_local_shadowing(self):
        config = {"role": "ROLE_ENGINE_A", "run_id": "r", "gpu_completion": True,
                  "implementation": c.IMPLEMENTATION, "run_root": "/new/run", "namespaces": {},
                  "bindings": {"EXPECTED_GPU_PLAN_A_SHA": c.PLAN_SHA}}
        channel = Mock()
        request = {"run_id": "r", "sequence": 0, "nonce": "nonce", "remaining_s": 1000, "action": "CANCEL"}
        with patch("agent.os.setsid"), patch("agent.signal.signal"), patch("agent.os.mkdir"), \
                patch("agent.Journal"), patch("agent.OuterGuards"), patch("agent.Session"), \
                patch("agent.uuid.uuid4", return_value=SimpleNamespace(hex="nonce")), \
                patch("agent.time.monotonic", return_value=0), \
                patch("agent.select.select", return_value=([channel], [], [])), \
                patch("agent.receive", return_value=request), patch("agent.packet") as packet:
            agent.serve(config, channel)
        self.assertNotIn("completion", agent.serve.__code__.co_varnames)
        self.assertEqual(packet.call_args.args[1]["kind"], "FINAL")
        self.assertIn("controller-cancel", packet.call_args.args[1]["reason"])

    def test_g1_only_controller_preserves_g0_failure_as_reference(self):
        owned = peers()
        with patch("controller.verified_file") as verify:
            ctl = controller.Controller(owned, gpu_completion=True, baseline=baseline(), clock=lambda: 0, start=0)
        self.assertEqual(verify.call_count, 7)
        ctl.wait = Mock(return_value={"rc": 0, "cleanup": {"complete": True}, "reason": "finished"})
        ctl.gap = Mock()
        result = ctl.run()
        self.assertEqual(result["completed_phases"], ["G1"])
        self.assertEqual(result["g0_reference_supervisor_status"], "FAILED")
        for role in ("ROLE_ENGINE_A", "ROLE_GPU_B"):
            self.assertEqual([x.kwargs["phase"] for x in owned[role].queue.call_args_list
                              if x.args == ("START",)], ["G1"])
        ctl.gap.assert_not_called()

    def test_g0_reference_is_real_product_complete_not_a_supervisor_pass(self):
        valid = baseline()
        c.validate_baseline(valid, Mock())
        for field, value in (("supervisor_status", "PASS"), ("product_complete", False), ("rc", -15)):
            bad = baseline()
            bad["G0"][field] = value
            with self.subTest(field=field), self.assertRaises(t.CoverageError):
                c.validate_baseline(bad, Mock())
        bad = baseline()
        bad["g0_archive_sha256"] = "bad"
        with self.assertRaises(t.CoverageError):
            c.validate_baseline(bad, Mock())

    def test_g1_full_gpu_cache_has_zero_transfer_and_fresh_cpu_is_not_prior_reuse(self):
        self.assertEqual(c.cache_accounting("G1", reused=c.ASSIGNED_BYTES, transferred=0,
            persisted=0, source_read=0, cached=c.ASSIGNED_BYTES, assigned=c.ASSIGNED_BYTES)["cache_state"], "warm")
        with self.assertRaises(t.CoverageError):
            c.cache_accounting("G1", reused=16011878400, transferred=1164595200,
                persisted=1164595200, source_read=1164595200, cached=c.ASSIGNED_BYTES, assigned=c.ASSIGNED_BYTES)
        cpu, _, _ = commands.product("ROLE_CPU_C", "CPU", Bindings(), gpu_completion=True)
        self.assertIn("--local-root", cpu)
        self.assertEqual(cpu[cpu.index("--memory-budget-mib") + 1], "106428")
        with self.assertRaises(t.CoverageError):
            commands.product("ROLE_ENGINE_A", "G0", Bindings(), gpu_completion=True)

    def test_twenty_minute_window_clips_fresh_caps_without_running_extension(self):
        with patch("controller.verified_file"):
            ctl = controller.Controller(peers(), gpu_completion=True, baseline=baseline(), clock=lambda: 0, start=0)
        self.assertEqual((ctl.deadline, ctl.phase_cap(1200), c.HISTORICAL_ACTIVE_S), (1200, 1150, 5744.2))
        for role in ("ROLE_ENGINE_A", "ROLE_GPU_B"):
            _, _, cap = commands.product(role, "G1", Bindings(), gpu_completion=True)
            self.assertEqual(cap, 1200)
        self.assertEqual(c.SETUP_MS, 1800000)  # Effective window remains the tighter bound.

    def test_only_two_remaining_requests_and_original_numeric_warning(self):
        self.assertEqual(c.PHASES, ("G1",))
        self.assertEqual(commands.requests("G1"), b"GEN 64 9419\nGEN 64 14773,6353\nQUIT\n")
        self.assertEqual((9 + 2, 576 + 128), (11, 704))
        self.assertGreater(0.0352, 0.03)  # Historical informational value is not a PASS.


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    def expired(*_):
        raise KeyboardInterrupt("focused-g1-exit-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = defaultTestLoader.loadTestsFromTestCase(G1ExitTests)
    forbidden = AssertionError("live operation forbidden in G1 exit fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.stat", side_effect=forbidden), patch("os.readlink", side_effect=forbidden), \
            patch("os.fork", side_effect=forbidden), patch("os.waitid", side_effect=forbidden), \
            patch("os.waitpid", side_effect=forbidden), patch("os.kill", side_effect=forbidden), \
            patch("os.pidfd_open", side_effect=forbidden), patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 12:
        raise SystemExit(1)
    print("G1_EXIT_FIXTURES_PASS_V1 tests=12 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
