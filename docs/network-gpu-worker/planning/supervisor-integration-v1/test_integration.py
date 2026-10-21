"""Synthetic sources only. No live process, socket, GPU, service or model."""
import copy
from contextlib import ExitStack
import json
import signal
import struct
import unittest
from unittest.mock import Mock, patch

import cache_status
import commands
import controller
import live_probe
import runtime
from outer import OuterGuards
from policy import (GIB, MIB, POLICIES, ReleaseGate, RoleGuard, Stamp, TrialOrder,
                    local_deadline, retention_deadline, accounting_deadline)
from resource_guard import CoverageError, GuardStop, EVENT_KEYS
from resource_telemetry import Observation


def sample(available=130 * GIB, rss=MIB, anon=MIB, free=6 * GIB):
    return {"host_bytes": {"MemAvailable": Observation(available, "ok", ""),
                           "SwapTotal": Observation(8 * GIB, "ok", ""),
                           "SwapFree": Observation(free, "ok", "")},
            "worker_memory_bytes": {"VmRSS": Observation(rss, "ok", ""),
                                     "RssAnon": Observation(anon, "ok", "")},
            "cgroup": {"membership": "/fixture", "mount": ("fixture",),
                       "namespace_receipt": {}, "rows": [
                           {"path": "/fixture", "directory_identity": (1, 2),
                            "maximum": None, "high": None, "current": MIB,
                            "coverage": "required-covered",
                            "events": {k: Observation(0, "ok", "") for k in EVENT_KEYS}}]}}


class Clock:
    def __init__(self, at=0, step=0):
        self.at, self.step = at, step

    def __call__(self):
        value = self.at
        self.at += self.step
        return value


class PolicyTests(unittest.TestCase):
    def test_distinct_role_admission(self):
        value = sample(available=12 * GIB)
        RoleGuard("ROLE_ENGINE_A", value).check("ROLE_ENGINE_A", value, True)
        with self.assertRaises(GuardStop):
            RoleGuard("ROLE_CPU_C", value).check("ROLE_CPU_C", value, True)

    def test_role_mixup(self):
        with self.assertRaises(CoverageError):
            RoleGuard("ROLE_ENGINE_A", sample()).check("ROLE_CPU_C", sample())

    def test_exact_published_caps(self):
        self.assertEqual(POLICIES["ROLE_CPU_C"].admission, 123384 * MIB)
        self.assertEqual(POLICIES["ROLE_CPU_C"].rss, 108024 * MIB)
        self.assertEqual(POLICIES["ROLE_ENGINE_A"].anonymous, 8 * GIB)
        self.assertEqual(POLICIES["ROLE_GPU_B"].rss, 2 * GIB)

    def test_swap_strict_boundary(self):
        baseline = sample()
        guard = RoleGuard("ROLE_ENGINE_A", baseline)
        guard.check("ROLE_ENGINE_A", sample(free=6 * GIB - 64 * MIB))
        with self.assertRaises(GuardStop):
            guard.check("ROLE_ENGINE_A", sample(free=6 * GIB - 64 * MIB - 1))

    def test_swap_anchor_never_rolls(self):
        guard = RoleGuard("ROLE_ENGINE_A", sample())
        guard.check("ROLE_ENGINE_A", sample(free=6 * GIB - 40 * MIB))
        with self.assertRaises(GuardStop):
            guard.check("ROLE_ENGINE_A", sample(free=6 * GIB - 80 * MIB))

    def test_missing_required_is_not_zero(self):
        value = sample()
        value["host_bytes"]["MemAvailable"] = Observation()
        with self.assertRaises(CoverageError):
            RoleGuard("ROLE_CPU_C", sample()).check("ROLE_CPU_C", value)

    def test_unlimited_still_requires_events(self):
        value = sample()
        value["cgroup"]["rows"][0]["events"]["oom"] = Observation()
        with self.assertRaises(CoverageError):
            RoleGuard("ROLE_GPU_B", sample()).check("ROLE_GPU_B", value)

    def test_new_event_stops(self):
        value = sample()
        value["cgroup"]["rows"][0]["events"]["high"] = Observation(1, "ok", "")
        with self.assertRaises(GuardStop):
            RoleGuard("ROLE_ENGINE_A", sample()).check("ROLE_ENGINE_A", value)

    def test_tightest_parent_limit(self):
        value = sample()
        row = copy.deepcopy(value["cgroup"]["rows"][0])
        row.update(path="/parent", maximum=10 * GIB, current=7 * GIB)
        value["cgroup"]["rows"].append(row)
        with self.assertRaises(GuardStop):
            RoleGuard("ROLE_ENGINE_A", value).check("ROLE_ENGINE_A", value)

    def test_rss_and_anonymous_distinct(self):
        with self.assertRaises(GuardStop):
            RoleGuard("ROLE_ENGINE_A", sample()).check("ROLE_ENGINE_A", sample(anon=8 * GIB + 1))


class ClockGateTests(unittest.TestCase):
    def test_foreign_clock_rejected(self):
        with self.assertRaises(CoverageError):
            local_deadline(Stamp("other", 10, 1000), 20, 600, "local")

    def test_wall_time_does_not_change_deadline(self):
        self.assertEqual(local_deadline(Stamp("local", 10, -1000000), 20, 600, "local"), 610)

    def test_attach_preserves_original_deadline(self):
        gate = ReleaseGate("run", "role", "phase", 500, 600)
        with self.assertRaises(GuardStop):
            gate.check(585)
        self.assertEqual(gate.deadline, 600)

    def test_release_requires_watchdog_and_coverage(self):
        gate = ReleaseGate("run", "role", "phase", 0, 600)
        with self.assertRaises(CoverageError):
            gate.release(gate.key, 1)
        gate.watchdog = True
        gate.covered(gate.key, 1)
        gate.release(gate.key, 2)
        self.assertTrue(gate.released)
        with self.assertRaises(CoverageError):
            gate.release(gate.key, 2)

    def test_stale_start_and_monitor_stall(self):
        gate = ReleaseGate("run", "role", "phase", 0, 600)
        with self.assertRaises(GuardStop):
            gate.check(11)
        gate.watchdog = True
        gate.covered(gate.key, 1)
        gate.release(gate.key, 2)
        with self.assertRaises(GuardStop):
            gate.check(5.01)

    def test_wrong_session_cannot_release(self):
        gate = ReleaseGate("run", "role", "phase", 0, 600)
        gate.watchdog = True
        with self.assertRaises(CoverageError):
            gate.covered(("old", "role", "phase"), 1)

    def test_retention_earliest_all_three_caps(self):
        stamp = lambda at: Stamp("local", at, 0)
        self.assertEqual(retention_deadline(stamp(0), stamp(7100), stamp(7110), 1800000, "local"), 7200)
        self.assertEqual(retention_deadline(stamp(0), stamp(100), stamp(110), 1800000, "local"), 1900)
        self.assertEqual(retention_deadline(stamp(0), stamp(100), stamp(110), 1000, "local"), 111)

    def test_retention_rejects_foreign_receipt(self):
        with self.assertRaises(CoverageError):
            retention_deadline(Stamp("a", 0, 0), Stamp("b", 1, 0), Stamp("a", 2, 0), 1, "a")

    def test_accounting_delay_clips_to_original_bounds(self):
        self.assertEqual(accounting_deadline(100, 200, 300), 130)
        self.assertEqual(accounting_deadline(100, 110, 300), 110)
        self.assertEqual(accounting_deadline(100, 200, 120), 105)


class StatusTests(unittest.TestCase):
    def test_wire_offsets_and_idle_zero_session(self):
        payload = bytearray(100)
        struct.pack_into("<IIQQQ", payload, 0, 1, 0, 0, 111597465600, 1000)
        struct.pack_into("<I", payload, 96, 1)
        header = struct.pack("<IHHQ", cache_status.MAGIC, 17, 0, 100)
        self.assertEqual(cache_status.decode(header, payload), 1000)
        payload[96] = 0
        with self.assertRaises(CoverageError):
            cache_status.decode(header, payload)

    def test_fragmented_reply(self):
        sock = Mock()
        sock.recv.side_effect = [b"a", b"b", b"c"]
        self.assertEqual(cache_status.read_exact(sock, 3, 1, Clock()), b"abc")

    def test_trickle_exhausts_total_not_per_read_budget(self):
        sock = Mock()
        sock.recv.return_value = b"x"
        with self.assertRaises(CoverageError):
            cache_status.read_exact(sock, 100, 1, Clock(step=.4))
        self.assertEqual(sock.recv.call_count, 3)

    def test_expired_deadline_performs_no_receive(self):
        sock = Mock()
        with self.assertRaises(CoverageError):
            cache_status.read_exact(sock, 1, 1, Clock(1))
        sock.recv.assert_not_called()

    def test_short_reply_is_refusal(self):
        sock = Mock()
        sock.recv.return_value = b""
        with self.assertRaises(CoverageError):
            cache_status.read_exact(sock, 1, 1, Clock())


class OwnershipTests(unittest.TestCase):
    def test_independent_probe_guard_does_not_walk_exiting_children(self):
        own = Mock()
        own.status.return_value = {"memory_bytes": {"VmRSS": Observation(MIB, "ok", "")}}
        with patch("live_probe.OwnedProcess.pin", return_value=own), \
                patch("live_probe.identity"), patch("live_probe.host_sample", return_value={}), \
                patch("live_probe.cgroup_snapshot", return_value={"coverage": "required-covered"}):
            value = live_probe.parent_sample({})
        own.children.assert_not_called()
        own.close.assert_called_once()
        self.assertEqual(value["worker_memory_bytes"], own.status.return_value["memory_bytes"])

    def job(self):
        job = object.__new__(runtime.Job)
        job.key = ("run", "ROLE_ENGINE_A", "B0")
        job.root, job.monitor = Mock(), Mock()
        job.root.identity.pid, job.monitor.identity.pid = 101, 102
        job.root.alive.return_value = False
        job.monitor.alive.return_value = False
        job.channel = Mock()
        job.last_sequence = -1
        job.gate = ReleaseGate(*job.key, 0, 100)
        job.items = {101: job.root, 102: job.monitor}
        job.children = {101, 102}
        job.reaped, job.transient_fds, job.transient_sockets = {}, set(), []
        job.fd_errors = []
        job.exec_gate, job.stop_result, job.deadline = None, None, 100
        job.adopt = Mock()
        return job

    def test_terminal_sampling_precedes_parent_poll(self):
        job = self.job()
        job.reap = Mock(return_value=0)
        with patch("runtime.time.monotonic", return_value=2), \
                patch("runtime.select.select", return_value=([job.channel], [], [])), \
                patch("runtime.receive", return_value={"kind": "EXITED", "key": job.key}):
            self.assertEqual(job.poll(), 0)

    def test_true_refusal_precedes_normal_exit(self):
        job = self.job()
        with patch("runtime.time.monotonic", return_value=2), \
                patch("runtime.select.select", return_value=([job.channel], [], [])), \
                patch("runtime.receive", return_value={"kind": "REFUSED", "key": job.key}):
            with self.assertRaises(CoverageError):
                job.poll()

    def test_parent_reaped_descendant_is_not_claimed_reaped(self):
        job = self.job()
        descendant = Mock()
        descendant.identity.pid = 103
        descendant.alive.return_value = False
        job.items[103] = descendant
        with patch("runtime.time.monotonic", Clock(step=.01)), \
                patch("runtime.os.waitpid", side_effect=[(101, 0), (102, 0), ChildProcessError()]):
            result = job.stop("fixture")
        self.assertTrue(result["complete"])
        self.assertEqual(result["exited_descendants_not_reaped_here"], [103])
        self.assertNotIn(103, result["reaped"])

    def test_escalation_and_real_parent_wait_receipts(self):
        job = self.job()
        for item in job.items.values():
            item.alive.return_value = True
            item.send.side_effect = lambda sig, p=item: setattr(p.alive, "return_value", sig != signal.SIGKILL)
        with patch("runtime.time.monotonic", Clock(step=.1)), patch("runtime.time.sleep"), \
                patch("runtime.OwnedTree", side_effect=lambda p: Mock(items={p.identity.pid: p}, refresh=lambda: [p])), \
                patch("runtime.os.waitpid", side_effect=[(101, 0), (102, 0), ChildProcessError()]):
            result = job.stop("cancel", grace=0)
        self.assertTrue(result["complete"])
        for item in job.items.values():
            self.assertEqual([c.args[0] for c in item.send.call_args_list], [signal.SIGTERM, signal.SIGKILL])
        self.assertIs(job.stop("again"), result)

    def test_partial_open_tracks_all_acquired_fds(self):
        captured = []
        def stop(job, reason):
            captured.append((reason, set(job.transient_fds)))
        with patch("runtime.subreaper"), patch("runtime.verify_initial_namespaces"), \
                patch("runtime.os.pipe2", return_value=(10, 11)), \
                patch("runtime.os.open", side_effect=[12, OSError("fixture")]), \
                patch.object(runtime.Job, "stop", stop):
            with self.assertRaises(runtime.PartialLaunch):
                runtime.Job(("r", "ROLE_ENGINE_A", "B0"), ["/generic/bin"], {}, 1, {}, None, None, 100, "/generic/run")
        self.assertEqual(captured, [("partial-launch", {10, 11, 12})])
        for stage in ("first_fork", "root_pin", "second_fork", "monitor_pin", "monitor_release"):
            with self.subTest(stage=stage), ExitStack() as stack:
                tracked = []
                writes = []
                def stopped(job, reason):
                    tracked.append((set(job.transient_fds), set(job.children)))
                    return {"complete": True}
                stack.enter_context(patch("runtime.subreaper"))
                stack.enter_context(patch("runtime.verify_initial_namespaces"))
                stack.enter_context(patch("runtime.os.pipe2", side_effect=[(10, 11), (14, 15)]))
                stack.enter_context(patch("runtime.os.open", side_effect=[12, 13]))
                stack.enter_context(patch("runtime.os.close"))
                forks = [OSError("fixture")] if stage == "first_fork" else \
                        [101, OSError("fixture")] if stage == "second_fork" else [101, 102]
                stack.enter_context(patch("runtime.os.fork", side_effect=forks))
                root, monitor = Mock(), Mock()
                pins = [OSError("fixture")] if stage == "root_pin" else \
                       [root, OSError("fixture")] if stage == "monitor_pin" else [root, monitor]
                stack.enter_context(patch("runtime.pin_child", side_effect=pins))
                stack.enter_context(patch("runtime.promote_exec", side_effect=lambda item, _: item))
                stack.enter_context(patch("runtime.socket.socketpair", return_value=(Mock(), Mock())))
                def write(fd, value):
                    writes.append(value)
                    raise OSError("fixture")
                stack.enter_context(patch("runtime.os.write", side_effect=write))
                stack.enter_context(patch.object(runtime.Job, "stop", stopped))
                with self.assertRaises(runtime.PartialLaunch):
                    runtime.Job(("r", "ROLE_ENGINE_A", "B0"), ["/generic/bin"], {}, 1,
                                {}, None, None, 100, "/generic/run")
                self.assertEqual(len(tracked), 1)
                self.assertIn(11, tracked[0][0])
                self.assertNotIn(b"G", writes)

    def test_fd_invalidated_even_when_close_fails(self):
        job = self.job()
        job.exec_gate = 7
        job.transient_fds.add(7)
        with patch("runtime.os.close", side_effect=OSError("fixture")):
            with self.assertRaises(OSError):
                job.close_fd(7)
        self.assertIsNone(job.exec_gate)
        self.assertNotIn(7, job.transient_fds)

    def test_required_monitor_failure_does_not_release(self):
        job = self.job()
        job.root.alive.return_value = True
        with patch("runtime.time.monotonic", return_value=2), \
                patch("runtime.select.select", return_value=([], [], [])):
            with self.assertRaises(CoverageError):
                job.poll()
        self.assertFalse(job.gate.released)

    def test_unexpected_pid_identity_change_refuses_sampling(self):
        job = self.job()
        job.root.alive.side_effect = CoverageError("owned-process-identity-changed")
        with patch("runtime.time.monotonic", return_value=2), \
                patch("runtime.select.select", return_value=([], [], [])):
            with self.assertRaises(CoverageError):
                job.poll()


class ControllerTests(unittest.TestCase):
    def peer(self):
        peer = object.__new__(controller.Peer)
        peer.role, peer.run_id = "ROLE_CPU_C", "run"
        peer.process = Mock()
        peer.buffer, peer.events = bytearray(), []
        peer.final_seen, peer.covered, peer.expected_finish = False, True, False
        return peer

    def test_other_role_failure_is_immediate(self):
        peer = self.peer()
        raw = {"kind": "FINAL", "run_id": "run", "role": peer.role,
               "reason": "guard-failed", "cleanup": {"complete": True}}
        with patch("controller.os.read", return_value=json.dumps(raw).encode() + b"\n"):
            with self.assertRaises(GuardStop):
                peer.pump(100)
        self.assertFalse(peer.covered)

    def test_expected_final_requires_successful_cleanup(self):
        peer = self.peer()
        peer.expected_finish = True
        raw = {"kind": "FINAL", "run_id": "run", "role": peer.role,
               "reason": "finished", "cleanup": {"complete": False}}
        with patch("controller.os.read", return_value=json.dumps(raw).encode() + b"\n"):
            with self.assertRaises(GuardStop):
                peer.pump(100)

    def test_cpu_incomplete_stop_and_failure_final_rejected(self):
        with self.assertRaises(GuardStop):
            controller.require_cleanup({"cleanup": {"complete": False}})
        with self.assertRaises(GuardStop):
            controller.require_cleanup({"cleanup": {"complete": True}, "reason": "failed"}, final=True)

    def test_close_failure_does_not_skip_other_owners(self):
        owners = [Mock(), Mock(), Mock()]
        owners[0].close.side_effect = OSError("fixture")
        with self.assertRaises(CoverageError):
            controller.close_all(owners)
        for owner in owners:
            owner.close.assert_called_once()

    def test_no_dependent_phase_after_failure(self):
        order = TrialOrder()
        with self.assertRaises(GuardStop):
            order.finish("B0", 1, True)
        with self.assertRaises(CoverageError):
            order.permit("B1", True, False)

    def test_missing_cpu_or_gpu_coverage_blocks_release(self):
        with self.assertRaises(CoverageError):
            TrialOrder().permit("B0", False, False)
        order = TrialOrder()
        order.next = 2
        with self.assertRaises(CoverageError):
            order.permit("G0", True, False)

    def test_failure_during_gap_prevents_next_start(self):
        peers = {"ROLE_CPU_C": Mock()}
        ctl = controller.Controller(peers, clock=Clock())
        ctl.pump = Mock(side_effect=GuardStop("cpu-failed"))
        with self.assertRaises(GuardStop):
            ctl.gap()
        peers["ROLE_CPU_C"].queue.assert_not_called()

    def test_failure_during_another_role_wait(self):
        cpu, engine = Mock(), Mock()
        engine.event.return_value = None
        ctl = controller.Controller({"ROLE_CPU_C": cpu, "ROLE_ENGINE_A": engine}, clock=Clock())
        ctl.pump = Mock(side_effect=GuardStop("cpu-failed"))
        with self.assertRaises(GuardStop):
            ctl.wait("ROLE_ENGINE_A", "DONE", "B0")
        engine.queue.assert_not_called()


class BindingTests(unittest.TestCase):
    def test_query_selection_mismatch_and_duplicates_refused(self):
        with self.assertRaises(CoverageError):
            OuterGuards("ROLE_ENGINE_A", "/generic/tool", ["A", "B"], selection="B,A")
        with self.assertRaises(CoverageError):
            OuterGuards("ROLE_GPU_B", "/generic/tool", ["A", "A"], selection="A,A")

    def test_engine_environment_is_explicit(self):
        class Bindings(dict):
            def __missing__(self, key):
                return "/generic/" + key
        _, env, cap = commands.product("ROLE_ENGINE_A", "B0", Bindings())
        self.assertEqual(cap, 1200)
        self.assertNotIn("STRATA_PRIMARY_DEVICE", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "/generic/GPU_SELECT_A")
        self.assertEqual(commands.REQUESTS.count(b"GEN 64"), 3)

    def test_gpu_return_can_be_delayed_without_resetting_baseline(self):
        outer = OuterGuards("ROLE_ENGINE_A", "/generic/tool", ["A", "B"], selection="A,B")
        outer.baseline = [12 * GIB, 12 * GIB]
        outer.sample = Mock(side_effect=[{"gpu_free_bytes": [11 * GIB, 12 * GIB]},
                                         {"gpu_free_bytes": [12 * GIB, 12 * GIB]}])
        self.assertFalse(outer.returned())
        self.assertTrue(outer.returned())
        self.assertEqual(outer.baseline, [12 * GIB, 12 * GIB])


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    # Any accidental process/network execution in synthetic tests fails loudly.
    with patch("os.fork", side_effect=AssertionError("live fork forbidden")), \
            patch("subprocess.Popen", side_effect=AssertionError("live process forbidden")), \
            patch("os.kill", side_effect=AssertionError("live signal forbidden")), \
            patch("signal.pidfd_send_signal", side_effect=AssertionError("live signal forbidden")), \
            patch("socket.socket", side_effect=AssertionError("live socket forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 44:
        raise SystemExit(1)
    print("SUPERVISOR_INTEGRATION_FIXTURES_PASS_V1 tests=" + str(result.testsRun) +
          " live_signals=0 workloads=0")


if __name__ == "__main__":
    main()
