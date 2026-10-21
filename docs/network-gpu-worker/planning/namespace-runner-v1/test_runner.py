"""Runner synthetic fixture sources; authored, NOT RUN."""
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import admission
import stage_entry
import runner
import owned_child


class RunnerTests(unittest.TestCase):
    def receipt(self, **changes):
        value = {"schema": 1, "role": "ROLE_CPU_C", "host_identity": "SYNTHETIC_HOST",
                 "machine_id_sha256": "a" * 64, "boot_id": "00000000-0000-0000-0000-000000000000",
                 "issued_utc": "2000-01-01T00:00:00Z", "source_id": "SYNTHETIC_SOURCE",
                 "method": "synthetic fixture only", "namespaces": {k: [1, 2] for k in ("pid", "mnt", "cgroup")}}
        value.update(changes)
        raw = json.dumps(value).encode()
        pins = {"digest": hashlib.sha256(raw).hexdigest(), "source": "SYNTHETIC_SOURCE",
                "host": "SYNTHETIC_HOST", "machine": "a" * 64}
        return raw, pins

    def test_synthetic_entry_has_no_live_gate(self):
        with patch("stage_entry.runpy.run_path") as run, patch("builtins.print"), \
                patch("admission.live_admission", side_effect=AssertionError("live forbidden")):
            stage_entry.synthetic()
        self.assertEqual([Path(c.args[0]).name for c in run.call_args_list],
                         ["test_runner.py", "test_integration.py"])
        evidence, budget = Mock(), Mock()
        budget.limited.return_value = nullcontext()
        args = SimpleNamespace(mode="synthetic-only", manifest_sha256="a" * 64, interpreter_sha256="b" * 64)
        facts = {"executable": "/generic/python", "executable_identity": [1, 2]}
        with patch("runner.artifact_admission", return_value=facts), \
                patch("runner.stage", return_value={"status": "PASS"}), \
                patch("runner.live_admission", side_effect=AssertionError("live forbidden")):
            report = {"stages": []}
            self.assertEqual(runner.execute(args, evidence, budget, report), "PASS_SYNTHETIC_ONLY")
        self.assertEqual(report["live_result"]["status"], "NOT_REQUESTED")

    def test_missing_receipt_probes_no_live_facts(self):
        with patch("admission.os.stat", side_effect=AssertionError("live forbidden")):
            value = admission.live_admission(None, {}, 0)
        self.assertEqual(value["reason"], "receipt-required")

    def test_untrusted_receipt_probes_no_live_facts(self):
        with patch("admission.regular_bytes", side_effect=AssertionError("read forbidden")):
            value = admission.live_admission("/generic/receipt", {}, 0)
        self.assertEqual(value["reason"], "externally-approved-pins-required")

    def test_permission_is_partial_observation(self):
        with patch("admission.regular_bytes", side_effect=PermissionError(13, "denied")):
            value = admission.live_admission("/generic/receipt", {k: "pin" for k in ("digest", "host", "source", "machine")}, 0)
        self.assertEqual(value["observations"][0]["status"], "permission-denied")
        self.assertEqual(value["status"], "BLOCKED")

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            admission.strict_json(b'{"key":1,"key":2}')

    def test_valid_externally_pinned_receipt(self):
        raw, pins = self.receipt()
        self.assertEqual(admission.parse_receipt(raw, pins, 946684801)["role"], "ROLE_CPU_C")

    def test_wrong_host(self):
        raw, pins = self.receipt(host_identity="OTHER_SYNTHETIC_HOST")
        with self.assertRaisesRegex(admission.Blocked, "wrong-host"):
            admission.parse_receipt(raw, pins, 946684801)

    def test_wrong_approved_source(self):
        raw, pins = self.receipt(source_id="OTHER_SYNTHETIC_SOURCE")
        with self.assertRaisesRegex(admission.Blocked, "untrusted"):
            admission.parse_receipt(raw, pins, 946684801)

    def test_digest_is_exact_bytes(self):
        raw, pins = self.receipt()
        with self.assertRaisesRegex(admission.Blocked, "digest-mismatch"):
            admission.parse_receipt(raw + b"\n", pins, 946684801)

    def test_boolean_namespace_is_not_integer(self):
        raw, pins = self.receipt(namespaces={k: [True, 2] for k in ("pid", "mnt", "cgroup")})
        with self.assertRaisesRegex(admission.Blocked, "malformed"):
            admission.parse_receipt(raw, pins, 946684801)

    def test_wrong_role(self):
        raw, pins = self.receipt(role="ROLE_ENGINE_A")
        with self.assertRaisesRegex(admission.Blocked, "wrong-role"):
            admission.parse_receipt(raw, pins, 946684801)

    def test_schema_extra_key_and_boolean_version(self):
        for changes in ({"extra": 1}, {"schema": True}):
            raw, pins = self.receipt(**changes)
            with self.assertRaisesRegex(admission.Blocked, "malformed"):
                admission.parse_receipt(raw, pins, 946684801)

    def test_future_receipt(self):
        raw, pins = self.receipt()
        with self.assertRaisesRegex(admission.Blocked, "future"):
            admission.parse_receipt(raw, pins, 946684799)

    def test_stale_receipt(self):
        raw, pins = self.receipt()
        with self.assertRaisesRegex(admission.Blocked, "stale"):
            admission.parse_receipt(raw, pins, 946771201)

    def test_reboot_mismatch(self):
        raw, pins = self.receipt()
        with patch("admission.regular_bytes", return_value=raw), \
                patch("admission.machine_digest", return_value="a" * 64), \
                patch("admission.proc_text", return_value="OTHER_BOOT"), \
                patch("admission.os.stat", return_value=SimpleNamespace(st_dev=1, st_ino=2)):
            result = admission.live_admission("/generic/receipt", pins, 946684801)
        self.assertEqual(result["reason"], "stale-reboot")

    def test_live_namespace_mismatch(self):
        raw, pins = self.receipt()
        with patch("admission.regular_bytes", return_value=raw), \
                patch("admission.machine_digest", return_value="a" * 64), \
                patch("admission.proc_text", return_value="00000000-0000-0000-0000-000000000000"), \
                patch("admission.os.stat", return_value=SimpleNamespace(st_dev=1, st_ino=3)):
            result = admission.live_admission("/generic/receipt", pins, 946684801)
        self.assertEqual(result["reason"], "namespace-mismatch")

    def test_live_fact_permission_partial_evidence(self):
        raw, pins = self.receipt()
        with patch("admission.regular_bytes", return_value=raw), \
                patch("admission.machine_digest", return_value="a" * 64), \
                patch("admission.proc_text", side_effect=PermissionError(13, "denied")), \
                patch("admission.os.stat", return_value=SimpleNamespace(st_dev=1, st_ino=2)):
            result = admission.live_admission("/generic/receipt", pins, 946684801)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(len(result["observations"]), 7)
        self.assertIn("permission-denied", result["reason"])

    def test_zero_size_proc_file_is_readable(self):
        with patch("admission.os.open", return_value=7), patch("admission.os.close"), \
                patch("admission.os.fstat", return_value=SimpleNamespace(st_mode=0o100444, st_size=0)), \
                patch("admission.os.read", side_effect=[b"synthetic\n", b""]):
            self.assertEqual(admission.proc_text("/generic/proc", 128), "synthetic")

    def test_nonregular_receipt_rejected(self):
        with patch("admission.os.open", return_value=7), patch("admission.os.close"), \
                patch("admission.os.fstat", return_value=SimpleNamespace(st_mode=0o010600, st_size=10)):
            with self.assertRaises(admission.Blocked):
                admission.regular_bytes("/generic/receipt", 16384)

    def test_full_stage_must_fit_before_reserve(self):
        budget = runner.Budget(0)
        with patch("runner.time.monotonic", return_value=85):
            budget.admit_stage()
        with patch("runner.time.monotonic", return_value=85.001):
            with self.assertRaises(admission.Blocked):
                budget.admit_stage()
        with patch("runner.time.monotonic", return_value=146), patch("runner.signal.setitimer") as timer:
            with self.assertRaises(admission.Blocked):
                with budget.limited(5):
                    self.fail("late admission released")
        timer.assert_not_called()

    def test_no_sealing_write_after_deadline(self):
        evidence = object.__new__(runner.Evidence)
        evidence.budget = runner.Budget(0)
        with patch("runner.time.monotonic", return_value=210), patch("runner.os.open") as opened:
            with self.assertRaises(runner.Deadline):
                evidence.write("REPORT.json", b"{}")
        opened.assert_not_called()
        with self.assertRaises(runner.Deadline):
            admission.observe("timed", Mock(side_effect=runner.Deadline("expired")))

    def test_runner_pass_preserved_when_integration_fails(self):
        raw = ("RUNNER_FIXTURES_BEGIN_V1\n" + runner.RUNNER_MARKER + "\nINTEGRATION_44_BEGIN_V1\n").encode()
        result = runner.classify("synthetic", 1, raw, {"complete": True})
        self.assertEqual(result["runner_fixtures"], "PASS")
        self.assertEqual(result["integration_44"], "FAIL")
        self.assertEqual(result["status"], "FAIL")

    def test_duplicate_markers_are_not_pass(self):
        raw = ("RUNNER_FIXTURES_BEGIN_V1\n" + (runner.RUNNER_MARKER + "\n") * 2 +
               "INTEGRATION_44_BEGIN_V1\n" + runner.INTEGRATION_MARKER + "\n").encode()
        self.assertEqual(runner.classify("synthetic", 0, raw, {"complete": True})["status"], "FAIL")

    def test_pid_reuse_or_start_change_rejected(self):
        child = object.__new__(owned_child.Child)
        child.procfd, child.identity, child.executable_identity = 7, (11, 100), (1, 2)
        with patch("owned_child.parse_stat", return_value=(11, 22, 101)), \
                patch("owned_child.read_text"), patch("owned_child.os.getpid", return_value=22), \
                patch("owned_child.os.stat", return_value=SimpleNamespace(st_dev=1, st_ino=2)):
            with self.assertRaises(owned_child.OwnershipError):
                child.verify()
        child.reaped, child.pidfd = False, 7
        child.alive = Mock(side_effect=[True, False])
        child.verify = Mock(side_effect=FileNotFoundError("terminal exe"))
        child.harvest = Mock(return_value=0)
        self.assertEqual(child.poll(), 0)
        child.alive = Mock(return_value=True)
        child.verify = Mock(side_effect=owned_child.OwnershipError("identity-changed"))
        with self.assertRaises(owned_child.OwnershipError):
            child.poll()

    def test_cleanup_failure_prevents_success(self):
        raw = ("RUNNER_FIXTURES_BEGIN_V1\n" + runner.RUNNER_MARKER +
               "\nINTEGRATION_44_BEGIN_V1\n" + runner.INTEGRATION_MARKER + "\n").encode()
        self.assertEqual(runner.classify("synthetic", 0, raw, {"complete": False})["status"], "FAIL")
        child, evidence, budget = Mock(), Mock(), Mock()
        child.streams, child.rc, child.pid, child.identity, child.started = {}, 0, 11, (11, 1), 0
        child.poll.return_value = 0
        child.stop.return_value = {"complete": True}
        evidence.log.side_effect = [7, 8]
        budget.work_end = 150
        with patch("runner.Child", return_value=child), \
                patch("runner.select.select", return_value=([], [], [])), \
                patch("runner.os.close", side_effect=[OSError("fixture"), None]) as close:
            result = runner.stage("synthetic", [], (1, 2), evidence, budget, {"stages": []})
        self.assertEqual(close.call_count, 2)
        self.assertEqual(result["reason"], "log-close-failed")
        self.assertIn("end", result)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests)
    with patch("os.fork", side_effect=AssertionError("live fork forbidden")), \
            patch("os.kill", side_effect=AssertionError("live signal forbidden")), \
            patch("os.pidfd_open", side_effect=AssertionError("live pidfd forbidden")), \
            patch("signal.pidfd_send_signal", side_effect=AssertionError("live signal forbidden")), \
            patch("subprocess.Popen", side_effect=AssertionError("live process forbidden")), \
            patch("socket.socket", side_effect=AssertionError("live socket forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 25:
        raise SystemExit(1)
    print("NAMESPACE_RUNNER_FIXTURES_PASS_V1 tests=25 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
