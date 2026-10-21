"""Runner synthetic fixture sources; authored, NOT RUN."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import admission
import stage_entry


class RunnerTests(unittest.TestCase):
    def test_synthetic_entry_has_no_live_gate(self):
        with patch("stage_entry.runpy.run_path") as run, patch("builtins.print"), \
                patch("admission.live_admission", side_effect=AssertionError("live forbidden")):
            stage_entry.synthetic()
        self.assertEqual([Path(c.args[0]).name for c in run.call_args_list],
                         ["test_runner.py", "test_integration.py"])

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


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RunnerTests)
    with patch("os.fork", side_effect=AssertionError("live fork forbidden")), \
            patch("os.kill", side_effect=AssertionError("live signal forbidden")), \
            patch("os.pidfd_open", side_effect=AssertionError("live pidfd forbidden")), \
            patch("signal.pidfd_send_signal", side_effect=AssertionError("live signal forbidden")), \
            patch("subprocess.Popen", side_effect=AssertionError("live process forbidden")), \
            patch("socket.socket", side_effect=AssertionError("live socket forbidden")):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 5:
        raise SystemExit(1)
    print("NAMESPACE_RUNNER_FIXTURES_PASS_V1 tests=5 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
