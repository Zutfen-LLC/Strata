"""Report-admission checks only; these tests do not claim device execution."""
import csv
import json
import signal
import sys
import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("portable_model_smoke_v1", pathlib.Path(__file__).with_name("portable_model_smoke_v1.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class PlacementAdmission(unittest.TestCase):
    def rows(self):
        result = []
        for layer in range(49):
            backend = "Vulkan0" if layer < 32 else "Vulkan1"
            if layer == 48:
                operations = [("global-head", "MUL_MAT")]
            elif layer % 4 == 3:
                operations = [("qsa-attend", "MUL_MAT")]
            else:
                operations = [("gdn-attention", "GATED_DELTA_NET"), ("gdn-attention", "SSM_CONV")]
            for phase, operation in operations:
                result.append(dict(layer=str(layer), phase=phase, op=operation, tensor="fixture",
                                   type="F32", backend=backend, metadata="0", fallback=""))
        result.append(dict(layer="0", phase="host", op="ROUTE_AND_MERGE", tensor="fixture", type="F32",
                           backend="CPU", metadata="0", fallback=""))
        return result

    def admit(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "placements.tsv"
            with path.open("w") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            return smoke.placements(path, [0, 1], 32)

    def test_contiguous_both_devices_and_intentional_host_work(self):
        self.assertTrue(all(count > 0 for count in self.admit(self.rows()).values()))
        self.assertEqual(len(smoke.PROMPT), 38)

    def test_cpu_tensor_fallback_rejected(self):
        rows = self.rows()
        rows[0]["backend"] = "CPU"
        with self.assertRaisesRegex(ValueError, "backend/fallback"):
            self.admit(rows)

    def test_head_on_wrong_die_rejected(self):
        rows = self.rows()
        next(row for row in rows if row["layer"] == "48")["backend"] = "Vulkan0"
        with self.assertRaisesRegex(ValueError, "backend/fallback"):
            self.admit(rows)

    def test_qsa_projection_without_attention_is_rejected(self):
        rows = self.rows()
        next(row for row in rows if row["layer"] == "3")["phase"] = "qsa-project"
        with self.assertRaisesRegex(ValueError, "no executed QSA attention"):
            self.admit(rows)


class HostAdmission(unittest.TestCase):
    WORKING_SET = 188225033248

    def host(self, available=249 << 30):
        return dict(MemTotal=251 << 30, MemAvailable=available, SwapTotal=32 << 30,
                    SwapFree=31 << 30, SwapUsed=1 << 30, pswpin=1234, pswpout=5678,
                    cgroup={"limits": [], "available": None}, effective_available=available)

    def test_full_working_set_anonymous_and_reserve_terms(self):
        terms = smoke.admit_host(self.host(), self.WORKING_SET)
        self.assertEqual(terms["required_available_bytes"], self.WORKING_SET + (32 << 30) + (8 << 30))
        self.assertEqual(terms["anonymous_cap_bytes"], 32 << 30)
        self.assertEqual(terms["reserve_bytes"], 8 << 30)

    def test_low_host_memory_prevents_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory) / "evidence"
            argv = ["smoke", "--binary", sys.executable, "--evidence", str(evidence), "--timeout-seconds", "1"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(smoke, "model_working_set", return_value={"working_set_bound_bytes": self.WORKING_SET}), \
                    mock.patch.object(smoke, "host_memory", return_value=self.host(214 << 30)), \
                    mock.patch.object(smoke.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(ValueError, "insufficient host/cgroup headroom"):
                    smoke.main()
                launch.assert_not_called()
            summary = json.loads((evidence / "summary.json").read_text())
            self.assertFalse(summary["launched"])
            self.assertEqual(summary["baseline_host"]["MemAvailable"], 214 << 30)
            self.assertEqual(summary["host_admission"]["required_available_bytes"], self.WORKING_SET + (40 << 30))

    def test_historical_swap_is_allowed_without_growth(self):
        baseline = self.host()
        smoke.guard_host(self.host(), baseline, {"VmSwap": 0})

    def test_child_swap_growth_stops(self):
        with self.assertRaisesRegex(ValueError, "child VmSwap"):
            smoke.guard_host(self.host(), self.host(), {"VmSwap": 4096})

    def test_system_swapout_growth_stops(self):
        current = self.host()
        current["pswpout"] += 1
        with self.assertRaisesRegex(ValueError, "system swap-out"):
            smoke.guard_host(current, self.host(), {"VmSwap": 0})

    def test_reserve_breach_stops_during_loading_and_kills_only_owned_group(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory) / "evidence"
            argv = ["smoke", "--binary", sys.executable, "--evidence", str(evidence), "--timeout-seconds", "1"]
            child = mock.Mock(pid=31337, returncode=None)
            child.poll.return_value = None
            def stopped(timeout):
                child.returncode = -signal.SIGTERM
                return child.returncode
            child.wait.side_effect = stopped
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(smoke, "model_working_set", return_value={"working_set_bound_bytes": self.WORKING_SET}), \
                    mock.patch.object(smoke, "host_memory", side_effect=[self.host(), self.host(7 << 30)]), \
                    mock.patch.object(smoke, "vram", return_value={"used": 0, "total": 8 << 30}), \
                    mock.patch.object(smoke, "rss", return_value={"VmSwap": 0, "VmRSS": 1024}), \
                    mock.patch.object(smoke.subprocess, "Popen", return_value=child), \
                    mock.patch.object(smoke.os, "killpg") as kill:
                with self.assertRaisesRegex(ValueError, "8 GiB reserve"):
                    smoke.main()
                kill.assert_called_once_with(31337, signal.SIGTERM)
                child.wait.assert_called_once_with(timeout=5)
            summary = json.loads((evidence / "summary.json").read_text())
            self.assertTrue(summary["launched"])
            self.assertEqual(summary["exit_status"], -signal.SIGTERM)
            self.assertEqual(summary["last_host"]["MemAvailable"], 7 << 30)
            sample = json.loads((evidence / "memory.jsonl").read_text().splitlines()[0])
            self.assertEqual(sample["host"]["MemAvailable"], 7 << 30)

    def test_finite_ancestor_cgroup_limits_are_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            parent = root / "limited"
            leaf = parent / "session"
            leaf.mkdir(parents=True)
            (leaf / "memory.max").write_text("max")
            (parent / "memory.max").write_text(str(220 << 30))
            (parent / "memory.current").write_text(str(10 << 30))
            limits = smoke.cgroup_limits(leaf, root)
            self.assertEqual(limits[0]["available"], 210 << 30)
            current = self.host()
            current["cgroup"] = {"limits": limits, "available": 210 << 30}
            current["effective_available"] = 210 << 30
            with self.assertRaisesRegex(ValueError, "insufficient host/cgroup headroom"):
                smoke.admit_host(current, self.WORKING_SET)
            current["effective_available"] = 7 << 30
            with self.assertRaisesRegex(ValueError, "8 GiB reserve"):
                smoke.guard_host(current, self.host(), {"VmSwap": 0})

    def test_working_set_is_sum_of_unique_stat_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for number in range(1, 7):
                (root / f"Qwen3.8-Flash-Next-Q8_0-0000{number}-of-00006.gguf").write_bytes(b"x" * number)
            with mock.patch.object(smoke, "MODEL_ROOT", root):
                self.assertEqual(smoke.model_working_set()["working_set_bound_bytes"], 21)


if __name__ == "__main__":
    unittest.main()
