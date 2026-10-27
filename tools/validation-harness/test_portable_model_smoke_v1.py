"""Report-admission checks only; these tests do not claim device execution."""
import csv
import importlib.util
import pathlib
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
