"""Real CPU native/ggml graph host checks. No GPU/model/network required."""
import csv
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def run(binary, *args, ok=True):
    result = subprocess.run([binary, *map(str, args)], capture_output=True, text=True, timeout=45)
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def main(binary):
    with tempfile.TemporaryDirectory(prefix="amd-q8-host-") as temp:
        root = Path(temp)
        outputs = [root / "first", root / "second"]
        for out in outputs:
            run(binary, "--backend", "cpu", "--out-dir", out)
            assert (out / "COMPLETED").is_file()
            assert (out / "expert.q8_0").stat().st_size == 5222400
            manifest = dict(line.split("=", 1) for line in (out / "manifest.txt").read_text().splitlines())
            assert manifest["weight_fnv1a64"] == "e6d5f545e2145434"
            assert manifest["signal_input_fnv1a64"] == "b46eda2ec728b1b9"
            assert manifest["zero_input_fnv1a64"] == "9213f0ec13614325"
            assert manifest["mode"] == "raw_f32"
            with (out / "metrics.tsv").open() as file:
                metrics = list(csv.DictReader(file, delimiter="\t"))
            assert len(metrics) == 24
            for row in metrics:
                assert int(row["values"]) == (2560 if row["stage"] == "out" else 640)
                assert math.isfinite(float(row["max_abs"]))
                assert math.isfinite(float(row["relative_L2"]))
                if row["comparison"].endswith("selected_vs_cpu_graph"):
                    assert int(row["bit_different_values"]) == 0
            for case in ("signal", "zero"):
                assert (out / f"{case}.input.f32").stat().st_size == 10240
                assert (out / f"{case}.native_input.q8_0").stat().st_size == 2720
                assert (out / f"{case}.native_hidden.q8_0").stat().st_size == 680
                for role in ("native", "cpu_graph", "selected"):
                    for stage, count in (("gate", 640), ("up", 640), ("hidden", 640), ("out", 2560)):
                        data = (out / f"{case}.{role}.{stage}.f32").read_bytes()
                        assert len(data) == count * 4
                        values = struct.unpack(f"={count}f", data)
                        assert all(math.isfinite(v) for v in values)
                        if case == "zero":
                            assert all(v == 0 for v in values)
            signal = (out / "signal.native.out.f32").read_bytes()
            assert any(v != 0 for v in struct.unpack("=2560f", signal))
        # Deterministic fixture and actual reference/backend artifacts across processes.
        names = {p.name for p in outputs[0].iterdir()}
        assert names == {p.name for p in outputs[1].iterdir()}
        for name in names:
            assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes(), name
        hybrids = [root / "hybrid-first", root / "hybrid-second"]
        for out in hybrids:
            run(binary, "--backend", "cpu", "--native-q8-boundaries", "--out-dir", out)
            assert (out / "COMPLETED").is_file()
            manifest = dict(line.split("=", 1) for line in (out / "manifest.txt").read_text().splitlines())
            assert manifest["mode"] == "hybrid_native_q8_boundaries"
            with (out / "metrics.tsv").open() as file:
                metrics = list(csv.DictReader(file, delimiter="\t"))
            assert len(metrics) == 24
            for row in metrics:
                assert math.isfinite(float(row["max_abs"])) and math.isfinite(float(row["relative_L2"]))
                # Same CPU dot implementation plus production quantizer/expression.
                # This is a host seam test, not a Vulkan bit-parity requirement.
                if row["comparison"].endswith("selected_vs_native"):
                    assert int(row["bit_different_values"]) == 0, row
            for case in ("signal", "zero"):
                for boundary, native_name, f32_name, count in (
                        ("input", "native_input", "selected_input", 2560),
                        ("hidden", "native_hidden", "selected_down_input", 640)):
                    packed = (out / f"{case}.selected_{boundary}.q8_0").read_bytes()
                    assert packed == (out / f"{case}.{native_name}.q8_0").read_bytes()
                    values = []
                    for at in range(0, len(packed), 34):
                        scale, *codes = struct.unpack("<e32b", packed[at:at+34])
                        values.extend(scale * code for code in codes)
                    assert len(values) == count
                    assert struct.pack(f"<{count}f", *values) == (out / f"{case}.{f32_name}.f32").read_bytes()
                for stage in ("gate", "up", "hidden", "out"):
                    assert (out / f"{case}.selected.{stage}.f32").read_bytes() == (out / f"{case}.native.{stage}.f32").read_bytes()
        hybrid_names = {p.name for p in hybrids[0].iterdir()}
        assert hybrid_names == {p.name for p in hybrids[1].iterdir()}
        for name in hybrid_names:
            assert (hybrids[0] / name).read_bytes() == (hybrids[1] / name).read_bytes(), name
        before = (outputs[0] / "expert.q8_0").read_bytes()
        run(binary, "--backend", "cpu", "--out-dir", outputs[0], ok=False)
        assert (outputs[0] / "expert.q8_0").read_bytes() == before
        for args in (("--backend", "other"), ("--backend", "vulkan"),
                     ("--backend", "cpu", "--device", "0"),
                     ("--backend", "cpu", "--native-q8-boundaries", "--native-q8-boundaries"),
                     ("--backend", "vulkan", "--device", "-1"),
                     ("--backend", "vulkan", "--device", "16")):
            target = root / "invalid"
            run(binary, *args, "--out-dir", target, ok=False)
            assert not target.exists()
    print("PASS: raw/hybrid expert references, packed/dequantized boundaries, deterministic evidence, finite metrics, zero control, CLI refusal")


if __name__ == "__main__":
    main(sys.argv[1])
