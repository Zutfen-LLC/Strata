"""Read-only exact raw-F32 projection comparison for PR #25's retained files."""
from pathlib import Path
import math
import struct
import sys


def floats(data, count):
    assert len(data) == count * 4, (len(data), count)
    values = struct.unpack(f"<{count}f", data)
    assert all(math.isfinite(v) for v in values), "nonfinite saved row"
    return values


def compare(root):
    assert (root / "COMPLETED").is_file(), "missing completed expert receipt"
    manifest = dict(line.split("=", 1) for line in (root / "manifest.txt").read_text().splitlines())
    assert (manifest["H"], manifest["FF"], manifest["T"]) == ("2560", "640", "1")
    assert manifest["backend"].startswith("Vulkan"), "selected backend is not Vulkan"
    assert manifest.get("mode", "raw_f32") == "raw_f32", "raw-F32 comparator cannot qualify a hybrid run"
    assert manifest["weight_fnv1a64"] == "e6d5f545e2145434"
    assert manifest["signal_input_fnv1a64"] == "b46eda2ec728b1b9"
    blob = (root / "expert.q8_0").read_bytes()
    assert len(blob) == 5222400, "wrong expert blob size"
    x = floats((root / "signal.input.f32").read_bytes(), 2560)
    units = [int(v * 128) for v in x]
    assert all(v / 128 == f and abs(v) <= 63 for v, f in zip(units, x)), "wrong input grid"
    matched = True
    print(f"ARTIFACT {root} backend={manifest['backend']}", flush=True)
    for stage, offset in (("gate", 0), ("up", 1740800)):
        predicted = []
        for row in range(640):
            total = 0
            for block in range(80):
                at = offset + row * 2720 + block * 34
                assert blob[at:at+2] == b"\x00\x18", "weight scale is not 2^-9"
                codes = memoryview(blob)[at+2:at+34].cast("b")
                assert all(abs(v) <= 15 for v in codes), "wrong weight-code range"
                total += sum(codes[j] * units[block * 32 + j] for j in range(32))
            # Absolute partial-sum bound is 2560*15*63 < 2^24. These dyadic
            # products/sums fit binary32 exactly, independently of reduction order.
            assert abs(total) <= 2560 * 15 * 63 < 2**24
            predicted.append(total / 65536)
        expected_bytes = struct.pack("<640f", *predicted)
        saved_bytes = (root / f"signal.selected.{stage}.f32").read_bytes()
        saved = floats(saved_bytes, 640)
        different = sum(saved_bytes[i:i+4] != expected_bytes[i:i+4]
                        for i in range(0, len(expected_bytes), 4))
        errors = [a - b for a, b in zip(saved, predicted)]
        base = sum(v * v for v in predicted)
        assert base > 0, "degenerate raw projection"
        rel = math.sqrt(sum(v * v for v in errors) / base)
        exact = saved_bytes == expected_bytes
        print(f"{stage} exact_raw_F32={int(exact)} bit_different_values={different}/640 "
              f"max_abs={max(map(abs, errors)):.17g} relative_L2={rel:.17g}", flush=True)
        matched &= exact
    return matched


assert len(sys.argv) == 3, "pass existing die0 and die1 output directories"
outcomes = [compare(Path(name)) for name in sys.argv[1:]]
print("BOTH_EXACT_RAW_F32=" + str(int(all(outcomes))), flush=True)
sys.exit(0 if all(outcomes) else 1)
