"""Real loopback frame/client/service path, with explicit CPU graph substitution."""
import csv
import math
import struct
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="amd-loopback-host-") as root:
    out = Path(root) / "evidence"
    result = subprocess.run([sys.argv[1], "--backend", "cpu", "--native-q8-boundaries",
                             "--loopback-worker-check", "--out-dir", str(out)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOOPBACK_ACCEPTED backend=cpu requests=8 experts=2 prepares=1 fetches=2" in result.stdout
    assert "released_bytes=0 live_allocations=0" in result.stdout
    assert (out / "LOOPBACK_COMPLETED").is_file() and (out / "COMPLETED").is_file()
    assert "bounded_t1_through_t8=1 oversized_refused=1" in result.stdout
    assert "SYNTHETIC_CAPTURE_SELF_CHECK T2=1 T1=1" in result.stdout
    assert (out / "synthetic-capture-result" / "CAPTURE_ROWS_COMPLETED").is_file()
    for request in range(1, 9):
        assert (out / f"loopback.request{request}.rows.f32").read_bytes() == (out / f"loopback.request{request}.direct.f32").read_bytes()

    # ISA preflight is CPU-only and must agree with the capture diagnostic guard.
    isa = subprocess.run([sys.argv[1], "--native-isa-check"], capture_output=True, text=True, timeout=10)
    native = out / "synthetic-capture-result"
    if "qualified_avx2_non_vnni=1" in isa.stdout:
        assert isa.returncode == 0, isa.stdout + isa.stderr
        assert result.stdout.count("CAPTURE_NATIVE_OBSERVED ") == 1
        info = dict(line.split("=", 1) for line in (native / "capture.native.txt").read_text().splitlines())
        assert (info["record"], info["entry"], info["token"], info["expert"]) == ("0", "0", "0", "1")
        assert info["native_isa_flags"] == "7"
        assert info["input_q8_equal"] == info["hidden_q8_equal"] == "1"
        assert info["CPU_model_parity"] == "unestablished"
        with (native / "capture.native.metrics.tsv").open() as file:
            metrics = list(csv.DictReader(file, delimiter="\t"))
        assert [row["stage"] for row in metrics] == ["gate", "up", "hidden", "out"]
        for row in metrics:
            stage = row["stage"]
            a = (native / f"capture.selected.{stage}.f32").read_bytes()
            b = (native / f"capture.native.{stage}.f32").read_bytes()
            count = 2560 if stage == "out" else 640
            assert len(a) == len(b) == count * 4
            assert int(row["bit_different_values"]) == sum(a[i:i+4] != b[i:i+4] for i in range(0, len(a), 4))
            av, bv = struct.unpack(f"<{count}f", a), struct.unpack(f"<{count}f", b)
            delta = [float(x) - y for x, y in zip(av, bv)]
            base = sum(float(y) * y for y in bv)
            assert float(row["max_abs"]) == max(map(abs, delta))
            assert math.isclose(float(row["relative_L2"]), math.sqrt(sum(d*d for d in delta) / base), abs_tol=1e-15)
        for boundary in ("input", "hidden"):
            assert (native / f"capture.native_{boundary}.q8_0").read_bytes() == (native / f"capture.selected_{boundary}.q8_0").read_bytes()
    else:
        assert isa.returncode != 0 and "qualified_avx2_non_vnni=0" in isa.stdout
        assert not (native / "capture.native.metrics.tsv").exists()
        assert "CAPTURE_NATIVE_OBSERVED " not in result.stdout

    # Exercise the public opt-in with the same synthetic sealed files as well.
    cli_out = Path(root) / "native-cli"
    cli = subprocess.run([sys.argv[1], "--backend", "cpu", "--native-q8-boundaries",
                          "--capture", str(out / "synthetic.capture"),
                          "--plan", str(out / "synthetic.plan"),
                          "--local-manifest", str(out / "synthetic.local"),
                          "--blob-dir", str(out / "synthetic-blobs"),
                          "--capture-native-check", "--out-dir", str(cli_out)],
                         capture_output=True, text=True, timeout=30)
    if isa.returncode == 0:
        assert cli.returncode == 0, cli.stdout + cli.stderr
        assert cli.stdout.count("CAPTURE_NATIVE_OBSERVED ") == 1
        assert (cli_out / "CAPTURE_ROWS_COMPLETED").is_file()
        for name in ("capture.native.metrics.tsv", "capture.native.txt", "capture.original_input.f32"):
            assert (cli_out / name).read_bytes() == (native / name).read_bytes()
    else:
        assert cli.returncode != 0 and "qualified_avx2_non_vnni=0" in cli.stdout
        assert not (cli_out / "CAPTURE_ROWS_COMPLETED").exists()
        assert not (cli_out / "capture-cache").exists()
