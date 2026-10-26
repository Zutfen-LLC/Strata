"""Real loopback frame/client/service path, with explicit CPU graph substitution."""
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
