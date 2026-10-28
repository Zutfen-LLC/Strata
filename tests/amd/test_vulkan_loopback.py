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
    assert "LOOPBACK_ACCEPTED backend=cpu requests=3 experts=2 prepares=1 fetches=2" in result.stdout
    assert "released_bytes=0 live_allocations=0" in result.stdout
    assert (out / "LOOPBACK_COMPLETED").is_file() and (out / "COMPLETED").is_file()
    expected = ((out / "signal.expert3.direct_hybrid.out.f32").read_bytes()
                + (out / "signal.selected.out.f32").read_bytes())
    for request in range(1, 4):
        assert (out / f"loopback.request{request}.rows.f32").read_bytes() == expected
