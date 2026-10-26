"""Explicit CPU substitution tests adapter/cache/lifecycle; no Vulkan initialization."""
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="amd-adapter-host-") as root:
    out = Path(root) / "evidence"
    result = subprocess.run([sys.argv[1], "--backend", "cpu", "--native-q8-boundaries",
                             "--executor-check", "--out-dir", str(out)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "ADAPTER_COMPLETED").is_file()
    assert "index=2 row_bytes=10240 bit_differences=0" in result.stdout
    assert "released_bytes=0 live_allocations=0" in result.stdout
    assert (out / "COMPLETED").is_file()
