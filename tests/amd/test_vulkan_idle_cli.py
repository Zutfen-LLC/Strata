#!/usr/bin/env python3
"""Check idle option parsing without starting a worker or opening a device."""
import subprocess
import sys

binary = sys.argv[1]
for value in ("-1", "86400001", "abc", "1x", "18446744073709551616"):
    result = subprocess.run([binary, "--idle-ms", value], capture_output=True)
    assert result.returncode == 2 and b"usage:" in result.stderr, value
# Valid values reach plan loading rather than failing option parsing.
for value in ("0", "1", "86400000"):
    result = subprocess.run([binary, "--idle-ms", value, "--plan", "/nonexistent-strata-plan",
                             "--cache-root", "/nonexistent-strata-cache"], capture_output=True)
    assert result.returncode == 2 and b"usage:" not in result.stderr, value
