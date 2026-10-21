"""Explicit synthetic targets only. NOT EXECUTED by the author/reviewers."""
from contextlib import ExitStack
import os
import signal
import subprocess
import unittest
from unittest.mock import patch

import resource_guard as guard
import resource_telemetry as telemetry

TARGETS = ("resource_telemetry_test.TelemetryTests", "resource_guard_test.GuardTests")


def forbidden(*args, **kwargs):
    raise AssertionError("fixture attempted a live operation")


def main():
    suite = unittest.defaultTestLoader.loadTestsFromNames(TARGETS)
    # Test-specific doubles may override these traps; restoration restores traps.
    with ExitStack() as stack:
        for obj, name in ((os, "kill"), (os, "pidfd_open"), (signal, "pidfd_send_signal"),
                          (subprocess, "Popen"), (telemetry, "read_text"),
                          (guard, "read_text"), (guard, "host_sample"),
                          (guard, "namespace_identity")):
            stack.enter_context(patch.object(obj, name, side_effect=forbidden, create=True))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 21 or result.skipped:
        print("RESOURCE_MONITOR_FIXTURES_FAIL_V1")
        return 1
    print("RESOURCE_MONITOR_FIXTURES_PASS_V1 tests=21 live_signals=0 workloads=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
