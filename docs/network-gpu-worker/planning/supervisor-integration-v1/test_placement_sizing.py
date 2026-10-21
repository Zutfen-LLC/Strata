"""Focused sizing contract; authored NOT RUN here. No live prerequisites.

This renders argv and decodes synthetic bytes only. It does not rerun ranking,
read plans/model assets, create a process/context/socket or change guards.
"""
from pathlib import Path
import resource
import signal
import struct
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import agent
import cache_status
import commands
from outer import OuterGuards
from policy import ENGINE_CAPS, GPU_CAPS, MIB, POLICIES
from resource_guard import CoverageError, GuardStop


class Bindings(dict):
    def __missing__(self, key):
        return "/generic/" + key


def option(argv, name):
    return argv[argv.index(name) + 1]


class SizingTests(unittest.TestCase):
    def test_all_engine_phases_keep_fixed_plan_and_new_helper(self):
        for phase in ("B0", "B1", "G0", "G1"):
            with self.subTest(phase=phase):
                argv, _, cap = commands.product("ROLE_ENGINE_A", phase, Bindings())
                self.assertEqual(option(argv, "--expert-cache"), "1056")
                self.assertEqual(option(argv, "--expert-cache-device1"), "2247")
                self.assertEqual(option(argv, "--vram-reserve-mib"), "1152")
                self.assertEqual(option(argv, "--expert-profile"), "/generic/PROFILE_A")
                self.assertEqual(cap, ENGINE_CAPS[phase])
                self.assertEqual("--remote-gpu-plan" in argv, phase.startswith("G"))
                if phase.startswith("G"):
                    self.assertEqual(option(argv, "--remote-gpu-plan"), "/generic/GPU_PLAN_A")
        self.assertEqual(agent.PLAN_SHA,
                         "a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578")

    def test_cpu_budget_is_tight_ceiling_for_derived_full_complement(self):
        engine, _, _ = commands.product("ROLE_ENGINE_A", "B0", Bindings())
        cpu, _, cap = commands.product("ROLE_CPU_C", "CPU", Bindings())
        full = 48 * 512 - int(option(engine, "--expert-cache")) - int(option(engine, "--expert-cache-device1"))
        payload = full * 5222400
        budget = int(option(cpu, "--memory-budget-mib"))
        self.assertEqual((full, payload, full - 3289), (21273, 111096115200, 17984))
        self.assertEqual(budget, 105950)
        self.assertEqual(budget, (payload + MIB - 1) // MIB)
        self.assertEqual(budget * MIB - payload, 512000)
        self.assertLess(budget * MIB, payload + 5222400)  # Even one extra blob must refuse.
        self.assertLessEqual(budget, 107000)
        self.assertEqual((cap, option(cpu, "--local-setup-ms"), option(cpu, "--residency-ms")),
                         (7200, "600000", "1800000"))

    def test_idle_status_accepts_new_full_arena_and_rejects_old(self):
        header = struct.pack("<IHHQ", cache_status.MAGIC, 17, 0, 100)
        payload = bytearray(100)
        struct.pack_into("<IIQQQ", payload, 0, 1, 0, 0, 111096115200, 1000)
        struct.pack_into("<I", payload, 96, 1)
        self.assertEqual(cache_status.decode(header, payload), 1000)
        struct.pack_into("<Q", payload, 16, 111012556800)
        with self.assertRaises(CoverageError):
            cache_status.decode(header, payload)

    def test_resource_floors_and_remote_cli_are_unchanged(self):
        policy = POLICIES["ROLE_CPU_C"]
        self.assertEqual((policy.admission, policy.rss, policy.headroom),
                         ((107000 + 16384) * MIB, 108024 * MIB, 8192 * MIB))
        guard = OuterGuards("ROLE_ENGINE_A", device_ids=("primary", "helper"), selection="primary,helper")
        guard.check({"gpu_free_bytes": [1152 * MIB, 512 * MIB]})
        for primary, helper in ((1152 * MIB - 1, 512 * MIB), (1152 * MIB, 512 * MIB - 1)):
            with self.subTest(primary=primary, helper=helper), self.assertRaises(GuardStop):
                guard.check({"gpu_free_bytes": [primary, helper]})
        for phase in ("G0", "G1"):
            argv, _, cap = commands.product("ROLE_GPU_B", phase, Bindings())
            self.assertEqual(option(argv, "--plan"), "/generic/GPU_PLAN_B")
            self.assertEqual(option(argv, "--cache-budget-mib"), "18432")
            self.assertEqual(option(argv, "--reserve-mib"), "4096")
            self.assertEqual(cap, GPU_CAPS[phase])


def main():
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MIB, MIB))
    def expired(*_):
        raise KeyboardInterrupt("focused-sizing-10s-wall-cap")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 10)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SizingTests)
    forbidden = AssertionError("live operation forbidden in sizing fixture")
    with patch("os.open", side_effect=forbidden), patch("os.scandir", side_effect=forbidden), \
            patch("os.readlink", side_effect=forbidden), patch("os.fork", side_effect=forbidden), \
            patch("os.kill", side_effect=forbidden), patch("os.pidfd_open", side_effect=forbidden), \
            patch("signal.pidfd_send_signal", side_effect=forbidden), \
            patch("subprocess.Popen", side_effect=forbidden), patch("socket.socket", side_effect=forbidden):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped or result.testsRun != 4:
        raise SystemExit(1)
    print("PLACEMENT_SIZING_FIXTURES_PASS_V1 tests=4 live_signals=0 workloads=0", flush=True)


if __name__ == "__main__":
    main()
