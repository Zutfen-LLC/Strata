"""Synthetic fixture sources; NOT EXECUTED. No live sampling/signalling."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import resource_guard as g
import resource_telemetry as t


def mount_info(root="/", point="/CGROUP_ROOT"):
    return t.ok("29 1 0:42 " + root + " " + point + " rw - cgroup2 cgroup rw\n")


def row(path, current=0, maximum=None, high=None):
    return {"path": path, "directory_identity": (1, len(path)),
            "current": current, "maximum": maximum, "high": high,
            "coverage": "required-covered", "events": {k: t.ok(0) for k in g.EVENT_KEYS}}


def group():
    return {"membership": "/parent/leaf", "mount": g.full_mount(mount_info()),
            "namespace_receipt": {"cgroup": (1, 2), "mnt": (1, 3)},
            "coverage": "required-covered",
            "rows": [row("/CGROUP_ROOT/parent/leaf"), row("/CGROUP_ROOT/parent")]}


def sample(free=6 * g.GIB):
    return {"host_bytes": {"SwapTotal": t.ok(8 * g.GIB), "SwapFree": t.ok(free),
                           "MemAvailable": t.ok(124 * g.GIB)},
            "worker_memory_bytes": {"VmRSS": t.ok(g.GIB), "VmSwap": t.Observation()},
            "cgroup": group()}


class GuardTests(unittest.TestCase):
    def test_strict_64mib_boundary_single_sample(self):
        guard = g.CpuGuard(sample())
        guard.check(sample(6 * g.GIB - 65536 * 1024))
        with self.assertRaisesRegex(t.CoverageError, "greater-than-64MiB"):
            guard.check(sample(6 * g.GIB - 65536 * 1024 - 1))
        # Reported triggering decline is 82176 KiB = 80.25 MiB.
        self.assertEqual((6214112 - 6131936) / 1024, 80.25)
        with self.assertRaisesRegex(t.CoverageError, "greater-than-64MiB"):
            guard.check(sample(6 * g.GIB - 82176 * 1024))

    def test_fixed_baseline_not_rolling_or_debounced(self):
        guard = g.CpuGuard(sample())
        guard.check(sample(6 * g.GIB - 40 * g.MIB))
        with self.assertRaises(t.CoverageError):
            guard.check(sample(6 * g.GIB - 65 * g.MIB))
        guard.check(sample(7 * g.GIB))  # recovery does not re-anchor upward
        guard.check(sample(6 * g.GIB - 64 * g.MIB))
        self.assertEqual(guard.swap_baseline, 6 * g.GIB)
        new_session = g.CpuGuard(sample(5 * g.GIB))
        self.assertEqual(new_session.swap_baseline, 5 * g.GIB)

    def test_existing_swap_occupancy_is_not_extra_allowance(self):
        guard = g.CpuGuard(sample(g.GIB))
        with self.assertRaises(t.CoverageError):
            guard.check(sample(g.GIB - 65 * g.MIB))
        no_swap = sample(0)
        no_swap["host_bytes"]["SwapTotal"] = t.ok(0)
        g.CpuGuard(no_swap).check(no_swap)

    def test_required_missing_or_malformed_blocks_admission_and_active(self):
        for field in ("SwapTotal", "SwapFree", "MemAvailable"):
            for value in (t.Observation(), t.malformed("fixture")):
                bad = sample()
                bad["host_bytes"][field] = value
                with self.assertRaises(t.CoverageError):
                    g.CpuGuard(bad)
                with self.assertRaises(t.CoverageError):
                    g.CpuGuard(sample()).check(bad)
        bad = sample()
        bad["worker_memory_bytes"]["VmRSS"] = t.Observation()
        with self.assertRaises(t.CoverageError):
            g.CpuGuard(bad)

    def test_new_diagnostics_do_not_create_guard_policy(self):
        baseline = sample()
        baseline.update(psi=t.malformed("unavailable"), vmstat_pages={"pswpout": t.Observation()})
        guard = g.CpuGuard(baseline)
        active = copy.deepcopy(baseline)
        active["worker_memory_bytes"]["VmSwap"] = t.ok(999 * g.MIB)
        active["psi"] = t.ok({"some": {"avg10": 100}})
        active["vmstat_pages"] = {"pswpout": t.ok(1000000000)}
        guard.check(active)

    def test_admission_stop_rss_and_tightest_ancestor(self):
        bad = sample()
        bad["host_bytes"]["MemAvailable"] = t.ok(g.CPU_ADMISSION_BYTES - 1)
        with self.assertRaisesRegex(t.CoverageError, "host-headroom"):
            g.CpuGuard(bad)
        active = sample()
        active["host_bytes"]["MemAvailable"] = t.ok(8 * g.GIB - 1)
        with self.assertRaises(t.CoverageError):
            g.CpuGuard(sample()).check(active)
        cg = group()
        cg["rows"][0].update(current=2 * g.GIB, maximum=140 * g.GIB)
        cg["rows"][1].update(current=10 * g.GIB, high=130 * g.GIB)
        self.assertEqual(g.headroom(cg), 120 * g.GIB)
        bad = sample()
        bad["cgroup"] = cg
        with self.assertRaisesRegex(t.CoverageError, "cgroup-headroom"):
            g.CpuGuard(bad)
        rss = sample()
        rss["worker_memory_bytes"]["VmRSS"] = t.ok(g.CPU_RSS_CAP_BYTES)
        g.CpuGuard(rss)
        rss["worker_memory_bytes"]["VmRSS"] = t.ok(g.CPU_RSS_CAP_BYTES + 1)
        with self.assertRaises(t.CoverageError):
            g.CpuGuard(rss)

    def test_unlimited_does_not_disable_events(self):
        baseline = sample()
        self.assertIsNone(g.headroom(baseline["cgroup"]))
        for event in g.EVENT_KEYS:
            changed = copy.deepcopy(baseline)
            changed["cgroup"]["rows"][1]["events"][event] = t.ok(1)
            with self.assertRaisesRegex(t.CoverageError, "new-memory-event"):
                g.CpuGuard(baseline).check(changed)
        missing = copy.deepcopy(baseline)
        missing["cgroup"]["rows"][0]["events"]["high"] = t.Observation()
        with self.assertRaises(t.CoverageError):
            g.CpuGuard(missing)
        initial = copy.deepcopy(baseline)
        initial["cgroup"]["rows"][0]["events"]["max"] = t.ok(3)
        with self.assertRaisesRegex(t.CoverageError, "counter-decreased"):
            g.CpuGuard(initial).check(baseline)

    def test_migration_limit_and_directory_replacement_stop(self):
        for key, value in (("membership", "/other"), ("mount", ("new",)),
                           ("namespace_receipt", {})):
            active = sample()
            active["cgroup"][key] = value
            with self.assertRaises(t.CoverageError):
                g.CpuGuard(sample()).check(active)
        for key, value in (("directory_identity", (9, 9)), ("maximum", 200 * g.GIB)):
            active = sample()
            active["cgroup"]["rows"][0][key] = value
            with self.assertRaises(t.CoverageError):
                g.CpuGuard(sample()).check(active)

    def test_membership_mount_root_escaping_and_ancestors(self):
        leaf = g.membership(t.ok("0::/parent/leaf\n"))
        mount = g.full_mount(mount_info(point="/CGROUP\\040ROOT"))
        self.assertEqual(g.ancestors(leaf, mount),
                         ["/CGROUP ROOT/parent/leaf", "/CGROUP ROOT/parent", "/CGROUP ROOT"])
        self.assertEqual(g.ancestors(g.membership(t.ok("0::/\n")), mount), ["/CGROUP ROOT"])
        for raw in ("1:memory:/leaf\n", "0::/leaf\n1:cpu:/leaf\n", "0::/../leaf\n",
                    "0::relative\n", "0::/parent//leaf\n"):
            with self.assertRaises(t.CoverageError):
                g.membership(t.ok(raw))
        for mount_text in (mount_info(root="/hidden"), t.ok(mount_info().value * 2),
                           t.Observation(), t.ok("broken")):
            with self.assertRaises(t.CoverageError):
                g.full_mount(mount_text)
        with self.assertRaises(t.CoverageError):
            g.ancestors(g.canonical("/" + "/".join(["x"] * 64)), g.full_mount(mount_info()))

    def test_initial_namespace_proof_not_just_pid1(self):
        process = SimpleNamespace(proc_fd=77)
        expected = {"cgroup": (1, 2), "mnt": (1, 3)}
        with patch.object(g, "namespace_identity", side_effect=[(1, 2), (1, 2), (1, 3), (1, 3)]):
            g.verify_namespaces(process, expected)
        with patch.object(g, "namespace_identity", return_value=(9, 9)):
            with self.assertRaises(t.CoverageError):
                g.verify_namespaces(process, expected)

    def test_actual_cgroup_discovery_required_missing_vs_true_root(self):
        process = Mock(proc_fd=77)
        process.read.return_value = t.ok("0::/leaf\n")
        files = {"memory.current": "0", "memory.max": "max", "memory.high": "max",
                 "cgroup.type": "domain", "cgroup.controllers": "memory cpu",
                 "memory.events": "high 0\nmax 0\noom 0\noom_kill 0\n"}
        def read(name, fd=None):
            if name == "/proc/self/mountinfo":
                return mount_info()
            return t.ok(files[name]) if name in files else t.Observation()
        with patch.object(g, "verify_namespaces"), patch.object(g, "read_text", side_effect=read), \
             patch.object(g.os, "open", return_value=88), patch.object(g.os, "close"), \
             patch.object(g.os, "fstat", return_value=SimpleNamespace(st_dev=1, st_ino=2)):
            actual = g.cgroup_snapshot(process, {})
            self.assertEqual(len(actual["rows"]), 2)
            self.assertEqual(actual["rows"][0]["coverage"], "required-covered")
            self.assertEqual(actual["rows"][1]["events"]["high"].status, "unsupported")
            del files["memory.events"]
            with self.assertRaises(t.CoverageError):
                g.cgroup_snapshot(process, {})

    def test_owned_stop_never_touches_unrelated_or_reused_process(self):
        root = Mock(identity=SimpleNamespace(pid=41))
        child = Mock(identity=SimpleNamespace(pid=42))
        root.alive.return_value = child.alive.return_value = False
        root.send.return_value = "sent"
        child.send.side_effect = t.CoverageError("identity-changed")
        tree = SimpleNamespace(items={41: root, 42: child})
        result = g.stop_owned(tree, grace_seconds=0)
        root.send.assert_called_once_with(g.signal.SIGTERM)
        child.send.assert_called_once_with(g.signal.SIGTERM)
        self.assertEqual([r[0] for r in result], [41, 42])

    def test_failed_admission_and_bad_deadline_stop_without_admitted_marker(self):
        for deadline in (0, float("nan"), "bad", 130):
            tree, journal = Mock(), Mock()
            with patch.object(g.time, "monotonic", return_value=100), \
                 patch.object(g, "capture", side_effect=t.CoverageError("missing-events")), \
                 patch.object(g, "stop_owned", return_value=[]) as stop:
                g.monitor(tree, None, {}, deadline, journal)
                stop.assert_called_once()
                self.assertEqual(journal.emit.call_args.args[0], "RESOURCE_MONITOR_STOP_V1")
                self.assertEqual(journal.emit.call_count, 1)
                tree.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
