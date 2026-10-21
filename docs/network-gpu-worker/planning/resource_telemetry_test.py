"""Synthetic sources only. NOT EXECUTED. No live processes or signals."""
import unittest
from unittest.mock import Mock, patch

import resource_telemetry as t


def stat_text(pid=41, ppid=7, start=900):
    rest = ["S", str(ppid)] + ["0"] * 17 + [str(start)]
    return t.ok(str(pid) + " (comm with ) spaces) " + " ".join(rest))


class TelemetryTests(unittest.TestCase):
    def test_kib_units_and_missing_not_zero(self):
        fields = t.kib_fields(t.ok("VmRSS: 2 kB\nVmSwap: 0 kB\n"), t.STATUS_FIELDS)
        self.assertEqual(fields["VmRSS"].value, 2048)
        self.assertEqual(fields["VmSwap"], t.ok(0))
        self.assertEqual(fields["RssAnon"].status, "unavailable")
        self.assertIsNone(fields["RssAnon"].value)
        for raw in ("VmRSS: 2 MB", "VmRSS: -1 kB", "VmRSS: 1 kB\nVmRSS: 1 kB"):
            self.assertEqual(t.kib_fields(t.ok(raw), ("VmRSS",))["VmRSS"].status, "malformed")

    def test_counter_units_and_unavailable(self):
        values = t.counters(t.ok("pswpin 17\npswpout 29\n"), ("pswpin", "pswpout"))
        self.assertEqual(values, {"pswpin": t.ok(17), "pswpout": t.ok(29)})
        absent = t.counters(t.Observation(None, "unavailable", "EACCES"), ("pswpin",))
        self.assertEqual(absent["pswpin"].reason, "EACCES")
        self.assertEqual(t.counters(t.ok("pswpout NaN"), ("pswpout",))["pswpout"].status,
                         "malformed")
        self.assertEqual(t.counter_delta(t.ok(1), t.ok(2)).reason, "counter-decreased")

    def test_first_observation_and_interval_are_separate(self):
        def sample(value):
            return {"vmstat_pages": {"pswpout": value}}
        base = t.DiagnosticBaselines(sample(t.ok(10)))
        first = sample(t.ok(13))
        base.annotate(first)
        second = sample(t.ok(17))
        base.annotate(second)
        self.assertEqual(second["vmstat_since_observation_start_pages"]["pswpout"], t.ok(7))
        self.assertEqual(second["vmstat_interval_pages"]["pswpout"], t.ok(4))
        missing = sample(t.Observation())
        base.annotate(missing)
        final = sample(t.ok(19))
        base.annotate(final)
        self.assertEqual(final["vmstat_since_observation_start_pages"]["pswpout"], t.ok(9))
        self.assertEqual(final["vmstat_interval_pages"]["pswpout"].status, "unavailable")

    def test_psi_diagnostic_schema(self):
        row = "some avg10=1.00 avg60=2.00 avg300=3.00 total=40\n"
        parsed = t.psi(t.ok(row))
        self.assertEqual(parsed.value["some"]["total_us"], 40)
        self.assertEqual(parsed.value["full"]["status"], "unsupported")
        self.assertEqual(t.psi(t.ok(row.replace("1.00", "nan"))).status, "malformed")
        self.assertEqual(t.psi(t.Observation()).status, "unavailable")

    def test_stat_comm_and_field_offsets(self):
        self.assertEqual(t.parse_stat(stat_text()), (41, 7, 900))
        with self.assertRaises(t.CoverageError):
            t.parse_stat(t.ok("41 (short) S 7"))

    def test_pid_reuse_discards_status_and_never_signals_replacement(self):
        proc = t.OwnedProcess(t.Identity(41, 900, 1, 2), 101, 102)
        with patch.object(proc, "alive", return_value=True), \
             patch.object(t, "read_text", return_value=stat_text(start=901)) as read, \
             patch.object(t.os, "stat", return_value=Mock(st_dev=1, st_ino=2)), \
             patch.object(t.signal, "pidfd_send_signal", create=True) as send:
            with self.assertRaises(t.CoverageError):
                proc.status()
            with self.assertRaises(t.CoverageError):
                proc.send(15)
            self.assertTrue(all(call.args[0] == "stat" for call in read.call_args_list))
            send.assert_not_called()

    def test_safe_signal_uses_held_pidfd_only(self):
        proc = t.OwnedProcess(t.Identity(41, 900, 1, 2), 101, 102)
        with patch.object(proc, "alive", return_value=True), \
             patch.object(proc, "verify"), \
             patch.object(t.signal, "pidfd_send_signal", create=True) as send, \
             patch.object(t.os, "kill", side_effect=AssertionError("numeric kill forbidden")):
            self.assertEqual(proc.send(15), "sent")
            send.assert_called_once_with(102, 15, None, 0)
        with patch.object(proc, "alive", return_value=False), \
             patch.object(t.signal, "pidfd_send_signal", create=True) as send:
            self.assertEqual(proc.send(15), "already-exited")
            send.assert_not_called()

    def test_missing_pidfd_blocks_without_fallback(self):
        with patch.object(t.os, "pidfd_open", None, create=True):
            # A noncallable exported interface is also unavailable.
            with self.assertRaises(t.CoverageError):
                t.OwnedProcess.pin(41)

    def test_namespace_refusal_precedes_pidfd_pin_and_proc_access(self):
        with patch.object(t.os, "pidfd_open", create=True) as pin, \
             patch.object(t.signal, "pidfd_send_signal", create=True), \
             patch.object(t, "verify_initial_namespaces", side_effect=t.CoverageError("pid-view")), \
             patch.object(t.os, "open") as opened:
            with self.assertRaises(t.CoverageError):
                t.OwnedProcess.pin(41, expected=t.Identity(41, 900, 1, 2), namespace_receipt={})
            pin.assert_not_called()
            opened.assert_not_called()

    def test_idempotent_close_attempts_both_and_all_owners(self):
        proc = t.OwnedProcess(t.Identity(41, 900, 1, 2), 101, 102)
        with patch.object(t.os, "close", side_effect=[OSError("fixture"), None]) as close:
            with self.assertRaises(OSError):
                proc.close()
            proc.close()
            self.assertEqual([c.args[0] for c in close.call_args_list], [101, 102])
        self.assertIsNone(proc.proc_fd)
        self.assertIsNone(proc.pid_fd)
        first, second = Mock(), Mock()
        first.identity.pid, second.identity.pid = 41, 42
        first.close.side_effect = OSError("fixture")
        tree = t.OwnedTree(first)
        tree.items[42] = second
        with self.assertRaises(OSError):
            tree.close()
        second.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
