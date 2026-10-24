#!/usr/bin/env python3
"""Remote-only regression fixtures. Fake backend: never invoke ethtool or change a host."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("worker_nic", Path(__file__).resolve().parents[2] / "tools/worker_nic.py")
nic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nic)


def state():
    return {"identity": {"interface": "fixture0", "driver": "e1000e", "driver_version": "fixture",
                         "bus_info": "fixture-pci", "pci_vendor": "fixture-vendor", "pci_device": "fixture-device",
                         "host_id_sha256": "fixture-host", "permanent_mac": "02:00:00:00:00:01"},
            "boot_id": "fixture-boot", "coalesce": {"rx-usecs": "3", "tx-usecs": "0", "adaptive-rx": "off"},
            "driver_raw": "synthetic", "coalesce_raw": "synthetic", "coalesce_diagnostic": ""}


class Fake:
    def __init__(self):
        self.state = state()
        self.writes = []
        self.failure = False
        self.drift = False
        self.partial_value = None

    def read(self, interface):
        return copy.deepcopy(self.state)

    def set_rx(self, interface, value):
        self.writes.append((interface, value))
        self.state["coalesce"]["rx-usecs"] = value
        if self.drift:
            self.state["coalesce"]["tx-usecs"] = "9"
        if self.failure:
            if self.partial_value is not None:
                self.state["coalesce"]["rx-usecs"] = self.partial_value
            raise nic.ProfileError("fixture partial set")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.snapshot = Path(self.directory.name) / "original.json"
        self.backend = Fake()
        nic.save_snapshot(self.snapshot, self.backend.read("fixture0"))

    def apply(self, **kwargs):
        return nic.change(self.backend, "fixture0", self.snapshot, **kwargs)

    def test_release_default_and_explicit_off_never_write(self):
        self.assertFalse(self.apply()["changed"])
        self.assertFalse(self.apply(profile="off")["changed"])
        self.assertEqual(self.backend.writes, [])

    def test_apply_readback_idempotence_and_exact_restore(self):
        before = self.snapshot.read_bytes()
        self.assertTrue(self.apply(profile="low-latency")["verified"])
        self.assertFalse(self.apply(profile="low-latency")["changed"])
        self.assertTrue(self.apply(restore=True)["verified"])
        self.assertFalse(self.apply(restore=True)["changed"])
        self.assertEqual(self.backend.writes, [("fixture0", "0"), ("fixture0", "3")])
        self.assertEqual(self.backend.state, state())
        self.assertEqual(self.snapshot.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            nic.save_snapshot(self.snapshot, state())

    def test_already_zero_original_restores_zero(self):
        self.backend.state["coalesce"]["rx-usecs"] = "0"
        other = Path(self.directory.name) / "zero.json"
        nic.save_snapshot(other, self.backend.state)
        nic.change(self.backend, "fixture0", other, profile="low-latency")
        nic.change(self.backend, "fixture0", other, restore=True)
        self.assertEqual(self.backend.writes, [])

    def test_unsupported_driver_and_unrestorable_mode_never_write(self):
        for key, value in (("driver", "igc"), ("driver", "ixgbe"), ("pci_device", "")):
            self.backend.state = state()
            self.backend.state["identity"][key] = value
            with self.assertRaises(nic.ProfileError):
                self.apply(profile="low-latency")
        for value in ("n/a", "2", "100"):
            self.backend.state = state()
            self.backend.state["coalesce"]["rx-usecs"] = value
            with self.assertRaises(nic.ProfileError):
                self.apply(profile="low-latency")
        self.assertEqual(self.backend.writes, [])

    def test_wrong_boot_wrong_identity_and_unrelated_drift_never_write(self):
        for section, key, value in (("identity", "bus_info", "other"),
                                    ("identity", "host_id_sha256", "other-host"),
                                    ("identity", "permanent_mac", "02:00:00:00:00:02"),
                                    ("coalesce", "tx-usecs", "1")):
            self.backend.state = state()
            self.backend.state[section][key] = value
            with self.assertRaises(nic.ProfileError):
                self.apply(profile="low-latency")
        self.backend.state = state()
        self.backend.state["boot_id"] = "other-boot"
        with self.assertRaises(nic.ProfileError):
            self.apply(restore=True)
        self.assertEqual(self.backend.writes, [])

    def test_failed_or_divergent_readback_is_uncertain_and_preserves_original(self):
        before = self.snapshot.read_bytes()
        self.backend.failure = True
        with self.assertRaisesRegex(nic.ProfileError, "outcome uncertain"):
            self.apply(profile="low-latency")
        self.assertEqual(self.snapshot.read_bytes(), before)
        self.backend.failure = False
        self.assertTrue(self.apply(restore=True)["verified"])
        self.backend.drift = True
        with self.assertRaisesRegex(nic.ProfileError, "readback differs"):
            self.apply(profile="low-latency")
        self.assertEqual(self.snapshot.read_bytes(), before)

    def test_explicit_restore_recovers_unexpected_owned_value_after_partial_failure(self):
        self.backend.failure = True
        self.backend.partial_value = "1"
        with self.assertRaisesRegex(nic.ProfileError, "outcome uncertain"):
            self.apply(profile="low-latency")
        self.backend.failure = False
        with self.assertRaises(nic.ProfileError):
            self.apply(profile="low-latency")
        self.assertTrue(self.apply(restore=True)["verified"])
        self.assertEqual(self.backend.state["coalesce"], state()["coalesce"])
        self.assertEqual(self.backend.writes, [("fixture0", "0"), ("fixture0", "3")])
        self.backend.state["coalesce"]["rx-usecs"] = "1"
        self.backend.state["coalesce"]["tx-usecs"] = "9"
        with self.assertRaisesRegex(nic.ProfileError, "drift"):
            self.apply(restore=True)
        self.assertEqual(len(self.backend.writes), 2)

    def test_changed_snapshot_is_detected_before_restore(self):
        self.apply(profile="low-latency")
        envelope = json.loads(self.snapshot.read_text())
        envelope["original"]["coalesce"]["rx-usecs"] = "0"
        self.snapshot.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(nic.ProfileError, "checksum"):
            self.apply(restore=True)
        self.assertEqual(self.backend.writes, [("fixture0", "0")])

    def test_test_default_requires_exact_validated_record(self):
        config = Path(self.directory.name) / "test.json"
        config.write_text(json.dumps({"schema": 1, "validated_hardware": []}))
        with self.assertRaisesRegex(nic.ProfileError, "no exact validated"):
            self.apply(test_config=config)
        self.assertEqual(self.backend.writes, [])
        config.write_text(json.dumps({"schema": 1, "validated_hardware": [
            {"identity": state()["identity"], "evidence": "fixture-only validation record"}]}))
        self.assertEqual(self.apply(test_config=config)["action"], "apply")
        self.assertEqual(self.backend.writes, [("fixture0", "0")])
        with self.assertRaises(nic.ProfileError):
            self.apply(profile="off", test_config=config)
        self.backend.state["identity"]["host_id_sha256"] = "other-host"
        with self.assertRaisesRegex(nic.ProfileError, "no exact validated"):
            self.apply(test_config=config)
        self.backend.state["identity"] = state()["identity"]
        self.backend.state["identity"]["permanent_mac"] = "02:00:00:00:00:02"
        with self.assertRaisesRegex(nic.ProfileError, "no exact validated"):
            self.apply(test_config=config)
        self.assertEqual(len(self.backend.writes), 1)

    def test_text_parser_and_discovery_failures(self):
        self.assertEqual(nic.permanent_mac("Permanent address: 02:AA:00:00:00:01\n"), "02:aa:00:00:00:01")
        for invalid in ("Permanent address: 00:00:00:00:00:00", "Permanent address: ff:ff:ff:ff:ff:ff", "unavailable"):
            with self.assertRaises(nic.ProfileError):
                nic.permanent_mac(invalid)
        settings = nic.fields("Coalesce parameters for fixture0:\nAdaptive RX: off  TX: off\n"
                              "rx-usecs: 3\ntx-usecs: n/a\n")
        self.assertEqual(settings, {"adaptive-rx": "off", "adaptive-tx": "off", "rx-usecs": "3", "tx-usecs": "n/a"})
        with self.assertRaisesRegex(nic.ProfileError, "unrecognized"):
            nic.fields("unexpected driver setting syntax")
        with patch.object(nic.subprocess, "run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(nic.ProfileError, "provision"):
                nic.Ethtool().command("-c", "fixture0")
        with patch.object(nic.subprocess, "run", side_effect=nic.subprocess.TimeoutExpired("ethtool", 10)):
            with self.assertRaisesRegex(nic.ProfileError, "uncertain"):
                nic.Ethtool().command("-C", "fixture0", "rx-usecs", "0")

    def test_invalid_snapshot_and_cli_missing_snapshot_never_reach_backend(self):
        self.snapshot.write_text('{"schema": 1, "original": {}}')
        with self.assertRaisesRegex(nic.ProfileError, "invalid original"):
            self.apply(profile="low-latency")
        with patch.object(nic, "Ethtool") as backend:
            with self.assertRaises(SystemExit):
                nic.main(["apply", "--interface", "fixture0", "--profile", "low-latency"])
            backend.assert_not_called()
        self.assertEqual(self.backend.writes, [])


if __name__ == "__main__":
    unittest.main()
