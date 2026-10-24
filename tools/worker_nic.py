#!/usr/bin/env python3
"""Explicit Linux CPU-worker NIC profile. Never called by worker/inference startup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


class ProfileError(RuntimeError):
    pass


def permanent_mac(text):
    match = re.fullmatch(r"Permanent address:\s*([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})\s*", text.strip())
    if not match or match[1].lower() in ("00:00:00:00:00:00", "ff:ff:ff:ff:ff:ff") or int(match[1][:2], 16) & 1:
        raise ProfileError("permanent NIC address unavailable; inspect ethtool -P before validating this device")
    return match[1].lower()


def original_digest(state):
    # Accidental-corruption detection, not authentication against a file editor.
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def fields(text):
    """Retain every coalesce setting, including unsupported/n/a values, for drift checks."""
    result = {}
    for line in text.splitlines():
        paired = re.fullmatch(r"\s*(Adaptive|CQE mode) RX:\s*(\S+)\s+TX:\s*(\S+)\s*", line)
        if paired:
            if paired[1] == "CQE mode" and any(value not in ("on", "off", "n/a") for value in paired.group(2, 3)):
                raise ProfileError(f"unrecognized CQE mode value {line.strip()!r}; inspect before changing settings")
            prefix = "adaptive" if paired[1] == "Adaptive" else "cqe-mode"
            parsed = {prefix + "-rx": paired[2], prefix + "-tx": paired[3]}
        else:
            match = re.fullmatch(r"\s*([a-z][a-z0-9-]*):\s*(\S+)\s*", line)
            if match:
                parsed = {match[1]: match[2]}
            elif line.strip() and not line.startswith("Coalesce parameters for "):
                raise ProfileError(f"unrecognized coalesce output {line.strip()!r}; inspect before changing settings")
            else:
                continue
        for key, value in parsed.items():
            if key in result:
                raise ProfileError(f"duplicate ethtool field {key}; inspect output manually")
            result[key] = value
    return result


class Ethtool:
    def command(self, *args):
        try:
            run = subprocess.run(["ethtool", *args], capture_output=True, text=True, timeout=10,
                                 env={**os.environ, "LC_ALL": "C"})
        except FileNotFoundError as exc:
            raise ProfileError("ethtool is unavailable; ask the operator to provision it") from exc
        except subprocess.TimeoutExpired as exc:
            raise ProfileError("ethtool timed out; setting outcome may be uncertain; read back before retrying") from exc
        if run.returncode:
            raise ProfileError(f"ethtool {' '.join(args)} failed: {run.stderr.strip()}; "
                               "check driver support and operator CAP_NET_ADMIN permissions")
        return run.stdout

    def read(self, interface):
        if platform.system() != "Linux":
            raise ProfileError("NIC profile supports Linux only; use the OS/driver's operator tools")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,14}", interface):
            raise ProfileError("select an exact interface name from /sys/class/net")
        device = Path("/sys/class/net") / interface / "device"
        if not device.exists():
            raise ProfileError("interface has no physical device; select the LAN NIC, not a bridge/tunnel")
        driver_raw = self.command("-i", interface)
        driver = dict(line.split(": ", 1) for line in driver_raw.splitlines() if ": " in line)
        identity = {"interface": interface, "driver": driver.get("driver", ""),
                    "driver_version": driver.get("version", ""),
                    "firmware_version": driver.get("firmware-version", ""),
                    "bus_info": driver.get("bus-info", ""), "kernel_release": platform.release()}
        for key in ("vendor", "device", "subsystem_vendor", "subsystem_device", "revision"):
            path = device / key
            identity["pci_" + key] = path.read_text().strip() if path.exists() else ""
        try:
            machine = Path("/etc/machine-id").read_text().strip().lower()
        except OSError:
            machine = ""
        identity["host_id_sha256"] = (hashlib.sha256(machine.encode("ascii")).hexdigest()
                                      if re.fullmatch(r"[0-9a-f]{32}", machine) and int(machine, 16) else "")
        try:
            identity["permanent_mac"] = permanent_mac(self.command("-P", interface))
        except ProfileError:
            identity["permanent_mac"] = ""
        try:
            raw = self.command("-c", interface)
            settings = fields(raw)
            diagnostic = ""
        except ProfileError as exc:
            raw, settings, diagnostic = "", {}, str(exc)
        return {"identity": identity, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "coalesce": settings, "driver_raw": driver_raw, "coalesce_raw": raw,
                "coalesce_diagnostic": diagnostic}

    def set_rx(self, interface, value):
        # Only a validated e1000e mode code reaches here. Never copy it to another driver.
        self.command("-C", interface, "rx-usecs", str(value))


def capability(state, restoring=False):
    identity, settings = state["identity"], state["coalesce"]
    reasons = []
    if identity.get("driver") != "e1000e":
        reasons.append("only e1000e has a reviewed mode mapping; do not copy its codes to this driver")
    if not all(identity.get(k) for k in ("bus_info", "driver_version", "pci_vendor", "pci_device")):
        reasons.append("physical PCI/driver identity incomplete; inspect ethtool -i and sysfs")
    if not identity.get("host_id_sha256") or not identity.get("permanent_mac"):
        reasons.append("host/permanent NIC identity incomplete; inspect /etc/machine-id and ethtool -P")
    if restoring and not re.fullmatch(r"[0-9]+", settings.get("rx-usecs", "")):
        reasons.append("current rx-usecs is unreadable; inspect ethtool -c before restoring")
    elif not restoring and settings.get("rx-usecs") not in ("0", "3"):
        reasons.append("only original e1000e modes 0 and 3 have an exact restore path in this profile")
    if state.get("coalesce_diagnostic"):
        reasons.append(state["coalesce_diagnostic"])
    return {"supported": not reasons, "diagnostics": reasons,
            "setting": "rx-usecs", "target": "0", "target_meaning": "interrupt moderation disabled",
            "original_3_meaning": "e1000e dynamic-conservative mode, not 3 microseconds"}


def supported(state, restoring=False):
    info = capability(state, restoring)
    if not info["supported"]:
        raise ProfileError("; ".join(info["diagnostics"]))


def load_json(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ProfileError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ProfileError("expected an object with schema 1")
    return data


def save_snapshot(path, state):
    supported(state)
    # Never overwrite an original snapshot, including via a symlink. Persist it before any mutation.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump({"schema": 1, "original": state, "original_sha256": original_digest(state)},
                  output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def original_snapshot(path):
    envelope = load_json(path)
    original = envelope.get("original")
    if (not isinstance(original, dict) or not isinstance(original.get("identity"), dict) or
            not isinstance(original.get("coalesce"), dict) or not original.get("boot_id")):
        raise ProfileError("invalid original snapshot; do not apply or guess a restore value")
    if envelope.get("original_sha256") != original_digest(original):
        raise ProfileError("original snapshot checksum missing or mismatched; recover the intact original before changing settings")
    supported(original)
    return original


def check_identity(original, current, restoring=False):
    if original["identity"] != current["identity"] or original["boot_id"] != current["boot_id"]:
        raise ProfileError("snapshot hardware/driver/boot mismatch; inspect this interface and take a new snapshot")
    supported(current, restoring)


def selected_profile(profile, test_config, state):
    if test_config:
        if profile is not None:
            raise ProfileError("choose --profile or --test-config, not both")
        config = load_json(test_config)
        # The test configuration defaults on only for a fully matched, operator-validated record.
        records = config.get("validated_hardware", [])
        if not isinstance(records, list) or not any(
                isinstance(r, dict) and r.get("identity") == state["identity"] and
                isinstance(r.get("evidence"), str) and r["evidence"].strip() for r in records):
            raise ProfileError("test profile has no exact validated hardware record; validate apply/readback/restore "
                               "and matched inference, then record this identity and evidence")
        profile = config.get("profile", "low-latency")
    else:
        profile = profile or "off"  # release default
    if profile not in ("off", "low-latency"):
        raise ProfileError("profile must be off or low-latency")
    return profile


def change(backend, interface, snapshot_path, restore=False, profile=None, test_config=None):
    current = backend.read(interface)
    if not restore and selected_profile(profile, test_config, current) == "off":
        return {"action": "off", "changed": False, "readback": current}
    original = original_snapshot(snapshot_path)
    check_identity(original, current, restore)
    applied = {**original["coalesce"], "rx-usecs": "0"}
    # Refuse unrelated drift rather than overwriting another operator's settings.
    unchanged_other_fields = ({k: v for k, v in current["coalesce"].items() if k != "rx-usecs"} ==
                              {k: v for k, v in original["coalesce"].items() if k != "rx-usecs"})
    if not unchanged_other_fields or (not restore and current["coalesce"] not in (original["coalesce"], applied)):
        raise ProfileError("coalescing drift since snapshot; inspect ethtool -c; no setting changed by this command")
    expected = original["coalesce"] if restore else applied
    changed = current["coalesce"] != expected
    if changed:
        try:
            backend.set_rx(interface, expected["rx-usecs"])
            current = backend.read(interface)
            check_identity(original, current, restore)
            if current["coalesce"] != expected:
                raise ProfileError("coalesce readback differs from the requested full setting map")
        except (OSError, ProfileError) as exc:
            # Never silently report a failed/partial set as unchanged or automatically change more knobs.
            raise ProfileError(f"NIC setting outcome uncertain: {exc}. Original remains at {snapshot_path}; "
                               "run discover, inspect readback, then explicitly restore using that snapshot") from exc
    return {"action": "restore" if restore else "apply", "changed": changed,
            "verified": True, "readback": current}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("discover", "snapshot", "apply", "restore"))
    parser.add_argument("--interface", required=True)
    parser.add_argument("--snapshot", help="new original file with corruption checksum; existing file for apply/restore")
    parser.add_argument("--profile", choices=("off", "low-latency"), default=None)
    parser.add_argument("--test-config", help="explicit test configuration with exact validated hardware records")
    args = parser.parse_args(argv)
    if args.action != "apply" and (args.profile is not None or args.test_config):
        parser.error("profile selection belongs only to explicit apply")
    if args.action in ("snapshot", "restore") and not args.snapshot:
        parser.error("this action requires --snapshot")
    if args.action == "apply" and (args.profile == "low-latency" or args.test_config) and not args.snapshot:
        parser.error("low-latency apply requires an original --snapshot")
    backend = Ethtool()
    if args.action in ("discover", "snapshot"):
        state = backend.read(args.interface)
        if args.action == "snapshot":
            save_snapshot(args.snapshot, state)
        result = {"action": args.action, "state": state, "capability": capability(state)}
    else:
        result = change(backend, args.interface, args.snapshot, args.action == "restore",
                        args.profile, args.test_config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ProfileError) as exc:
        print(f"worker NIC: {exc}", file=sys.stderr)
        sys.exit(1)
