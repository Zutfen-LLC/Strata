"""Live trust contract. Import is inert; synthetic admission never calls this.

CLI pins must come from an independently approved administrator/source handoff.
This module checks their binding, NOT the authenticity of that human handoff.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import stat


class Blocked(ValueError):
    def __init__(self, status, detail):
        super().__init__(status + ":" + detail)
        self.status, self.detail = status, detail


def strict_json(raw):
    def unique(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate-key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite-json")))


def regular_bytes(path, cap, check=lambda: None):
    """Exact bounded bytes, no symlink/FIFO/device; detect in-read replacement."""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= cap:
            raise Blocked("malformed", "regular-file-size")
        result = bytearray()
        while len(result) <= cap:
            check()
            chunk = os.read(fd, min(65536, cap + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
        after = os.fstat(fd)
        stable = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if len(result) != before.st_size or stable(before) != stable(after):
            raise Blocked("mismatch", "file-changed-or-length")
        check()
        return bytes(result)
    finally:
        os.close(fd)


def observe(label, action, required=True):
    try:
        return {"name": label, "required": required, "status": "ok", "value": action()}
    except PermissionError as exc:
        return {"name": label, "required": required, "status": "permission-denied", "errno": exc.errno}
    except FileNotFoundError as exc:
        return {"name": label, "required": required, "status": "missing", "errno": exc.errno}
    except OSError as exc:
        return {"name": label, "required": required, "status": "os-error", "errno": exc.errno}
    except (ValueError, UnicodeError) as exc:
        return {"name": label, "required": required, "status": getattr(exc, "status", "malformed"),
                "detail": str(exc)}


def hex_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def parse_receipt(raw, pins, now):
    if not hex_digest(pins.get("digest")) or not hex_digest(pins.get("machine")) or \
            any(not isinstance(pins.get(k), str) or not pins[k].strip() for k in ("source", "host")):
        raise Blocked("untrusted", "externally-approved-pins-required")
    if hashlib.sha256(raw).hexdigest() != pins["digest"]:
        raise Blocked("digest-mismatch", "exact-receipt-bytes")
    try:
        value = strict_json(raw)
        keys = {"schema", "role", "host_identity", "machine_id_sha256", "boot_id", "issued_utc",
                "source_id", "method", "namespaces"}
        if not isinstance(value, dict) or set(value) != keys or type(value["schema"]) is not int or value["schema"] != 1:
            raise ValueError("receipt-schema")
        if value["role"] != "ROLE_CPU_C":
            raise Blocked("wrong-role", "receipt-role")
        if value["host_identity"] != pins["host"] or value["machine_id_sha256"] != pins["machine"]:
            raise Blocked("wrong-host", "approved-host-binding")
        if value["source_id"] != pins["source"]:
            raise Blocked("untrusted", "approved-source-mismatch")
        if not isinstance(value["method"], str) or not 1 <= len(value["method"]) <= 1024:
            raise ValueError("receipt-method")
        if not isinstance(value["boot_id"], str) or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value["boot_id"]) is None:
            raise ValueError("boot-schema")
        ns = value["namespaces"]
        if not isinstance(ns, dict) or set(ns) != {"pid", "mnt", "cgroup"} or \
                any(not isinstance(pair, list) or len(pair) != 2 or
                    any(type(v) is not int or v < 0 for v in pair) for pair in ns.values()):
            raise ValueError("namespace-schema")
        issued = value["issued_utc"]
        if not isinstance(issued, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", issued) is None:
            raise ValueError("utc-schema")
        at = datetime.strptime(issued, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        if at > now:
            raise Blocked("future", "receipt-issued-after-clock")
        if now - at > 86400:
            raise Blocked("stale", "receipt-older-than-24h")
        return value
    except Blocked:
        raise
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise Blocked("malformed", str(exc)) from None


def live_admission(receipt_path, pins, now, check=lambda: None):
    result = {"status": "BLOCKED", "observations": [], "reason": "receipt-required"}
    if not receipt_path:
        return result
    if not all(pins.get(k) for k in ("digest", "source", "host", "machine")):
        result["reason"] = "externally-approved-pins-required"
        return result  # No live facts are probed on an untrusted-input path.
    obs = observe("receipt", lambda: regular_bytes(receipt_path, 16384, check))
    raw = obs.pop("value", None)
    result["observations"].append(obs)
    if obs["status"] != "ok":
        result["reason"] = obs["status"]
        return result
    result["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    parsed = observe("receipt-binding", lambda: parse_receipt(raw, pins, now))
    value = parsed.pop("value", None)
    result["observations"].append(parsed)
    if parsed["status"] != "ok":
        result["reason"] = parsed["status"]
        return result
    facts = [observe("machine-id", lambda: hashlib.sha256(regular_bytes("/etc/machine-id", 128, check).strip()).hexdigest()),
             observe("boot-id", lambda: regular_bytes("/proc/sys/kernel/random/boot_id", 128, check).decode("ascii").strip())]
    for kind in ("pid", "mnt", "cgroup"):
        def pair(kind=kind):
            check()
            st = os.stat("/proc/self/ns/" + kind)
            return [st.st_dev, st.st_ino]
        facts.append(observe("namespace:" + kind, pair))
    result["observations"].extend(facts)
    if any(f["status"] != "ok" for f in facts):
        result["reason"] = next(f["status"] + ":" + f["name"] for f in facts if f["status"] != "ok")
        return result
    if facts[0]["value"] != pins["machine"]:
        result["reason"] = "wrong-live-host"
        return result
    if facts[1]["value"] != value["boot_id"]:
        result["reason"] = "stale-reboot"
        return result
    if any(f["value"] != value["namespaces"][f["name"].split(":")[1]] for f in facts[2:]):
        result["reason"] = "namespace-mismatch"
        return result
    # Actual applicable cgroup/headroom/events checks use unchanged live code.
    # Permission failures become observations, never an erased snapshot.
    def coverage():
        import live_probe
        sample = live_probe.parent_sample(value["namespaces"])
        live_probe.ProbeGuard(sample).check("ROLE_PROBE", sample)
        return "required-covered"
    checked = observe("live-guard-coverage", coverage)
    result["observations"].append(checked)
    if checked["status"] != "ok":
        result["reason"] = checked["status"]
        return result
    result.update(status="PASS", reason="approved-source-binding-and-live-coverage", namespaces=value["namespaces"])
    return result
