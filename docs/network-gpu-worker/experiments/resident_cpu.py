"""Explicit CPU residency transition using the existing independent watchdog.

No process starts on import. This adds no listener, credential, route or launcher.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import stat
import struct
import time

from cache_status import MAGIC, read_exact, remaining_timeout
from linux_owned import identity
from resource_guard import CoverageError, Journal
from resource_telemetry import Identity, OwnedProcess

NO_EXPIRY = (1 << 64) - 1
ARENA = 111597465600


class ResidentJournal(Journal):
    """Two bounded 64-MiB generations; unreadable/full storage still refuses."""
    limit = 64 * 1048576

    def __init__(self, path):
        self.path = path
        super().__init__(path)

    def emit(self, marker, **data):
        raw = (json.dumps({"marker": marker, **data}, default=asdict, sort_keys=True,
                          allow_nan=False) + "\n").encode("ascii")
        if len(raw) > 65536 or len(raw) > self.limit:
            raise CoverageError("resident-journal-record-cap")
        if self.bytes + len(raw) > self.limit:
            self.close()
            previous = self.path + ".previous"
            if os.path.lexists(previous):
                st = os.lstat(previous)
                if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_nlink != 1:
                    raise CoverageError("resident-journal-previous-ownership")
            os.replace(self.path, previous)
            super().__init__(self.path)
        super().emit(marker, **data)


def cache_status(port):
    """Idle-only, one-second loopback read; never SETUP/INFER/UNLOAD/STOP."""
    if type(port) is not int or not 0 < port < 65536:
        raise CoverageError("resident-port")
    end = time.monotonic() + 1
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        remaining_timeout(sock, end)
        sock.connect(("127.0.0.1", port))
        remaining_timeout(sock, end)
        sock.sendall(struct.pack("<IHHQ", MAGIC, 16, 0, 0))
        if struct.unpack("<IHHQ", read_exact(sock, 16, end)) != (MAGIC, 17, 0, 100):
            raise CoverageError("resident-cache-status-frame")
        payload = read_exact(sock, 100, end)
    version, state, session, arena, remaining = struct.unpack_from("<IIQQQ", payload)
    complete, = struct.unpack_from("<I", payload, 96)
    if version != 1 or state not in (0, 1, 2) or complete not in (0, 1):
        raise CoverageError("resident-cache-status-schema")
    return {"state": state, "session": session, "resident_bytes": arena,
            "idle_remaining_ms": remaining, "complete": bool(complete)}


def receipt(config, job, session):
    """Identity receipt; it deliberately makes no claim that weights are loaded."""
    return {"run_id": config["run_id"], "lifetime": "service-owned-no-idle-expiry",
            "port": int(config["bindings"]["CPU_PORT_C"]), "admission": asdict(session.admission),
            "baseline": session.baseline,
            "watchdog": asdict(identity(os.getpid(), session.namespaces)),
            "worker": asdict(job.root.identity), "monitor": asdict(job.monitor.identity),
            "resources": job.directory + "/resources.jsonl"}


def grant(config, job, session):
    """Own the guarded listener independently before model setup starts."""
    job.root.verify()
    job.monitor.verify()
    job.gate.check(time.monotonic())
    value = receipt(config, job, session)
    raw = (json.dumps(value, default=asdict, allow_nan=False) + "\n").encode("ascii")
    if len(raw) > 1048576:
        raise CoverageError("resident-receipt-cap")
    fd = os.open(config["run_root"] + "/resident.json",
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    # Explicit service ownership, never renewal of an engine/GPU lease.
    job.deadline = job.gate.deadline = float("inf")
    return {k: value[k] for k in ("lifetime", "watchdog", "worker", "monitor")}


def commit(config, job, session):
    """After engine EOF, confirm actual complete no-expiry residency for report."""
    job.root.verify()
    port = int(config["bindings"]["CPU_PORT_C"])
    status = cache_status(port)
    job.root.verify()
    if status != {"state": 0, "session": 0, "resident_bytes": ARENA,
                  "idle_remaining_ms": NO_EXPIRY, "complete": True}:
        raise CoverageError("complete-idle-no-expiry-arena-required")
    value = receipt(config, job, session)
    return {k: value[k] for k in ("lifetime", "watchdog", "worker", "monitor")}


def resource_row(path):
    """A bounded last complete sample, including the normal rotation interval."""
    for candidate in (path,path+".previous"):
        try:
            with open(candidate,"rb") as source:
                source.seek(0,os.SEEK_END);source.seek(max(0,source.tell()-131072))
                rows=source.read(131073)
        except FileNotFoundError:
            if candidate==path:continue
            raise
        complete=rows.split(b"\n")[:-1]
        if len(rows)>131072:raise CoverageError("resident-resource-tail-cap")
        if complete:return json.loads(complete[-1])
    raise CoverageError("resident-resource-tail-unavailable")


def live_status(config, *, query_cache=True):
    """Read current process identities, fresh coverage and worker cache status."""
    with open(config["run_root"] + "/resident.json", "rb") as source:
        raw = source.read(1048577)
    if not 0 < len(raw) <= 1048576:
        raise CoverageError("resident-receipt-cap")
    receipt = json.loads(raw)
    if receipt["run_id"] != config["run_id"] or receipt["lifetime"] != "service-owned-no-idle-expiry":
        raise CoverageError("resident-receipt-identity")
    owners = []
    try:
        for key in ("watchdog", "worker", "monitor"):
            expected = Identity(**receipt[key])
            owners.append(OwnedProcess.pin(expected.pid, expected=expected,
                                          namespace_receipt=config["namespaces"]))
        row = resource_row(receipt["resources"])
        if row.get("marker") != "INTEGRATION_RESOURCE_SAMPLE_V1" or \
                not 0 <= time.monotonic() - row["mono"] < 4 or \
                row["key"] != [config["run_id"], "ROLE_CPU_C", "CPU"]:
            raise CoverageError("resident-coverage-not-fresh")
        status = cache_status(receipt["port"]) if query_cache else None
        if status is not None and status["complete"]:
            if status["resident_bytes"] != ARENA or status["idle_remaining_ms"] != NO_EXPIRY:
                raise CoverageError("resident-arena-changed")
        elif status is not None and status["resident_bytes"] != 0:
            raise CoverageError("resident-incomplete-arena")
        for owner in owners:
            owner.verify()
        age = time.monotonic() - row["mono"]
        if not 0 <= age < 4:
            raise CoverageError("resident-coverage-no-longer-fresh")
        return {"live": True, "coverage_age_s": age,
                "lifetime": receipt["lifetime"], "cache": status,
                "identities": {k: receipt[k] for k in ("watchdog", "worker", "monitor")}}
    finally:
        for owner in owners:
            owner.close()
