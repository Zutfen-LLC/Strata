"""GPU-RESERVE-WARNING-V1: local runtime warnings; admission/remote guards intact."""
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import time

from resource_guard import CoverageError, GuardStop
from linux_owned import parent_death
from policy import GIB, MIB


def gpu_free(tool, device_ids):
    if not Path(tool).is_absolute() or not device_ids or \
            len(set(device_ids)) != len(device_ids) or \
            any(not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", v) for v in device_ids):
        raise CoverageError("explicit-selected-device-bindings-required")
    parent = os.getpid()
    process = subprocess.Popen([tool, "--id=" + ",".join(device_ids),
                                "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, close_fds=True,
                               env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                               preexec_fn=lambda: parent_death(parent))
    # This unreaped direct child cannot be replaced by a reused numeric PID.
    fd = None
    try:
        fd = os.pidfd_open(process.pid, 0)
        streams = {process.stdout.fileno(): bytearray(), process.stderr.fileno(): bytearray()}
        active = set(streams)
        end = time.monotonic() + 1
        while active:
            left = end - time.monotonic()
            if left <= 0:
                raise CoverageError("gpu-management-timeout")
            ready = select.select(list(active), [], [], left)[0]
            for stream in ready:
                chunk = os.read(stream, 4097 - len(streams[stream]))
                streams[stream].extend(chunk)
                if len(streams[stream]) > 4096:
                    raise CoverageError("gpu-management-output-cap")
                if not chunk:
                    active.remove(stream)
        process.wait(timeout=max(0.001, end - time.monotonic()))
        out, err = (bytes(v) for v in streams.values())
        if process.returncode:
            raise CoverageError("gpu-management-failed")
        rows = [row.split(",") for row in out.decode("ascii").splitlines()]
        if len(rows) != len(device_ids) or any(len(row) != 2 for row in rows):
            raise CoverageError("gpu-management-schema")
        found = {}
        for uuid, free in rows:
            uuid, free = uuid.strip(), free.strip()
            if uuid not in device_ids or uuid in found or not free.isdigit():
                raise CoverageError("gpu-management-identity")
            found[uuid] = int(free) * MIB
        return [found[uuid] for uuid in device_ids]
    finally:
        try:
            if fd is None:
                # pidfd_open failed before any wait/poll: own unreaped child,
                # constructor-only cleanup; no PID reuse is possible here.
                process.kill()
            if process.poll() is None and fd is not None:
                signal.pidfd_send_signal(fd, signal.SIGKILL, None, 0)
            process.wait(timeout=1)
        finally:
            process.stdout.close()
            process.stderr.close()
            if fd is not None:
                os.close(fd)


class OuterGuards:
    def __init__(self, role, tool=None, device_ids=(), cache=None, selection=None):
        self.role, self.tool, self.device_ids = role, tool, tuple(device_ids)
        self.cache = Path(cache) if cache else None
        self.baseline = None
        self.local_minimum = None
        self.local_below = [False, False]
        self.local_crossings = [0, 0]
        if role != "ROLE_CPU_C" and (len(self.device_ids) != (2 if role == "ROLE_ENGINE_A" else 1) or
                                     selection != ",".join(self.device_ids)):
            raise CoverageError("guard-workload-device-binding")

    def sample(self):
        if self.role == "ROLE_CPU_C":
            return {}
        values = {"gpu_free_bytes": gpu_free(self.tool, self.device_ids)}
        if self.role == "ROLE_GPU_B":
            if self.cache is None or not self.cache.is_absolute():
                raise CoverageError("cache-binding-required")
            path = self.cache if self.cache.exists() else self.cache.parent
            stats = os.statvfs(path)
            values["disk_free_bytes"] = stats.f_bavail * stats.f_frsize
        return values

    def check(self, values, admission=False, phase=None):
        if self.role == "ROLE_CPU_C":
            return
        free = values.get("gpu_free_bytes")
        count = 2 if self.role == "ROLE_ENGINE_A" else 1
        if not isinstance(free, list) or len(free) != count or \
                any(type(v) is not int or v < 0 for v in free):
            raise CoverageError("selected-gpu-guard-unavailable")
        floors = (1152 * MIB, 512 * MIB) if count == 2 else (4 * GIB,)
        below = [v < minimum for v, minimum in zip(free, floors)]
        if any(below) and (admission or self.role != "ROLE_ENGINE_A"):
            raise GuardStop("gpu-reserve-floor")
        if self.role == "ROLE_ENGINE_A":
            if admission:
                # Fresh admission is still strict. The forked phase monitor
                # inherits empty diagnostics; startup does not reset any guard.
                self.local_minimum = None
                self.local_below = [False, False]
                self.local_crossings = [0, 0]
                return None
            entered = [now and not old for now, old in zip(below, self.local_below)]
            first = [entry and count == 0 for entry, count in zip(entered, self.local_crossings)]
            self.local_crossings = [count + int(entry) for count, entry in zip(self.local_crossings, entered)]
            self.local_minimum = list(free) if self.local_minimum is None else \
                [min(old, now) for old, now in zip(self.local_minimum, free)]
            self.local_below = below
            return {"policy": "GPU-RESERVE-WARNING-V1", "free_bytes": list(free),
                    "minimum_sampled_free_bytes": list(self.local_minimum),
                    "reserve_bytes": list(floors), "below_reserve": list(below),
                    "crossings": list(self.local_crossings), "first_warning": first}
        if self.role == "ROLE_GPU_B":
            disk = values.get("disk_free_bytes")
            if type(disk) is not int or disk < 0:
                raise CoverageError("cache-filesystem-guard-unavailable")
            need = 23890755584 if admission and phase == "G0" else 4 * GIB
            if disk < need:
                raise GuardStop("cache-filesystem-headroom")
            if admission and (free[0] - 4096 * MIB - 256 * MIB - 1598720) // 5222400 < 3289:
                raise GuardStop("fixed-gpu-plan-does-not-fit")

    def admit_cache(self, phase, *, resume=False):
        if self.role != "ROLE_GPU_B":
            return
        if phase == "G0" and not resume:
            os.mkdir(self.cache, 0o700)  # No overwrite or deletion of prior assets.
        elif phase not in ("G0", "G1") or not self.cache.is_dir() or self.cache.is_symlink():
            raise CoverageError("warm-cache-binding-invalid")

    def returned(self):
        """Selected devices must return to their admitted free-memory baseline.

        The historical 11-second gaps remain; this adds a required verification,
        without altering placement or freeing anybody else's allocations.
        """
        if self.role == "ROLE_CPU_C" or self.baseline is None:
            return True
        current = self.sample()
        self.check(current)
        return all(now >= old for now, old in zip(current["gpu_free_bytes"], self.baseline))
