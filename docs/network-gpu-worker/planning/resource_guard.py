"""New ROLE_CPU_C replacement memory guard, revision 1. NOT EXECUTED.

Library integration seam, not an integrated patch to an unavailable driver.
No workload launch, numeric PID kill, network operation or placement changes.
"""
from dataclasses import asdict
import errno
import json
import math
import os
from pathlib import PurePosixPath
import re
import signal
import time

from resource_telemetry import (CoverageError, DiagnosticBaselines, Observation,
                                ProcessExited,
                                host_sample, read_text, required, counters,
                                verify_initial_namespaces)

MIB = 1048576
GIB = 1024 * MIB
SWAP_DECLINE_BYTES = 65536 * 1024
CPU_ADMISSION_BYTES = (107000 + 16384) * MIB
CPU_STOP_HEADROOM_BYTES = 8 * GIB
CPU_RSS_CAP_BYTES = 108024 * MIB
EVENT_KEYS = ("high", "max", "oom", "oom_kill")
MAX_ANCESTORS = 64
SAMPLE_SECONDS = 2.0


class GuardStop(CoverageError):
    """An evaluated established guard crossed its stopping boundary."""


def canonical(path):
    if not path.startswith("/") or any(p in (".", "..", "") for p in path.split("/")[1:]):
        if path != "/":
            raise CoverageError("noncanonical-cgroup-path")
    return PurePosixPath(path)


def membership(text):
    raw = required(text, "cgroup-membership")
    rows = [line.split(":", 2) for line in raw.splitlines()]
    if len(rows) != 1 or len(rows[0]) != 3 or rows[0][:2] != ["0", ""]:
        raise CoverageError("unsupported-v1-hybrid-or-membership-schema")
    return canonical(rows[0][2])


def mount_unescape(raw):
    escapes = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    decoded = re.sub(r"\\(040|011|012|134)", lambda m: escapes[m[1]], raw)
    if re.search(r"\\[0-9]{3}", decoded):
        raise CoverageError("mount-escape-schema")
    return decoded


def full_mount(text):
    mounts = []
    for line in required(text, "mountinfo").splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            raise CoverageError("mountinfo-schema")
        pre, post = left.split(), right.split()
        if len(pre) < 6 or len(post) < 3:
            raise CoverageError("mountinfo-schema")
        if post[0] == "cgroup":
            raise CoverageError("unsupported-hybrid-mounts")
        if post[0] == "cgroup2":
            root = canonical(mount_unescape(pre[3]))
            point = canonical(mount_unescape(pre[4]))
            mounts.append((pre[0], pre[2], str(root), str(point), post[2]))
    if len(mounts) != 1 or mounts[0][2] != "/":
        raise CoverageError("hidden-ancestors-or-ambiguous-cgroup2-mount")
    return mounts[0]


def ancestors(leaf, mount):
    """Map membership relative to mount root; include actual root explicitly."""
    path = PurePosixPath(mount[3]) / leaf.relative_to("/")
    root = PurePosixPath(mount[3])
    result = []
    while True:
        result.append(str(path))
        if len(result) > MAX_ANCESTORS:
            raise CoverageError("cgroup-depth-cap")
        if path == root:
            return result
        if root not in path.parents:
            raise CoverageError("cgroup-mount-containment")
        path = path.parent


def namespace_identity(name, dir_fd=None):
    st = os.stat(name, dir_fd=dir_fd)
    return st.st_dev, st.st_ino


def verify_namespaces(process, expected):
    # expected is a PRIVATE trusted receipt for the INITIAL host namespaces.
    # Equality to a container's PID 1 alone does not establish this prerequisite.
    for kind in ("cgroup", "mnt", "pid"):
        try:
            actual = namespace_identity("ns/" + kind, process.proc_fd)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ESRCH):
                process.reconcile_missing("ns/" + kind)
            raise
        monitor = namespace_identity("/proc/self/ns/" + kind)
        if actual != tuple(expected[kind]) or monitor != tuple(expected[kind]):
            raise CoverageError("initial-host-namespace-proof-required:" + kind)


def scalar(text, allow_max=False):
    value = required(text, "cgroup-scalar").strip()
    if value == "max" and allow_max:
        return None
    if not value.isdigit():
        raise CoverageError("cgroup-scalar-schema")
    return int(value)


def cgroup_snapshot(process, namespace_receipt):
    process.verify()
    verify_namespaces(process, namespace_receipt)
    raw_membership = process.read("cgroup")
    leaf = membership(raw_membership)
    mount_text = read_text("/proc/self/mountinfo")
    mount = full_mount(mount_text)
    rows = []
    paths = ancestors(leaf, mount)
    for index, path in enumerate(paths):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            row = {"path": path, "directory_identity": (st.st_dev, st.st_ino)}
            if index == len(paths) - 1:
                controllers = required(read_text("cgroup.controllers", fd), "root-controllers")
                if "memory" not in controllers.split():
                    raise CoverageError("memory-controller-unavailable")
                row["coverage"] = "root-not-resource-controlled"
                row["events"] = {k: Observation(None, "unsupported", "true-root-interface")
                                 for k in EVENT_KEYS}
                row.update(current=None, maximum=None, high=None)
            else:
                if required(read_text("cgroup.type", fd), "cgroup-type").strip() != "domain":
                    raise CoverageError("unsupported-threaded-cgroup")
                row["coverage"] = "required-covered"
                row["current"] = scalar(read_text("memory.current", fd))
                row["maximum"] = scalar(read_text("memory.max", fd), True)
                row["high"] = scalar(read_text("memory.high", fd), True)
                row["events"] = counters(read_text("memory.events", fd), EVENT_KEYS)
                for key in EVENT_KEYS:
                    required(row["events"][key], "memory.events:" + key)
            rows.append(row)
        except (CoverageError, OSError) as exc:
            raise CoverageError("required-cgroup-node:" + path + ":" + str(exc)) from None
        finally:
            os.close(fd)
    verify_namespaces(process, namespace_receipt)
    if process.read("cgroup") != raw_membership or read_text("/proc/self/mountinfo") != mount_text:
        raise CoverageError("cgroup-discovery-changed-during-sample")
    return {"membership": str(leaf), "mount": mount, "rows": rows,
            "coverage": "required-covered", "namespace_receipt": namespace_receipt}


def headroom(snapshot):
    values = []
    for row in snapshot["rows"]:
        if row["coverage"] == "root-not-resource-controlled":
            continue
        for key in ("maximum", "high"):
            if row[key] is not None:
                values.append(max(0, row[key] - row["current"]))
    # Unlimited means no finite cap; it does NOT bypass event coverage.
    return min(values) if values else None


def signature(snapshot):
    return (snapshot["membership"], snapshot["mount"], snapshot["namespace_receipt"],
            [(r["path"], r["directory_identity"], r["maximum"], r["high"], r["coverage"])
             for r in snapshot["rows"]])


class CpuGuard:
    """Immutable launch baseline. New instance only for a separately admitted session."""
    def __init__(self, sample, admission_swapfree_bytes=None):
        total, free = self.swap_values(sample)
        self.swap_baseline = free if admission_swapfree_bytes is None else admission_swapfree_bytes
        if type(self.swap_baseline) is not int or not 0 <= self.swap_baseline <= total:
            raise CoverageError("original-admission-swapfree-schema")
        self.cgroup_baseline = sample["cgroup"]
        self.check_memory(sample, CPU_ADMISSION_BYTES)
        self.check_rss(sample)
        self.check(sample)

    @staticmethod
    def swap_values(sample):
        total = required(sample["host_bytes"]["SwapTotal"], "SwapTotal")
        free = required(sample["host_bytes"]["SwapFree"], "SwapFree")
        if not 0 <= free <= total:
            raise CoverageError("swap-level-schema")
        return total, free

    @staticmethod
    def check_memory(sample, floor):
        if required(sample["host_bytes"]["MemAvailable"], "MemAvailable") < floor:
            raise GuardStop("host-headroom")
        room = headroom(sample["cgroup"])
        if room is not None and room < floor:
            raise GuardStop("cgroup-headroom")

    @staticmethod
    def check_rss(sample):
        rss = required(sample["worker_memory_bytes"]["VmRSS"], "worker-VmRSS")
        if rss > CPU_RSS_CAP_BYTES:
            raise GuardStop("worker-rss-ceiling")

    def check(self, sample):
        self.check_memory(sample, CPU_STOP_HEADROOM_BYTES)
        self.check_rss(sample)
        _, free = self.swap_values(sample)
        # Strict inequality: exactly 65536 KiB decline does NOT stop.
        if free < self.swap_baseline - SWAP_DECLINE_BYTES:
            raise GuardStop("swapfree-decline-greater-than-64MiB")
        if signature(sample["cgroup"]) != signature(self.cgroup_baseline):
            raise CoverageError("cgroup-coverage-or-limit-changed")
        for row, initial in zip(sample["cgroup"]["rows"], self.cgroup_baseline["rows"]):
            if row["coverage"] == "root-not-resource-controlled":
                continue
            for key in EVENT_KEYS:
                now = required(row["events"][key], "memory.events:" + key)
                baseline = required(initial["events"][key], "event-baseline:" + key)
                if now < baseline:
                    raise CoverageError("event-counter-decreased")
                if now > baseline:
                    raise GuardStop("new-memory-event:" + key)


def capture(tree, worker_identity, namespace_receipt):
    verify_initial_namespaces(namespace_receipt, tree.root.proc_fd, process=tree.root)
    items = tree.refresh()
    worker = next((p for p in items if p.identity == worker_identity), None)
    if worker is None:
        raise CoverageError("worker-not-in-owned-tree")
    sample = host_sample()
    sample["cgroup"] = cgroup_snapshot(worker, namespace_receipt)
    sample["processes"] = []
    for process in items:
        try:
            row = capture_process(process, namespace_receipt, sample["cgroup"]["membership"])
        except ProcessExited:
            if process is tree.root or process is worker:
                raise
            continue
        sample["processes"].append(row)
    sample["worker_memory_bytes"] = next(p["memory_bytes"] for p in sample["processes"]
                                         if p["identity"]["pid"] == worker_identity.pid)
    tree.root.verify()
    return sample


def capture_process(process, namespace_receipt, cgroup_membership):
    """One live owned row; only a held-pidfd-confirmed exit is terminal."""
    row = process.status()
    verify_namespaces(process, namespace_receipt)
    if str(membership(process.read("cgroup"))) != cgroup_membership:
        raise CoverageError("owned-descendant-outside-covered-group")
    process.verify()
    return row


class Journal:
    """New private JSONL file; a full/unwritable log is a coverage failure."""
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC |
                          os.O_NOFOLLOW, 0o600)
        self.bytes = 0

    def emit(self, marker, **data):
        raw = (json.dumps({"marker": marker, **data}, default=asdict, sort_keys=True,
                          allow_nan=False) + "\n").encode("ascii")
        if len(raw) > 65536 or self.bytes + len(raw) > 64 * MIB:
            raise CoverageError("journal-cap")
        if os.write(self.fd, raw) != len(raw):
            raise CoverageError("journal-short-write")
        self.bytes += len(raw)

    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


def stop_owned(tree, grace_seconds=15, clock=time.monotonic, pause=time.sleep):
    """TERM wrapper first, then pinned children; never signal a replacement PID."""
    outcomes = []
    items = list(tree.items.values())
    liveness_errors = set()

    def live(item):
        try:
            return item.alive()
        except Exception as exc:
            if item.identity.pid not in liveness_errors:
                outcomes.append((item.identity.pid, "LIVENESS", type(exc).__name__))
                liveness_errors.add(item.identity.pid)
            return True  # Unknown is not proof of exit; still attempt safe pidfd cleanup.

    for item in items:
        try:
            outcomes.append((item.identity.pid, "TERM", item.send(signal.SIGTERM)))
        except Exception as exc:
            outcomes.append((item.identity.pid, "TERM", type(exc).__name__))
    end = clock() + max(0, min(15, grace_seconds))
    while any(live(p) for p in items) and clock() < end:
        pause(min(0.1, max(0, end - clock())))
    for item in items:
        if live(item):
            try:
                outcomes.append((item.identity.pid, "KILL", item.send(signal.SIGKILL)))
            except Exception as exc:
                outcomes.append((item.identity.pid, "KILL", type(exc).__name__))
    for item in items:
        outcomes.append((item.identity.pid, "EXIT_OBSERVATION",
                         "unconfirmed" if live(item) else "pidfd-exited-not-reaped"))
    return outcomes


def monitor(tree, worker_identity, namespace_receipt, hard_deadline, journal, *,
            admission_swapfree_bytes=None, admission_monotonic=None):
    """Blocking library seam. Supervisor must gate SETUP on ADMITTED marker.

    hard_deadline is the earliest ORIGINAL worker/trial/phase cap, in this
    host's monotonic clock, including 15 s cleanup; never restart that clock.
    External supervisor must retain its independent watchdog and reap children.
    """
    started = time.monotonic()
    cleanup_deadline = started
    reason = "coverage-failure"
    coverage_status = "failed"
    try:
        if type(admission_swapfree_bytes) is not int or admission_swapfree_bytes < 0 or \
                not isinstance(admission_monotonic, (int, float)) or \
                not math.isfinite(admission_monotonic) or not admission_monotonic <= started:
            raise CoverageError("original-admission-baseline-receipt-required")
        if not isinstance(hard_deadline, (int, float)) or not math.isfinite(hard_deadline) or \
                not started + 15 < hard_deadline <= admission_monotonic + 7200:
            raise CoverageError("invalid-original-hard-deadline")
        cleanup_deadline = hard_deadline
        initial = capture(tree, worker_identity, namespace_receipt)
        guard = CpuGuard(initial, admission_swapfree_bytes)
        diagnostics = DiagnosticBaselines(initial)
        if time.monotonic() - started >= SAMPLE_SECONDS:
            raise CoverageError("admission-overran-two-seconds")
        journal.emit("RESOURCE_MONITOR_ADMITTED_V1", monotonic=started, sample=initial,
                     admission_monotonic=admission_monotonic,
                     swapfree_baseline_bytes=guard.swap_baseline)
        next_sample = started + SAMPLE_SECONDS
        while True:
            now = time.monotonic()
            if now >= hard_deadline - 15:
                raise GuardStop("original-hard-deadline-cleanup")
            time.sleep(min(max(0, next_sample - now), max(0, hard_deadline - 15 - now)))
            if time.monotonic() >= hard_deadline - 15:
                raise GuardStop("original-hard-deadline-cleanup")
            began = time.monotonic()
            sample = capture(tree, worker_identity, namespace_receipt)
            diagnostics.annotate(sample)
            journal.emit("RESOURCE_SAMPLE_V1", monotonic=began, sample=sample,
                         swapfree_baseline_bytes=guard.swap_baseline)
            guard.check(sample)
            if time.monotonic() - began >= SAMPLE_SECONDS:
                raise CoverageError("sample-overran-two-seconds")
            next_sample = began + SAMPLE_SECONDS
    except Exception as exc:
        reason = type(exc).__name__ + ":" + str(exc)
        coverage_status = "covered" if isinstance(exc, GuardStop) else "failed"
    finally:
        # Always attempt owned cleanup even if collection, logging or caller fails.
        try:
            try:
                outcomes = stop_owned(tree, min(15, max(0, cleanup_deadline - time.monotonic())))
            except Exception as exc:
                outcomes = [("cleanup", "unconfirmed", type(exc).__name__)]
            journal.emit("RESOURCE_MONITOR_STOP_V1", reason=reason, coverage_status=coverage_status,
                         cleanup_status="requires-supervisor-confirmation", outcomes=outcomes)
        finally:
            try:
                journal.close()
            finally:
                tree.close()
    return reason
