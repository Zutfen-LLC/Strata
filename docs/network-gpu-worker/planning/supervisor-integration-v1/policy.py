"""Explicit role policies and host-local lifetime contracts. NOT EXECUTED."""
from dataclasses import dataclass
import math

from resource_guard import (CoverageError, GuardStop, CpuGuard, EVENT_KEYS,
                            headroom, signature, SWAP_DECLINE_BYTES)
from resource_telemetry import required

MIB = 1048576
GIB = 1024 * MIB
PHASES = ("B0", "B1", "G0", "G1")
ENGINE_CAPS = dict(zip(PHASES, (1200, 600, 1800, 900)))
GPU_CAPS = {"G0": 1900, "G1": 1000}


@dataclass(frozen=True)
class Policy:
    role: str
    admission: int
    headroom: int
    rss: int
    anonymous: object = None


POLICIES = {
    "ROLE_CPU_C": Policy("ROLE_CPU_C", (107000 + 16384) * MIB, 8 * GIB, 108024 * MIB),
    "ROLE_ENGINE_A": Policy("ROLE_ENGINE_A", 12 * GIB, 4 * GIB, 24 * GIB, 8 * GIB),
    "ROLE_GPU_B": Policy("ROLE_GPU_B", 4 * GIB, 2 * GIB, 2 * GIB),
}


class RoleGuard:
    def __init__(self, role, baseline):
        if role not in POLICIES:
            raise CoverageError("unsupported-role")
        self.policy = POLICIES[role]
        self.baseline = baseline
        self.swap_baseline = CpuGuard.swap_values(baseline)[1]

    def check(self, role, sample, admission=False):
        if role != self.policy.role:
            raise CoverageError("role-policy-mismatch")
        if signature(sample["cgroup"]) != signature(self.baseline["cgroup"]):
            raise CoverageError("cgroup-coverage-or-limit-changed")
        minimum = self.policy.admission if admission else self.policy.headroom
        if required(sample["host_bytes"]["MemAvailable"], "MemAvailable") < minimum:
            raise GuardStop("role-host-headroom")
        room = headroom(sample["cgroup"])
        if room is not None and room < minimum:
            raise GuardStop("role-cgroup-headroom")
        memory = sample["worker_memory_bytes"]
        if required(memory["VmRSS"], "VmRSS") > self.policy.rss:
            raise GuardStop("role-rss")
        if self.policy.anonymous is not None and \
                required(memory["RssAnon"], "RssAnon") > self.policy.anonymous:
            raise GuardStop("role-anonymous")
        _, free = CpuGuard.swap_values(sample)
        if free < self.swap_baseline - SWAP_DECLINE_BYTES:
            raise GuardStop("fixed-admission-swap-decline-greater-than-64MiB")
        for row, old in zip(sample["cgroup"]["rows"], self.baseline["cgroup"]["rows"]):
            if row["coverage"] == "root-not-resource-controlled":
                continue
            for key in EVENT_KEYS:
                value = required(row["events"][key], "memory.events:" + key)
                anchor = required(old["events"][key], "baseline-memory.events:" + key)
                if value < anchor:
                    raise CoverageError("event-counter-decreased")
                if value > anchor:
                    raise GuardStop("new-memory-event:" + key)


@dataclass(frozen=True)
class Stamp:
    domain: str
    mono: float
    wall: float  # Correlation ONLY. Never used to derive a deadline.


def duration(value, cap):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= cap:
        raise CoverageError("invalid-duration")
    return value


def local_deadline(stamp, now, seconds, domain):
    if stamp.domain != domain or not math.isfinite(stamp.mono) or stamp.mono > now:
        raise CoverageError("foreign-or-invalid-clock-domain")
    return stamp.mono + duration(seconds, 7200)


def retention_deadline(admission, completion, idle_query_start, remaining_ms, domain):
    if any(s.domain != domain for s in (admission, completion, idle_query_start)):
        raise CoverageError("foreign-retention-clock")
    if not admission.mono <= completion.mono <= idle_query_start.mono or \
            type(remaining_ms) is not int or not 0 < remaining_ms <= 1800000:
        raise CoverageError("invalid-idle-receipt")
    # Query-start anchor is conservative: query/network latency consumes TTL.
    return min(admission.mono + 7200, completion.mono + 1800,
               idle_query_start.mono + remaining_ms / 1000)


def accounting_deadline(now, original, global_limit):
    return min(now + 30, original, global_limit - 15)


def bounded_job_deadline(now, original, global_limit):
    """Clip existing upper bounds; retain 15 seconds for owned cleanup."""
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for v in (now, original, global_limit)):
        raise CoverageError("invalid-phase-deadline")
    deadline = min(original, global_limit)
    if deadline <= now + 15:
        raise GuardStop("no-positive-phase-interval-after-cleanup")
    return deadline


class ReleaseGate:
    """No stale readiness, late heartbeat or partial launch can release exec."""
    def __init__(self, run_id, role, phase, now, deadline):
        self.key = (run_id, role, phase)
        self.ready_until = min(now + 10, deadline - 15)
        self.deadline = deadline
        self.covered_at = None
        self.watchdog = False
        self.released = False

    def covered(self, key, now):
        if key != self.key or self.released:
            raise CoverageError("stale-monitor-receipt")
        self.covered_at = now

    def release(self, key, now):
        if key != self.key or self.released or not self.watchdog or self.covered_at is None or \
                not self.covered_at <= now <= min(self.ready_until, self.covered_at + 4):
            raise CoverageError("release-refused")
        self.released = True

    def check(self, now):
        if now >= self.deadline - 15:
            raise GuardStop("original-deadline")
        if not self.released and now > self.ready_until:
            raise GuardStop("monitor-never-ready")
        if self.released and (self.covered_at is None or now - self.covered_at > 4):
            raise GuardStop("monitor-stalled")


class TrialOrder:
    def __init__(self, phases=PHASES):
        if tuple(phases) not in (PHASES, ("G0", "G1"), ("G1",)):
            raise CoverageError("unsupported-trial-schedule")
        self.phases = tuple(phases)
        self.next = 0
        self.failed = False

    def permit(self, phase, cpu_covered, gpu_covered):
        if self.failed or self.next >= len(self.phases) or self.phases[self.next] != phase or not cpu_covered or \
                (phase.startswith("G") and not gpu_covered):
            raise CoverageError("phase-release-refused")

    def finish(self, phase, rc, cleaned):
        if self.next >= len(self.phases) or self.phases[self.next] != phase or rc != 0 or not cleaned:
            self.failed = True
            raise GuardStop("phase-failed-no-dependent-launch")
        self.next += 1
