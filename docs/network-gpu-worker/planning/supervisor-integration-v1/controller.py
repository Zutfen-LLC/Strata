"""Serial trial coordination over existing private transport argv.

Source only. This code is NOT permission to run a trial or establish transport.
Every Peer/forward is an unreaped direct child; pidfds and wait() own cleanup.
"""
import json
import os
import select
import signal
import subprocess
import time

from resource_guard import CoverageError, GuardStop
from linux_owned import parent_death
from policy import ENGINE_CAPS, GPU_CAPS, TrialOrder, bounded_job_deadline
from agent import verified_file
import gpu_completion as completion


def close_all(owners):
    errors = []
    for owner in owners:
        try:
            owner.close()
        except BaseException as exc:
            errors.append(type(exc).__name__ + ":" + str(exc))
    if errors:
        raise CoverageError("incomplete-owned-cleanup:" + repr(errors))


def require_cleanup(receipt, *, final=False):
    if not receipt["cleanup"]["complete"] or (final and receipt.get("reason") != "finished"):
        raise GuardStop("incomplete-or-failed-cleanup-receipt")


def child_pidfd(process):
    try:
        return os.pidfd_open(process.pid, 0)
    except BaseException:
        # No wait/poll has happened: this is our unreaped direct child, so its
        # PID cannot be reused. This constructor-only failure is never a fallback
        # for signalling a sampled or arbitrary PID.
        try:
            process.kill()
            process.wait(timeout=1)
        finally:
            close_all(stream for stream in (process.stdin, process.stdout, process.stderr) if stream is not None)
        raise


def finish_process(process, fd):
    errors = []
    try:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except BaseException as exc:
            errors.append(type(exc).__name__)
        try:
            if process.poll() is None:
                try:
                    signal.pidfd_send_signal(fd, signal.SIGTERM, None, 0)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    signal.pidfd_send_signal(fd, signal.SIGKILL, None, 0)
                except ProcessLookupError:
                    pass
                process.wait(timeout=1)
        except BaseException as exc:
            errors.append(type(exc).__name__)
    finally:
        try:
            close_all(stream for stream in (process.stdout, process.stderr) if stream is not None)
        except BaseException as exc:
            errors.append(type(exc).__name__)
        try:
            os.close(fd)
        except OSError as exc:
            errors.append(type(exc).__name__)
    if errors:
        raise CoverageError("owned-transport-cleanup-incomplete:" + repr(errors))


class Peer:
    def __init__(self, role, run_id, argv):
        self.role, self.run_id = role, run_id
        parent = os.getpid()
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, close_fds=True,
                                        preexec_fn=lambda: parent_death(parent))
        self.pidfd = child_pidfd(self.process)
        self.closed = False
        try:
            os.set_blocking(self.process.stdout.fileno(), False)
            os.set_blocking(self.process.stdin.fileno(), False)
        except BaseException:
            self.close()
            raise
        self.buffer = bytearray()
        self.sequence = 0
        self.pending = None
        self.events = []
        self.final_seen = False
        self.covered = False
        self.expected_finish = False
        self.last_progress = time.monotonic()
        self.cleaning_until = 0

    def queue(self, action, **fields):
        if self.pending is not None:
            raise CoverageError("duplicate-pending-command")
        if action == "FINISH":
            self.expected_finish = True
        self.pending = {"action": action, **fields}

    def pump(self, remaining, success_age=0):
        data = os.read(self.process.stdout.fileno(), 4096)
        if not data:
            if self.final_seen:
                return
            raise CoverageError("peer-eof:" + self.role)
        self.buffer.extend(data)
        self.last_progress = time.monotonic()
        if len(self.buffer) > 8192:
            raise CoverageError("peer-output-cap")
        while b"\n" in self.buffer:
            raw, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            message = json.loads(raw)
            if message.get("run_id") != self.run_id or message.get("role") != self.role:
                raise CoverageError("stale-peer-identity")
            if message.get("kind") == "CHALLENGE":
                if message.get("sequence") != self.sequence:
                    raise CoverageError("stale-peer-sequence")
                request = {"run_id": self.run_id, "sequence": self.sequence,
                           "nonce": message["nonce"], "remaining_s": remaining,
                           **(self.pending or {"action": "HEARTBEAT"})}
                if request["action"] == "RETAIN":
                    request["success_age_s"] = success_age
                payload = (json.dumps(request, allow_nan=False) + "\n").encode("ascii")
                if len(payload) > 4096 or os.write(self.process.stdin.fileno(), payload) != len(payload):
                    raise CoverageError("controller-write-cap-or-short-write")
                self.pending = None
                self.sequence += 1
            else:
                if message.get("kind") == "CLEANING":
                    self.cleaning_until = time.monotonic() + 45
                if message.get("kind") == "LISTENING_COVERED":
                    self.covered = True
                if message.get("kind") in ("STOPPED", "FINAL"):
                    self.covered = False
                if message.get("kind") == "FINAL" and (not self.expected_finish or
                        message.get("reason") != "finished" or not message["cleanup"]["complete"]):
                    raise GuardStop("required-peer-failed:" + self.role)
                self.final_seen = self.final_seen or message.get("kind") == "FINAL"
                self.events.append(message)
                if len(self.events) > 32:
                    raise CoverageError("peer-event-cap")

    def event(self, kind, phase=None):
        for index, event in enumerate(self.events):
            if event.get("kind") == "FINAL" and kind != "FINAL":
                raise GuardStop("peer-stopped-before-phase-completion")
            if event.get("kind") == kind and (phase is None or event.get("phase") == phase):
                return self.events.pop(index)
        return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        fd, self.pidfd = self.pidfd, None
        finish_process(self.process, fd)


class Controller:
    def __init__(self, peers, *, retain=False, clock=time.monotonic, start=None,
                 gpu_completion=False, baseline=None):
        self.peers = peers
        self.clock = clock
        self.start = clock() if start is None else start
        if type(gpu_completion) is not bool or (gpu_completion and retain):
            raise CoverageError("bounded-completion-mode-or-retention")
        self.gpu_completion = gpu_completion
        self.baseline = completion.validate_baseline(baseline, verified_file) if gpu_completion else None
        self.engine_caps = completion.ENGINE_CAPS if gpu_completion else ENGINE_CAPS
        self.gpu_caps = completion.GPU_CAPS if gpu_completion else GPU_CAPS
        self.phases = tuple(self.engine_caps)
        self.deadline = self.start + (completion.WINDOW_S if gpu_completion else 5400)
        self.order = TrialOrder(self.phases)
        self.retain = retain
        self.success_at = None

    def pump(self):
        remaining = self.deadline - 45 - self.clock()
        if remaining <= 20:
            raise GuardStop("global-cleanup-reserve")
        active = [p for p in self.peers.values() if not p.closed and not p.final_seen]
        ready = select.select([p.process.stdout for p in active], [], [], 0.05)[0]
        for peer in self.peers.values():
            if not peer.closed and peer.process.stdout in ready:
                age = 0 if self.success_at is None else self.clock() - self.success_at
                peer.pump(remaining, age)
        if any(not p.closed and not p.final_seen and self.clock() > max(p.last_progress + 7, p.cleaning_until)
               for p in self.peers.values()):
            raise GuardStop("required-peer-stalled")

    def wait(self, role, kind, phase=None, cap=30):
        end = min(self.deadline - 45, self.clock() + cap)
        while self.clock() < end:
            result = self.peers[role].event(kind, phase)
            if result is not None:
                return result
            self.pump()
        raise GuardStop("bounded-peer-readiness-or-completion-timeout")

    def gap(self):
        end = self.clock() + 11
        while self.clock() < end:
            self.pump()

    def phase_cap(self, cap):
        now = self.clock()
        # The existing 45-second global reserve and five-second challenge
        # transit allowance remain real. No full-maximum-envelope admission.
        return bounded_job_deadline(now, now + cap, self.deadline - 45 - 5) - now

    def start_phase(self, peer, phase):
        peer.queue("START", phase=phase, **({"gpu_completion": True} if self.gpu_completion else {}))

    def run(self):
        cpu, engine, gpu = (self.peers[r] for r in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B"))
        try:
            cpu_cap = self.phase_cap(7200)
            self.start_phase(cpu, "CPU")
            self.wait(cpu.role, "LISTENING_COVERED", "CPU", min(450, cpu_cap))
            for phase in self.phases:
                if phase.startswith("G"):
                    gpu_cap = self.phase_cap(self.gpu_caps[phase])
                    self.start_phase(gpu, phase)
                    self.wait(gpu.role, "LISTENING_COVERED", phase, min(30, gpu_cap))
                self.order.permit(phase, cpu.covered, gpu.covered)
                engine_cap = self.phase_cap(self.engine_caps[phase])
                self.start_phase(engine, phase)
                done = self.wait(engine.role, "DONE", phase, engine_cap)
                self.order.finish(phase, done["rc"], done["cleanup"]["complete"])
                if phase == "G1":
                    self.success_at = self.clock()  # Cleanup consumes the retention grant.
                if phase.startswith("G"):
                    gpu.queue("STOP")
                    result = self.wait(gpu.role, "STOPPED", phase, 45)
                    require_cleanup(result)
                if phase in ("B0", "G0"):
                    self.gap()
            for peer in (engine, gpu):
                peer.queue("FINISH")
                final = self.wait(peer.role, "FINAL", cap=20)
                require_cleanup(final, final=True)
                peer.close()
            if self.retain:
                cpu.queue("RETAIN", completed_phases=["B0", "B1", "G0", "G1"])
                self.wait(cpu.role, "RETENTION_COMMITTED", cap=5)
            else:
                cpu.queue("STOP")
                stopped = self.wait(cpu.role, "STOPPED", "CPU", 45)
                require_cleanup(stopped)
                cpu.queue("FINISH")
                final = self.wait(cpu.role, "FINAL", cap=5)
                require_cleanup(final, final=True)
            result = {"complete": True, "retained_cpu": self.retain}
            if self.gpu_completion:
                result.update(completed_phases=list(self.phases), gpu_completion=True,
                              baseline_archive_sha256=self.baseline["archive_sha256"],
                              historical_active_s=completion.HISTORICAL_ACTIVE_S,
                              window_start_mono=self.start, window_deadline_mono=self.deadline,
                              new_request_allowance=3, new_token_allowance=192)
            return result
        except BaseException:
            # A live peer will receive CANCEL on its next fresh challenge. If
            # the controller/transport has failed, EOF or its five-second lease
            # independently stops that host's work. Never launch another phase.
            for peer in self.peers.values():
                if not peer.closed and not peer.final_seen:
                    peer.pending = {"action": "CANCEL"}
            end = min(self.deadline, self.clock() + 20)
            while self.clock() < end and any(not p.closed and not p.final_seen for p in self.peers.values()):
                try:
                    active = [p for p in self.peers.values() if not p.closed and not p.final_seen]
                    ready = select.select([p.process.stdout for p in active], [], [], 0.05)[0]
                    for peer in active:
                        if peer.process.stdout in ready:
                            peer.pump(max(21, self.deadline - self.clock()))
                except BaseException:
                    break
            raise
        finally:
            # Closing control channels cancels uncommitted remote work. A committed
            # CPU retention daemon independently enforces its original local cap.
            close_all(self.peers.values())


class Forward:
    """An exact, privately approved existing transport command; no shell.

    Establishment/endpoint readiness remains a separately approved prerequisite.
    This parent owns only the command it starts. No preexisting tunnel is stopped.
    """
    def __init__(self, argv):
        parent = os.getpid()
        self.process = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        close_fds=True, preexec_fn=lambda: parent_death(parent))
        self.pidfd = child_pidfd(self.process)
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        fd, self.pidfd = self.pidfd, None
        finish_process(self.process, fd)


def coordinated(peers, forwards, *, retain=False):
    """Caller creates at most four approved forwards and three Peer objects.

    All creation failures require closing every object already returned. A failed
    constructor owns its own partial child cleanup. No retries or replacements.
    """
    try:
        return Controller(peers, retain=retain).run()
    finally:
        close_all(forwards)


def launch(config):
    """Private, separately approved argv only; no generated SSH/network options."""
    roles = {"ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B"}
    if set(config["peers"]) != roles or len(config["forwards"]) > 4:
        raise CoverageError("bounded-transport-config")
    start = time.monotonic()
    forwards, peers = [], {}
    try:
        for argv in config["forwards"]:
            forwards.append(Forward(argv))
        for role, argv in config["peers"].items():
            peers[role] = Peer(role, config["run_id"], argv)
        return Controller(peers, retain=config.get("retain", False), start=start,
                          gpu_completion=config.get("gpu_completion", False),
                          baseline=config.get("baseline")).run()
    finally:
        close_all([*peers.values(), *forwards])
