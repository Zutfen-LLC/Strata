"""Detached host watchdog protocol. Static source; NOT EXECUTED.

No model window is authorized by possessing a config or these files.
The private config supplies frozen bindings, initial namespace provenance,
verified executable identities and existing transport. Never commit it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import time
import uuid

from resource_guard import CoverageError, Journal
from resource_telemetry import required, read_text
from cache_status import idle_deadline
from commands import product, REQUESTS
from outer import OuterGuards
from policy import ENGINE_CAPS, GPU_CAPS, Stamp, duration
from runtime import Job, Session, packet, receive


def owned_listener(process, port):
    process.verify()
    fd = os.open("fd", os.O_RDONLY | os.O_DIRECTORY, dir_fd=process.proc_fd)
    try:
        with os.scandir(fd) as entries:
            names = []
            for entry in entries:
                names.append(entry.name)
                if len(names) > 256:
                    raise CoverageError("owned-fd-cap")
        sockets = {os.readlink(name, dir_fd=fd) for name in names}
        table = required(process.read("net/tcp"), "owned-listener-table")
        for row in table.splitlines()[1:]:
            fields = row.split()
            if len(fields) < 10:
                raise CoverageError("tcp-table-schema")
            if fields[1] == "0100007F:" + format(port, "04X") and fields[3] == "0A" and \
                    "socket:[" + fields[9] + "]" in sockets:
                process.verify()
                return True
        return False
    finally:
        os.close(fd)


def verified_file(path, expected, cap):
    st = os.stat(path)
    if not Path(path).is_absolute() or not 0 < st.st_size <= cap or \
            not isinstance(expected, str) or len(expected) != 64:
        raise CoverageError("bounded-binary-provenance-required")
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as file:
        total = 0
        while True:
            chunk = file.read(min(1048576, cap + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > cap:
                raise CoverageError("provenance-read-cap")
            digest.update(chunk)
    current = os.stat(path)
    stable = lambda value: (value.st_dev, value.st_ino, value.st_size,
                            value.st_mtime_ns, value.st_ctime_ns)
    if digest.hexdigest() != expected or stable(current) != stable(st):
        raise CoverageError("live-binary-provenance-mismatch")


def serve(config, channel):
    role, run_id = config["role"], config["run_id"]
    if role not in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B"):
        raise CoverageError("role-config")
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    cancelled = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))
    root = config["run_root"]
    os.mkdir(root, 0o700)
    journal = Journal(root + "/watchdog.jsonl")
    outer = OuterGuards(role, config.get("gpu_tool"), config.get("gpu_query_ids", ()),
                        config["bindings"].get("CACHE_B"))
    session = Session(run_id, role, uuid.uuid4().hex, config["namespaces"], outer)
    job = None
    detached = False
    challenge = None
    challenge_end = 0
    next_challenge = 0
    sequence = 0
    global_limit = float("inf")
    expected_phase = 0
    phase_list = ("CPU",) if role == "ROLE_CPU_C" else tuple(ENGINE_CAPS if role == "ROLE_ENGINE_A" else GPU_CAPS)
    listen_deadline = None
    listener_sent = False
    try:
        while True:
            now = time.monotonic()
            if cancelled[0] or (not detached and now >= global_limit - 15):
                raise CoverageError("cancelled-or-global-cleanup-bound")
            if job is not None:
                rc = job.poll()
                if rc is not None:
                    result = job.stop("workload-exit")
                    journal.emit("JOB_REAPED_V1", phase=job.key[2], rc=rc, cleanup=result)
                    packet(channel, {"kind": "DONE", "run_id": run_id, "role": role,
                                     "phase": job.key[2], "rc": rc, "cleanup": result})
                    job = None
                    if rc != 0 or not result["complete"] or role != "ROLE_ENGINE_A":
                        raise CoverageError("unexpected-workload-exit-or-incomplete-cleanup")
                    expected_phase += 1
                elif job.gate.released and role != "ROLE_ENGINE_A" and not listener_sent:
                    port = int(config["bindings"]["CPU_PORT_C" if role == "ROLE_CPU_C" else "GPU_PORT_B"])
                    if owned_listener(job.root, port):
                        packet(channel, {"kind": "LISTENING_COVERED", "run_id": run_id,
                                         "role": role, "phase": job.key[2]})
                        listener_sent = True
                    elif now > listen_deadline:
                        raise CoverageError("owned-listener-never-ready")
            if detached:
                time.sleep(0.05)
                continue
            if challenge is None and now >= next_challenge:
                challenge = uuid.uuid4().hex
                challenge_end = now + 5
                packet(channel, {"kind": "CHALLENGE", "run_id": run_id, "role": role,
                                 "sequence": sequence, "nonce": challenge})
            if challenge is not None and now >= challenge_end:
                raise CoverageError("controller-lease-expired")
            if channel not in select.select([channel], [], [], 0.05)[0]:
                continue
            request = receive(channel)
            if request.get("run_id") != run_id or request.get("sequence") != sequence or \
                    request.get("nonce") != challenge or now >= challenge_end:
                raise CoverageError("stale-controller-request")
            # Fresh challenge bounds transit age; durations, never raw remote clocks.
            remaining = duration(request["remaining_s"], 5400)
            if remaining <= 20:
                raise CoverageError("insufficient-controller-budget")
            global_limit = min(global_limit, time.monotonic() + remaining - 5)
            action = request.get("action")
            challenge, next_challenge = None, time.monotonic() + 1
            sequence += 1
            if action == "CANCEL":
                raise CoverageError("controller-cancel")
            if action == "START":
                phase = request.get("phase")
                if job is not None or expected_phase >= len(phase_list) or phase != phase_list[expected_phase]:
                    raise CoverageError("phase-order-or-duplicate-start")
                session.admit()  # Fresh admission; frozen first-run baseline/events.
                argv, env, cap = product(role, phase, config["bindings"])
                verified_file(argv[0], config["binary_sha256"], 256 * 1048576)
                if role != "ROLE_CPU_C":
                    plan = config["bindings"]["GPU_PLAN_A" if role == "ROLE_ENGINE_A" else "GPU_PLAN_B"]
                    verified_file(plan, config["bindings"]["EXPECTED_GPU_PLAN_A_SHA"], 4 * 1048576)
                outer.check(outer.sample(), admission=True, phase=phase)
                outer.admit_cache(phase)
                directory = root + "/" + phase
                os.mkdir(directory, 0o700)
                input_r, input_w = os.pipe2(os.O_CLOEXEC)
                if role == "ROLE_ENGINE_A":
                    os.write(input_w, REQUESTS)
                os.close(input_w)
                launch = Stamp(session.domain, time.monotonic(), time.time())
                deadline = session.admission.mono + 7200 if role == "ROLE_CPU_C" else \
                    min(global_limit, launch.mono + cap)
                try:
                    job = Job((run_id, role, phase), argv, env, input_r, session.namespaces,
                              session.guard, outer, deadline, directory)
                finally:
                    os.close(input_r)
                journal.emit("ADMISSION_LAUNCH_V1", admission=vars(session.admission),
                             launch=vars(launch), deadline=deadline,
                             swapfree_bytes=session.guard.swap_baseline, key=job.key)
                listen_deadline = launch.mono + (450 if role == "ROLE_CPU_C" else 30)
                listener_sent = False
            elif action == "STOP":
                if job is None:
                    raise CoverageError("stop-without-owned-job")
                phase = job.key[2]
                result = job.stop("phase-stop")
                job = None
                expected_phase += 1
                packet(channel, {"kind": "STOPPED", "run_id": run_id, "role": role,
                                 "phase": phase, "cleanup": result})
                if not result["complete"]:
                    raise CoverageError("incomplete-stop")
            elif action == "RETAIN":
                if role != "ROLE_CPU_C" or job is None or not config.get("retention_approved") or \
                        request.get("completed_phases") != ["B0", "B1", "G0", "G1"]:
                    raise CoverageError("retention-not-approved-or-trial-incomplete")
                age = duration(request["success_age_s"] + 5, 5400)
                completion = Stamp(session.domain, time.monotonic() - age, time.time())
                port = int(config["bindings"]["CPU_PORT_C"])
                if not owned_listener(job.root, port):
                    raise CoverageError("retention-listener-ownership")
                job.deadline = idle_deadline(port, job.root, session.admission, completion,
                                             session.domain, engines_closed=True)
                job.gate.deadline = job.deadline
                if time.monotonic() >= job.deadline - 15:
                    raise CoverageError("retention-already-expiring")
                journal.emit("RETENTION_COMMITTED_V1", deadline=job.deadline,
                             completion=vars(completion), original_admission=vars(session.admission))
                packet(channel, {"kind": "RETENTION_COMMITTED", "run_id": run_id, "role": role,
                                 "bounded": True})
                detached = True  # EOF/controller exit cannot undo this finite grant.
            elif action != "HEARTBEAT":
                raise CoverageError("unknown-controller-action")
    finally:
        try:
            if job is not None:
                journal.emit("WATCHDOG_CLEANUP_V1", cleanup=job.stop("agent-exit"))
        finally:
            journal.close()
            channel.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config = json.loads(required(read_text(args.config), "private-config"))
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    pid = os.fork()
    if pid == 0:
        parent.close()
        try:
            serve(config, child)
        finally:
            os._exit(0)
    child.close()
    # Parent proxy uses no signal/group cleanup. EOF reaches independent guardian;
    # a committed retention daemon is subsequently reaped by initial PID 1.
    buffer = bytearray()
    try:
        while True:
            ready = select.select([0, parent], [], [], 1)[0]
            if parent in ready:
                value = receive(parent)
                raw = (json.dumps(value) + "\n").encode("ascii")
                if os.write(1, raw) != len(raw):
                    raise CoverageError("proxy-short-write")
            if 0 in ready:
                raw = os.read(0, 4096)
                if not raw:
                    return
                buffer.extend(raw)
                if len(buffer) > 8192:
                    raise CoverageError("proxy-input-cap")
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    packet(parent, json.loads(line))
    finally:
        parent.close()


if __name__ == "__main__":
    main()
