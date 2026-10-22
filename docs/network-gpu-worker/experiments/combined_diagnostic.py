"""One explicit profile over the retained 31156168 supervisor; inert on import.

This does not grant execution. Private configs require a separately approved
1500-second measurement window. CPU service residency is an explicit separate
lifetime grant; the pinned supervisor is staged with the published narrow patch.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

TAG = "GPU-COMBINED-V1-20261022"
SUPERVISOR = "311561689bb4b1f8a109821601e3a6bda07067e6"
WINDOW = 1500
HISTORICAL = 6651.9
STREAMS = {"G0": b"GEN 64 9419\nGEN 64 14773,6353\nQUIT\n",
           "G1": b"GEN 1 9419\nQUIT\n"}
_configured = False


def configure(root, config):
    """Bind constants once, before any owner starts; never extend running jobs."""
    global _configured
    if _configured:
        raise ValueError("diagnostic profile already initialized")
    if config.get("diagnostic") != TAG or config.get("supervisor_commit") != SUPERVISOR or \
            config.get("approved_window_s") != WINDOW or config.get("execution_approved") is not True:
        raise ValueError("separate diagnostic execution approval/config required")
    if not isinstance(config.get("diagnostic_revision"), str) or len(config["diagnostic_revision"]) != 40:
        raise ValueError("candidate revision receipt required")
    root = Path(root).resolve(strict=True)
    sys.path[:0] = [str(root), str(root.parent), str(Path(__file__).resolve().parent)]
    import agent
    import controller
    import gpu_completion as completion
    import policy
    if not getattr(agent, "SERVICE_RESIDENCY_V1", False) or not getattr(controller, "SERVICE_RESIDENCY_V1", False):
        raise ValueError("published supervisor residency patch required")
    # A separate small replay alongside the guarded 106-GiB arena is not a new
    # full-arena admission. Its own RSS/anonymous cap and host/swap/event checks
    # are strict; the resident CPU's original guard continues independently.
    policy.POLICIES["ROLE_REPLAY_C"] = policy.Policy("ROLE_REPLAY_C", 8704 * policy.MIB,
                                                    8 * policy.GIB, 512 * policy.MIB, 512 * policy.MIB)
    if config.get("role") == "ROLE_CPU_C" and config.get("service_residency") is True:
        import runtime
        from resident_cpu import ResidentJournal
        agent.Journal = runtime.Journal = ResidentJournal
    # The existing mode already supports G0/G1 serial jobs and one resident CPU.
    # Names below are private protocol labels, not new claims about old trials.
    completion.PHASES = ("G0", "G1")
    completion.ENGINE_CAPS = {p: WINDOW for p in STREAMS}
    completion.GPU_CAPS = {p: WINDOW for p in STREAMS}
    completion.WINDOW_S = WINDOW
    completion.HISTORICAL_ACTIVE_S = HISTORICAL
    original = agent.product

    def product(role, phase, bindings, *, gpu_completion=False):
        if not gpu_completion:
            raise ValueError("combined profile requires completion mode")
        argv, env, cap = original(role, phase, bindings, gpu_completion=True)
        if role == "ROLE_CPU_C":
            pos = argv.index("--residency-ms")
            argv[pos:pos + 2] = ["--residency-forever"]
        if phase == "G1" and role in ("ROLE_ENGINE_A", "ROLE_GPU_B"):
            directory = str(Path(config["run_root"]) / "capture")
            os.mkdir(directory, 0o700)
            env["STRATA_GPU_MERGE_CAPTURE_DIR" if role == "ROLE_ENGINE_A" else "STRATA_GPU_CAPTURE_DIR"] = directory
        return argv, env, cap

    agent.product = product
    agent.requests = lambda phase: STREAMS[phase]
    _configured = True
    return agent, controller


def replay(agent, config, remaining, *, entered=None, argv_count=5, cap=90):
    """Reuse the real CPU-role Session/Job guard for a short 05 replay."""
    from linux_owned import parent_death
    from outer import OuterGuards
    from resource_guard import Journal
    from runtime import Job, Session
    import uuid
    entered = time.monotonic() if entered is None else entered
    if config.get("role") != "ROLE_CPU_C" or not 20 < remaining <= WINDOW or argv_count not in (5,7) or cap not in (90,120):
        raise ValueError("bounded 05 replay config")
    parent_death(os.getppid())
    root = config["run_root"]
    os.mkdir(root, 0o700)
    journal = Journal(root + "/watchdog.jsonl")
    outer = OuterGuards("ROLE_CPU_C")
    session = Session(config["run_id"], "ROLE_REPLAY_C", uuid.uuid4().hex, config["namespaces"], outer)
    job = None
    stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    try:
        session.admit()
        argv = config["replay_argv"]
        if not isinstance(argv, list) or len(argv) != argv_count or any(not isinstance(x, str) or not Path(x).is_absolute() for x in argv):
            raise ValueError("exact absolute replay argv required")
        agent.verified_file(argv[0], config["binary_sha256"], 256 * 1048576)
        deadline = entered + min(cap, remaining - 5)
        if time.monotonic() >= deadline - 15:
            raise ValueError("replay preflight consumed safe interval")
        fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
        try:
            job = Job((config["run_id"], "ROLE_REPLAY_C", "REPLAY"), argv,
                      {"OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                      fd, session.namespaces, session.guard, outer, deadline, root)
        finally:
            os.close(fd)
        while True:
            if stopped[0] or time.monotonic() >= deadline - 15:
                raise ValueError("replay deadline/cancel")
            rc = job.poll()
            if rc is not None:
                cleanup = job.stop("replay-exit")
                if rc or not cleanup["complete"]:
                    raise ValueError("replay nonzero/incomplete cleanup")
                return {"replay_rc": rc, "cleanup": cleanup}
            time.sleep(0.05)
    finally:
        try:
            if job is not None:
                result = job.stop("replay-final")
                journal.emit("COMBINED_REPLAY_FINAL_V1", cleanup=result)
                if not result["complete"]:
                    raise ValueError("replay cleanup incomplete")
        finally:
            journal.close()


def main():
    entered = time.monotonic()  # Config/import/admission/hash time consumes the grant.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--supervisor-root", required=True)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--agent-config")
    modes.add_argument("--controller-config")
    modes.add_argument("--replay-config")
    modes.add_argument("--resident-status")
    ap.add_argument("--remaining-s", type=float)
    args = ap.parse_args()
    path = args.agent_config or args.controller_config or args.replay_config or args.resident_status
    with open(path, "rb") as source:
        raw = source.read(1048577)
    if not 0 < len(raw) <= 1048576:
        raise ValueError("bounded private config")
    config = json.loads(raw)
    agent, controller = configure(args.supervisor_root, config)
    if args.agent_config:
        cpu = config.get("role") == "ROLE_CPU_C"
        if config.get("gpu_completion") is not True or \
                (cpu and (config.get("retention_approved") is not True or config.get("service_residency") is not True)) or \
                (not cpu and (config.get("retention_approved") or config.get("service_residency"))):
            raise ValueError("diagnostic agent mode/retention")
        sys.argv = [agent.__file__, "--config", path]
        agent.main()
    elif args.controller_config:
        if config.get("gpu_completion") is not True or config.get("retain") is not True or \
                config.get("service_residency") is not True:
            raise ValueError("diagnostic controller mode/retention")
        result = controller.launch(config)
        result.update(diagnostic=TAG, diagnostic_revision=config["diagnostic_revision"],
                      phases={"G0": "PERF_AB_64", "G1": "CAPTURE_A_1"},
                      new_request_allowance=3, new_token_allowance=129,
                      followup="collect captures and guarded 05 replay within original deadline minus 45s")
        print(json.dumps(result, allow_nan=False))
    elif args.resident_status:
        if config.get("role") != "ROLE_CPU_C" or config.get("service_residency") is not True:
            raise ValueError("resident CPU config required")
        from resident_cpu import live_status
        print(json.dumps(live_status(config), allow_nan=False))
    else:
        if args.remaining_s is None:
            raise ValueError("fresh remaining duration required")
        print(json.dumps(replay(agent, config, args.remaining_s, entered=entered), allow_nan=False))


if __name__ == "__main__":
    main()
