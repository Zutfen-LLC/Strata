"""Attached B/G practical evaluation. Preparation is not execution permission."""
import argparse
import hashlib
import json
from pathlib import Path
import select
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import first_layer_profile as attached
from paired_tasks import TAG, PHASES, WINDOW, REQUEST_CAP, TOKEN_CAP, read_json, requests, validate_plan

SUPERVISOR = attached.SUPERVISOR
_configured = False


class OutputClock:
    """Watchdog-side output observation, no device tracing or product wrapper."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.offset, self.tail, self.events = 0, b"", []

    def sample(self):
        path = self.directory / "workload.out"
        size = path.stat().st_size
        if not self.offset <= size <= 1048576:
            raise ValueError("bounded stable workload output")
        with open(path, "rb") as source:
            source.seek(self.offset)
            raw = source.read(1048577 - self.offset)
        self.offset += len(raw)
        lines = (self.tail + raw).split(b"\n")
        self.tail = lines.pop()
        changed = False
        for line in lines:
            if line.startswith((b"READY", b"DONE ")):
                if len(self.events) >= 6:
                    raise ValueError("bounded paired timing events")
                self.events.append({"observed_mono": time.monotonic(), "line": line.decode("ascii")})
                changed = True
        if changed:
            with open(self.directory / "output-clock.json", "w", encoding="ascii") as output:
                json.dump({"events": self.events, "scope": "file observation at watchdog polls; includes poll/guard jitter, not wire RTT"}, output)


def configure(root, config):
    global _configured
    if _configured:
        raise ValueError("paired profile already configured")
    if config.get("diagnostic") != TAG or config.get("supervisor_commit") != SUPERVISOR or \
            config.get("approved_window_s") != WINDOW or config.get("execution_approved") is not True or \
            len(config.get("diagnostic_revision", "")) != 40:
        raise ValueError("separate paired model-execution approval/config required")
    plan_raw = Path(config["plan"]).read_bytes()
    if not 0 < len(plan_raw) <= 1048576 or hashlib.sha256(plan_raw).hexdigest() != config["plan_sha256"]:
        raise ValueError("identical private token plan binding")
    plan = json.loads(plan_raw)
    validate_plan(plan)
    root = Path(root).resolve(strict=True)
    sys.path[:0] = [str(root), str(root.parent), str(Path(__file__).resolve().parent)]
    import agent
    import controller
    import gpu_completion as completion
    if not getattr(agent, "FIRST_LAYER_PHASES_V1", False) or not getattr(controller, "SERVICE_RESIDENCY_V1", False):
        raise ValueError("retained published supervisor patches required")
    completion.PHASES = PHASES
    completion.WINDOW_S = WINDOW
    completion.ENGINE_CAPS = {p: WINDOW for p in PHASES}
    completion.GPU_CAPS = {"G0": WINDOW}
    attached.WINDOW = WINDOW  # Only reuse its read-only resident observer, never its trace configure().
    original_product, original_job = agent.product, agent.Job

    def product(role, phase, bindings, *, gpu_completion=False):
        if role == "ROLE_CPU_C":
            raise ValueError("paired attach must never start a CPU worker")
        if not gpu_completion:
            raise ValueError("bounded completion mode required")
        argv, env, cap = original_product(role, phase, bindings, gpu_completion=True)
        if any("TRACE" in key or "CAPTURE" in key or key == "STRATA_LOGPOS" for key in env):
            raise ValueError("clean uninstrumented base environment required")
        if role == "ROLE_ENGINE_A":
            env.pop("STRATA_DUMP_FIRST_LOGITS", None)
            env["STRATA_REQUIRE_CPU_RESIDENT_HIT"] = "1"
            if "--dump-routing" in argv:
                pos = argv.index("--dump-routing")
                del argv[pos:pos + 2]
            if phase.endswith("1"):
                env["STRATA_DUMP_FIRST_LOGITS"] = str(Path(config["run_root"]) / (phase + ".first-logits"))
                env["STRATA_LOGPOS"] = str(Path(config["run_root"]) / (phase + ".logpos"))
        return argv, env, cap

    def job(*args, **kwargs):
        result = original_job(*args, **kwargs)
        if result.key[1] == "ROLE_ENGINE_A":
            clock, original_poll = OutputClock(result.directory), result.poll

            def poll():
                clock.sample()
                rc = original_poll()
                clock.sample()
                return rc
            result.poll = poll
        return result

    agent.product, agent.requests, agent.Job = product, lambda phase: requests(plan, phase), job
    _configured = True
    return agent, controller


def launch(controller, config):
    from policy import TrialOrder
    if set(config["peers"]) != {"ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B"} or len(config["forwards"]) > 4:
        raise ValueError("existing bounded role transports required")

    class Paired(controller.Controller):
        def __init__(self, peers, start):
            self.peers, self.clock, self.start = peers, time.monotonic, start
            self.deadline, self.phases, self.order = start + WINDOW, PHASES, TrialOrder(PHASES)
            self.engine_caps = {p: WINDOW for p in PHASES}
            self.gpu_caps, self.gpu_completion, self.success_at = {"G0": WINDOW}, True, None

        def run(self):
            cpu, engine, gpu = (self.peers[r] for r in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B"))
            try:
                self.start_phase(cpu, "CPU")  # Existing service observation, never product start.
                proof = self.wait(cpu.role, "LISTENING_COVERED", "CPU", 10)
                for phase in PHASES:
                    if phase == "G0":
                        self.start_phase(gpu, phase)
                        self.wait(gpu.role, "LISTENING_COVERED", phase, min(30, self.phase_cap(WINDOW)))
                    self.order.permit(phase, cpu.covered, gpu.covered)
                    self.start_phase(engine, phase)
                    done = self.wait(engine.role, "DONE", phase, self.phase_cap(WINDOW))
                    self.order.finish(phase, done["rc"], done["cleanup"]["complete"])
                    if phase != PHASES[-1]:
                        self.gap()
                gpu.queue("STOP")
                controller.require_cleanup(self.wait(gpu.role, "STOPPED", "G0", 45))
                for peer in (engine, gpu, cpu):
                    peer.queue("FINISH")
                    final = self.wait(peer.role, "FINAL", cap=10)
                    controller.require_cleanup(final, final=True)
                    if peer is cpu:
                        resident = final["resident"]
                    peer.close()
                return {"complete": True, "diagnostic": TAG, "window_s": WINDOW,
                        "new_elapsed_s": self.clock() - self.start, "request_cap": REQUEST_CAP,
                        "generated_token_cap": TOKEN_CAP, "historical_requests": 17,
                        "historical_generated_tokens": 835, "historical_numerical_verdict": "NUMERICAL_FAIL",
                        "cpu_attached": proof["attached"], "cpu_resident": resident}
            except BaseException:
                for peer in self.peers.values():
                    if not peer.closed and not peer.final_seen:
                        peer.pending = {"action": "CANCEL"}
                end = min(self.deadline, self.clock() + 20)
                while self.clock() < end and any(not p.closed and not p.final_seen for p in self.peers.values()):
                    try:
                        active = [p for p in self.peers.values() if not p.closed and not p.final_seen]
                        for stream in select.select([p.process.stdout for p in active], [], [], .05)[0]:
                            next(p for p in active if p.process.stdout is stream).pump(max(21, self.deadline - self.clock()))
                    except BaseException:
                        break
                raise
            finally:
                controller.close_all(self.peers.values())

    start, forwards, peers = time.monotonic(), [], {}
    try:
        for argv in config["forwards"]:
            forwards.append(controller.Forward(argv))
        for role, argv in config["peers"].items():
            peers[role] = controller.Peer(role, config["run_id"], argv)
        return Paired(peers, start).run()
    finally:
        controller.close_all([*peers.values(), *forwards])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-root", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("agent-config", "observe-config", "controller-config"):
        modes.add_argument("--" + name)
    args = parser.parse_args()
    path = args.agent_config or args.observe_config or args.controller_config
    config = read_json(path)
    agent, controller = configure(args.supervisor_root, config)
    if args.observe_config:
        if config.get("role") != "ROLE_CPU_C":
            raise ValueError("resident observer role required")
        agent.serve = attached.observe
        sys.argv = [agent.__file__, "--config", path]
        agent.main()
    elif args.agent_config:
        if config.get("role") not in ("ROLE_ENGINE_A", "ROLE_GPU_B") or config.get("gpu_completion") is not True or \
                config.get("retention_approved") or config.get("service_residency"):
            raise ValueError("engine/GPU agents only; resident service untouched")
        sys.argv = [agent.__file__, "--config", path]
        agent.main()
    else:
        print(json.dumps(launch(controller, config), allow_nan=False))


if __name__ == "__main__":
    main()
