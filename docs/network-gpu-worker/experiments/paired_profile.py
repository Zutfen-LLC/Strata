"""Attached B/G practical evaluation. Preparation is not execution permission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import first_layer_profile as attached
import paired_g_continuation as gpu_only
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
    continuation = config.get("diagnostic") == gpu_only.TAG
    phases = gpu_only.PHASES if continuation else PHASES
    if config.get("diagnostic") not in (TAG, gpu_only.TAG) or config.get("supervisor_commit") != SUPERVISOR or \
            config.get("approved_window_s") != WINDOW or config.get("execution_approved") is not True or \
            len(config.get("diagnostic_revision", "")) != 40:
        raise ValueError("separate paired model-execution approval/config required")
    plan_raw = Path(config["plan"]).read_bytes()
    if not 0 < len(plan_raw) <= 1048576 or hashlib.sha256(plan_raw).hexdigest() != config["plan_sha256"]:
        raise ValueError("identical private token plan binding")
    plan = json.loads(plan_raw)
    validate_plan(plan)
    if continuation:
        if config["plan_sha256"] != gpu_only.PLAN_SHA:
            raise ValueError("exact retained G-only plan required")
        if config.get("role") == "ROLE_ENGINE_A" and config["binary_sha256"] != config["retained_engine_sha256"]:
            raise ValueError("accepted baseline engine binary must remain unchanged")
    root = Path(root).resolve(strict=True)
    sys.path[:0] = [str(root), str(root.parent), str(Path(__file__).resolve().parent)]
    import agent
    import controller
    import gpu_completion as completion
    if not getattr(agent, "FIRST_LAYER_PHASES_V1", False) or not getattr(controller, "SERVICE_RESIDENCY_V1", False):
        raise ValueError("retained published supervisor patches required")
    if continuation and not getattr(controller, "PEER_DIAGNOSTICS_V1", False):
        raise ValueError("published peer-diagnostics patch required")
    completion.PHASES = phases
    completion.WINDOW_S = WINDOW
    completion.ENGINE_CAPS = {p: WINDOW for p in phases}
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

    def phase_requests(phase):
        if phase not in phases:
            raise ValueError("phase is outside the approved continuation")
        return requests(plan, phase)

    agent.product, agent.requests, agent.Job = product, phase_requests, job
    _configured = True
    return agent, controller


def launch(controller, config):
    start, started_utc = time.monotonic(), time.time()  # Before controller/relay startup.
    from policy import TrialOrder
    continuation = config.get("diagnostic") == gpu_only.TAG
    phases = gpu_only.PHASES if continuation else PHASES
    tag = gpu_only.TAG if continuation else TAG
    request_cap, token_cap = (gpu_only.REQUEST_CAP, gpu_only.TOKEN_CAP) if continuation else (REQUEST_CAP, TOKEN_CAP)
    if set(config["peers"]) != {"ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B"} or len(config["forwards"]) > 4:
        raise ValueError("existing bounded role transports required")

    class Paired(controller.Controller):
        def __init__(self, peers, start):
            self.peers, self.clock, self.start = peers, time.monotonic, start
            self.deadline, self.phases, self.order = start + WINDOW, phases, TrialOrder(phases)
            self.engine_caps = {p: WINDOW for p in phases}
            self.gpu_caps, self.gpu_completion, self.success_at = {"G0": WINDOW}, True, None

        def run(self):
            cpu, engine, gpu = (self.peers[r] for r in ("ROLE_CPU_C", "ROLE_ENGINE_A", "ROLE_GPU_B"))
            try:
                self.start_phase(cpu, "CPU")  # Existing service observation, never product start.
                proof = self.wait(cpu.role, "LISTENING_COVERED", "CPU", 10)
                for phase in phases:
                    if phase == "G0":
                        self.start_phase(gpu, phase)
                        self.wait(gpu.role, "LISTENING_COVERED", phase, min(30, self.phase_cap(WINDOW)))
                    self.order.permit(phase, cpu.covered, gpu.covered)
                    self.start_phase(engine, phase)
                    done = self.wait(engine.role, "DONE", phase, self.phase_cap(WINDOW))
                    self.order.finish(phase, done["rc"], done["cleanup"]["complete"])
                    if phase != phases[-1]:
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
                return {"complete": True, "diagnostic": tag, "window_s": WINDOW,
                        "new_elapsed_s": self.clock() - self.start, "request_cap": request_cap,
                        "generated_token_cap": token_cap, "historical_requests": 24 if continuation else 17,
                        "historical_generated_tokens": 888 if continuation else 835,
                        "separate_run_comparison": continuation, "historical_numerical_verdict": "NUMERICAL_FAIL",
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

    forwards, peers, clock_receipt, evidence = [], {}, None, None
    def save_clock():
        if evidence is not None:
            temporary = evidence / "execution-clock.json.tmp"
            with open(temporary, "w", encoding="ascii") as output:
                json.dump(clock_receipt, output, allow_nan=False)
            os.replace(temporary, evidence / "execution-clock.json")
    try:
        if continuation:
            evidence = Path(config["controller_evidence"])
            if not evidence.is_absolute():
                raise ValueError("new absolute controller evidence directory required")
            os.mkdir(evidence, 0o700)
            clock_receipt = {"diagnostic": tag, "start_mono": start, "start_utc": started_utc,
                             "deadline_mono": start + WINDOW, "window_s": WINDOW,
                             "cleanup_reserve_s": 45, "status": "RUNNING"}
            save_clock()
            baseline = config["retained_b"]
            proof = gpu_only.check_baseline(gpu_only.retained_plan(config["plan"]), baseline["root"], baseline["engine_config"])
            if proof["engine_binary_sha256"] != config["retained_engine_sha256"]:
                raise ValueError("retained B/new engine binary binding mismatch")
            with open(evidence / "retained-b-check.json", "w", encoding="ascii") as output:
                json.dump(proof, output, allow_nan=False)
        for argv in config["forwards"]:
            forwards.append(controller.Forward(argv))
        for role, argv in config["peers"].items():
            kwargs = {"diagnostic_path": str(evidence / (role + ".json"))} if continuation else {}
            peers[role] = controller.Peer(role, config["run_id"], argv, **kwargs)
        result = Paired(peers, start).run()
        if clock_receipt is not None:
            clock_receipt["status"] = "COMPLETE"
        return result
    except BaseException as exc:
        if clock_receipt is not None:
            clock_receipt.update(status="FAILED", error=type(exc).__name__ + ":" + str(exc))
        raise
    finally:
        try:
            controller.close_all([*peers.values(), *forwards])
        except BaseException as exc:
            if clock_receipt is not None:
                clock_receipt.update(status="FAILED", cleanup_error=type(exc).__name__ + ":" + str(exc))
            raise
        finally:
            if clock_receipt is not None:
                clock_receipt.update(end_mono=time.monotonic())
                clock_receipt["elapsed_s"] = clock_receipt["end_mono"] - start
                save_clock()


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
