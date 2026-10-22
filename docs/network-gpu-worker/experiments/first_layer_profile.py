"""Two fresh first-token engines attached to an already owned CPU arena.

Preparation is not execution permission. No operation occurs on import.
"""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import sys
import time
import uuid

TAG="GPU-FIRST-LAYER-V2-20261022"
SUPERVISOR="311561689bb4b1f8a109821601e3a6bda07067e6"
WINDOW=600
PHASES=("B0","G0")
_configured=False


def read_config(path):
    with open(path,"rb") as source: raw=source.read(1048577)
    if not 0<len(raw)<=1048576: raise ValueError("bounded private config")
    return json.loads(raw)


def configure(root,config):
    global _configured
    if _configured: raise ValueError("first-layer profile already configured")
    if config.get("diagnostic")!=TAG or config.get("supervisor_commit")!=SUPERVISOR or \
            config.get("approved_window_s")!=WINDOW or config.get("execution_approved") is not True or \
            not isinstance(config.get("diagnostic_revision"),str) or len(config["diagnostic_revision"])!=40:
        raise ValueError("explicit first-layer execution approval/config required")
    root=Path(root).resolve(strict=True)
    sys.path[:0]=[str(root),str(root.parent),str(Path(__file__).resolve().parent)]
    import agent,controller,gpu_completion as completion,policy
    if not getattr(agent,"FIRST_LAYER_PHASES_V1",False) or ("B0","G0") not in policy.FIRST_LAYER_SCHEDULES_V1:
        raise ValueError("first-layer supervisor phase patch required")
    completion.PHASES=PHASES
    completion.WINDOW_S=WINDOW
    completion.ENGINE_CAPS={p:WINDOW for p in PHASES}
    completion.GPU_CAPS={"G0":WINDOW}
    completion.HISTORICAL_ACTIVE_S=7541.0+117 # Approximate failed-trace increment; not a new precise receipt.
    policy.POLICIES["ROLE_REPLAY_C"]=policy.Policy("ROLE_REPLAY_C",8704*policy.MIB,8*policy.GIB,512*policy.MIB,512*policy.MIB)
    original=agent.product
    def product(role,phase,bindings,*,gpu_completion=False):
        if role=="ROLE_CPU_C": raise ValueError("attach diagnostic must never start a CPU worker")
        if not gpu_completion: raise ValueError("first-layer completion mode required")
        argv,env,cap=original(role,phase,bindings,gpu_completion=True)
        directory=str(Path(config["run_root"])/("trace-"+phase))
        os.mkdir(directory,0o700)
        if role=="ROLE_ENGINE_A":
            env.update(STRATA_FIRST_LAYER_TRACE_DIR=directory,STRATA_REQUIRE_CPU_RESIDENT_HIT="1",
                       STRATA_VERIFY_ASYNC_LAUNCH="1",STRATA_VERIFY_TRACE="1")
        else: env["STRATA_GPU_LAYER_ROWS_DIR"]=directory
        return argv,env,cap
    def requests(phase):
        if phase not in PHASES: raise ValueError("first-layer phase")
        return b"GEN 1 9419\nQUIT\n"
    agent.product,agent.requests=product,requests
    _configured=True
    return agent,controller


def resident_check(config,*,idle):
    from resident_cpu import live_status,ARENA,NO_EXPIRY
    status=live_status(read_config(config["resident_config"]),query_cache=idle)
    if idle and status["cache"]!={"state":0,"session":0,"resident_bytes":ARENA,
                                   "idle_remaining_ms":NO_EXPIRY,"complete":True}:
        raise ValueError("existing complete idle CPU arena required; no cold fallback")
    return status


def observe(config,channel):
    """Only observes existing pinned identities; owns no service or model child."""
    from linux_owned import parent_death
    from runtime import packet,receive
    from policy import duration
    parent_death(os.getppid())
    role,run_id="ROLE_CPU_C",config["run_id"]
    sequence=0;started=False;deadline=time.monotonic()+WINDOW
    cleanup={"complete":True,"reaped":{},"errors":[],"resident_service_untouched":True}
    reason="finished"
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    try:
        initial=resident_check(config,idle=True)
        while not stopped[0]:
            resident_check(config,idle=False) # No cache socket while an engine holds its one connection.
            nonce=uuid.uuid4().hex;end=min(deadline-15,time.monotonic()+5)
            packet(channel,{"kind":"CHALLENGE","run_id":run_id,"role":role,"sequence":sequence,"nonce":nonce})
            if channel not in select.select([channel],[],[],max(0,end-time.monotonic()))[0]:
                raise ValueError("resident observer lease expired")
            request=receive(channel)
            if request.get("run_id")!=run_id or request.get("nonce")!=nonce or request.get("sequence")!=sequence or time.monotonic()>=end:
                raise ValueError("stale resident observer request")
            deadline=min(deadline,time.monotonic()+duration(request["remaining_s"],WINDOW)-5)
            sequence+=1;action=request.get("action")
            if action=="START" and request.get("phase")=="CPU" and not started:
                packet(channel,{"kind":"LISTENING_COVERED","run_id":run_id,"role":role,"phase":"CPU","attached":initial})
                started=True
            elif action=="FINISH" and started:
                final=resident_check(config,idle=True)
                packet(channel,{"kind":"FINAL","run_id":run_id,"role":role,"reason":"finished","cleanup":cleanup,"resident":final})
                return
            elif action=="CANCEL": raise ValueError("measurement cancelled; resident service untouched")
            elif action!="HEARTBEAT": raise ValueError("observer cannot start/unload/stop the resident service")
            time.sleep(min(1,max(0,deadline-15-time.monotonic())))
        raise ValueError("observer cancelled; resident service untouched")
    except BaseException as exc:
        reason=type(exc).__name__+":"+str(exc)
        packet(channel,{"kind":"FINAL","run_id":run_id,"role":role,"reason":reason,"cleanup":cleanup})
    finally: channel.close()


def launch(controller,config):
    from policy import TrialOrder
    roles={"ROLE_CPU_C","ROLE_ENGINE_A","ROLE_GPU_B"}
    if set(config["peers"])!=roles or len(config["forwards"])>4: raise ValueError("existing bounded role transports required")
    class Attached(controller.Controller):
        def __init__(self,peers,start):
            self.peers,self.clock,self.start=peers,time.monotonic,start
            self.deadline=start+WINDOW;self.phases=PHASES;self.order=TrialOrder(PHASES)
            self.engine_caps={p:WINDOW for p in PHASES};self.gpu_caps={"G0":WINDOW}
            self.gpu_completion=True;self.success_at=None
        def run(self):
            cpu,engine,gpu=(self.peers[r] for r in ("ROLE_CPU_C","ROLE_ENGINE_A","ROLE_GPU_B"))
            try:
                self.start_phase(cpu,"CPU") # Observer proof only; never CPU product launch.
                attached=self.wait(cpu.role,"LISTENING_COVERED","CPU",10)
                for phase in PHASES:
                    if phase=="G0":
                        self.start_phase(gpu,phase);self.wait(gpu.role,"LISTENING_COVERED",phase,min(30,self.phase_cap(WINDOW)))
                    self.order.permit(phase,cpu.covered,gpu.covered)
                    self.start_phase(engine,phase)
                    done=self.wait(engine.role,"DONE",phase,self.phase_cap(WINDOW))
                    self.order.finish(phase,done["rc"],done["cleanup"]["complete"])
                    if phase=="G0":
                        gpu.queue("STOP");controller.require_cleanup(self.wait(gpu.role,"STOPPED",phase,45))
                    else:self.gap()
                for peer in (engine,gpu,cpu):
                    peer.queue("FINISH");final=self.wait(peer.role,"FINAL",cap=10)
                    controller.require_cleanup(final,final=True)
                    if peer is cpu: resident=final["resident"]
                    peer.close()
                return {"complete":True,"diagnostic":TAG,"window_start_mono":self.start,"window_deadline_mono":self.deadline,
                        "historical_active_s_before_failed_trace":7541.0,"failed_trace_active_s_approx":117,
                        "historical_requests":15,"historical_generated_tokens":833,"failed_trace_started_requests":1,
                        "new_request_cap":2,"new_token_cap":2,"cpu_attached":attached["attached"],"cpu_resident":resident}
            except BaseException:
                for peer in self.peers.values():
                    if not peer.closed and not peer.final_seen:peer.pending={"action":"CANCEL"}
                end=min(self.deadline,self.clock()+20)
                while self.clock()<end and any(not p.closed and not p.final_seen for p in self.peers.values()):
                    try:
                        active=[p for p in self.peers.values() if not p.closed and not p.final_seen]
                        for stream in select.select([p.process.stdout for p in active],[],[],.05)[0]:
                            next(p for p in active if p.process.stdout is stream).pump(max(21,self.deadline-self.clock()))
                    except BaseException:break
                raise
            finally:controller.close_all(self.peers.values())
    start=time.monotonic();forwards=[];peers={}
    try:
        for argv in config["forwards"]:forwards.append(controller.Forward(argv))
        for role,argv in config["peers"].items():peers[role]=controller.Peer(role,config["run_id"],argv)
        return Attached(peers,start).run()
    finally:controller.close_all([*peers.values(),*forwards])


def main():
    entered=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--supervisor-root",required=True)
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ("agent-config","observe-config","controller-config","replay-config"):modes.add_argument("--"+mode)
    parser.add_argument("--remaining-s",type=float);args=parser.parse_args()
    path=args.agent_config or args.observe_config or args.controller_config or args.replay_config
    config=read_config(path);agent,controller=configure(args.supervisor_root,config)
    if args.observe_config:
        if config.get("role")!="ROLE_CPU_C":raise ValueError("05 observer role required")
        agent.serve=observe;sys.argv=[agent.__file__,"--config",path];agent.main()
    elif args.agent_config:
        if config.get("role")=="ROLE_CPU_C" or config.get("gpu_completion") is not True:raise ValueError("engine/GPU agents only")
        sys.argv=[agent.__file__,"--config",path];agent.main()
    elif args.controller_config:print(json.dumps(launch(controller,config),allow_nan=False))
    else:
        from combined_diagnostic import replay
        if args.remaining_s is None:raise ValueError("fresh original-window remainder required")
        print(json.dumps(replay(agent,config,args.remaining_s,entered=entered,argv_count=7,cap=120),allow_nan=False))


if __name__=="__main__":main()
