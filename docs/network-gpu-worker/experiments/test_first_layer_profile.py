"""Focused attach/phase/file-lifecycle mocks. Authored NOT RUN; no live workloads."""
import argparse
import json
from pathlib import Path
import resource
import signal
import sys
import tempfile
from unittest import TestCase,TextTestRunner,defaultTestLoader
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import first_layer_profile as profile
ROOT=None
CONFIG={"diagnostic":profile.TAG,"supervisor_commit":profile.SUPERVISOR,"approved_window_s":600,
        "execution_approved":True,"diagnostic_revision":"a"*40,"run_root":"/private/new","run_id":"new"}
class Cases(TestCase):
    @classmethod
    def setUpClass(cls):cls.agent,cls.controller=profile.configure(ROOT,CONFIG)
    def test_streams_caps_and_no_cpu_start(self):
        import gpu_completion as completion
        from policy import TrialOrder
        self.assertEqual((completion.ENGINE_CAPS,completion.GPU_CAPS),({"B0":600,"G0":600},{"G0":600}))
        self.assertEqual(TrialOrder(profile.PHASES).phases,("B0","G0"))
        for phase in profile.PHASES:self.assertEqual(self.agent.requests(phase),b"GEN 1 9419\nQUIT\n")
        with self.assertRaisesRegex(ValueError,"never start a CPU"):self.agent.product("ROLE_CPU_C","CPU",{},gpu_completion=True)
    def test_product_capture_flags_and_required_resident_hit(self):
        class B(dict):
            def __missing__(self,key):return "/private/"+key
        for role,phase in (("ROLE_ENGINE_A","B0"),("ROLE_ENGINE_A","G0"),("ROLE_GPU_B","G0")):
            with patch("os.mkdir") as mkdir:argv,env,cap=self.agent.product(role,phase,B(),gpu_completion=True)
            mkdir.assert_called_once_with("/private/new/trace-"+phase,0o700);self.assertEqual(cap,600)
            if role=="ROLE_ENGINE_A":
                self.assertEqual(env["STRATA_REQUIRE_CPU_RESIDENT_HIT"],"1")
                self.assertIn("STRATA_FIRST_LAYER_TRACE_DIR",env)
                self.assertEqual("--remote-gpu-experts" in argv,phase=="G0")
            else:self.assertIn("STRATA_GPU_LAYER_ROWS_DIR",env)
            self.assertNotIn("STRATA_GPU_CAPTURE_DIR",env)
    def test_resident_idle_proof_and_busy_observation(self):
        import resident_cpu
        full={"state":0,"session":0,"resident_bytes":resident_cpu.ARENA,"idle_remaining_ms":resident_cpu.NO_EXPIRY,"complete":True}
        cfg={"resident_config":"/private/old-config"}
        with patch.object(profile,"read_config",return_value={}),patch.object(resident_cpu,"live_status",return_value={"cache":full}) as status:
            profile.resident_check(cfg,idle=True);status.assert_called_once_with({},query_cache=True)
        with patch.object(profile,"read_config",return_value={}),patch.object(resident_cpu,"live_status",return_value={"cache":None}) as status:
            profile.resident_check(cfg,idle=False);status.assert_called_once_with({},query_cache=False)
            with self.assertRaisesRegex(ValueError,"no cold fallback"):profile.resident_check(cfg,idle=True)
    def observer_case(self,last):
        import runtime,linux_owned
        channel=Mock();messages=[];challenge={};count=[0]
        def packet(_,value):
            messages.append(value)
            if value["kind"]=="CHALLENGE":challenge.update(value)
        def receive(_):
            count[0]+=1
            return {"run_id":"new","nonce":challenge["nonce"],"sequence":count[0]-1,"remaining_s":500,
                    "action":"START" if count[0]==1 else last,"phase":"CPU","gpu_completion":True}
        with patch.object(linux_owned,"parent_death"),patch.object(profile.signal,"signal"), \
                patch.object(profile,"resident_check",return_value={"live":True}) as checked, \
                patch.object(runtime,"packet",side_effect=packet),patch.object(runtime,"receive",side_effect=receive), \
                patch.object(profile.select,"select",return_value=([channel],[],[])),patch.object(profile.time,"sleep"):
            profile.observe(CONFIG,channel)
        self.assertTrue(messages[-1]["cleanup"]["resident_service_untouched"])
        self.assertTrue(any(call.kwargs=={"idle":False} for call in checked.call_args_list))
        return messages
    def test_observer_finishes_without_touching_resident_service(self):
        messages=self.observer_case("FINISH")
        self.assertEqual(messages[-1]["reason"],"finished")
        self.assertTrue(any(row["kind"]=="LISTENING_COVERED" for row in messages))
    def test_observer_refuses_stop_without_signalling_service(self):
        messages=self.observer_case("STOP")
        self.assertIn("cannot start/unload/stop",messages[-1]["reason"])
    def test_partial_tail_rotation_and_real_file_failure(self):
        from resident_cpu import resource_row
        with tempfile.TemporaryDirectory() as root:
            path=Path(root,"resources.jsonl");path.write_bytes(b'{"sequence":1}\n{"partial":')
            self.assertEqual(resource_row(str(path))["sequence"],1)
            path.rename(str(path)+".previous");path.write_bytes(b"")
            self.assertEqual(resource_row(str(path))["sequence"],1)
            path.write_bytes(b"{bad}\n")
            with self.assertRaises(json.JSONDecodeError):resource_row(str(path))
            with patch("builtins.open",side_effect=PermissionError("fixture denied")),self.assertRaises(PermissionError):resource_row(str(path))
    def test_gpu_agent_first_phase_is_g0_not_baseline(self):
        import gpu_completion as completion
        from resource_guard import GuardStop
        class B(dict):
            def __missing__(self,key):return "/private/"+key
        bindings=B(EXPECTED_GPU_PLAN_A_SHA=self.agent.PLAN_SHA)
        cfg={**CONFIG,"role":"ROLE_GPU_B","gpu_completion":True,"implementation":completion.IMPLEMENTATION,
             "bindings":bindings,"binary_sha256":"b"*64,"namespaces":{}}
        channel=Mock();challenge={};job=Mock(key=("new","ROLE_GPU_B","G0"));job.poll.side_effect=GuardStop("fixture done")
        job.stop.return_value={"complete":True,"reaped":{},"errors":[]};session=Mock(domain="fixture");session.admission.mono=1
        def packet(_,value):
            if value["kind"]=="CHALLENGE":challenge.update(value)
        def receive(_):return {"run_id":"new","nonce":challenge["nonce"],"sequence":0,"remaining_s":500,"action":"START","phase":"G0","gpu_completion":True}
        with patch.object(self.agent.os,"setsid"),patch.object(self.agent.signal,"signal"),patch.object(self.agent.os,"mkdir"), \
                patch.object(self.agent.os,"pipe2",return_value=(101,102)),patch.object(self.agent.os,"close"), \
                patch.object(self.agent,"Journal"),patch.object(self.agent,"OuterGuards"),patch.object(self.agent,"Session",return_value=session), \
                patch.object(self.agent,"Job",return_value=job) as constructor,patch.object(self.agent,"verified_file"), \
                patch.object(self.agent,"packet",side_effect=packet),patch.object(self.agent,"receive",side_effect=receive), \
                patch.object(self.agent.select,"select",return_value=([channel],[],[])),patch.object(self.agent.time,"monotonic",return_value=100):
            self.agent.serve(cfg,channel)
        constructor.assert_called_once();self.assertEqual(constructor.call_args.args[0],("new","ROLE_GPU_B","G0"))
def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument("--supervisor-root",required=True);ROOT=parser.parse_args().supervisor_root
    resource.setrlimit(resource.RLIMIT_AS,(128*1048576,128*1048576));resource.setrlimit(resource.RLIMIT_CPU,(5,5));signal.alarm(10)
    forbidden=AssertionError("live workload/transport forbidden in focused fixture")
    with patch("subprocess.Popen",side_effect=forbidden),patch("os.fork",side_effect=forbidden), \
            patch("socket.socket",side_effect=forbidden),patch("signal.pidfd_send_signal",side_effect=forbidden):
        result=TextTestRunner(verbosity=2).run(defaultTestLoader.loadTestsFromTestCase(Cases))
    if not result.wasSuccessful() or result.skipped or result.testsRun!=7:raise SystemExit(1)
    print("FIRST_LAYER_PROFILE_PASS_V1 tests=7 workloads=0")
if __name__=="__main__":main()
