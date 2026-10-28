import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import portable_resident_v1 as M
from portable_resident_v1 import Resident

FAKE=r'''
import json,sys,time
print(json.dumps(dict(ready=True,schema=1)),flush=True)
for i,line in enumerate(sys.stdin,1):
    if line=='QUIT\n': break
    if sys.argv[1]=='hang': time.sleep(20)
    if sys.argv[1]=='fail': sys.exit(2)
    n,t=line.split()[1:]
    tokens=list(map(int,t.split(',')))
    for p,t in enumerate(tokens): print(json.dumps(dict(position=p,token=t,generated=False,greedy_next=7)))
    print(json.dumps(dict(stop='max_new',position=len(tokens))))
    print(json.dumps(dict(done=i,prompt_count=len(tokens),output_count=0,prompt_ms=0,cpu_hits_delta=10,cpu_pool_gu_ms_delta=(float('nan') if sys.argv[1]=='nan' else -1.25 if sys.argv[1]=='negative' else 1.25),cpu_pool_quant_ms_delta=.25,cpu_pool_down_ms_delta=2.5,cpu_pool_elapsed_ms_delta=4.1,cpu_verifications_delta=0,cpu_copied_bytes_delta=0,cpu_digest_bytes_delta=0,source_invalidations_delta=0,model_step_ms_delta=18.,gdn_layer_ms_delta=10.,qsa_layer_ms_delta=5.,output_head_ms_delta=2.,gdn_attention_exchange_ms_delta=3.,gdn_attention_compute_ms_delta=2.,gdn_ffn_ms_delta=6.,gdn_outer_residual_ms_delta=1.,gdn_attention_surrounding_ms_delta=1.,gdn_ffn_non_provider_ms_delta=1.,gdn_attention_exchange_calls_delta=36*len(tokens),gdn_attention_compute_calls_delta=36*len(tokens),gdn_ffn_calls_delta=36*len(tokens),model_residual_ms_delta=1.,gdn_provider_ms_delta=5.,qsa_provider_ms_delta=2.5,model_step_calls_delta=len(tokens),gdn_layer_calls_delta=36*len(tokens),qsa_layer_calls_delta=12*len(tokens),output_head_calls_delta=len(tokens),expert_provider_ms_delta=7.5,expert_provider_calls_delta=len(tokens)*48,gpu_infer_rows_ms_delta=[1.0,float('nan') if sys.argv[1]=='nan_gpu' else -1.0 if sys.argv[1]=='negative_gpu' else 2.0],gpu_infer_rows_calls_delta=[1,1],gpu_entries_per_device_delta=[1,1])))
    sys.stdout.flush()
'''
class HarnessTest(unittest.TestCase):
    def run_fake(self,mode,action):
        with tempfile.TemporaryDirectory() as d:
            runner=Resident([sys.executable,'-c',FAKE,mode],Path(d),load_seconds=2,request_seconds=.3,total_seconds=5)
            try:
                runner.ready()
                action(runner)
            finally:
                runner.close()
                self.assertIsNotNone(runner.child.returncode)
    def test_model_phase_reconciliation(self):
        m=dict(model_step_calls_delta=2,gdn_layer_calls_delta=72,qsa_layer_calls_delta=24,output_head_calls_delta=2,
               model_step_ms_delta=18.,gdn_layer_ms_delta=10.,qsa_layer_ms_delta=5.,output_head_ms_delta=2.,
               gdn_attention_exchange_ms_delta=3.,gdn_attention_compute_ms_delta=2.,gdn_ffn_ms_delta=6.,gdn_outer_residual_ms_delta=1.,gdn_attention_surrounding_ms_delta=1.,gdn_ffn_non_provider_ms_delta=1.,gdn_attention_exchange_calls_delta=72,gdn_attention_compute_calls_delta=72,gdn_ffn_calls_delta=72,model_residual_ms_delta=1.,gdn_provider_ms_delta=5.,qsa_provider_ms_delta=2.5,expert_provider_ms_delta=7.5)
        M.model_phase_fields(m,2)
        for key,value in [('gdn_attention_exchange_calls_delta',71),('gdn_attention_compute_calls_delta',71),('gdn_ffn_calls_delta',71),
                          ('gdn_attention_exchange_ms_delta',float('inf')),('gdn_attention_compute_ms_delta',4.),
                          ('gdn_ffn_ms_delta',4.),('gdn_attention_exchange_ms_delta',5.),
                          ('gdn_attention_compute_ms_delta',float('nan')),('gdn_ffn_ms_delta',float('inf')),
                          ('gdn_outer_residual_ms_delta',float('nan')),('gdn_attention_surrounding_ms_delta',2.),
                          ('gdn_ffn_non_provider_ms_delta',2.),('gdn_outer_residual_ms_delta',2.),
                          ('gdn_attention_surrounding_ms_delta',float('nan')),('gdn_ffn_non_provider_ms_delta',-1.),
                          ('gdn_layer_calls_delta',71),('qsa_provider_ms_delta',6.),('model_step_ms_delta',15.),
                          ('model_residual_ms_delta',2.),('gdn_layer_ms_delta',float('nan')),('model_residual_ms_delta',float('nan'))]:
            bad=dict(m);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): M.model_phase_fields(bad,2)
    def test_four_requests(self):
        def action(r):
            rows=[r.request([5,7],0,i) for i in range(1,5)]
            self.assertTrue(all(v['rows']==rows[0]['rows'] for v in rows))
            r.finish()
        self.run_fake('ok',action)
    def test_invalid_pool_delta_reaps(self):
        def action(r):
            with self.assertRaisesRegex(ValueError,'invalid timer'): r.request([5],0,1)
        for mode in ('nan','negative'):
            with self.subTest(mode=mode): self.run_fake(mode,action)
    def test_invalid_executor_delta_reaps(self):
        def action(r):
            with self.assertRaisesRegex(ValueError,'invalid GPU timing'): r.request([5],0,1)
        for mode in ('nan_gpu','negative_gpu'):
            with self.subTest(mode=mode): self.run_fake(mode,action)
    def test_timeout_reaps(self):
        def action(r):
            with self.assertRaises(TimeoutError): r.request([5],0,1)
        self.run_fake('hang',action)
    def test_failure_reaps(self):
        def action(r):
            with self.assertRaises(RuntimeError): r.request([5],0,1)
        self.run_fake('fail',action)
    def test_guard_failure_reaps(self):
        def action(r):
            def stop(_): raise ValueError('guard STOP')
            r.tick=stop
            with self.assertRaisesRegex(ValueError,'guard STOP'): r.request([5],0,1)
        self.run_fake('ok',action)
    def main_run(self,pressure,timings=False,resident_weights=False,dense_weights=False,state_resident=False,tensor_caps=None):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'out'; tokens=root/'tokens.json';tokens.write_text('[5,7]')
            plans=[root/'p0',root/'p1']
            for plan in plans: plan.write_text('frozen')
            real=M.Resident
            commands=[]
            def runner(command,out,tick):
                commands.append(command)
                return real([sys.executable,'-c',FAKE,'ok'],out,tick,load_seconds=2,request_seconds=2,total_seconds=5)
            host=dict(effective_available=1<<40,pswpout=0)
            args=['resident','--binary',sys.executable,'--binary-sha256',M.Q.bundle.sha(sys.executable),
                  '--out',str(out),'--tokens-file',str(tokens),'--max-new','0',
                  '--plan',str(plans[0]),'--plan',str(plans[1])]
            if tensor_caps is not None: args+=['--expert-tensor-mib',tensor_caps]
            if timings: args+=['--expert-timings']
            if resident_weights: args+=['--gdn-resident-weights']
            if dense_weights: args+=['--dense-resident-weights']
            if state_resident: args+=['--gdn-resident-state']
            with mock.patch.object(sys,'argv',args), mock.patch.object(M,'Resident',side_effect=runner), \
                 mock.patch.object(M.S,'host_memory',return_value=host), \
                 mock.patch.object(M.S,'model_working_set',return_value=dict(working_set_bound_bytes=0)), \
                 mock.patch.object(M.S,'verified_devices',return_value=[dict(total=8<<30)]*2), \
                 mock.patch.object(M.S,'sample_vram',return_value=[dict(used=0,total=8<<30)]*2), \
                 mock.patch.object(M.S,'rss',return_value=dict(VmRSS=1,VmHWM=1,VmSwap=0)), \
                 mock.patch.object(M.S,'guard_host',side_effect=ValueError('pressure') if pressure else None), \
                 mock.patch.object(M.S,'placements'), \
                 mock.patch.object(M.L,'devices',return_value={0:dict(entries=4),1:dict(entries=4)}):
                if pressure:
                    with self.assertRaisesRegex(ValueError,'pressure'): M.main()
                else: M.main()
            receipt=json.loads((out/'summary.json').read_text())
            self.assertTrue(receipt['child_reaped'])
            self.assertEqual(receipt['status'],'STOP' if pressure else 'complete')
            self.assertEqual(receipt['last_process']['VmRSS'],1)
            self.assertEqual(len(commands),1)
            self.assertIn('--stdio',commands[0])
            self.assertEqual('--expert-timings' in commands[0],timings)
            self.assertEqual(commands[0].count('--gdn-resident-weights'),int(resident_weights))
            self.assertEqual(commands[0].count('--dense-resident-weights'),int(dense_weights))
            self.assertEqual(commands[0].count('--gdn-resident-state'),int(state_resident))
            self.assertEqual(commands[0][commands[0].index('--expert-tensor-mib')+1],tensor_caps or '2404,2404')
            if not pressure:
                self.assertEqual(len(receipt['requests']),4)
                self.assertEqual(receipt['excluded_warmups'],1)
                self.assertEqual(receipt['exit_code'],0)
    def test_main_four_receipts(self): self.main_run(False)
    def test_main_pressure_receipt(self): self.main_run(True)
    def test_main_timing_forwarding(self): self.main_run(False,True)
    def test_main_resident_weight_forwarding(self): self.main_run(False,True,True)
    def test_main_dense_weight_forwarding(self): self.main_run(False,True,False,True)
    def test_main_state_forwarding(self): self.main_run(False,state_resident=True)
    def test_main_state_and_dense_forwarding(self): self.main_run(False,dense_weights=True,state_resident=True)
    def test_main_explicit_default_tensor_caps(self): self.main_run(False,tensor_caps='2404,2404')
    def test_main_tensor_caps_with_state_and_dense(self):
        self.main_run(False,dense_weights=True,state_resident=True,tensor_caps='3606,3606')
    def test_invalid_tensor_caps_before_launch(self):
        for caps in ('3606','3606,3606,1','0,3606','3606,0','-1,3606','8193,3606','3606,8193',
                     '3606.0,3606','3606,NaN','3606, 3606',',3606','3606,','3606,+1'):
            with self.subTest(caps=caps), tempfile.TemporaryDirectory() as d:
                out=Path(d)/'out'
                args=['resident','--binary',sys.executable,'--binary-sha256','0'*64,
                      '--out',str(out),'--tokens-file',str(Path(d)/'missing-tokens'),'--max-new','32',
                      '--plan','p0','--plan','p1','--expert-tensor-mib',caps]
                with mock.patch.object(sys,'argv',args), mock.patch.object(sys,'stderr'), \
                     mock.patch.object(M,'Resident') as child, self.assertRaises(SystemExit) as stopped:
                    M.main()
                self.assertEqual(stopped.exception.code,2)
                child.assert_not_called()
                self.assertFalse(out.exists())
    def test_main_both_weight_flags(self): self.main_run(False,True,True,True)
if __name__=='__main__': unittest.main()
