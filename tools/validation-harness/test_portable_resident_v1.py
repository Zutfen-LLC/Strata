import copy
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
    if line=='QUIT\n': sys.exit(70 if sys.argv[1]=='fatal_shutdown' else 0)
    if sys.argv[1]=='hang': time.sleep(20)
    if sys.argv[1]=='fail': sys.exit(2)
    n,t=line.split()[1:]
    tokens=list(map(int,t.split(',')))
    for p,t in enumerate(tokens): print(json.dumps(dict(position=p,token=t,generated=False,greedy_next=7)))
    print(json.dumps(dict(stop='max_new',position=len(tokens))))
    print(json.dumps(dict(done=i,prompt_count=len(tokens),output_count=0,prompt_ms=0,cpu_hits_delta=10,cpu_pool_gu_ms_delta=(float('nan') if sys.argv[1]=='nan' else -1.25 if sys.argv[1]=='negative' else 1.25),cpu_pool_quant_ms_delta=.25,cpu_pool_down_ms_delta=2.5,cpu_pool_elapsed_ms_delta=4.1,cpu_verifications_delta=0,cpu_copied_bytes_delta=0,cpu_digest_bytes_delta=0,source_invalidations_delta=0,model_step_ms_delta=18.,gdn_layer_ms_delta=10.,qsa_layer_ms_delta=5.,output_head_ms_delta=2.,gdn_attention_exchange_ms_delta=3.,gdn_attention_compute_ms_delta=2.,gdn_ffn_ms_delta=6.,gdn_outer_residual_ms_delta=1.,gdn_attention_surrounding_ms_delta=1.,gdn_ffn_non_provider_ms_delta=1.,gdn_attention_exchange_calls_delta=36*len(tokens),gdn_attention_compute_calls_delta=36*len(tokens),gdn_ffn_calls_delta=36*len(tokens),model_residual_ms_delta=1.,gdn_provider_ms_delta=5.,qsa_provider_ms_delta=2.5,model_step_calls_delta=len(tokens),gdn_layer_calls_delta=36*len(tokens),qsa_layer_calls_delta=12*len(tokens),output_head_calls_delta=len(tokens),expert_provider_ms_delta=7.5,expert_provider_calls_delta=len(tokens)*48,gpu_infer_rows_ms_delta=[1.0,float('nan') if sys.argv[1]=='nan_gpu' else -1.0 if sys.argv[1]=='negative_gpu' else 2.0],gpu_infer_rows_calls_delta=[1,1],gpu_entries_per_device_delta=[1,1],gdn_candidate_readback_ms_delta=.2,gdn_candidate_finite_ms_delta=.3,gdn_state_commit_ms_delta=(.5 if sys.argv[1]=='production' else 0),gdn_candidate_readback_calls_delta=(36 if sys.argv[1]=='production' else 109)*len(tokens),gdn_candidate_finite_calls_delta=(36 if sys.argv[1]=='production' else 109)*len(tokens),gdn_state_commit_calls_delta=(36*len(tokens) if sys.argv[1]=='production' else 0))))
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
        detailed=dict(m,gdn_candidate_readback_ms_delta=.2,gdn_candidate_finite_ms_delta=.3,
                      gdn_state_commit_ms_delta=.5,gdn_candidate_readback_calls_delta=218,
                      gdn_candidate_finite_calls_delta=218,gdn_state_commit_calls_delta=72)
        M.model_phase_fields(detailed,2)
        M.model_phase_fields(dict(detailed,gdn_state_commit_ms_delta=0,gdn_state_commit_calls_delta=0),2)
        for key,value in [('gdn_candidate_readback_ms_delta',float('nan')),
                          ('gdn_candidate_finite_ms_delta',-1),('gdn_state_commit_ms_delta',float('inf')),
                          ('gdn_candidate_readback_ms_delta',.8),('gdn_candidate_finite_ms_delta',.9),
                          ('gdn_state_commit_ms_delta',1.1),('gdn_candidate_readback_calls_delta',217),
                          ('gdn_candidate_finite_calls_delta',True),('gdn_state_commit_calls_delta',1),
                          ('gdn_state_commit_calls_delta',0)]:
            with self.subTest(candidate=key),self.assertRaises(ValueError):
                M.model_phase_fields(dict(detailed,**{key:value}),2)
        production = dict(detailed,gdn_candidate_readback_calls_delta=72,gdn_candidate_finite_calls_delta=72)
        M.model_phase_fields(production,2,'device_production')
        for policy, sample in [('device_production',production),('eager_diagnostic',detailed)]:
            for field in ('gdn_candidate_readback_calls_delta','gdn_candidate_finite_calls_delta'):
                for wrong in (0,71,73,218 if policy=='device_production' else 72):
                    with self.subTest(policy=policy,field=field,wrong=wrong),self.assertRaises(ValueError):
                        M.model_phase_fields(dict(sample,**{field:wrong}),2,policy)
            for field in ('gdn_candidate_readback_calls_delta','gdn_candidate_finite_calls_delta',
                          'gdn_state_commit_calls_delta','gdn_state_commit_ms_delta'):
                missing=dict(sample);del missing[field]
                with self.subTest(policy=policy,missing=field),self.assertRaises(ValueError):
                    M.model_phase_fields(missing,2,policy)
        for commits in (0,1,71,73):
            with self.subTest(commits=commits),self.assertRaises(ValueError):
                M.model_phase_fields(dict(production,gdn_state_commit_calls_delta=commits),2,'device_production')
        with self.assertRaisesRegex(ValueError,'incomplete GDN'):
            M.model_phase_fields(m,2,'device_production')
        with self.assertRaisesRegex(ValueError,'unknown GDN'):
            M.model_phase_fields(production,2,'unknown')
        del detailed['gdn_candidate_finite_calls_delta']
        with self.assertRaisesRegex(ValueError,'incomplete GDN'): M.model_phase_fields(detailed,2)
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
    def test_unconfirmed_drain_exit_cannot_finish_cleanly(self):
        def action(r):
            r.request([5],0,1) # valid output does not establish safe shutdown
            with self.assertRaisesRegex(RuntimeError,'resident failed on exit'): r.finish()
            self.assertEqual(r.child.returncode,70)
        self.run_fake('fatal_shutdown',action)
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
    def main_run(self,pressure,timings=False,resident_weights=False,dense_weights=False,state_resident=False,tensor_caps=None,production=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'out'; tokens=root/'tokens.json';tokens.write_text('[5,7]')
            plans=[root/'p0',root/'p1']
            for plan in plans: plan.write_text('frozen')
            real=M.Resident
            commands=[]
            def runner(command,out,tick):
                commands.append(command)
                return real([sys.executable,'-c',FAKE,'production' if production else 'ok'],out,tick,load_seconds=2,request_seconds=2,total_seconds=5)
            host=dict(effective_available=1<<40,pswpout=0)
            args=['resident','--binary',sys.executable,'--binary-sha256',M.Q.bundle.sha(sys.executable),
                  '--out',str(out),'--tokens-file',str(tokens),'--max-new','0',
                  '--plan',str(plans[0]),'--plan',str(plans[1])]
            if tensor_caps is not None: args+=['--expert-tensor-mib',tensor_caps]
            if timings: args+=['--expert-timings']
            if resident_weights: args+=['--gdn-resident-weights']
            if dense_weights: args+=['--dense-resident-weights']
            if state_resident: args+=['--gdn-resident-state']
            if production: args+=['--gdn-production']
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
            self.assertEqual(commands[0].count('--gdn-production'),int(production))
            self.assertEqual(receipt['gdn_policy'],'device_production' if production else 'eager_diagnostic')
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
    def test_main_production_policy(self):
        self.main_run(False,timings=True,state_resident=True,production=True)
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

class SuiteControls(unittest.TestCase):
    def fixtures(self, root):
        from test_portable_quality_v1 import Tokenizer
        tok=Tokenizer()
        tok.tokens=tok.tokens+['']*(248320-len(tok.tokens))
        suite=root/'suite'; M.Q.prepare(tok,suite)
        prompts,binding=M.P.frozen(tok,suite)
        reference=root/'reference'; reference.mkdir()
        checks=root/'checks'; checks.mkdir()
        binary=root/'binary'; binary.write_bytes(b'qualified fixture only')
        plans=[root/'die0.plan',root/'die1.plan']
        for i,p in enumerate(plans): p.write_text(str(i))
        pins={str(p):M.Q.bundle.sha(p) for p in [binary]+plans}
        (checks/'input-identities.sha256').write_text(''.join(f'{v}  {k}\n' for k,v in pins.items()))
        summary=dict(status='complete',suite=binding,binary_sha256=pins[str(binary)],
                     expert_plans=[f'{i}:{p}' for i,p in enumerate(plans)],requests=[])
        records=[]
        for index,prompt in enumerate(prompts,1):
            # Case 9 deliberately differs from all others, like a changed fresh
            # backend answer: the resident must follow its own per-case target.
            generated=[9,248046] if index==9 else [248046]
            ids=prompt+generated
            rows=[dict(position=i,token=t,generated=i>=len(prompt),greedy_next=ids[i+1] if i+1<len(ids) else 0)
                  for i,t in enumerate(ids)]
            stop=dict(stop='eos',position=len(rows))
            records.append(dict(rows=rows,stop=stop))
            case=reference/f'case-{index:02d}';case.mkdir()
            argv=[str(binary),'--model',str(binary),'--manifest',M.S.MANIFEST,'--vulkan','0,1',
                  '--split','32','--device-cap-mib','7680,7680','--threads','4','--expert-workers','12',
                  '--host-cap-mib','32768','--tokens',','.join(map(str,prompt)),'--max-new','32','--context','256',
                  '--expert-tensor-mib','5529,5529','--expert-workspace-mib','128,128',
                  '--expert-free-reserve-mib','512,512']
            for spec in summary['expert_plans']: argv+=['--expert-plan',spec]
            M.Q.write(case/'argv.json',argv)
            (case/'stdout.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows+[stop]))
            summary['requests'].append(dict(index=index,tokens=generated,finish='eos'))
        M.Q.write(reference/'summary.json',summary)
        return tok,suite,prompts,binding,reference,binary,plans,records

    def metrics(self,prompt,record,index):
        n=len(record['rows']); outputs=n-len(prompt)
        m=dict(done=index,prompt_count=len(prompt),output_count=outputs,
               setup_reset_ms=0.,prompt_ms=1.,decode_ms=0.,selection_prompt_ms=0.,
               selection_decode_ms=0.,terminal_feed_ms=0.,selection_prompt_steps=len(prompt)-1,
               selection_decode_steps=outputs,cpu_hits_delta=1,cpu_verifications_delta=0,
               cpu_copied_bytes_delta=0,cpu_digest_bytes_delta=0,source_invalidations_delta=0,
               cpu_pool_gu_ms_delta=1.,cpu_pool_quant_ms_delta=1.,cpu_pool_down_ms_delta=1.,cpu_pool_elapsed_ms_delta=3.,
               model_step_calls_delta=n,gdn_layer_calls_delta=36*n,qsa_layer_calls_delta=12*n,
               output_head_calls_delta=n,gdn_attention_exchange_calls_delta=36*n,
               gdn_attention_compute_calls_delta=36*n,gdn_ffn_calls_delta=36*n,
               model_step_ms_delta=18.,gdn_layer_ms_delta=10.,qsa_layer_ms_delta=5.,output_head_ms_delta=2.,
               gdn_attention_exchange_ms_delta=3.,gdn_attention_compute_ms_delta=2.,gdn_ffn_ms_delta=6.,
               gdn_outer_residual_ms_delta=1.,gdn_attention_surrounding_ms_delta=1.,gdn_ffn_non_provider_ms_delta=1.,
               model_residual_ms_delta=1.,gdn_provider_ms_delta=5.,qsa_provider_ms_delta=2.5,
               expert_provider_ms_delta=7.5,expert_provider_calls_delta=48*n,
               gpu_infer_rows_ms_delta=[1.,2.],gpu_infer_rows_calls_delta=[1,1],gpu_entries_per_device_delta=[1,1])
        return m

    def test_main_suite_one_child_and_owned_failure_cleanup(self):
        child_code=r"""
import json,sys
samples=json.load(open(sys.argv[1]))
print(json.dumps(dict(ready=True,schema=1,load_ms=1.)),flush=True)
for index,line in enumerate(sys.stdin,1):
    if line=='QUIT\n': break
    sample=samples[index-1]
    assert line==sample['command']
    for row in sample['rows']+[sample['stop'],sample['metrics']]: print(json.dumps(row))
    sys.stdout.flush()
"""
        for failure in (False,True):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(root)
                fresh=json.loads((ref/'summary.json').read_text());fresh['qsa_reuse_attention']=True;fresh['qsa_policy']='device_production';fresh['overlap_shared_cpu']=True
                (ref/'summary.json').write_text(json.dumps(fresh))
                for case in ref.glob('case-*'):
                    path=case/'argv.json'
                    path.write_text(json.dumps(json.loads(path.read_text())+['--overlap-shared-cpu','--dense-resident-weights','--gdn-resident-state','--qsa-reuse-attention','--qsa-production']))
                out=root/'out';samples=[]
                for index in range(1,25):
                    prompt=prompts[(index-1)%12];record=copy.deepcopy(records[(index-1)%12])
                    record['metrics']=self.metrics(prompt,record,index)
                    if index<=12: record['metrics']['cpu_verifications_delta']=index
                    if failure and index==13: record['metrics']['cpu_copied_bytes_delta']=1
                    record['command']='GEN 32 '+','.join(map(str,prompt))+'\n'
                    samples.append(record)
                payload=root/'samples.json';payload.write_text(json.dumps(samples))
                real=M.Resident;children=[];commands=[];deadlines=[]
                def runner(command,out,tick,**bounds):
                    commands.append(command);deadlines.append(bounds)
                    child=real([sys.executable,'-c',child_code,str(payload)],out,tick,**bounds)
                    children.append(child)
                    return child
                host=dict(effective_available=1<<40,pswpout=0)
                args=['resident','--binary',str(binary),'--binary-sha256',M.Q.bundle.sha(binary),
                      '--out',str(out),'--suite',str(suite),'--gguf',str(binary),'--reference-root',str(ref),
                      '--expert-tensor-mib','5529,5529','--plan',str(plans[0]),'--plan',str(plans[1]),
                      '--overlap-shared-cpu','--dense-resident-weights','--gdn-resident-state','--qsa-reuse-attention','--qsa-production']
                with mock.patch.object(sys,'argv',args),mock.patch.object(M,'Resident',side_effect=runner), \
                     mock.patch.object(M.Q,'tokenizer',return_value=tok), \
                     mock.patch.object(M.S,'host_memory',return_value=host), \
                     mock.patch.object(M.S,'model_working_set',return_value=dict(working_set_bound_bytes=0)), \
                     mock.patch.object(M.S,'verified_devices',return_value=[dict(total=8<<30)]*2), \
                     mock.patch.object(M.S,'sample_vram',return_value=[dict(used=0,total=8<<30)]*2), \
                     mock.patch.object(M.S,'rss',return_value=dict(VmRSS=1,VmHWM=1,VmSwap=0)), \
                     mock.patch.object(M.S,'guard_host'),mock.patch.object(M.S,'placements'), \
                     mock.patch.object(M.L,'devices',return_value={0:dict(entries=4),1:dict(entries=4)}):
                    if failure:
                        with self.assertRaisesRegex(ValueError,'warm request'): M.main()
                    else: M.main()
                self.assertEqual(len(children),1)
                self.assertIsNotNone(children[0].child.returncode)
                self.assertEqual(deadlines,[dict(load_seconds=180,request_seconds=600,total_seconds=3600)])
                argv=commands[0]
                self.assertEqual(argv[argv.index('--max-requests')+1],'25')
                self.assertEqual(argv[argv.index('--expert-tensor-mib')+1],'5529,5529')
                self.assertIn('--expert-timings',argv)
                self.assertEqual(argv.count('--overlap-shared-cpu'),1)
                self.assertEqual(argv.count('--dense-resident-weights'),1)
                self.assertEqual(argv.count('--gdn-resident-state'),1)
                self.assertEqual(argv.count('--qsa-reuse-attention'),1)
                self.assertEqual(argv.count('--qsa-production'),1)
                self.assertNotIn('--gdn-resident-weights',argv)
                receipt=json.loads((out/'summary.json').read_text())
                self.assertTrue(receipt['child_reaped'])
                self.assertIs(receipt['overlap_shared_cpu'],True)
                self.assertEqual(receipt['overlap_environment'],M.P.overlap_environment())
                self.assertTrue(all(p['overlap_shared_cpu'] is True for p in receipt['passes']))
                self.assertIs(receipt['qsa_reuse_attention'],True)
                self.assertEqual(receipt['qsa_policy'],'device_production')
                self.assertTrue(all(p['qsa_policy']=='device_production' for p in receipt['passes']))
                self.assertTrue(all(p['qsa_reuse_attention'] is True for p in receipt['passes']))
                self.assertEqual(receipt['status'],'STOP' if failure else 'complete')
                self.assertEqual(len(list(out.glob('request-*.json'))),13 if failure else 24)
                if failure: self.assertFalse((out/'request-14.json').exists())

    def test_reference_binding_and_mutations(self):
        for mutation in ('none','suite','binary','order','plan','argv','rows','summary'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(Path(d))
                summary=json.loads((ref/'summary.json').read_text())
                if mutation=='suite': summary['suite']={}
                if mutation=='binary': summary['binary_sha256']='0'*64
                if mutation=='order': summary['requests'].reverse()
                if mutation=='plan': plans[1].write_text('mutated')
                if mutation=='summary': summary['requests'][8]['tokens']=[248046]
                if mutation=='argv':
                    path=ref/'case-09/argv.json';argv=json.loads(path.read_text());argv[argv.index('--max-new')+1]='31';path.write_text(json.dumps(argv))
                if mutation=='rows':
                    path=ref/'case-09/stdout.jsonl';path.write_text(path.read_text().replace('"token": 9,','"token": 10,',1))
                (ref/'summary.json').write_text(json.dumps(summary))
                args=(ref,binding,prompts,len(tok.tokens),M.Q.bundle.sha(binary),plans,'5529,5529')
                if mutation=='none': self.assertEqual(M.reference_rows(*args),records)
                else:
                    with self.assertRaises(ValueError): M.reference_rows(*args)
                # frozen() remains the authority, including mutation rejection.
                (suite/'requests').write_text('GEN 32 1\n')
                with self.assertRaises(ValueError): M.P.frozen(tok,suite)

    def test_reference_residency_policy_mismatch_and_duplicates(self):
        for flag in M.P.RESIDENCY_FLAGS:
            for recorded, expected in ((0, True), (1, False), (2, True), (1, True)):
                with self.subTest(flag=flag,recorded=recorded,expected=expected), tempfile.TemporaryDirectory() as d:
                    tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(Path(d))
                    if flag == '--overlap-shared-cpu':
                        summary=json.loads((ref/'summary.json').read_text());summary['overlap_shared_cpu']=expected
                        (ref/'summary.json').write_text(json.dumps(summary))
                    if flag == '--qsa-production':
                        summary=json.loads((ref/'summary.json').read_text());summary['qsa_policy']='device_production' if expected else 'eager_diagnostic'
                        (ref/'summary.json').write_text(json.dumps(summary))
                    if flag == '--qsa-reuse-attention':
                        summary=json.loads((ref/'summary.json').read_text());summary['qsa_reuse_attention']=expected
                        (ref/'summary.json').write_text(json.dumps(summary))
                    for case in ref.glob('case-*'):
                        path=case/'argv.json'
                        path.write_text(json.dumps(json.loads(path.read_text())+[flag]*recorded))
                    args=(ref,binding,prompts,len(tok.tokens),M.Q.bundle.sha(binary),plans,'5529,5529')
                    if recorded == int(expected):
                        self.assertEqual(M.reference_rows(*args,residency={flag:expected}),records)
                    else:
                        with self.assertRaisesRegex(ValueError,'residency policy mismatch'):
                            M.reference_rows(*args,residency={flag:expected})

    def test_qsa_summary_binding_mutations(self):
        for expected in (False, True):
            for recorded in ('missing', False, True, 0, 1, 'true', None):
                with self.subTest(expected=expected,recorded=recorded),tempfile.TemporaryDirectory() as d:
                    tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(Path(d))
                    summary=json.loads((ref/'summary.json').read_text())
                    if recorded != 'missing': summary['qsa_reuse_attention']=recorded
                    (ref/'summary.json').write_text(json.dumps(summary))
                    if expected:
                        for case in ref.glob('case-*'):
                            path=case/'argv.json'
                            path.write_text(json.dumps(json.loads(path.read_text())+['--qsa-reuse-attention']))
                    args=(ref,binding,prompts,len(tok.tokens),M.Q.bundle.sha(binary),plans,'5529,5529')
                    if recorded is expected or (not expected and recorded == 'missing'):
                        self.assertEqual(M.reference_rows(*args,residency={'--qsa-reuse-attention':expected}),records)
                    else:
                        with self.assertRaisesRegex(ValueError,'QSA summary'):
                            M.reference_rows(*args,residency={'--qsa-reuse-attention':expected})

    def test_overlap_summary_binding_mutations(self):
        for expected in (False, True):
            for recorded in ('missing', False, True, 0, 1, 'true', None):
                with self.subTest(expected=expected,recorded=recorded),tempfile.TemporaryDirectory() as d:
                    tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(Path(d))
                    summary=json.loads((ref/'summary.json').read_text())
                    if recorded != 'missing': summary['overlap_shared_cpu']=recorded
                    (ref/'summary.json').write_text(json.dumps(summary))
                    if expected:
                        for case in ref.glob('case-*'):
                            path=case/'argv.json'
                            path.write_text(json.dumps(json.loads(path.read_text())+['--overlap-shared-cpu']))
                    args=(ref,binding,prompts,len(tok.tokens),M.Q.bundle.sha(binary),plans,'5529,5529')
                    if recorded is expected or (not expected and recorded == 'missing'):
                        self.assertEqual(M.reference_rows(*args,residency={'--overlap-shared-cpu':expected}),records)
                    else:
                        with self.assertRaisesRegex(ValueError,'shared CPU overlap summary'):
                            M.reference_rows(*args,residency={'--overlap-shared-cpu':expected})

    def test_qsa_production_summary_mutations(self):
        for enabled in (False, True):
            expected='device_production' if enabled else 'eager_diagnostic'
            for recorded in ('missing','eager_diagnostic','device_production',True,False,0,None,{}):
                with self.subTest(enabled=enabled,recorded=recorded),tempfile.TemporaryDirectory() as d:
                    tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(Path(d))
                    summary=json.loads((ref/'summary.json').read_text())
                    if recorded != 'missing': summary['qsa_policy']=recorded
                    (ref/'summary.json').write_text(json.dumps(summary))
                    if enabled:
                        for case in ref.glob('case-*'):
                            path=case/'argv.json'
                            path.write_text(json.dumps(json.loads(path.read_text())+['--qsa-production']))
                    args=(ref,binding,prompts,len(tok.tokens),M.Q.bundle.sha(binary),plans,'5529,5529')
                    if recorded == expected or (not enabled and recorded == 'missing'):
                        self.assertEqual(M.reference_rows(*args,residency={'--qsa-production':enabled}),records)
                    else:
                        with self.assertRaisesRegex(ValueError,'QSA production summary'):
                            M.reference_rows(*args,residency={'--qsa-production':enabled})

    def test_suite_phase_policy_binding(self):
        for policy, multiplier in [('eager_diagnostic',109),('device_production',36)]:
            for recorded in (36,109):
                with self.subTest(policy=policy,recorded=recorded),tempfile.TemporaryDirectory() as d:
                    tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(Path(d))
                    out=Path(d)/'out';out.mkdir()
                    def request(prompt,max_new,index):
                        record=copy.deepcopy(records[(index-1)%12])
                        m=self.metrics(prompt,record,index);n=len(record['rows'])
                        m.update(gdn_candidate_readback_ms_delta=.2,gdn_candidate_finite_ms_delta=.3,
                                 gdn_state_commit_ms_delta=.5,gdn_candidate_readback_calls_delta=recorded*n,
                                 gdn_candidate_finite_calls_delta=recorded*n,gdn_state_commit_calls_delta=36*n)
                        record['metrics']=m
                        return record
                    runner=mock.Mock(request=mock.Mock(side_effect=request));receipt={'gdn_policy':policy}
                    if recorded==multiplier:
                        M.run_suite(runner,out,prompts,records,binding,len(tok.tokens),'binary',receipt)
                        self.assertEqual(runner.request.call_count,24)
                    else:
                        with self.assertRaisesRegex(ValueError,'candidate diagnostic containment/counts'):
                            M.run_suite(runner,out,prompts,records,binding,len(tok.tokens),'binary',receipt)
                        self.assertEqual(runner.request.call_count,1)
                    saved=json.loads((out/'pass-1'/'summary.json').read_text())
                    self.assertEqual(saved['gdn_policy'],policy)
                    self.assertEqual(saved['status'],'complete' if recorded==multiplier else 'STOP')

    def test_two_pass_order_warm_policy_and_fail_stop(self):
        for failure in ('none','warm','invalidate','reference9','count','timer','done','child'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                tok,suite,prompts,binding,ref,binary,plans,records=self.fixtures(Path(d))
                out=Path(d)/'out';out.mkdir();calls=[]
                def request(prompt,max_new,index):
                    calls.append((prompt,max_new,index))
                    if failure=='child' and index==3: raise RuntimeError('child failure')
                    record=copy.deepcopy(records[(index-1)%12])
                    record['metrics']=self.metrics(prompt,record,index)
                    m=record['metrics']
                    if index<=12: m.update(cpu_verifications_delta=index,cpu_copied_bytes_delta=10,cpu_digest_bytes_delta=20)
                    if failure=='warm' and index==13: m['cpu_verifications_delta']=1
                    if failure=='invalidate' and index==2: m['source_invalidations_delta']=1
                    if failure=='reference9' and index==9: record['rows'][0]['greedy_next']+=1
                    if failure=='count' and index==2: m['model_step_calls_delta']-=1
                    if failure=='timer' and index==2: m['cpu_pool_elapsed_ms_delta']=float('nan')
                    if failure=='done' and index==2: m['done']=1
                    return record
                runner=mock.Mock(request=mock.Mock(side_effect=request));receipt={'gdn_policy':'eager_diagnostic'}
                if failure=='none': M.run_suite(runner,out,prompts,records,binding,len(tok.tokens),'binary',receipt)
                else:
                    with self.assertRaises((ValueError,RuntimeError)):
                        M.run_suite(runner,out,prompts,records,binding,len(tok.tokens),'binary',receipt)
                limit=dict(none=24,warm=13,invalidate=2,reference9=9,count=2,timer=2,done=2,child=3)[failure]
                self.assertEqual(calls,[(prompts[(i-1)%12],32,i) for i in range(1,limit+1)])
                saved=json.loads((out/f"pass-{2 if failure in ('none','warm') else 1}"/'summary.json').read_text())
                self.assertEqual(saved['status'],'complete' if failure=='none' else 'STOP')
                if failure=='none':
                    self.assertEqual([r['index'] for r in saved['requests']],list(range(1,13)))
                    self.assertEqual(saved['requests'][8]['tokens'],[9,248046])
                    self.assertEqual(saved['consumed_rows'],sum(map(len,prompts))+13)

if __name__=='__main__': unittest.main()
