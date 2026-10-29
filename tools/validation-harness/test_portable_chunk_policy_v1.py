import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import portable_quality_v1 as P
import portable_resident_v1 as R
import test_portable_resident_v1 as ResidentTests


def metrics(size=4,prompt=10,consumed=12):
    c=(prompt-1+size-1)//size; f=c+consumed-prompt+1
    return dict(chunk_policy='layer_major_v1',prompt_chunk_size=size,gdn_subphases_available=False,
        logical_tokens_delta=consumed,prompt_chunks_delta=c,prompt_chunk_tokens_delta=prompt-1,
        chunk_layer_calls_delta=48*c,chunk_ffn_pre_calls_delta=48*c,chunk_provider_calls_delta=48*c,chunk_ffn_post_calls_delta=48*c,
        model_forward_calls_delta=f,model_layer_calls_delta=48*f,ffn_pre_calls_delta=48*f,provider_calls_delta=48*f,ffn_post_calls_delta=48*f)


class ChunkPolicy(unittest.TestCase):
    def test_flags_and_canonical_argv(self):
        for size in range(2,9):
            flags=P.residency_flags(SimpleNamespace(prompt_chunk_size=size))
            self.assertEqual(flags,['--prompt-head-elision','--prompt-chunk-size',str(size)])
            self.assertEqual(P.chunk_argv(flags),size)
        for value in (True,False,1,9,'4',None):
            with self.assertRaises(ValueError):P.chunk_size(SimpleNamespace(prompt_chunk_size=value))
        for argv in (['--prompt-chunk-size'],['--prompt-chunk-size','0'],['--prompt-chunk-size','04'],
                     ['--prompt-chunk-size','4','--prompt-chunk-size','4']):
            with self.assertRaises(ValueError):P.chunk_argv(argv)

    def test_token_binding_and_unavailable_logits(self):
        row=dict(position=0,token=5,generated=False,greedy_next=None,schema=2,head_policy='prompt_head_elision',
                 logits_available=False,chunk_policy='layer_major_v1',prompt_chunk_size=4)
        P.token_receipt(row,0,3,248320,'prompt_head_elision',4)
        for key,value in (('prompt_chunk_size',True),('prompt_chunk_size',3),('chunk_policy','foreign'),
                          ('greedy_next',7),('logits_available',True),('schema',3)):
            with self.subTest(key=key),self.assertRaises(ValueError):P.token_receipt(dict(row,**{key:value}),0,3,248320,'prompt_head_elision',4)
        with self.assertRaises(ValueError):P.token_receipt(row,0,3,248320,'prompt_head_elision')
        with self.assertRaises(ValueError):P.token_receipt(row,0,3,248320,'eager_diagnostic',4)
        for size in (1,9,True,'4'):
            with self.assertRaises(ValueError):P.chunk_binding(row,size)
        last=dict(row,position=2,logits_available=True,greedy_next=7)
        P.token_receipt(last,2,3,248320,'prompt_head_elision',4)
        with self.assertRaises(ValueError):P.token_receipt(dict(last,greedy_next=None),2,3,248320,'prompt_head_elision',4)

    def test_actual_execution_counter_mutations(self):
        sample=metrics()
        self.assertEqual(P.chunk_fields(sample,10,12,4),6)
        for key in P.CHUNK_COUNTERS:
            for value in (sample[key]+1,True,None):
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):P.chunk_fields(dict(sample,**{key:value}),10,12,4)
            missing=dict(sample);del missing[key]
            with self.assertRaises(ValueError):P.chunk_fields(missing,10,12,4)
        for bad in (dict(sample,gdn_subphases_available=True),dict(sample,prompt_chunk_size=3),dict(sample,chunk_policy='foreign')):
            with self.assertRaises(ValueError):P.chunk_fields(bad,10,12,4)
        with self.assertRaises(ValueError):P.chunk_fields(sample,10,12,0)

    def test_coarse_timing_and_no_fake_t1_subphases(self):
        sample=metrics()
        sample.update(prompt_count=10,heads_skipped_delta=9,heads_executed_delta=3,model_step_calls_delta=6,
            gdn_layer_calls_delta=216,qsa_layer_calls_delta=72,output_head_calls_delta=3,
            model_step_ms_delta=18.,gdn_layer_ms_delta=10.,qsa_layer_ms_delta=5.,output_head_ms_delta=2.,
            model_residual_ms_delta=1.,gdn_provider_ms_delta=5.,qsa_provider_ms_delta=2.5,expert_provider_ms_delta=7.5)
        R.model_phase_fields(sample,12,'device_production','prompt_head_elision',4)
        for key,value in (('model_step_calls_delta',12),('gdn_layer_calls_delta',432),('output_head_calls_delta',12),
                          ('gdn_attention_compute_calls_delta',0),('gdn_candidate_readback_calls_delta',0),
                          ('gdn_provider_ms_delta',11.),('model_residual_ms_delta',2.)):
            with self.subTest(key=key),self.assertRaises(ValueError):R.model_phase_fields(dict(sample,**{key:value}),12,'device_production','prompt_head_elision',4)

    def test_resident_done_and_row_mutations(self):
        rows=[dict(position=i,token=t,generated=False,greedy_next=7 if i==2 else None,schema=2,
                   head_policy='prompt_head_elision',logits_available=i==2,chunk_policy='layer_major_v1',prompt_chunk_size=2)
              for i,t in enumerate([5,6,7])]
        done=dict(metrics(2,3,3),done=1,prompt_count=3,output_count=0,receipt_schema=2,
                  head_policy='prompt_head_elision',heads_skipped_delta=2,heads_executed_delta=1)
        samples=rows+[dict(stop='max_new',position=3),done]
        def request(records):
            runner=object.__new__(R.Resident);runner.child=mock.Mock();runner.request_seconds=1
            runner.head_policy='prompt_head_elision';runner.prompt_chunk_size=2;runner.row=mock.Mock(side_effect=records)
            return runner.request([5,6,7],0,1)
        self.assertEqual(request(samples)['rows'],rows)
        for key,value in (('prompt_chunk_size',3),('chunk_policy','foreign'),('prompt_chunks_delta',2),('provider_calls_delta',144)):
            bad=copy.deepcopy(samples);bad[-1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):request(bad)
        bad=copy.deepcopy(samples);bad[0]['prompt_chunk_size']=3
        with self.assertRaises(ValueError):request(bad)

    def test_nonbenchmark_fresh_chunk_run_requests_and_consumes_timings(self):
        from contextlib import ExitStack
        for batched,causal,missing in ((False,False,False),(True,False,False),(True,False,True),(True,True,False),(True,True,True)):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);binary=root/'binary';binary.write_bytes(b'mocked child')
                args=SimpleNamespace(suite=root/'suite',binary=binary,binary_sha256=P.Q.bundle.sha(binary),
                    devices='0,1',pci=','.join(P.S.PCI),split=32,device_cap_mib=7680,
                    out=root/'out',prompt_chunk_size=2,benchmark=False,dump_executed_logits=True,gdn_batch_input_projections=batched,gdn_causal_chunk=causal,gdn_production=causal)
                rows=[dict(position=i,token=t,generated=i==3,greedy_next=None if i<2 else 248046,
                    schema=2,head_policy='prompt_head_elision',logits_available=i>=2,
                    chunk_policy='layer_major_v1',prompt_chunk_size=2) for i,t in enumerate([5,6,7,248046])]
                timing=metrics(2,3,4)
                if batched and not missing:timing['gdn_batch_input_projections']=True
                if causal and not missing:timing['gdn_causal_chunk']=True
                def launch(argv,stdout,stderr,**kwargs):
                    self.assertEqual(argv.count(P.GDN_PROJECTION_FLAG),int(batched))
                    self.assertEqual(argv.count(P.GDN_CAUSAL_FLAG),int(causal))
                    self.assertEqual(argv.count('--timings'),1)
                    Path(argv[argv.index('--timings')+1]).write_text(json.dumps(timing))
                    import array
                    self.assertEqual(argv.count('--dump-executed-logits'),1)
                    requested=Path(argv[argv.index('--dump-executed-logits')+1])
                    self.assertEqual(list(requested.iterdir()),[])
                    raw=array.array('f',[0.])*248320;raw[248046]=1.
                    for position in (2,3):(requested/f'pos-{position}.f32').write_bytes(raw.tobytes())
                    diagnostic=dict(schema=1,format='executed_heads_f32le_v1',head_policy='prompt_head_elision',
                        chunk_policy='layer_major_v1',prompt_chunk_size=2,vocabulary=248320,prompt_tokens=[5,6,7],
                        output_count=1,stop='eos',frames=[dict(position=p,token=t,role=r,file=f'pos-{p}.f32')
                            for p,t,r in [(2,7,'final_prompt'),(3,248046,'terminal')]])
                    (requested/'manifest.json').write_text(json.dumps(diagnostic))
                    stdout.write(''.join(json.dumps(row)+'\n' for row in rows+[dict(stop='eos',position=4)]));stdout.flush()
                    return SimpleNamespace(pid=123456789,poll=lambda:0)
                with ExitStack() as stack:
                    for name,value in [('frozen',([[5,6,7]],{}))]:stack.enter_context(mock.patch.object(P,name,return_value=value))
                    for name,value in [('host_memory',{'effective_available':1<<40}),('model_working_set',{'working_set_bound_bytes':0}),
                        ('verified_devices',[{'total':9<<30}]*2),('vram',{'used':0,'total':9<<30}),('rss',{}),('placements',{'fixture':1})]:
                        stack.enter_context(mock.patch.object(P.S,name,return_value=value))
                    stack.enter_context(mock.patch.object(P.S,'guard_host'))
                    stack.enter_context(mock.patch.object(P.S,'admit_host'))
                    stack.enter_context(mock.patch.object(P.subprocess,'Popen',side_effect=launch))
                    if missing:
                        with self.assertRaisesRegex(ValueError,'projection policy mismatch'):
                            P.run_portable(args,SimpleNamespace(tokens=range(248320)))
                        continue
                    P.run_portable(args,SimpleNamespace(tokens=range(248320)))
                summary=json.loads((args.out/'summary.json').read_text())
                self.assertEqual(summary['status'],'complete')
                self.assertEqual(summary.get('gdn_batch_input_projections',False),batched)
                self.assertEqual(summary.get('gdn_causal_chunk',False),causal)
                self.assertEqual(summary['requests'][0]['chunk_work'],timing)
                self.assertNotIn('timing',summary['requests'][0])
                self.assertEqual(summary['requests'][0]['executed_logits_manifest']['prompt_chunk_size'],2)

    def test_fresh_reference_summary_argv_and_rows(self):
        for mutation in ('none','missing_summary','wrong_summary','bool_summary','argv_size','argv_duplicate','row_size','off'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                tok,suite,prompts,binding,ref,binary,plans,records=ResidentTests.SuiteControls().fixtures(root)
                summary=json.loads((ref/'summary.json').read_text())
                summary.update(head_policy='prompt_head_elision',prompt_chunk_size=4,chunk_policy='layer_major_v1')
                for index,(record,prompt) in enumerate(zip(records,prompts),1):
                    case=ref/f'case-{index:02d}'
                    argv=json.loads((case/'argv.json').read_text())+['--prompt-head-elision','--prompt-chunk-size','4']
                    if index==1 and mutation=='argv_size':argv[-1]='3'
                    if index==1 and mutation=='argv_duplicate':argv+=['--prompt-chunk-size','4']
                    (case/'argv.json').write_text(json.dumps(argv))
                    for i,row in enumerate(record['rows']):
                        row.update(schema=2,head_policy='prompt_head_elision',logits_available=i+1>=len(prompt),
                                   prompt_chunk_size=4,chunk_policy='layer_major_v1')
                        if i+1<len(prompt):row['greedy_next']=None
                    if index==1 and mutation=='row_size':record['rows'][0]['prompt_chunk_size']=3
                    (case/'stdout.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in record['rows']+[record['stop']]))
                if mutation=='missing_summary':del summary['prompt_chunk_size']
                if mutation=='wrong_summary':summary['prompt_chunk_size']=3
                if mutation=='bool_summary':summary['prompt_chunk_size']=True
                (ref/'summary.json').write_text(json.dumps(summary))
                args=(ref,binding,prompts,len(tok.tokens),R.Q.bundle.sha(binary),plans,'5529,5529')
                if mutation=='none':self.assertEqual(R.reference_rows(*args,residency={'--prompt-head-elision':True},prompt_chunk_size=4),records)
                else:
                    with self.assertRaises(ValueError):R.reference_rows(*args,residency={'--prompt-head-elision':True},prompt_chunk_size=0 if mutation=='off' else 4)

    def test_projection_policy_arguments_and_strict_types(self):
        import argparse
        parser=argparse.ArgumentParser();P.residency_arguments(parser)
        self.assertEqual(P.residency_flags(parser.parse_args([])),[])
        args=parser.parse_args(['--prompt-chunk-size','8',P.GDN_PROJECTION_FLAG])
        flags=P.residency_flags(args)
        self.assertEqual(flags.count(P.GDN_PROJECTION_FLAG),1)
        self.assertTrue(P.projection_argv(flags))
        for value in (1,0,'true',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                P.projection_policy(SimpleNamespace(prompt_chunk_size=8,gdn_batch_input_projections=value))
        with self.assertRaises(ValueError):P.residency_flags(parser.parse_args([P.GDN_PROJECTION_FLAG]))
        for argv in ([P.GDN_PROJECTION_FLAG],flags+[P.GDN_PROJECTION_FLAG]):
            with self.assertRaises(ValueError):P.projection_argv(argv)
        for enabled in (False,True):
            P.projection_binding({'gdn_batch_input_projections':enabled},enabled)
            for value in (1,0,'true',None,not enabled):
                with self.subTest(enabled=enabled,value=value),self.assertRaises(ValueError):
                    P.projection_binding({'gdn_batch_input_projections':value},enabled)
        P.projection_binding({},False)
        with self.assertRaises(ValueError):P.projection_binding({},True)

    def test_resident_projection_runtime_receipt_binding(self):
        rows=[dict(position=i,token=t,generated=False,greedy_next=7 if i==2 else None,schema=2,
            head_policy='prompt_head_elision',logits_available=i==2,chunk_policy='layer_major_v1',prompt_chunk_size=2)
            for i,t in enumerate([5,6,7])]
        done=dict(metrics(2,3,3),done=1,prompt_count=3,output_count=0,receipt_schema=2,
            head_policy='prompt_head_elision',heads_skipped_delta=2,heads_executed_delta=1)
        def request(enabled,receipt):
            runner=object.__new__(R.Resident);runner.child=mock.Mock();runner.request_seconds=1
            runner.head_policy='prompt_head_elision';runner.prompt_chunk_size=2;runner.gdn_batch_input_projections=enabled
            runner.row=mock.Mock(side_effect=rows+[dict(stop='max_new',position=3),receipt])
            return runner.request([5,6,7],0,1)
        self.assertEqual(request(False,done)['rows'],rows)
        self.assertEqual(request(True,dict(done,gdn_batch_input_projections=True))['rows'],rows)
        for enabled,value in ((True,'missing'),(True,False),(True,1),(True,'true'),(False,True)):
            bad=dict(done)
            if value!='missing':bad['gdn_batch_input_projections']=value
            with self.subTest(enabled=enabled,value=value),self.assertRaises(ValueError):request(enabled,bad)
        with tempfile.TemporaryDirectory() as directory,mock.patch.object(R.subprocess,'Popen') as child:
            for argv in ([P.GDN_PROJECTION_FLAG],['--prompt-chunk-size','8',P.GDN_PROJECTION_FLAG,P.GDN_PROJECTION_FLAG]):
                with self.assertRaises(ValueError):R.Resident(argv,Path(directory))
            child.assert_not_called()

    def test_projection_fresh_reference_policy_mutations(self):
        for mutation in ('none','summary_missing','summary_false','summary_number','argv_missing','argv_duplicate',
                         'timing_missing','timing_false','timing_number','saved_timing','off'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                tok,suite,prompts,binding,ref,binary,plans,records=ResidentTests.SuiteControls().fixtures(root)
                summary=json.loads((ref/'summary.json').read_text())
                summary.update(head_policy='prompt_head_elision',prompt_chunk_size=8,chunk_policy='layer_major_v1',
                               gdn_batch_input_projections=True)
                for index,(record,prompt) in enumerate(zip(records,prompts),1):
                    case=ref/f'case-{index:02d}'
                    argv=json.loads((case/'argv.json').read_text())+['--prompt-head-elision','--prompt-chunk-size','8',P.GDN_PROJECTION_FLAG]
                    if index==1 and mutation=='argv_missing':argv.remove(P.GDN_PROJECTION_FLAG)
                    if index==1 and mutation=='argv_duplicate':argv.append(P.GDN_PROJECTION_FLAG)
                    (case/'argv.json').write_text(json.dumps(argv))
                    for i,row in enumerate(record['rows']):
                        row.update(schema=2,head_policy='prompt_head_elision',logits_available=i+1>=len(prompt),
                                   prompt_chunk_size=8,chunk_policy='layer_major_v1')
                        if i+1<len(prompt):row['greedy_next']=None
                    (case/'stdout.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in record['rows']+[record['stop']]))
                    timing=dict(metrics(8,len(prompt),len(record['rows'])),gdn_batch_input_projections=True)
                    summary['requests'][index-1]['chunk_work']=copy.deepcopy(timing)
                    if index==1 and mutation=='saved_timing':summary['requests'][0]['chunk_work']['gdn_batch_input_projections']=False
                    if index==1 and mutation=='timing_missing':del timing['gdn_batch_input_projections']
                    if index==1 and mutation=='timing_false':timing['gdn_batch_input_projections']=False
                    if index==1 and mutation=='timing_number':timing['gdn_batch_input_projections']=1
                    (case/'timings.json').write_text(json.dumps(timing))
                if mutation=='summary_missing':del summary['gdn_batch_input_projections']
                if mutation=='summary_false':summary['gdn_batch_input_projections']=False
                if mutation=='summary_number':summary['gdn_batch_input_projections']=1
                (ref/'summary.json').write_text(json.dumps(summary))
                args=(ref,binding,prompts,len(tok.tokens),R.Q.bundle.sha(binary),plans,'5529,5529')
                policy={'--prompt-head-elision':True,P.GDN_PROJECTION_FLAG:mutation!='off'}
                if mutation=='none':self.assertEqual(R.reference_rows(*args,residency=policy,prompt_chunk_size=8),records)
                else:
                    with self.assertRaises(ValueError):R.reference_rows(*args,residency=policy,prompt_chunk_size=8)

    def test_causal_fresh_reference_policy_mutations(self):
        for mutation in ('none','summary_missing','summary_false','summary_number','argv_missing','argv_duplicate',
                         'timing_missing','timing_false','timing_number','saved_timing','off'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                tok,suite,prompts,binding,ref,binary,plans,records=ResidentTests.SuiteControls().fixtures(root)
                summary=json.loads((ref/'summary.json').read_text())
                summary.update(head_policy='prompt_head_elision',prompt_chunk_size=8,chunk_policy='layer_major_v1',
                               gdn_batch_input_projections=True,gdn_causal_chunk=True,gdn_policy='device_production')
                for index,(record,prompt) in enumerate(zip(records,prompts),1):
                    case=ref/f'case-{index:02d}'
                    argv=json.loads((case/'argv.json').read_text())+['--prompt-head-elision','--prompt-chunk-size','8',P.GDN_PROJECTION_FLAG,P.GDN_CAUSAL_FLAG,'--gdn-production']
                    if index==1 and mutation=='argv_missing':argv.remove(P.GDN_CAUSAL_FLAG)
                    if index==1 and mutation=='argv_duplicate':argv.append(P.GDN_CAUSAL_FLAG)
                    (case/'argv.json').write_text(json.dumps(argv))
                    for i,row in enumerate(record['rows']):
                        row.update(schema=2,head_policy='prompt_head_elision',logits_available=i+1>=len(prompt),
                                   prompt_chunk_size=8,chunk_policy='layer_major_v1')
                        if i+1<len(prompt):row['greedy_next']=None
                    (case/'stdout.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in record['rows']+[record['stop']]))
                    timing=dict(metrics(8,len(prompt),len(record['rows'])),gdn_batch_input_projections=True,gdn_causal_chunk=True)
                    summary['requests'][index-1]['chunk_work']=copy.deepcopy(timing)
                    if index==1 and mutation=='saved_timing':summary['requests'][0]['chunk_work']['gdn_causal_chunk']=False
                    if index==1 and mutation=='timing_missing':del timing['gdn_causal_chunk']
                    if index==1 and mutation=='timing_false':timing['gdn_causal_chunk']=False
                    if index==1 and mutation=='timing_number':timing['gdn_causal_chunk']=1
                    (case/'timings.json').write_text(json.dumps(timing))
                if mutation=='summary_missing':del summary['gdn_causal_chunk']
                if mutation=='summary_false':summary['gdn_causal_chunk']=False
                if mutation=='summary_number':summary['gdn_causal_chunk']=1
                (ref/'summary.json').write_text(json.dumps(summary))
                args=(ref,binding,prompts,len(tok.tokens),R.Q.bundle.sha(binary),plans,'5529,5529')
                policy={'--prompt-head-elision':True,P.GDN_PROJECTION_FLAG:True,P.GDN_CAUSAL_FLAG:mutation!='off','--gdn-production':True}
                if mutation=='none':self.assertEqual(R.reference_rows(*args,residency=policy,prompt_chunk_size=8),records)
                else:
                    with self.assertRaises(ValueError):R.reference_rows(*args,residency=policy,prompt_chunk_size=8)

    def test_resident_main_projection_forwarding_and_early_rejection(self):
        import sys
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokens=root/'tokens.json';tokens.write_text('[5,6,7]')
            plans=[root/'p0',root/'p1']
            for plan in plans:plan.write_text('fixture')
            out=root/'out'
            argv=['resident','--binary',sys.executable,'--binary-sha256',R.Q.bundle.sha(sys.executable),
                  '--out',str(out),'--tokens-file',str(tokens),'--max-new','0',
                  '--plan',str(plans[0]),'--plan',str(plans[1]),'--prompt-chunk-size','8',P.GDN_PROJECTION_FLAG,P.GDN_CAUSAL_FLAG,'--gdn-production']
            commands=[]
            def launch(command,path,tick):
                commands.append(command);(path/'stderr.log').write_text('fixture')
                runner=mock.Mock();runner.child.returncode=0;runner.ready.return_value={'ready':True,'schema':1}
                runner.configure_mock(**{'cleanup_guard_error':None})
                del runner.cleanup_guard_error
                def request(tokens,max_new,index):
                    return dict(rows=[{'fixture':True}],stop={'stop':'max_new'},metrics=dict(
                        gdn_batch_input_projections=True,gdn_causal_chunk=True,gpu_entries_per_device_delta=[1,1],cpu_verifications_delta=0,
                        cpu_copied_bytes_delta=0,cpu_digest_bytes_delta=0,source_invalidations_delta=0))
                runner.request.side_effect=request
                return runner
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(sys,'argv',argv))
                stack.enter_context(mock.patch.object(R,'Resident',side_effect=launch))
                for module,name,value in ((R.S,'host_memory',dict(effective_available=1<<40,pswpout=0)),
                    (R.S,'model_working_set',dict(working_set_bound_bytes=0)),(R.S,'verified_devices',[dict(total=8<<30)]*2),
                    (R.S,'sample_vram',[dict(used=0,total=8<<30)]*2),(R.S,'placements',{}),
                    (R.L,'devices',{0:dict(entries=4),1:dict(entries=4)})):
                    stack.enter_context(mock.patch.object(module,name,return_value=value))
                R.main()
            self.assertEqual(commands[0].count(P.GDN_PROJECTION_FLAG),1)
            self.assertTrue(P.projection_argv(commands[0]))
            self.assertTrue(P.causal_argv(commands[0]))
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['status'],'complete');self.assertIs(summary['gdn_batch_input_projections'],True)
            self.assertIs(summary['gdn_causal_chunk'],True)
            self.assertEqual(len(summary['requests']),4);self.assertEqual(summary['excluded_warmups'],1)
            invalid=[x for x in argv if x not in (P.GDN_CAUSAL_FLAG,'--gdn-production')]
            i=invalid.index('--prompt-chunk-size');del invalid[i:i+2]
            with mock.patch.object(sys,'argv',invalid),mock.patch.object(sys,'stderr'),mock.patch.object(R,'Resident') as child:
                with self.assertRaises(SystemExit):R.main()
                child.assert_not_called()


class CausalPolicy(unittest.TestCase):
    def test_policy_argv_and_strict_receipts(self):
        import argparse
        parser=argparse.ArgumentParser();P.residency_arguments(parser)
        valid=['--prompt-chunk-size','8',P.GDN_PROJECTION_FLAG,'--gdn-production',P.GDN_CAUSAL_FLAG]
        flags=P.residency_flags(parser.parse_args(valid))
        self.assertTrue(P.causal_argv(flags))
        for flag in (P.GDN_PROJECTION_FLAG,'--gdn-production'):
            with self.assertRaises(ValueError):P.residency_flags(parser.parse_args([x for x in valid if x!=flag]))
            with self.assertRaises(ValueError):P.causal_argv([x for x in flags if x!=flag])
        with self.assertRaises(ValueError):P.causal_argv(flags+[P.GDN_CAUSAL_FLAG])
        for enabled in (True,False):
            P.causal_binding({'gdn_causal_chunk':enabled},enabled)
            for value in (1,0,'true',None,not enabled):
                with self.assertRaises(ValueError):P.causal_binding({'gdn_causal_chunk':value},enabled)
        with self.assertRaises(ValueError):P.causal_binding({},True)
        P.causal_binding({},False)

    def test_resident_rejects_foreign_causal_policy(self):
        rows=[dict(position=i,token=t,generated=False,greedy_next=7 if i==2 else None,schema=2,
            head_policy='prompt_head_elision',logits_available=i==2,chunk_policy='layer_major_v1',prompt_chunk_size=2)
            for i,t in enumerate([5,6,7])]
        done=dict(metrics(2,3,3),done=1,prompt_count=3,output_count=0,receipt_schema=2,
            head_policy='prompt_head_elision',heads_skipped_delta=2,heads_executed_delta=1,gdn_batch_input_projections=True)
        for enabled,value in ((True,None),(True,False),(True,1),(False,True)):
            runner=object.__new__(R.Resident);runner.child=mock.Mock();runner.request_seconds=1
            runner.head_policy='prompt_head_elision';runner.prompt_chunk_size=2;runner.gdn_batch_input_projections=True
            runner.gdn_causal_chunk=enabled
            receipt=dict(done,**({} if value is None else {'gdn_causal_chunk':value}))
            runner.row=mock.Mock(side_effect=rows+[dict(stop='max_new',position=3),receipt])
            with self.assertRaisesRegex(ValueError,'causal chunk policy mismatch'):runner.request([5,6,7],0,1)
