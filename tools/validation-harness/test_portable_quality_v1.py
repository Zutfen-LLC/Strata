"""Host-only controls; no model, network, GPU or child process launch."""
import portable_model_smoke_v1 as B

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import portable_quality_v1 as P


class Tokenizer:
    tokens = [chr(i) for i in range(256)]
    def encode(self, text, parse_special=False): return list(text.encode())
    def decode(self, ids, errors='strict'): return bytes(ids).decode(errors=errors)


class QualityAdapter(unittest.TestCase):
    def test_frozen_unchanged_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'suite'
            P.Q.prepare(Tokenizer(), root)
            prompts, binding = P.frozen(Tokenizer(), root)
            self.assertEqual(len(prompts), 12)
            self.assertEqual(binding['scorer_sha256'], P.Q.bundle.sha(P.Q.__file__))
            p = root/'requests'; p.write_text(p.read_text().replace('GEN 32', 'GEN 31', 1))
            with self.assertRaises(ValueError): P.frozen(Tokenizer(), root)

    def test_greedy_receipts_and_prompt_boundary(self):
        rows = [dict(position=0,token=1,generated=False,greedy_next=2),
                dict(position=1,token=2,generated=True,greedy_next=3),
                dict(position=2,token=248046,generated=True,greedy_next=0)]
        rows[1]['greedy_next'] = 248046
        stop = dict(stop='eos',position=3)
        self.assertEqual(P.completion(rows+[stop],[1],248320),([2,248046],'eos'))
        for field, value in [('position',3),('token',3),('generated',False)]:
            bad=copy.deepcopy(rows);bad[1][field]=value
            with self.assertRaises(ValueError): P.completion(bad+[stop],[1],248320)
        with self.assertRaises(ValueError): P.completion(rows+[dict(stop='context',position=3)],[1],248320)

    def test_head_elision_receipts_are_explicit_and_policy_bound(self):
        rows = [dict(position=0,token=1,generated=False,greedy_next=None),
                dict(position=1,token=3,generated=False,greedy_next=248046),
                dict(position=2,token=248046,generated=True,greedy_next=0)]
        for i,row in enumerate(rows):
            row.update(schema=2,head_policy='prompt_head_elision',logits_available=i!=0)
        records=rows+[dict(stop='eos',position=3)]
        self.assertEqual(P.completion(records,[1,3],248320,'prompt_head_elision'),([248046],'eos'))
        with self.assertRaises(ValueError): P.completion(records,[1,3],248320)
        for pos,key,value in ((0,'greedy_next',0),(0,'logits_available',True),(1,'greedy_next',None),
                              (1,'logits_available',False),(0,'schema',3),(0,'head_policy','foreign'),
                              (0,'generated',True),(0,'position',False),(0,'schema',True)):
            bad=copy.deepcopy(records);bad[pos][key]=value
            with self.assertRaises(ValueError): P.completion(bad,[1,3],248320,'prompt_head_elision')
        bad=copy.deepcopy(records);del bad[0]['schema']
        with self.assertRaises(ValueError): P.completion(bad,[1,3],248320,'prompt_head_elision')
        with self.assertRaises(ValueError): P.completion(records,[1,3],248320,'foreign')
        # The last prompt and terminal generated head must still be available.
        one=[dict(position=0,token=3,generated=False,greedy_next=248046,schema=2,
                  head_policy='prompt_head_elision',logits_available=True),dict(rows[2],position=1),
             dict(stop='eos',position=2)]
        self.assertEqual(P.completion(one,[3],248320,'prompt_head_elision'),([248046],'eos'))

    def test_native_owned_lifecycle_and_failure_cleanup(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        for failure in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                child=SimpleNamespace(pid=54321,poll=lambda:None)
                calls=[]
                def engine(name,cfg,argv,line,measured,peers,dump):
                    calls.append((name,line,measured,peers,dump))
                    if name=='load':return dict(error=None,engine_rc=0)
                    if failure:return dict(error='request failed',engine_rc=1)
                    return dict(error=None,engine_rc=0,tokens=[2],done=dict(finish='stop',output_count=1))
                driver=SimpleNamespace(
                    prepare_argv=lambda n,l:('/manifest',['LOAD'],['LOCAL'],['unused']),
                    set_single_value=Mock(),require_once=Mock(),port_open=Mock(side_effect=[False,True,False]),
                    subprocess=SimpleNamespace(Popen=Mock(return_value=child)),
                    stop_process_group=Mock(return_value=0),engine_stage=engine,
                    stage_ok=lambda r,measured:r.get('error') is None and r.get('engine_rc')==0,
                    WORKER_START_SECONDS=1)
                ndc=dict(assets=dict(worker='/worker'),worker_argv=['@worker','--port','17431','--residency-forever'])
                summary={'requests':[]}
                with patch.object(P.Q.compare_logits,'read',return_value=[0.]*248320):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'native case1 failed'):
                            P.owned_native(driver,ndc,{},[[1],[1]],summary,root)
                    else:P.owned_native(driver,ndc,{},[[1],[1]],summary,root)
                driver.subprocess.Popen.assert_called_once()
                driver.stop_process_group.assert_called_once_with(child,'owned CPU worker')
                self.assertEqual(calls[0][0],'load')
                self.assertEqual(len(calls),2 if failure else 3)
                if not failure:self.assertEqual(len(summary['requests']),2)
                self.assertEqual(summary['cpu_worker_rc'],0)

    def test_native_occupied_endpoint_never_launches_or_stops(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            driver=SimpleNamespace(prepare_argv=lambda n,l:('/manifest',[],[],[]),
                set_single_value=Mock(),require_once=Mock(),port_open=lambda p:True,
                subprocess=SimpleNamespace(Popen=Mock()),stop_process_group=Mock())
            ndc=dict(assets=dict(worker='/worker'),worker_argv=['@worker'])
            with self.assertRaisesRegex(ValueError,'already occupied'):
                P.owned_native(driver,ndc,{},[[1]],{'requests':[]},Path(d))
            driver.subprocess.Popen.assert_not_called()
            driver.stop_process_group.assert_not_called()

    def test_guard_failure_reaps_only_owned_child_and_keeps_failure(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); suite=root/'suite'; P.Q.prepare(Tokenizer(),suite)
            binary=root/'binary';binary.write_bytes(b'not an executable: mock launch only')
            args=SimpleNamespace(suite=suite,binary=binary,binary_sha256=P.Q.bundle.sha(binary),out=root/'run',
                                 devices='0,1',pci=','.join(P.S.PCI),split=32,device_cap_mib=7680,
                                 dense_resident_weights=True,gdn_resident_state=True,gdn_production=True,qsa_reuse_attention=True,qsa_production=True,prompt_head_elision=True)
            child=SimpleNamespace(pid=123456789,poll=lambda:None,wait=lambda timeout:0)
            with patch.object(P.S,'host_memory',return_value={'effective_available':1<<40}), \
                 patch.object(P.S,'model_working_set',return_value={'working_set_bound_bytes':0}), \
                 patch.object(P.S,'verified_devices',return_value=[dict(pci=p,free=9<<30,total=9<<30) for p in B.PCI]), \
                 patch.object(P.S,'vram',side_effect=lambda pci:dict(pci=pci,used=0,total=9<<30)), \
                 patch.object(P.S,'rss',return_value={}), \
                 patch.object(P.S,'guard_host',side_effect=ValueError('pressure')), \
                 patch.object(P.subprocess,'Popen',return_value=child) as launch, \
                 patch.object(P.os,'killpg') as kill:
                with self.assertRaisesRegex(ValueError,'pressure'): P.run_portable(args,Tokenizer())
                launch.assert_called_once()
                argv=launch.call_args.args[0]
                self.assertEqual(argv.count('--dense-resident-weights'),1)
                self.assertEqual(argv.count('--prompt-head-elision'),1)
                self.assertEqual(argv.count('--gdn-resident-state'),1)
                self.assertEqual(argv.count('--gdn-production'),1)
                self.assertEqual(argv.count('--qsa-reuse-attention'),1)
                self.assertEqual(argv.count('--qsa-production'),1)
                self.assertNotIn('--gdn-resident-weights',argv)
                kill.assert_called_once_with(child.pid,P.signal.SIGTERM)
            receipt=json.loads((args.out/'summary.json').read_text())
            self.assertEqual(receipt['status'],'STOP')
            self.assertIs(receipt['qsa_reuse_attention'],True)
            self.assertEqual(receipt['qsa_policy'],'device_production')
            self.assertEqual(receipt['requests'],[])
            self.assertEqual(receipt['error'],'pressure')

    def test_fresh_cli_exposes_shared_device_options(self):
        import sys
        for options,devices,split,cap in (([], '0,1',32,7680),
            (['--devices','3,2','--pci','0000:21:00.0,0000:41:00.0',
              '--split','24','--device-cap-mib','4096'], '3,2',24,4096)):
            argv=['portable_quality_v1.py','--gguf','metadata','portable',
                  '--binary','binary','--binary-sha256','digest','--suite','suite','--out','out']+options
            with patch.object(sys,'argv',argv), patch.object(P.Q,'tokenizer'), \
                 patch.object(P,'run_portable') as run:
                self.assertEqual(P.main(),0)
                args=run.call_args.args[0]
                self.assertEqual((args.devices,args.split,args.device_cap_mib),(devices,split,cap))
                if options:self.assertEqual(args.pci,'0000:21:00.0,0000:41:00.0')

    def test_selected_devices_bind_fresh_launch_observation_and_placement(self):
        import argparse
        from types import SimpleNamespace
        for benchmark in (False, True):
            for custom in (False, True):
                with self.subTest(benchmark=benchmark, custom=custom), tempfile.TemporaryDirectory() as d:
                    root = Path(d)
                    binary = root/'binary'; binary.write_bytes(b'mocked child')
                    parser = argparse.ArgumentParser()
                    P.S.device_arguments(parser)
                    options = (['--devices','3,2','--pci','0000:21:00.0,0000:41:00.0',
                                '--split','24','--device-cap-mib','8192,4096'] if custom else [])
                    args = parser.parse_args(options)
                    args.__dict__.update(suite=root/'suite', binary=binary,
                        binary_sha256=P.Q.bundle.sha(binary), out=root/'out', benchmark=benchmark)
                    selected, physical = P.S.device_selection(parser, args)
                    mapping = [dict(index=i,pci=p,free=9<<30,total=9<<30) for i,p in zip(selected,physical)]
                    child = SimpleNamespace(pid=123456789,poll=lambda:0)
                    def launch(argv, stdout, stderr, **kwargs):
                        stdout.write('{}\n'); stdout.flush()
                        if benchmark:
                            Path(argv[argv.index('--timings')+1]).write_text(json.dumps(dict(
                                setup_ms=1,prompt_ms=2,decode_ms=3,prompt_count=2,output_count=1,
                                selection_prompt_ms=0,selection_decode_ms=0,terminal_feed_ms=0,
                                selection_prompt_steps=1,selection_decode_steps=1)))
                        return child
                    with patch.object(P,'frozen',return_value=([[1,2]],{})), \
                         patch.object(P,'completion',return_value=([248046],'eos')), \
                         patch.object(P.S,'host_memory',return_value={'effective_available':1<<40}), \
                         patch.object(P.S,'model_working_set',return_value={'working_set_bound_bytes':0}), \
                         patch.object(P.S,'verified_devices',return_value=mapping) as verify, \
                         patch.object(P.S,'vram',side_effect=lambda pci:dict(pci=pci,used=0,total=9<<30)) as observe, \
                         patch.object(P.S,'rss',return_value={}), patch.object(P.S,'guard_host'), \
                         patch.object(P.S,'placements',return_value={}) as placement, \
                         patch.object(P.subprocess,'Popen',side_effect=launch) as popen:
                        P.run_portable(args, SimpleNamespace(tokens=range(248320)))
                    verify.assert_called_once_with(binary,selected,physical)
                    self.assertEqual({call.args[0] for call in observe.call_args_list},set(physical))
                    self.assertEqual(len(observe.call_args_list),4*len(popen.call_args_list))
                    for call in popen.call_args_list:
                        argv = call.args[0]
                        for flag,value in (('--vulkan',args.devices),('--split',str(args.split)),
                                           ('--device-cap-mib',P.S.cap_argv(args))):
                            self.assertEqual(argv[argv.index(flag)+1],value)
                    for call in placement.call_args_list:
                        self.assertEqual(call.args[1:],(selected,args.split))
                    summary=json.loads((args.out/'summary.json').read_text())
                    self.assertEqual(summary['selected_devices'],mapping)
                    self.assertEqual(summary['device_policy']['pci'],physical)
                    self.assertEqual(summary['status'],'complete')

    def test_selected_identity_failure_stops_before_model_launch(self):
        import argparse
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); binary=root/'binary'; binary.write_bytes(b'mocked child')
            parser=argparse.ArgumentParser(); P.S.device_arguments(parser)
            args=parser.parse_args([])
            args.__dict__.update(suite=root/'suite',binary=binary,
                binary_sha256=P.Q.bundle.sha(binary),out=root/'out')
            with patch.object(P,'frozen',return_value=([[1]],{})), \
                 patch.object(P.S,'host_memory',return_value={'effective_available':1<<40}), \
                 patch.object(P.S,'model_working_set',return_value={'working_set_bound_bytes':0}), \
                 patch.object(P.S,'verified_devices',side_effect=ValueError('PCI mismatch')), \
                 patch.object(P.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'PCI mismatch'):
                    P.run_portable(args,SimpleNamespace(tokens=[]))
                launch.assert_not_called()
            self.assertEqual(json.loads((args.out/'summary.json').read_text())['error'],'PCI mismatch')

    def test_score_outcomes_preserve_gold_and_failed_numeric_status(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);suite=root/'suite';P.Q.prepare(Tokenizer(),suite)
            _,binding=P.frozen(Tokenizer(),suite)
            native=[]
            for i,case in enumerate(P.Q.CASES,1):
                answer=case[3] if isinstance(case[3],str) else json.dumps(case[3],separators=(',',':'))
                native.append(dict(index=i,tokens=list(answer.encode()),finish='stop'))
            run=root/'portable';run.mkdir()
            for label,baseline_bad,candidate_bad,expected in [
                ('pass',False,False,'screen-pass'),('regress',False,True,'screen-regression'),
                ('baseline',True,False,'inconclusive-baseline')]:
                a=copy.deepcopy(native);b=copy.deepcopy(native)
                if baseline_bad:a[0]['tokens']=list(b'wrong')
                if candidate_bad:b[0]['tokens']=list(b'wrong')
                (run/'summary.json').write_text(json.dumps(dict(status='complete',suite=binding,requests=b)))
                args=SimpleNamespace(suite=suite,portable_root=run,native_root=root/'native',native_sha256='sha',out=root/label)
                with patch.object(P,'native_rows',return_value=a):code=P.score(Tokenizer(),args)
                report=json.loads(args.out.read_text())
                self.assertEqual(report['result'],expected)
                self.assertEqual(code,int(label!='pass'))
                self.assertIn('FAILED',report['numerical_parity'])

    def test_residency_flags_default_off_and_independently_forwarded(self):
        from types import SimpleNamespace
        self.assertEqual(P.residency_flags(SimpleNamespace()),[])
        for flag in P.RESIDENCY_FLAGS:
            args=SimpleNamespace(**{flag[2:].replace('-','_'):True})
            self.assertEqual(P.residency_flags(args),[flag])

    def test_strict_scorer_reused(self):
        self.assertTrue(P.Q.correct('391<|im_end|>','exact','391'))
        self.assertFalse(P.Q.correct('391.','exact','391'))
        self.assertFalse(P.Q.correct('{"count":3,"count":3}','json',{'count':3}))



class BenchmarkControls(unittest.TestCase):
    def test_only_first_frozen_case_repeated_and_bound(self):
        prompts, binding = P.benchmark_prompts([[1, 2], [3]], {'sha': 'frozen'}, True)
        self.assertEqual(prompts, [[1, 2]] * 4)
        self.assertEqual(binding['excluded_warmup'], 1)
        self.assertEqual(binding['measured_trials'], 3)
        self.assertEqual(binding['sha'], 'frozen')

    def test_single_case_has_one_measured_request_and_no_warmup(self):
        prompts,binding=P.benchmark_prompts([[1,2],[3]],{'sha':'frozen'},True,single=True)
        self.assertEqual(prompts,[[1,2]])
        self.assertEqual(binding['excluded_warmup'],0)
        self.assertEqual(binding['measured_trials'],1)
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(out=Path(d),benchmark=False,single_case=True)
            result=dict(status='complete',requests=[dict(index=1)])
            P.benchmark_receipt(args,result)
            saved=json.loads((Path(d)/'benchmark.json').read_text())
            self.assertFalse(saved['requests'][0]['excluded_warmup'])
            self.assertIn('no excluded warmup',saved['cache_policy'])

    def test_reject_nonfinite_and_mismatched_counts(self):
        good = dict(setup_ms=1., prompt_ms=2., decode_ms=3., prompt_count=2, output_count=1)
        self.assertEqual(P.timing_fields(good, [1, 2], [3]), good)
        for key, value in [('prompt_ms', float('nan')), ('decode_ms', -1),
                           ('setup_ms', float('inf')), ('output_count', 2), ('prompt_count', 3)]:
            bad = dict(good); bad[key] = value
            with self.assertRaises(ValueError): P.timing_fields(bad, [1, 2], [3])

    def test_warmup_explicit_and_failure_retained(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(out=Path(d), benchmark=True)
            result = dict(status='STOP', error='pressure', requests=[dict(index=i) for i in range(1, 4)])
            P.benchmark_receipt(args, result)
            saved = json.loads((Path(d)/'benchmark.json').read_text())
            self.assertEqual([r['excluded_warmup'] for r in saved['requests']], [True, False, False])
            self.assertEqual(saved['error'], 'pressure')
            self.assertIn('FAILED', saved['numerical_parity'])

    def test_four_owned_native_trials_keep_done_timings_without_logit_dump(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); calls=[]
            def stage(name,cfg,argv,line,measured,peers,dump):
                calls.append((name,line,dump))
                if name=='load': return dict(error=None,engine_rc=0,setup_s=1.)
                return dict(error=None,engine_rc=0,setup_s=2.,tokens=[2],
                    done=dict(raw='DONE 1 1 3.0 4.0 stop',finish='stop',output_count=1,prompt_ms=3.,decode_ms=4.))
            child=SimpleNamespace(pid=54321,poll=lambda:None)
            driver=SimpleNamespace(prepare_argv=lambda n,l:('/manifest',[],[],[]),
                set_single_value=Mock(),require_once=Mock(),port_open=Mock(side_effect=[False,True,False]),
                subprocess=SimpleNamespace(Popen=Mock(return_value=child)),
                stop_process_group=Mock(return_value=0),engine_stage=stage,
                stage_ok=lambda r,measured:r.get('error') is None,WORKER_START_SECONDS=1)
            ndc=dict(assets=dict(worker='/worker'),worker_argv=['@worker'])
            summary={'requests':[]}
            with patch.object(P.Q.compare_logits,'read') as read:
                P.owned_native(driver,ndc,{},[[1]]*4,summary,root,benchmark=True)
                read.assert_not_called()
            self.assertEqual(len(calls),5)
            self.assertTrue(all(not c[2] for c in calls))
            self.assertEqual([r['timing']['decode_ms'] for r in summary['requests']],[4.]*4)
            driver.stop_process_group.assert_called_once_with(child,'owned CPU worker')

if __name__=='__main__': unittest.main()
