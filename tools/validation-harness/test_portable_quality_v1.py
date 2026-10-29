"""Host-only controls; no model, network, GPU or child process launch."""
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
                                 dense_resident_weights=True,gdn_resident_state=True,gdn_production=True)
            child=SimpleNamespace(pid=123456789,poll=lambda:None,wait=lambda timeout:0)
            with patch.object(P.S,'host_memory',return_value={'effective_available':1<<40}), \
                 patch.object(P.S,'model_working_set',return_value={'working_set_bound_bytes':0}), \
                 patch.object(P.S,'vram',return_value={'used':0,'total':9<<30}), \
                 patch.object(P.S,'rss',return_value={}), \
                 patch.object(P.S,'guard_host',side_effect=ValueError('pressure')), \
                 patch.object(P.subprocess,'Popen',return_value=child) as launch, \
                 patch.object(P.os,'killpg') as kill:
                with self.assertRaisesRegex(ValueError,'pressure'): P.run_portable(args,Tokenizer())
                launch.assert_called_once()
                argv=launch.call_args.args[0]
                self.assertEqual(argv.count('--dense-resident-weights'),1)
                self.assertEqual(argv.count('--gdn-resident-state'),1)
                self.assertEqual(argv.count('--gdn-production'),1)
                self.assertNotIn('--gdn-resident-weights',argv)
                kill.assert_called_once_with(child.pid,P.signal.SIGTERM)
            receipt=json.loads((args.out/'summary.json').read_text())
            self.assertEqual(receipt['status'],'STOP')
            self.assertEqual(receipt['requests'],[])
            self.assertEqual(receipt['error'],'pressure')

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
