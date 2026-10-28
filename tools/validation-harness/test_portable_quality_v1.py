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

    def test_native_clone_preserves_runtime_and_resident(self):
        cfg=dict(attach_root='/resident',attached_worker_idle=True,final_worker='retain',failure_worker='retain',
                 arms=[dict(name='old',restart=False,requests=[0],reference='old')],
                 assets=dict(engine='/engine',worker='/worker',manifest='/manifest'),
                 argv=['/engine','--native','/model','--prefill','64'],worker_argv=['/worker','--port','1'],env={'X':'Y'})
        original=copy.deepcopy(cfg)
        with patch.object(P.Q.bundle,'sha',return_value='sha'):
            result=P.native_configuration(cfg,'/suite','/output','sha')
            self.assertEqual(result['argv'][1:],cfg['argv'][1:])
            self.assertEqual(result['env'],cfg['env'])
            self.assertEqual(result['attach_root'],'/resident')
            self.assertFalse(result['arms'][0]['restart'])
            self.assertEqual(result['arms'][0]['requests'],list(range(12)))
            self.assertEqual(cfg,original)
            for key,value in [('attach_root',None),('failure_worker','stop')]:
                bad=copy.deepcopy(cfg);bad[key]=value
                with self.assertRaises(ValueError): P.native_configuration(bad,'/suite','/out','sha')

    def test_guard_failure_reaps_only_owned_child_and_keeps_failure(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); suite=root/'suite'; P.Q.prepare(Tokenizer(),suite)
            binary=root/'binary';binary.write_bytes(b'not an executable: mock launch only')
            args=SimpleNamespace(suite=suite,binary=binary,binary_sha256=P.Q.bundle.sha(binary),out=root/'run')
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

    def test_strict_scorer_reused(self):
        self.assertTrue(P.Q.correct('391<|im_end|>','exact','391'))
        self.assertFalse(P.Q.correct('391.','exact','391'))
        self.assertFalse(P.Q.correct('{"count":3,"count":3}','json',{'count':3}))


if __name__=='__main__': unittest.main()
