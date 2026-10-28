"""Frozen v2 inventory and scoring policy tests. No model or generated code is executed."""
import array
import collections
import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import quality_v2 as V
from test_quality import ByteTokenizer


class V2Tests(unittest.TestCase):
    def test_inventory_independent_and_auditable(self):
        cases = V.inventory()
        self.assertEqual(len(cases), 100)
        self.assertEqual(collections.Counter(c['category'] for c in cases),
                         dict(arithmetic=17, extraction=17, structured=17, instruction=17, reasoning=16, code=16))
        self.assertFalse({c['prompt'] for c in cases} & {c[1] for c in V.Q.CASES})
        for case in cases:
            answer = json.dumps(case['expected']) if case['semantic_rule'] == 'json' else case['expected']
            with self.subTest(case=case['id']):
                self.assertTrue(all(V.assess(case, answer).values()))
                self.assertFalse(V.assess(case, answer + ' explanation')['semantic_correct'])
                self.assertLessEqual(len(ByteTokenizer().encode(answer)), 32)
                self.assertTrue(case['rationale'])

    def test_semantics_and_formatting_are_separate_and_narrow(self):
        cases = {c['id']: c for c in V.inventory()}
        self.assertEqual(V.assess(cases['reasoning-01'], ' YES<|im_end|>\n'),
                         dict(semantic_correct=True, format_compliant=False))
        self.assertEqual(V.assess(cases['arithmetic-01'], '999'),
                         dict(semantic_correct=False, format_compliant=True))
        self.assertEqual(V.assess(cases['arithmetic-01'], '+0124.0'),
                         dict(semantic_correct=True, format_compliant=False))
        self.assertEqual(V.assess(cases['arithmetic-09'], '14/16'),
                         dict(semantic_correct=True, format_compliant=False))
        self.assertTrue(V.assess(cases['arithmetic-10'], '3.250')['semantic_correct'])
        self.assertFalse(V.assess(cases['arithmetic-10'], '3.251')['semantic_correct'])
        for answer in ('silent', 'SILENT.', '```SILENT```'):
            self.assertFalse(V.assess(cases['instruction-01'], answer)['semantic_correct'])
        self.assertFalse(V.assess(cases['instruction-08'], 'left right')['semantic_correct'])
        self.assertTrue(V.assess(cases['instruction-08'], ' \nleft\nright\n')['semantic_correct'])
        for answer in ('{"legs":4,"animal":"fox","legs":4}', '{"legs":4.0,"animal":"fox"}',
                       '{"legs":true,"animal":"fox"}', '{"animal":"fox","legs":4,"extra":0}'):
            self.assertFalse(V.assess(cases['structured-01'], answer)['semantic_correct'])
        for answer in ('NaN', '1e999', '{"x":1,"x":2}', '```{} ```'):
            self.assertFalse(V.assess(cases['structured-08'], answer)['format_compliant'])

    @staticmethod
    def row(name, a, b, entries=1, category='reasoning', af=True, bf=True):
        return dict(id=name, category=category, vulkan_entries=entries,
                    baseline=dict(semantic_correct=a, format_compliant=af),
                    candidate=dict(semantic_correct=b, format_compliant=bf))

    def test_paired_policy_does_not_require_perfect_baseline_or_offset_losses(self):
        rows = [self.row('shared', False, False), self.row('gain', False, True), self.row('same', True, True)]
        result = V.summarize(rows)
        self.assertEqual(result['result'], 'screen-no-observed-regression')
        self.assertEqual(result['offload_evidence']['semantic']['shared_failures'], 1)
        rows.append(self.row('loss', True, False))
        result = V.summarize(rows)
        self.assertEqual(result['result'], 'screen-semantic-regression')
        self.assertEqual(result['offload_evidence']['semantic']['case_ids']['regressions'], ['loss'])
        self.assertEqual(result['offload_evidence']['semantic']['improvements'], 1)
        self.assertEqual(result['categories']['reasoning']['exercised']['semantic']['total'], 4)

    def test_unexercised_pairs_excluded_and_format_regressions_visible(self):
        rows = [self.row('active', True, True), self.row('inactive-loss', True, False, entries=0)]
        result = V.summarize(rows)
        self.assertEqual(result['result'], 'screen-coverage-limited')
        self.assertEqual(result['offload_evidence']['semantic']['regressions'], 0)
        self.assertEqual(result['all_cases']['semantic']['regressions'], 1)
        self.assertEqual(result['nonparticipating_case_ids'], ['inactive-loss'])
        self.assertEqual(V.summarize([rows[1]])['result'], 'screen-unexercised')
        self.assertEqual(V.summarize([self.row('format-loss', True, True, af=True, bf=False)])['result'],
                         'screen-format-regression')

    def test_freeze_exclusive_plan_and_source_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); plan = root / 'plan'; plan.write_bytes(b'synthetic plan')
            V.prepare(ByteTokenizer(), root / 'frozen', plan)
            frozen = json.loads((root / 'frozen' / 'quality_manifest').read_text())
            self.assertEqual(frozen['version'], 2)
            self.assertEqual(frozen['plan_sha256'], V.Q.bundle.sha(plan))
            self.assertEqual(frozen['sources'], V.sources())
            self.assertEqual(len((root / 'frozen' / 'requests').read_text().splitlines()), 100)
            with self.assertRaises(FileExistsError): V.prepare(ByteTokenizer(), root / 'frozen', plan)

    def test_v2_preflight_is_explicit_bounded_and_legacy_limit_unchanged(self):
        import test_harness
        fixture = test_harness.HarnessTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root = fixture.root; plan = root / 'plan'; plan.write_bytes(b'synthetic plan')
        V.prepare(ByteTokenizer(), root / 'v2', plan)
        cfg = fixture.cfg
        cfg['assets'].update(requests=str(root / 'v2' / 'requests'), quality_manifest=str(root / 'v2' / 'quality_manifest'))
        cfg.update(evaluation_mode='quality', quality_suite='v2-100', capture_first_logits=True,
                   attach_root=str(root), attached_worker_idle=True)
        cfg['argv'] += ['--greedy', '--max-context', '256']
        cfg['arms'][0].update(restart=False, requests=list(range(100)))
        def validate():
            with patch.object(V.Q.bundle.shutil, 'which', return_value='/synthetic/nvidia-smi'):
                return V.Q.bundle.validate(cfg, pathlib.Path(V.Q.bundle.__file__).parent)
        self.assertEqual(len(validate()), 100)
        cfg['argv'][-1] = '255'
        with self.assertRaisesRegex(ValueError, 'context at least 256'): validate()
        cfg['argv'][-1] = '256'
        cfg['evaluation_mode'] = 'exact'
        with self.assertRaisesRegex(ValueError, 'explicit quality mode'): validate()
        cfg.pop('quality_suite'); cfg.pop('evaluation_mode'); cfg['arms'][0]['requests'] = []
        with self.assertRaisesRegex(ValueError, '1..64'): validate()
        cfg.update(evaluation_mode='quality', quality_suite='v2-100')
        cfg['arms'][0]['requests'] = list(range(100))
        path = root / 'v2' / 'requests'
        path.write_text(path.read_text().replace('GEN 32 ', 'GEN 33 ', 1))
        frozen_path = root / 'v2' / 'quality_manifest'
        frozen = json.loads(frozen_path.read_text()); frozen['requests_sha256'] = V.Q.bundle.sha(path)
        frozen_path.write_text(json.dumps(frozen))
        with self.assertRaisesRegex(ValueError, 'GEN 32'): validate()

    def test_full_private_report_and_frozen_source_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); plan = root / 'plan'; plan.write_bytes(b'synthetic plan')
            tok = ByteTokenizer(); frozen_root = root / 'frozen'; V.prepare(tok, frozen_root, plan)
            configs = []; roots = []
            for side in ('cpu', 'vk'):
                run = root / side; run.mkdir(); roots.append(run)
                for helper in V.Q.bundle.HELPERS: (run / helper).write_text('synthetic helper')
                assets = {name: str(frozen_root / name) for name in ('requests', 'quality_manifest')}
                for key in ('engine', 'worker', 'manifest'):
                    path = root / key
                    if not path.exists(): path.write_text(key)
                    assets[key] = str(path)
                argv = [assets['engine'], '--greedy']
                if side == 'vk': argv += ['--remote-gpu-worker', 'vulkan@127.0.0.1:1234', str(plan)]
                arm = dict(name='warm', requests=list(range(100)), workers=1, loader_threads=1,
                           caller_mask=[0], pool_masks=[[0]], engine_main_mask=[0])
                configs.append(dict(evaluation_mode='quality', quality_suite='v2-100', assets=assets, argv=argv,
                                    worker_argv=[assets['worker']], env={}, worker_env={}, model_paths=[],
                                    worker_bytes=1, request_seconds=1, arms=[arm]))
                rows = []
                for i, case in enumerate(V.inventory()):
                    text = json.dumps(case['expected']) if case['semantic_rule'] == 'json' else case['expected']
                    if i == 0: text = '0'  # same baseline imperfection, not an automatic inconclusive verdict
                    ids = tok.encode(text)
                    rows.append(dict(index=i+1, tokens=ids, output_count=len(ids), finish='stop', prompt_s=1, decode_s=1))
                    (run / f'warm-first-logits.{i}').write_bytes(array.array('f', [1 if side == 'cpu' else 2]*256).tobytes())
                (run / 'summary.json').write_text(json.dumps({'arms': [{'arm':'warm','requests':rows}]}))
                (run / 'warm').mkdir()
                reports = [f'worker[0] Vulkan/native-Q8-host-boundaries approximate READY transport=mTLS '
                           f'model=1 layout=2:3 placement=4:5 session=6 entries={i} failures=0' for i in range(101)]
                (run / 'warm' / 'stderr.log').write_text('\n'.join(reports))
            with patch.object(V.Q.bundle, 'verify', side_effect=configs):
                self.assertEqual(V.score(tok, *roots, root / 'report'), 0)
            report = json.loads((root / 'report').read_text())
            self.assertEqual(report['offload_evidence']['semantic']['baseline_correct'], 99)
            self.assertEqual(report['offload_evidence']['semantic']['shared_failures'], 1)
            self.assertEqual(len(report['cases']), 100)
            self.assertFalse(report['cases'][0]['fixed_numeric_diagnostic']['numeric_pass'])
            plan.write_bytes(b'changed plan')
            with patch.object(V.Q.bundle, 'verify', side_effect=configs):
                with self.assertRaisesRegex(ValueError, 'frozen v2'): V.score(tok, *roots, root / 'bad-report')
            plan.write_bytes(b'synthetic plan')
            frozen_path = frozen_root / 'quality_manifest'
            frozen = json.loads(frozen_path.read_text()); frozen['sources']['quality_v2.py'] = 'altered'
            frozen_path.write_text(json.dumps(frozen))
            with patch.object(V.Q.bundle, 'verify', side_effect=configs):
                with self.assertRaisesRegex(ValueError, 'frozen v2'): V.score(tok, *roots, root / 'bad-source')


if __name__ == '__main__': unittest.main()
