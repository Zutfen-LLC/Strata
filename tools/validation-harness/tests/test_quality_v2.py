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
            self.assertEqual(report['placement_plan_sha256'], V.Q.bundle.sha(plan))
            self.assertIn('one Vulkan endpoint', report['scope'])
            self.assertNotIn('expert302', report['scope'])
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


class MultiEndpointTests(unittest.TestCase):
    """Synthetic complete retained arms; no hardware or model execution."""
    @staticmethod
    def plan(path, expert, model=1):
        import struct
        data = struct.pack('<8sQ6HQQI', b'SGPUPLN1', model, 48, 512, 2560, 640, 8, 8, 2, 3, 1)
        data += struct.pack('<HHQQQ', 0, expert, 5222400, 7, 8)
        identities = []
        for seed, values in ((0xcbf29ce484222325, data), (0x9e3779b97f4a7c15, data[::-1])):
            for byte in values: seed = ((seed ^ byte) * 1099511628211) % 2**64
            identities.append(seed)
        path.write_bytes(data + struct.pack('<QQ', *identities))
        return path

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name); self.tok = ByteTokenizer()
        self.plans = [self.plan(self.root / f'plan{i}', i) for i in range(2)]
        self.binding = V.plan_binding(self.plans)
        frozen = self.root / 'frozen'; V.prepare(self.tok, frozen, self.plans)
        self.configs = []; self.roots = []
        for side in ('cpu', 'vk'):
            run = self.root / side; run.mkdir(); self.roots.append(run)
            for helper in V.Q.bundle.HELPERS: (run / helper).write_text('synthetic helper')
            assets = {name: str(frozen / name) for name in ('requests', 'quality_manifest')}
            for key in ('engine', 'worker', 'manifest'):
                path = self.root / key; path.write_text(key); assets[key] = str(path)
            argv = [assets['engine'], '--greedy']
            if side == 'vk':
                for i, plan in enumerate(self.plans):
                    argv += ['--remote-gpu-worker', f'vulkan@127.0.0.1:{1234+i}', str(plan)]
            arm = dict(name='warm', requests=list(range(100)), workers=1, loader_threads=1,
                       caller_mask=[0], pool_masks=[[0]], engine_main_mask=[0])
            self.configs.append(dict(evaluation_mode='quality', quality_suite='v2-100', assets=assets,
                argv=argv, worker_argv=[assets['worker']], env={}, worker_env={}, model_paths=[],
                worker_bytes=1, request_seconds=1, arms=[arm]))
            rows = []
            for i, case in enumerate(V.inventory()):
                text = json.dumps(case['expected']) if case['semantic_rule'] == 'json' else case['expected']
                ids = self.tok.encode(text)
                rows.append(dict(index=i+1, tokens=ids, output_count=len(ids), finish='stop', prompt_s=1, decode_s=1))
                (run / f'warm-first-logits.{i}').write_bytes(array.array('f', [1]*256).tobytes())
            (run / 'summary.json').write_text(json.dumps({'arms': [{'arm':'warm','requests':rows}]}))
            (run / 'warm').mkdir()
        self.log = self.roots[1] / 'warm' / 'stderr.log'
        self.log.write_text(self.reports())

    def reports(self, inactive=False):
        lines = []
        for snapshot in range(101):
            for index, plan in enumerate(self.binding['plans']):
                value = 0 if inactive and index == 1 else snapshot
                lines.append(f'worker[{index}] Vulkan/native-Q8-host-boundaries approximate READY transport=mTLS '
                    f'model={plan["model"]} layout={plan["layout"]} placement={plan["placement"]} '
                    f'session={index+1} assigned={plan["assigned_bytes"]} requests={value} entries={value} failures=0')
        return '\n'.join(lines)

    def score(self, name='report'):
        with patch.object(V.Q.bundle, 'verify', side_effect=self.configs):
            result = V.score(self.tok, *self.roots, self.root / name)
        return result, json.loads((self.root / name).read_text())

    def test_two_disjoint_plans_and_report(self):
        result, report = self.score()
        self.assertEqual(result, 0)
        self.assertEqual(report['placement_plans'], self.binding['plans'])
        self.assertTrue(report['endpoint_coverage_complete'])
        self.assertNotIn('placement_plan_sha256', report)
        self.assertEqual(report['cases'][97]['id'], 'code-14')
        self.assertEqual(report['cases'][97]['vulkan_entries_by_endpoint'], [1, 1])
        self.assertEqual(report['cases'][97]['vulkan_entries'], 2)
        for index, endpoint in enumerate(report['endpoint_participation']):
            self.assertEqual(endpoint['worker_index'], index)
            self.assertEqual(endpoint['total_entries'], 100)
            self.assertEqual(endpoint['exercised_cases'], 100)
            self.assertEqual(endpoint['assigned_experts'], 1)
        frozen = json.loads((self.root / 'frozen' / 'quality_manifest').read_text())
        self.assertEqual(frozen['plans'], self.binding['plans'])
        self.assertNotIn('plan_sha256', frozen)

    def test_plan_order_content_overlap_and_runtime_identity_refused(self):
        original = self.log.read_text()
        for field, replacement in [('model=1', 'model=9'), ('layout=2:3', 'layout=3:2'),
                                   ('session=2', 'session=0'), ('assigned=5222400', 'assigned=1'),
                                   ('transport=mTLS', 'transport=plaintext-test'), ('failures=0', 'failures=1')]:
            self.log.write_text(original.replace(field, replacement, 1))
            with self.subTest(field=field), self.assertRaises(ValueError): self.score('bad')
        placement = self.binding['plans'][1]['placement']
        self.log.write_text(original.replace('placement='+placement, 'placement=1:2', 1))
        with self.assertRaisesRegex(ValueError, 'identity/plan'): self.score('bad-placement')
        self.log.write_text(original)
        self.configs[1]['argv'][-1], self.configs[1]['argv'][-4] = self.configs[1]['argv'][-4], self.configs[1]['argv'][-1]
        with self.assertRaisesRegex(ValueError, 'frozen v2'): self.score('bad-order')
        with self.assertRaisesRegex(ValueError, 'overlap'): V.plan_binding([self.plans[0], self.plans[0]])
        data = self.plans[0].read_bytes(); self.plans[0].write_bytes(data[:-1]+bytes([data[-1]^1]))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): V.plan_binding(self.plans)

    def test_inactive_endpoint_is_coverage_limited(self):
        self.log.write_text(self.reports(inactive=True))
        result, report = self.score()
        self.assertEqual((result, report['result']), (1, 'screen-coverage-limited'))
        self.assertFalse(report['endpoint_coverage_complete'])
        self.assertEqual(report['offload_evidence']['semantic']['total'], 100)
        self.assertEqual(report['endpoint_participation'][1]['exercised_case_ids'], [])
        self.assertEqual(len(report['endpoint_participation'][1]['nonparticipating_case_ids']), 100)
        self.assertEqual(report['cases'][0]['vulkan_entries_by_endpoint'], [1, 0])

    def test_counter_reset_missing_duplicate_and_reordered_coverage_refused(self):
        original = self.reports(); lines = original.splitlines()
        variants = [original.replace('requests=100 entries=100', 'requests=0 entries=0', 1),
                    original.replace('requests=100 entries=100', 'requests=99 entries=100', 1),
                    original.replace('requests=100 entries=100', 'requests=100 entries=0', 1),
                    original.replace('session=2 assigned=5222400 requests=100', 'session=3 assigned=5222400 requests=100'),
                    '\n'.join(lines[:-1]), original+'\n'+lines[-1],
                    '\n'.join([lines[1], lines[0]]+lines[2:]),
                    original.replace('worker[1]', 'worker[0]')]
        for text in variants:
            with self.subTest(last=text[-120:]), self.assertRaises(ValueError):
                V.endpoint_participation(text, 100, self.binding['plans'])

    def test_each_case_need_not_exercise_each_endpoint(self):
        text = self.reports(inactive=True)
        lines = text.splitlines()
        # Only the last case exercises endpoint 1; no per-case all-endpoint requirement.
        lines[-1] = lines[-1].replace('requests=0 entries=0', 'requests=1 entries=1')
        self.log.write_text('\n'.join(lines))
        result, report = self.score()
        self.assertEqual(result, 0)
        self.assertTrue(report['endpoint_coverage_complete'])
        self.assertEqual(report['endpoint_participation'][1]['exercised_cases'], 1)

    def test_valid_changed_plan_and_different_model_refused(self):
        self.plan(self.plans[1], 2)
        with self.assertRaisesRegex(ValueError, 'frozen v2'): self.score('changed-plan')
        self.plan(self.plans[1], 1, model=9)
        with self.assertRaisesRegex(ValueError, 'model/layout/geometry'): V.plan_binding(self.plans)

    def test_regression_precedes_inactive_endpoint_coverage(self):
        self.log.write_text(self.reports(inactive=True))
        path = self.roots[1] / 'summary.json'
        summary = json.loads(path.read_text())
        summary['arms'][0]['requests'][0]['tokens'] = self.tok.encode('999')
        summary['arms'][0]['requests'][0]['output_count'] = 3
        path.write_text(json.dumps(summary))
        result, report = self.score()
        self.assertEqual((result, report['result']), (1, 'screen-semantic-regression'))
        self.assertFalse(report['endpoint_coverage_complete'])

    def test_single_plan_schema_and_default_comparator_stay_single(self):
        self.assertEqual(V.plan_binding(self.plans[0]), dict(plan_sha256=V.Q.bundle.sha(self.plans[0])))
        with self.assertRaises(ValueError): V.Q.comparable(*self.configs)
        V.Q.comparable(*self.configs, gpu_workers=2)


if __name__ == '__main__': unittest.main()
