"""Host-only quality policy regressions. No model, GPU, sockets or generated-code execution."""
import array
import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import quality as Q
import run as driver


class ByteTokenizer:
    tokens = [chr(i) for i in range(256)]

    def encode(self, text, parse_special=False): return list(text.encode())
    def decode(self, ids, errors='strict'): return bytes(ids).decode(errors=errors)


class QualityTests(unittest.TestCase):
    def test_strict_scoring(self):
        self.assertTrue(Q.correct(' 391<|im_end|>', 'exact', '391'))
        for answer in ('391.', '<think>391</think>391', '```391```', '391 extra'):
            self.assertFalse(Q.correct(answer, 'exact', '391'))
        self.assertTrue(Q.correct('{"count":3,"city":"Oslo"}', 'json', {'city': 'Oslo', 'count': 3}))
        for answer in ('{"count":true}', '{"count":3.0}', '{"count":3,"count":3}', '{"count":NaN}'):
            self.assertFalse(Q.correct(answer, 'json', {'count': 3}))

    def test_prepare_exclusive_and_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp) / 'frozen'
            Q.prepare(ByteTokenizer(), out)
            frozen = json.loads((out / 'quality_manifest').read_text())
            self.assertEqual(len(frozen['cases']), 12)
            self.assertEqual(frozen['requests_sha256'], Q.bundle.sha(out / 'requests'))
            self.assertEqual(len((out / 'requests').read_text().splitlines()), 12)
            with self.assertRaises(FileExistsError): Q.prepare(ByteTokenizer(), out)

    def test_exact_default_still_raises_and_quality_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            reference = root / 'reference'
            reference.write_text(json.dumps({'requests': [{'index': 1, 'tokens': [1], 'worker_entries': 1}]}))
            (root / 'x-first-logits.0').write_bytes(b'abcd')
            expected = root / 'expected'; expected.write_bytes(b'efgh')
            cfg = {'assets': {'ref': str(reference), 'logits': str(expected)}}
            arm = {'name': 'x', 'reference': 'ref', 'first_logits_reference': 'logits'}
            actual = {'requests': [{'index': 1, 'tokens': [2], 'worker_entries': 1}]}
            with self.assertRaisesRegex(RuntimeError, 'tokens'): driver.check_references(cfg, arm, actual, root)
            cfg['evaluation_mode'] = 'quality'
            driver.check_references(cfg, arm, actual, root)
            checks = json.loads((root / 'x-exact-diagnostics.json').read_text())['checks']
            self.assertFalse(checks[0]['exact_tokens_and_entries_pass'])
            self.assertFalse(checks[1]['exact_first_logits_pass'])
            cfg.pop('evaluation_mode'); arm.pop('reference')
            with self.assertRaisesRegex(RuntimeError, 'logits'): driver.check_references(cfg, arm, actual, root)

    def test_quality_completion_is_fatal_on_nonfinite_or_invalid_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / 'logits'
            path.write_bytes(array.array('f', [1, 2]).tobytes())
            driver.C.check_quality_completion({'finish': 'stop'}, [1], path)
            for finish, tokens in [('nonfinite', [1]), ('trace', []), ('cancel', [1]),
                                   ('stop', [-1]), ('stop', [2]), ('length', [1]*33)]:
                with self.assertRaises(RuntimeError):
                    driver.C.check_quality_completion({'finish': finish}, tokens, path)
            path.write_bytes(array.array('f', [1, float('nan')]).tobytes())
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                driver.C.check_quality_completion({'finish': 'stop'}, [1], path)

    def test_preflight_requires_explicit_complete_frozen_quality(self):
        import test_harness
        fixture = test_harness.HarnessTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        cfg = fixture.cfg
        cfg['evaluation_mode'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'unknown evaluation'): Q.bundle.validate(cfg, pathlib.Path(Q.bundle.__file__).parent)
        cfg['evaluation_mode'] = 'quality'
        with self.assertRaisesRegex(ValueError, 'first-logit capture'): Q.bundle.validate(cfg, pathlib.Path(Q.bundle.__file__).parent)
        cfg['capture_first_logits'] = True
        with self.assertRaisesRegex(ValueError, 'greedy'): Q.bundle.validate(cfg, pathlib.Path(Q.bundle.__file__).parent)
        cfg['argv'].append('--greedy')
        with self.assertRaisesRegex(ValueError, 'ordered twelve'): Q.bundle.validate(cfg, pathlib.Path(Q.bundle.__file__).parent)
        frozen = fixture.root / 'frozen'; Q.prepare(ByteTokenizer(), frozen)
        cfg['assets']['requests'] = str(frozen / 'requests')
        cfg['assets']['quality_manifest'] = str(frozen / 'quality_manifest')
        cfg['arms'][0].update(restart=False, requests=list(range(12)))
        cfg.update(attach_root=str(fixture.root), attached_worker_idle=True)
        # validate imports are ordinarily checked against the helper source, not fixture root.
        with patch.object(Q.bundle.shutil, 'which', return_value='/synthetic/nvidia-smi'):
            Q.bundle.validate(cfg, pathlib.Path(Q.bundle.__file__).parent)
        (frozen / 'requests').write_text('GEN 1 2\n')
        with self.assertRaisesRegex(ValueError, 'frozen manifest'):
            Q.bundle.validate(cfg, pathlib.Path(Q.bundle.__file__).parent)

    @staticmethod
    def report(entries, failure=0, transport='mTLS', placement='1:2'):
        return ('worker[0] Vulkan/native-Q8-host-boundaries approximate READY transport=' + transport +
                f' model=1 layout=2:3 placement={placement} session=4 entries={entries} failures={failure}')

    def test_participation_identity_and_transport(self):
        text = '\n'.join(self.report(i) for i in range(13))
        self.assertEqual(Q.participation(text, 12), [1]*12)
        for bad in (text.replace('mTLS', 'plaintext-test'), text.replace('failures=0', 'failures=1'),
                    text + '\n' + self.report(13), text.replace('entries=12', 'entries=0'),
                    text.replace('entries=12', 'entries=12').replace('session=4 entries=12', 'session=5 entries=12')):
            with self.assertRaises(ValueError): Q.participation(bad, 12)

    def test_score_retains_baseline_failures_and_numeric_failure(self):
        # Synthetic answers demonstrate policy only; they are not model evidence.
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); frozen = root / 'frozen'; tok = ByteTokenizer()
            Q.prepare(tok, frozen)
            configs = []; roots = []
            for side in ('cpu', 'vk'):
                run = root / side; run.mkdir(); roots.append(run)
                for helper in Q.bundle.HELPERS: (run / helper).write_text('synthetic helper')
                assets = {'requests': str(frozen / 'requests'), 'quality_manifest': str(frozen / 'quality_manifest')}
                for key in ('engine', 'worker', 'manifest'):
                    path = root / key
                    if not path.exists(): path.write_text(key)
                    assets[key] = str(path)
                argv = [assets['engine'], '--greedy']
                if side == 'vk': argv += ['--remote-gpu-worker', 'vulkan@127.0.0.1:1234', '/plan']
                arm = dict(name='warm', requests=list(range(12)), workers=1, loader_threads=1,
                           caller_mask=[0], pool_masks=[[0]], engine_main_mask=[0])
                configs.append(dict(assets=assets, argv=argv, worker_argv=[assets['worker']], env={}, worker_env={}, model_paths=[],
                                    worker_bytes=1, request_seconds=1, arms=[arm]))
                rows = []
                for i, (_, _, rule, expected) in enumerate(Q.CASES):
                    text = expected if rule == 'exact' else json.dumps(expected)
                    if side == 'cpu' and i == 0: text = 'wrong'
                    ids = tok.encode(text)
                    rows.append(dict(index=i+1, tokens=ids, output_count=len(ids), finish='stop', prompt_s=1, decode_s=1))
                    (run / f'warm-first-logits.{i}').write_bytes(array.array('f', [1, 2 if side == 'cpu' else 3]).tobytes())
                (run / 'summary.json').write_text(json.dumps({'arms': [{'arm': 'warm', 'requests': rows}]}))
                (run / 'warm').mkdir()
                (run / 'warm' / 'stderr.log').write_text('\n'.join(self.report(i) for i in range(13)))
            with patch.object(Q.bundle, 'verify', side_effect=configs):
                self.assertEqual(Q.score(tok, *roots, root / 'result'), 1)
            result = json.loads((root / 'result').read_text())
            self.assertEqual(result['result'], 'inconclusive-baseline')
            self.assertEqual((result['baseline_score'], result['candidate_score']), (11, 12))
            self.assertFalse(result['cases'][0]['fixed_numeric_diagnostic']['numeric_pass'])
            self.assertTrue(result['cases'][0]['correctness_disagreement'])
            # Frozen asset and setting mismatches must never become an answer score.
            changed = copy.deepcopy(configs[1]); changed['argv'].append('--different-arithmetic')
            with self.assertRaises(ValueError): Q.comparable(configs[0], changed)
            (roots[1] / 'failure.json').write_text('{}')
            with patch.object(Q.bundle, 'verify', side_effect=configs):
                with self.assertRaisesRegex(ValueError, 'failed runtime'): Q.score(tok, *roots, root / 'no-result')


if __name__ == '__main__': unittest.main()
