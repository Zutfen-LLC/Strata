"""Host-only topology/plan binding checks; no GPU or model execution."""
import argparse
import csv
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import portable_model_smoke_v1 as S
import portable_local_expert_smoke_v1 as L
import portable_resident_v1 as R
import test_portable_model_smoke_v1 as SM
import test_portable_resident_v1 as RM


def plan(path, pair, model=1):
    data = struct.pack('<8sQ6HQQI', b'SGPUPLN1', model, 48, 512, 2560, 640, 8, 8, 2, 3, 1)
    data += struct.pack('<HHQQQ', *pair, 5222400, 7, 8)
    ids = []
    for seed, values in ((0xcbf29ce484222325, data), (0x9e3779b97f4a7c15, data[::-1])):
        for byte in values:
            seed = ((seed ^ byte) * 1099511628211) % 2**64
        ids.append(seed)
    path.write_bytes(data + struct.pack('<QQ', *ids))
    return path


class TopologyControls(unittest.TestCase):
    def test_shared_arguments_default_and_strict_argv(self):
        p = argparse.ArgumentParser()
        S.device_arguments(p)
        default = p.parse_args([])
        self.assertEqual(S.dense_topology(default), 'split')
        self.assertEqual(S.topology_argv(default), [])
        self.assertEqual(S.argv_topology(['engine', '--split', '32']), 'split')
        primary = p.parse_args(['--dense-topology', 'primary', '--split', '0'])
        self.assertEqual(S.topology_argv(primary), ['--dense-topology', 'primary'])
        self.assertEqual(S.device_selection(p, primary)[0], [0, 1])
        for argv in (['--dense-topology'], ['--dense-topology', 'foreign'],
                     ['--dense-topology', 'split', '--dense-topology', 'primary']):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                S.argv_topology(argv)

    def test_primary_dense_head_on_slot_zero_and_no_helper_dense_required(self):
        rows = SM.PlacementAdmission().rows()
        for row in rows:
            if row['backend'].startswith('Vulkan'):
                row['backend'] = 'Vulkan3'
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'placements.tsv'
            def write():
                with path.open('w') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
                    w.writeheader(); w.writerows(rows)
            write()
            counts = S.placements(path, [3, 2], 32, 'primary')
            self.assertGreater(counts['3'], 0)
            self.assertEqual(counts['2'], 0)
            for layer in ('47', '48'):
                target = next(r for r in rows if r['layer'] == layer and r['backend'] == 'Vulkan3')
                target['backend'] = 'Vulkan2'; write()
                with self.assertRaisesRegex(ValueError, 'backend/fallback'):
                    S.placements(path, [3, 2], 32, 'primary')
                target['backend'] = 'Vulkan3'

    def test_exact_plan_identity_and_separate_helper_load(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = [plan(root/'p0', (0, 7)), plan(root/'p1', (47, 9))]
            binding = L.primary_plans(paths)
            log = '\n'.join(f'expert_device stage=loaded position=0 slot={i} backend=Vulkan{device} '
                            'numerics=VULKAN_HYBRID_Q8 assigned=1 weight_bytes=5222400'
                            for i, device in enumerate((3, 2)))
            self.assertEqual(set(L.primary_plan_reports(log, paths, binding, [3, 2])), {0, 1})
            for bad in (log.replace('assigned=1', 'assigned=2'), log.replace('Vulkan2', 'Vulkan3'),
                        log.splitlines()[0], log+'\n'+log.splitlines()[0]):
                with self.subTest(log=bad), self.assertRaises(ValueError):
                    L.primary_plan_reports(bad, paths, binding, [3, 2])
            # Same counts and bytes, different helper ID: frozen binding rejects.
            plan(paths[1], (47, 10))
            with self.assertRaisesRegex(ValueError, 'plans changed'):
                L.primary_plan_reports(log, paths, binding, [3, 2])
            plan(paths[1], (0, 7))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                L.primary_plans(paths)
            plan(paths[1], (47, 9), model=2)
            with self.assertRaisesRegex(ValueError, 'model/layout/geometry'):
                L.primary_plans(paths)

    def test_primary_smoke_forwards_and_discovery_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); binary = root/'binary'; binary.write_bytes(b'fixture')
            paths = [plan(root/'p0', (0, 7)), plan(root/'p1', (47, 9))]
            argv = ['smoke', '--binary', str(binary), '--binary-sha256', L.Q.bundle.sha(binary),
                    '--out', str(root/'out'), '--phase', 'smoke', '--dense-topology', 'primary',
                    '--plan', str(paths[0]), '--plan', str(paths[1])]
            with patch('sys.argv', argv), patch.object(S, 'host_memory', side_effect=ValueError('prelaunch')):
                with self.assertRaisesRegex(ValueError, 'prelaunch'):
                    L.main()
            command = json.loads((root/'out/planned-argv.json').read_text())
            receipt = json.loads((root/'out/summary.json').read_text())
            self.assertEqual(S.argv_topology(command), 'primary')
            self.assertEqual(receipt['device_policy']['dense_topology'], 'primary')
            self.assertEqual(receipt['primary_expert_plans'], L.primary_plans(paths))
            discover = argv[:argv.index('--plan')]
            discover[discover.index('--phase')+1] = 'discover'
            with patch('sys.argv', discover), patch.object(L.subprocess, 'Popen') as launch, patch('sys.stderr'):
                with self.assertRaises(SystemExit): L.main()
                launch.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'split-only'):
                L.assignments('', dense_topology='primary')

    def test_reference_topology_summary_and_argv_must_agree(self):
        for mutation in ('none', 'summary', 'argv', 'duplicate', 'plan', 'default-request'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                tok, _, prompts, binding, ref, binary, paths, records = RM.SuiteControls().fixtures(root)
                plan(paths[0], (0, 7)); plan(paths[1], (47, 9))
                (root/'checks/input-identities.sha256').write_text(''.join(
                    f'{R.Q.bundle.sha(p)}  {p}\n' for p in [binary]+paths))
                summary = json.loads((ref/'summary.json').read_text())
                policy = dict(summary['device_policy'], dense_topology='primary')
                summary['device_policy'] = policy
                summary['primary_expert_plans'] = L.primary_plans(paths)
                if mutation == 'summary': del summary['device_policy']['dense_topology']
                if mutation == 'plan': summary['primary_expert_plans']['plans'][1]['sha256'] = '0'*64
                (ref/'summary.json').write_text(json.dumps(summary))
                for i in range(1, 13):
                    path = ref/f'case-{i:02d}/argv.json'
                    command = json.loads(path.read_text()) + ['--dense-topology', 'primary']
                    if mutation == 'argv' and i == 2: command = command[:-2]
                    if mutation == 'duplicate' and i == 2: command += ['--dense-topology', 'primary']
                    path.write_text(json.dumps(command))
                args = (ref, binding, prompts, len(tok.tokens), R.Q.bundle.sha(binary), paths, '5529,5529')
                if mutation == 'none':
                    self.assertEqual(R.reference_rows(*args, device_policy=dict(policy, split=0)), records)
                else:
                    with self.assertRaises(ValueError):
                        R.reference_rows(*args, device_policy=None if mutation == 'default-request' else dict(policy, dense_topology='primary'))

    def test_historical_absence_and_explicit_split_references_match(self):
        with tempfile.TemporaryDirectory() as d:
            tok, _, prompts, binding, ref, binary, paths, records = RM.SuiteControls().fixtures(Path(d))
            args = (ref, binding, prompts, len(tok.tokens), R.Q.bundle.sha(binary), paths, '5529,5529')
            self.assertEqual(R.reference_rows(*args), records)
            for i in range(1, 13):
                path = ref/f'case-{i:02d}/argv.json'
                path.write_text(json.dumps(json.loads(path.read_text()) + ['--dense-topology', 'split']))
            self.assertEqual(R.reference_rows(*args), records)
            with self.assertRaisesRegex(ValueError, 'dense topology'):
                R.reference_rows(*args, device_policy={'dense_topology': 'primary'})


if __name__ == '__main__':
    unittest.main()
