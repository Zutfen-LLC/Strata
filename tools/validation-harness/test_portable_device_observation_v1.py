"""Fake observations only: no GPU, model, nvidia-smi or child execution."""
import portable_model_smoke_v1 as B

import argparse
import pathlib
import json
import signal
import sys
import subprocess
import tempfile
import unittest
from unittest import mock
import portable_model_smoke_v1 as S
import portable_local_expert_smoke_v1 as L


class DeviceObservation(unittest.TestCase):
    def args(self, *extra):
        parser = argparse.ArgumentParser()
        S.device_arguments(parser)
        args = parser.parse_args(extra)
        return args, S.device_selection(parser, args)

    def test_amd_defaults_unchanged(self):
        args, selection = self.args()
        self.assertEqual(selection, ([0, 1], ['0000:07:00.0', '0000:0b:00.0']))
        self.assertEqual((args.split, args.device_cap_mib, args.manifest, args.model_root),
                         (32, 7680, S.MANIFEST, None))

    def test_explicit_selection_and_paths(self):
        args, selection = self.args('--devices', '2,0', '--pci', '0000:03:00.0,0000:01:00.0',
                                    '--model-root', '/existing/model', '--manifest', '/existing/manifest', '--split', '31')
        self.assertEqual(selection[0], [2, 0])
        self.assertEqual(args.model_root, pathlib.Path('/existing/model'))
        self.assertEqual(args.split, 31)

    def test_cap_bounds_and_selected_capacity(self):
        for cap in ('0', '-1', '1,2,3', '1,', str(1 << 100)):
            with self.subTest(cap=cap), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                self.args('--device-cap-mib', cap)
        initial = [dict(pci=p,total=8 << 30,used=9 << 20) for p in S.PCI]
        mapping = [dict(pci=p,total=8 << 30,free=(8 << 30)-(9 << 20)) for p in S.PCI]
        budgets = S.admit_vram(initial, mapping, 7680 << 20)
        self.assertEqual([b['effective_limit_bytes'] >> 20 for b in budgets], [7671,7671])
        # The real conservative ledger fits even though policy ceiling + reserve does not.
        self.assertLess(7550 << 20, budgets[0]['effective_limit_bytes'])
        for capacity in (7 << 30, 0):
            with self.assertRaisesRegex(ValueError, 'capacity/free'):
                S.admit_vram(initial, [dict(d,total=capacity) for d in mapping], 7680 << 20)
        constrained = S.admit_vram(initial, [dict(d,free=7800 << 20) for d in mapping], 7680 << 20)
        self.assertEqual(constrained[0]['effective_limit_bytes'], 7288 << 20)
        with self.assertRaises(ValueError):
            S.admit_vram(initial, [dict(d,free=512 << 20) for d in mapping], 7680 << 20)

    def test_asymmetric_caps_follow_reversed_devices(self):
        args, selection = self.args('--devices','2,0','--pci',','.join(reversed(S.PCI)),
                                    '--device-cap-mib','11776,7680')
        self.assertEqual(S.device_caps(args), [11776 << 20,7680 << 20])
        self.assertEqual(S.cap_argv(args), '11776,7680')
        physical = selection[1]
        initial = [dict(pci=p,total=t << 20,used=9 << 20) for p,t in zip(physical,[12288,8192])]
        mapping = [dict(pci=d['pci'],total=d['total'],free=d['total']-d['used']) for d in initial]
        budgets = S.admit_vram(initial,mapping,S.device_caps(args))
        self.assertEqual([b['effective_limit_bytes'] >> 20 for b in budgets],[11767,7671])
        self.assertEqual(S.budget_argv(budgets)[0], '--device-free-bytes')
        with self.assertRaises(ValueError): S.admit_vram(initial,mapping,list(reversed(S.device_caps(args))))
        S.guard_vram([dict(d,used=d['used']+(7550 << 20)) for d in initial],initial,budgets)
        for sample in ([dict(d,used=d['total']-(511 << 20)) for d in initial],
                       [dict(d,pci=S.PCI[0]) for d in initial]):
            with self.assertRaises(ValueError): S.guard_vram(sample,initial,budgets)

    def test_larger_reserve_is_outside_ledger_once(self):
        initial = [dict(pci=p,total=8192 << 20,used=9 << 20) for p in S.PCI]
        mapping = [dict(pci=d['pci'],total=d['total'],free=d['total']-d['used']) for d in initial]
        reserves = S.device_reserves(argparse.Namespace(expert_free_reserve_mib='1024,256'))
        self.assertEqual(reserves,[1024 << 20,512 << 20])
        budgets = S.admit_vram(initial,mapping,7680 << 20,reserves)
        self.assertEqual([b['effective_limit_bytes'] >> 20 for b in budgets],[7159,7671])
        S.guard_vram([dict(d,used=d['used']+(7000 << 20)) for d in initial],initial,budgets)

    def test_missing_or_contradictory_memory_evidence(self):
        initial = [dict(pci=p,total=8 << 30,used=0) for p in S.PCI]
        mapping = [dict(pci=p,total=8 << 30,free=8 << 30) for p in S.PCI]
        for changed in ([dict(d,free=-1) for d in mapping], [dict(d,free=9 << 30) for d in mapping],
                        [dict(d,pci='wrong') for d in mapping], mapping[:1]):
            with self.assertRaises(ValueError): S.admit_vram(initial,changed,7680 << 20)
        with self.assertRaises(KeyError): S.admit_vram(initial,[dict(pci=p,total=8 << 30) for p in S.PCI],7680 << 20)

    def mapping(self, output):
        with mock.patch.object(S.subprocess, 'run', return_value=mock.Mock(stdout=output)) as run, \
                mock.patch.object(pathlib.Path, 'read_text', return_value='0x030000'):
            result = S.verified_devices('/binary', [2, 0], ['0000:03:00.0', '0000:01:00.0'])
            self.assertEqual(run.call_args.args[0], ['/binary', '--list-vulkan'])
            return result

    OUTPUT = '0\t0000:01:00.0\t8000000000\t8589934592\tNVIDIA RTX3060\n2\t0000:03:00.0\t8000000000\t8589934592\tNVIDIA RTX3060\n'

    def test_same_backend_mapping_in_selected_order(self):
        result = self.mapping(self.OUTPUT)
        self.assertEqual([v['index'] for v in result], [2, 0])
        self.assertEqual([v['pci'] for v in result], ['0000:03:00.0', '0000:01:00.0'])
        self.assertEqual(len(self.mapping(self.OUTPUT.replace('NVIDIA RTX3060', 'AMD RADV'))), 2)

    def test_mapping_mismatch_missing_and_software_rejected(self):
        for output in (self.OUTPUT.replace('03:00.0', '04:00.0'),
                       self.OUTPUT.replace('NVIDIA RTX3060', 'llvmpipe'),
                       self.OUTPUT.replace('0000:03:00.0', '-'), '', self.OUTPUT+self.OUTPUT):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.mapping(output)

    def test_amd_sysfs_observation(self):
        with mock.patch.object(pathlib.Path, 'read_text', side_effect=['0x1002', '1024', str(8 << 30)]):
            self.assertEqual(S.vram(S.PCI[0]), dict(pci=S.PCI[0], used=1024, total=8 << 30))

    def test_nvidia_uses_physical_identity_not_index(self):
        with mock.patch.object(pathlib.Path, 'read_text', return_value='0x10de'), \
                mock.patch.object(S.subprocess, 'run', return_value=mock.Mock(stdout='00000000:03:00.0, 1, 12288\n')) as run:
            value = S.vram('0000:03:00.0')
            self.assertIn('--id=0000:03:00.0', run.call_args.args[0])
            self.assertEqual((value['used'], value['total']), (1 << 20, 12288 << 20))

    def test_nvidia_bad_or_missing_telemetry_rejected(self):
        for output in ('', '00000000:01:00.0, 1, 12288', '00000000:03:00.0, N/A, 12288',
                       '00000000:03:00.0, 12289, 12288', '00000000:03:00.0, 0, 0'):
            with mock.patch.object(pathlib.Path, 'read_text', return_value='0x10de'), \
                    mock.patch.object(S.subprocess, 'run', return_value=mock.Mock(stdout=output)), self.assertRaises(ValueError):
                S.vram('0000:03:00.0')
        for failure in (FileNotFoundError('nvidia-smi'), subprocess.CalledProcessError(1, 'nvidia-smi'),
                        subprocess.TimeoutExpired('nvidia-smi', 5)):
            with mock.patch.object(pathlib.Path, 'read_text', return_value='0x10de'), \
                    mock.patch.object(S.subprocess, 'run', side_effect=failure), self.assertRaises(type(failure)):
                S.vram('0000:03:00.0')

    def test_intel_fails_with_adapter_requirement(self):
        with mock.patch.object(pathlib.Path, 'read_text', return_value='0x8086'), \
                self.assertRaisesRegex(ValueError, 'unsupported guarded VRAM telemetry.*adapter required'):
            S.vram('0000:03:00.0')

    def test_capacity_drift_stops(self):
        with mock.patch.object(S, 'vram', return_value=dict(total=7 << 30, used=0)), \
                self.assertRaisesRegex(ValueError, 'capacity changed'):
            S.sample_vram(S.PCI, [dict(pci=p,free=8<<30,total=8<<30) for p in B.PCI])

    def test_failed_telemetry_reaps_owned_child_and_preserves_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory) / 'evidence'
            child = mock.Mock(pid=31337, returncode=-signal.SIGTERM)
            child.poll.return_value = None
            host = dict(effective_available=1 << 40, pswpout=0)
            argv = ['smoke', '--binary', sys.executable, '--evidence', str(evidence), '--timeout-seconds', '1']
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(S, 'model_working_set', return_value={'working_set_bound_bytes': 0}), \
                    mock.patch.object(S, 'host_memory', return_value=host), \
                    mock.patch.object(S, 'rss', return_value={}), \
                    mock.patch.object(S, 'verified_devices', return_value=[dict(pci=p,free=8<<30,total=8<<30) for p in B.PCI]), \
                    mock.patch.object(S, 'sample_vram', side_effect=[[dict(pci=p,used=0,total=8<<30) for p in B.PCI], ValueError('telemetry unavailable')]), \
                    mock.patch.object(S.subprocess, 'Popen', return_value=child), \
                    mock.patch.object(S.os, 'killpg') as kill:
                with self.assertRaisesRegex(ValueError, 'telemetry unavailable'):
                    S.main()
                kill.assert_called_once_with(31337, signal.SIGTERM)
                child.wait.assert_called_once_with(timeout=5)
            summary = json.loads((evidence / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'STOP')
            self.assertEqual(summary['error'], 'telemetry unavailable')

    def test_failed_mapping_prevents_model_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = pathlib.Path(directory) / 'evidence'
            argv = ['smoke', '--binary', sys.executable, '--evidence', str(evidence), '--timeout-seconds', '1']
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(S, 'model_working_set', return_value={'working_set_bound_bytes': 0}), \
                    mock.patch.object(S, 'host_memory', return_value=dict(effective_available=1 << 40)), \
                    mock.patch.object(S, 'verified_devices', side_effect=ValueError('PCI mismatch')), \
                    mock.patch.object(S.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'PCI mismatch'):
                    S.main()
                launch.assert_not_called()
            self.assertFalse(json.loads((evidence / 'summary.json').read_text())['launched'])

    def test_local_expert_uses_selected_backend_and_split(self):
        log = 'expert_device stage=token-complete slot=0 backend=Vulkan2 numerics=VULKAN_HYBRID_Q8'
        self.assertEqual(L.devices(log, 'token-complete', [2, 0])[0]['backend'], 'Vulkan2')
        routes = '\n'.join(f'expert_route position=0 layer={l} ids=9,8,7,6,5,4,3,2,1,0' for l in range(48))
        self.assertTrue(all(l >= 31 for l, _ in L.assignments(routes, 31)[1]))


if __name__ == '__main__':
    unittest.main()
