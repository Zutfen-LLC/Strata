"""Fake observations only: no GPU, model, nvidia-smi or child execution."""
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
        for cap in ('0', '-1', '8193', str(1 << 100)):
            with self.subTest(cap=cap), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                self.args('--device-cap-mib', cap)
        initial = [dict(total=8 << 30, used=0)] * 2
        S.admit_vram(initial, [dict(total=8 << 30)]*2, 7680 << 20)
        for capacity in (7 << 30, 0):
            with self.assertRaisesRegex(ValueError, 'capacity/free'):
                S.admit_vram(initial, [dict(total=capacity)]*2, 7680 << 20)
        with self.assertRaises(ValueError):
            S.admit_vram([dict(total=8 << 30, used=1 << 30)]*2, [dict(total=8 << 30)]*2, 7680 << 20)

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
            S.sample_vram(S.PCI, [dict(total=8 << 30)]*2)

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
                    mock.patch.object(S, 'verified_devices', return_value=[dict(total=8 << 30)]*2), \
                    mock.patch.object(S, 'sample_vram', side_effect=[[dict(total=8 << 30, used=0)]*2, ValueError('telemetry unavailable')]), \
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
