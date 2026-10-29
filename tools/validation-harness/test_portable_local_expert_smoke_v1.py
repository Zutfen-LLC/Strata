import portable_model_smoke_v1 as B
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import portable_local_expert_smoke_v1 as M

class LocalSmoke(unittest.TestCase):
    def test_sixteen_total_per_die_and_route_order(self):
        log='\n'.join(f'expert_route position=0 layer={l} ids=9,8,7,6,5,4,3,2,1,0' for l in range(48))
        a,b=M.assignments(log)
        self.assertEqual(len(a),16);self.assertEqual(len(b),16)
        self.assertEqual(a[:2],[(0,0),(0,1)])
        self.assertTrue(all(l<32 for l,e in a));self.assertTrue(all(l>=32 for l,e in b))
        with self.assertRaises(ValueError):M.assignments('')

    def test_device_backend_is_strict(self):
        log='expert_device stage=token-complete position=0 slot=0 backend=Vulkan0 numerics=VULKAN_HYBRID_Q8 entries=1'
        self.assertEqual(M.devices(log,'token-complete')[0]['entries'],'1')
        with self.assertRaises(ValueError):M.devices(log.replace('backend=Vulkan0','backend=CPU'),'token-complete')

    def test_existing_smoke_forwards_larger_plans_and_separate_caps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binary=root/'binary';binary.write_bytes(b'mocked')
            plans=[root/'retained0',root/'retained1']
            for plan in plans:plan.write_bytes(b'engine validates plan metadata')
            argv=['smoke','--binary',str(binary),'--binary-sha256',M.Q.bundle.sha(binary),
                  '--out',str(root/'out'),'--phase','smoke','--plan',str(plans[0]),'--plan',str(plans[1]),
                  '--tensor-mib','5800,5810','--workspace-mib','128,128','--free-reserve-mib','512,512']
            with patch('sys.argv',argv),patch.object(M.S,'host_memory',side_effect=ValueError('prelaunch stop')), \
                 patch.object(M.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'prelaunch stop'):M.main()
                launch.assert_not_called()
            command=json.loads((root/'out/planned-argv.json').read_text())
            self.assertEqual(command[command.index('--expert-tensor-mib')+1],'5800,5810')
            self.assertEqual(command[command.index('--expert-workspace-mib')+1],'128,128')
            self.assertEqual(command[command.index('--expert-free-reserve-mib')+1],'512,512')
            self.assertEqual(command.count('--expert-plan'),2)

    def test_pressure_failure_retains_stop_and_reaps_owned_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binary=root/'binary';binary.write_bytes(b'mocked child')
            child=SimpleNamespace(pid=123456789,poll=lambda:None,wait=lambda timeout=None:-15)
            argv=['smoke','--binary',str(binary),'--binary-sha256',M.Q.bundle.sha(binary),
                  '--out',str(root/'out'),'--phase','discover']
            with patch('sys.argv',argv),patch.object(M.S,'host_memory',side_effect=[
                     {'effective_available':1<<40,'SwapUsed':10,'pswpout':2},
                     {'effective_available':1<<39,'SwapUsed':20,'pswpout':3}]), \
                 patch.object(M.S,'model_working_set',return_value={'working_set_bound_bytes':0}), \
                 patch.object(M.S,'vram',side_effect=[dict(pci=p,used=u,total=8<<30) for p,u in zip(B.PCI*2,[0,0,100,200])]), \
                 patch.object(M.S,'verified_devices',return_value=[dict(pci=p,free=8<<30,total=8<<30) for p in B.PCI]), \
                 patch.object(M.S,'rss',return_value=dict(VmRSS=1000,VmHWM=2000,VmSwap=30)), \
                 patch.object(M.S,'guard_host',side_effect=ValueError('pressure')), \
                 patch.object(M.subprocess,'Popen',return_value=child),patch.object(M.os,'killpg') as kill:
                with self.assertRaisesRegex(ValueError,'pressure'):M.main()
                kill.assert_called_once_with(child.pid,M.signal.SIGTERM)
            receipt=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(receipt['status'],'STOP');self.assertEqual(receipt['error'],'pressure')
            self.assertEqual(receipt['exit_code'],-15);self.assertTrue(receipt['child_reaped'])
            self.assertEqual(receipt['sample_count'],1)
            self.assertEqual(receipt['minimum_effective_available'],1<<39)
            self.assertEqual(receipt['sampled_peak_vram'],[100,200])
            self.assertEqual(receipt['sampled_peak_growth'],[100,200])
            self.assertEqual(receipt['minimum_vram_free'],[(8<<30)-100,(8<<30)-200])
            self.assertEqual(receipt['peak_process'],dict(VmRSS=1000,VmHWM=2000,VmSwap=30))
            self.assertEqual(receipt['peak_swap_used'],20)
            self.assertEqual(receipt['maximum_swap_out_pages_delta'],1)
            self.assertEqual(receipt['baseline_host']['pswpout'],2)
            self.assertEqual(receipt['last_host']['pswpout'],3)

if __name__=='__main__':unittest.main()
