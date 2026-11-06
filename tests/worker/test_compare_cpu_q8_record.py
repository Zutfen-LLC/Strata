import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('comparison',root/'tools/compare_cpu_q8_record.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Comparison(unittest.TestCase):
    def test_record_order_and_packed_padding(self):
        with tempfile.TemporaryDirectory() as d:
            cpu=Path(d)/'cpu';cpu.mkdir();vk=Path(d)/'layer'
            original=list(range(19,-1,-1))
            arrays=[list(range(20)),list(range(20)),list(range(21)),list(range(20)),
                    [i//10 for i in original],original]
            meta='record 0 model 1 placement 1 2 layer 0 T 2 K 10\n'+('0'*32+' ')*4+'\n'
            meta+=''.join(' '.join(map(str,a))+' \n' for a in arrays)
            Path(str(vk)+'.metadata').write_text(meta)
            for suffix,size in [('.input.q8_1',2*80*36),('.gate.f32',20*640*4),
                                ('.up.f32',20*640*4),('.hidden.q8_1',20*20*36),
                                ('.output.f32',20*2560*4)]:
                Path(str(vk)+suffix).write_bytes(bytes(size))
            for i in range(20):
                for suffix,size in [('input.q8_0',80*34),('gate.f32',640*4),('up.f32',640*4),
                                    ('hidden.q8_0',20*34),('row.f32',2560*4)]:
                    (cpu/f'entry-{i}.{suffix}').write_bytes(bytes(size))
            for n in ('union.f32','vulkan-union.f32'):(cpu/n).write_bytes(bytes(2*2560*4))
            self.assertTrue(m.run(cpu,vk)['union']['equal'])
            Path(str(vk)+'.gate.f32').write_bytes(b'\x01'+bytes(20*640*4-1))
            report=m.run(cpu,vk)
            self.assertEqual(report['entries'][0]['original_entry'],19)
            self.assertFalse(report['entries'][0]['gate']['equal'])
            Path(str(vk)+'.output.f32').write_bytes(b'bad')
            with self.assertRaises(ValueError):m.run(cpu,vk)
    def test_unused_q8_sum_is_not_compared(self):
        a=bytes(34);b=b'\0\0\x12\x34'+bytes(32)
        x,y=m.q8_fields(a,b);self.assertEqual(x,y)
        with self.assertRaises(ValueError):m.q8_fields(a,b[:-1])

if __name__=='__main__':unittest.main()
