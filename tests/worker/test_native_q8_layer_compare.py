import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('comparison', Path(__file__).resolve().parents[2] / 'tools/compare_native_q8_layer.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
class Comparison(unittest.TestCase):
    def test_exact_failures_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a', Path(folder)/'b'
            for prefix in (a, b):
                Path(str(prefix)+'.metadata').write_bytes(b'record 0 model 1 placement 1 2 layer 0 T 1 K 1\ndigests\n0\n0\n0 1\n0\n0\n0\n')
                for suffix, size in zip(m.SUFFIXES, (2880,2560,2560,720,10240)):
                    Path(str(prefix)+suffix).write_bytes(bytes(size) if suffix.endswith('q8_1') else struct.pack('<f',1.0)*(size//4))
            self.assertTrue(m.compare(a,b)['exact_consumed_boundaries'])
            Path(str(b)+'.output.f32').write_bytes(struct.pack('<I',0x3f800001)+struct.pack('<f',1.0)*2559)
            self.assertFalse(m.compare(a,b)['exact_consumed_boundaries'])
            Path(str(b)+'.output.f32').write_bytes(struct.pack('<I',0x7f800000)+struct.pack('<f',1.0)*2559)
            self.assertEqual(m.compare(a,b)['stages']['.output.f32']['nonfinite'], ['vulkan'])
            for groups in (20,24,25):
                entries=groups
                header=f'record 0 model 1 placement 1 2 layer 0 T 3 K 10\ndigests\n'
                rows=[list(range(groups)),list(range(groups)),list(range(groups+1)),list(range(entries)),[i//10 for i in range(entries)],list(range(entries))]
                metadata=header+''.join(' '.join(map(str,row))+'\n' for row in rows)
                for prefix in (a,b):
                    Path(str(prefix)+'.metadata').write_text(metadata)
                    for suffix,size in zip(m.SUFFIXES,(3*2880,entries*2560,entries*2560,entries*720,entries*10240)):
                        Path(str(prefix)+suffix).write_bytes(bytes(size))
                if groups<=24:self.assertTrue(m.compare(a,b)['exact_consumed_boundaries'])
                else:
                    with self.assertRaises(ValueError):m.compare(a,b)
            Path(str(b)+'.metadata').write_bytes(b'changed entry order\n')
            with self.assertRaises(ValueError):m.compare(a,b)
if __name__ == '__main__':unittest.main()
