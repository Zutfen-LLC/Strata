import importlib.util
import os
import subprocess
from pathlib import Path
import struct
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('comparison', Path(__file__).resolve().parents[2] / 'tools/compare_native_q8_layer.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
class Comparison(unittest.TestCase):
    def test_actual_cpp_emitted_headers(self):
        executable = os.environ.get('STRATA_Q8_METADATA_FIXTURE_EXE')
        if not executable:
            self.skipTest('CMake supplies the actual C++ metadata emitter')
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder)/'emitted'
            subprocess.run([executable,'--metadata-fixture',str(fixture)],check=True,capture_output=True)
            numeric = Path(str(fixture)+'.numeric').read_bytes()
            legacy = Path(str(fixture)+'.legacy').read_bytes()
            self.assertIn(b' T 2 K 10\n', numeric)
            self.assertIn(b' T \x02 K \x0a\n', legacy)
            self.assertEqual(len(legacy.splitlines()),9)
            a,b = Path(folder)/'a',Path(folder)/'b'
            for metadata,kind in ((legacy,'legacy_uint8_stream'),(numeric,'numeric_header')):
                for prefix in (a,b):
                    Path(str(prefix)+'.metadata').write_bytes(metadata)
                    for suffix,size in zip(m.SUFFIXES,(5760,51200,51200,14400,204800)):
                        Path(str(prefix)+suffix).write_bytes(bytes(size))
                result=m.compare(a,b)
                self.assertTrue(result['exact_consumed_boundaries'])
                self.assertEqual(result['metadata_format'],kind)
                self.assertEqual(Path(str(a)+'.metadata').read_bytes(),metadata)
                for bad in (metadata+b'\n',metadata.replace(b' K ',b' K  ')):
                    for prefix in (a,b):Path(str(prefix)+'.metadata').write_bytes(bad)
                    with self.assertRaises(ValueError):m.compare(a,b)
            Path(str(a)+'.metadata').write_bytes(legacy)
            Path(str(b)+'.metadata').write_bytes(numeric)
            with self.assertRaises(ValueError):m.compare(a,b)

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
