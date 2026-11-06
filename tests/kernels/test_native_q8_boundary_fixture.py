import importlib.util
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('fixture', ROOT / 'tools/native_q8_boundary_fixture.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

class BoundaryFixtureTest(unittest.TestCase):
    def test_blocks_are_fixed_finite_32_values_and_contain_boundaries(self):
        blocks = fixture.blocks()
        self.assertEqual(set(blocks), {'control', 'signed_subnormal', 'normal_boundary', 'rounding'})
        for bits in blocks.values():
            self.assertEqual(len(bits), 32)
            self.assertTrue(all((x & 0x7f800000) != 0x7f800000 for x in bits))
        self.assertTrue({1, 0x80000001, 0x007fffff, 0x807fffff} <= set(blocks['signed_subnormal']))
        self.assertTrue({0x007fffff, 0x00800000, 0x00800001} <= set(blocks['normal_boundary']))
        self.assertTrue({0x3effffff, 0x3f000000, 0x3f000001} <= set(blocks['rounding']))
        self.assertEqual(struct.unpack('<f', struct.pack('<I', blocks['rounding'][0]))[0], 127.)
    def test_comparator_detects_each_section_and_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = pathlib.Path(d)/'a', pathlib.Path(d)/'b'
            a.write_bytes(bytes(144)); b.write_bytes(bytes(144))
            cmd = [sys.executable, str(ROOT/'tools/native_q8_boundary_compare.py'), '--producer', 'expert', '--cuda', str(a), '--vulkan', str(b)]
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            for index in (0, 2, 4, 35, 36, 143):
                raw=bytearray(144); raw[index]=1; b.write_bytes(raw)
                self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            b.write_bytes(bytes(143)); self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)

    def test_generation_is_repeatable_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = pathlib.Path(d)/'a', pathlib.Path(d)/'b'
            for out in (a,b): subprocess.run([sys.executable, str(ROOT/'tools/native_q8_boundary_fixture.py'), str(out)], check=True)
            self.assertEqual({p.name:p.read_bytes() for p in a.iterdir()}, {p.name:p.read_bytes() for p in b.iterdir()})
            result = subprocess.run([sys.executable, str(ROOT/'tools/native_q8_boundary_fixture.py'), str(a)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

if __name__ == '__main__': unittest.main()
