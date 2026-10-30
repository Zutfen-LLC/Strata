import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('compare', pathlib.Path(__file__).resolve().parents[2] / 'tools/compare_native_router.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparisonTest(unittest.TestCase):
    def test_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = [pathlib.Path(tmp) / name for name in ('a', 'b')]
            for path in (a, b):
                path.mkdir()
                (path / 'report.json').write_text(json.dumps(dict(tokens=1, experts=512, k=10, qualified=False, sentinels_ok=True, valid_output=True)))
                (path / 'ids.i32').write_bytes(struct.pack('<10I', *range(10)))
                (path / 'weights.f32').write_bytes(struct.pack('<10f', *([0.1] * 10)))
                (path / 'view-0.bin').write_bytes(bytes((3 + 512 + 9) * 4))
            self.assertTrue(MODULE.compare(a, b)['exact'])
            words = bytearray((b / 'weights.f32').read_bytes())
            words[0] ^= 1
            (b / 'weights.f32').write_bytes(words)
            self.assertEqual(MODULE.compare(a, b)['weights.f32']['different_bits'], 1)
            self.assertFalse(MODULE.compare(a, b)['exact'])
            (b / 'weights.f32').write_bytes((a / 'weights.f32').read_bytes())
            (b / 'ids.i32').write_bytes(struct.pack('<10I', *reversed(range(10))))
            self.assertFalse(MODULE.compare(a, b)['exact'])
            (b / 'view-0.bin').write_bytes(bytes((3 + 512 + 8) * 4))
            with self.assertRaises(AssertionError):
                MODULE.compare(a, b)


if __name__ == '__main__':
    unittest.main()
