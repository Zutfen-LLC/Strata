import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('compare', ROOT/'tools/compare_native_shared.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class SharedCompare(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        h, f, t = 64, 32, 1
        self.sizes = compare.counts(h, f, t)
        data = struct.pack('<7I', 0x3153484e, 1, h, f, t, 8, 1)
        data += bytes(sum(self.sizes[n]*4 for n in compare.NAMES[:5]))
        self.fixture = self.root/'fixture.bin'
        self.fixture.write_bytes(data)
        for directory in ('native', 'vulkan'):
            target = self.root/directory
            target.mkdir()
            (target/'report.json').write_text(json.dumps(dict(h=h, f=f, tokens=t,
                observation_only=True, qualified=False, sentinels_ok=True, finite=True)))
            for name, size in self.sizes.items():
                (target/(name+'.bin')).write_bytes(struct.pack('<I', 0xa5a5a5a5) if name == 'dummy' else bytes(size*4))

    def run_compare(self):
        return compare.compare(self.fixture, self.root/'native', self.root/'vulkan')

    def change(self, name, at=0, word=0x3f800000):
        path = self.root/'vulkan'/(name+'.bin')
        data = bytearray(path.read_bytes())
        struct.pack_into('<I', data, at, word)
        path.write_bytes(data)

    def test_identical(self):
        result = self.run_compare()
        self.assertTrue(result['exact_arithmetic_checks_passed'])
        self.assertFalse(result['qualified'])

    def test_scalar_dot_must_match(self):
        self.change('raw')
        self.assertFalse(self.run_compare()['exact_arithmetic_checks_passed'])

    def test_projection_must_match_same_input(self):
        self.change('gate')
        self.assertFalse(self.run_compare()['exact_arithmetic_checks_passed'])

    def test_intrinsic_differences_are_observations(self):
        self.change('h')
        self.change('sigmoid')
        self.change('out')
        result = self.run_compare()
        self.assertTrue(result['exact_arithmetic_checks_passed'])
        self.assertEqual(result['first_differing_stage'], 'h')

    def test_unused_q8_sum_half(self):
        self.change('qx', word=0x3f800000)  # high half only, low scale unchanged
        result = self.run_compare()
        self.assertTrue(result['qx_consumed_equal'])
        self.assertFalse(result['stages']['qx']['exact'])
        self.assertTrue(result['exact_arithmetic_checks_passed'])

    def test_consumed_quant_difference(self):
        self.change('qx', at=4)
        self.assertFalse(self.run_compare()['exact_arithmetic_checks_passed'])

    def test_down_conditional_on_same_quant_input(self):
        self.change('down')
        self.assertFalse(self.run_compare()['exact_arithmetic_checks_passed'])
        self.change('h')
        self.change('qh', at=4)
        result = self.run_compare()
        self.assertTrue(result['exact_arithmetic_checks_passed'])
        self.assertFalse(result['qh_consumed_equal'])

    def test_same_hidden_requires_same_quantization(self):
        self.change('qh', at=4)
        self.assertFalse(self.run_compare()['exact_arithmetic_checks_passed'])

    def test_same_scale_inputs_require_same_product(self):
        self.change('out')
        self.assertFalse(self.run_compare()['exact_arithmetic_checks_passed'])

    def test_mutated_input_refused(self):
        self.change('ws')
        with self.assertRaisesRegex(ValueError, 'captured input'):
            self.run_compare()

    def test_bad_guard_report_refused(self):
        path = self.root/'vulkan/report.json'
        report = json.loads(path.read_text())
        report['sentinels_ok'] = False
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'receipt flags'):
            self.run_compare()

    def test_missing_or_truncated_capture_refused(self):
        (self.root/'native/raw.bin').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'capture length'):
            self.run_compare()

    def test_nonfinite_capture_refused(self):
        self.change('out', word=0x7f800000)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.run_compare()


if __name__ == '__main__':
    unittest.main()
