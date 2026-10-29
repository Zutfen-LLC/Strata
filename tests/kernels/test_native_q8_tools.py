"""Interchange/comparison and verbatim-oracle tooling tests; no GPU claims."""
import importlib.util
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Tools(unittest.TestCase):
    def test_verbatim_native_definitions(self):
        oracle = module('native_q8_oracle')
        text = (ROOT / 'src/kernels/cuda/iq_kernels.cu').read_text()
        generated = oracle.generate()
        for name in oracle.NAMES:
            definition = oracle.definition(text, name)
            self.assertIn(definition, generated)
            self.assertIn(name, definition)
        self.assertIn('__fmul_rn', oracle.definition(text, 'swiglu_q8_1_entries_kernel'))
        self.assertIn('for (int o = 16;', oracle.definition(text, 'q8_1_store'))

    def test_strict_shader_creator_and_gate_remain_unchanged(self):
        source = (ROOT / 'src/core/strata_vk_native_stages.inc').read_text()
        begin = source.index('StrataVkNativeStages* strata_vk_native_stages_create(')
        end = source.index('StrataVkNativeStages* strata_vk_native_stages_create_observation(', begin)
        self.assertEqual(hashlib.sha256(source[begin:end].encode()).hexdigest(),
            '0f3b4c73985d46e52994ef5a84551e504f52bc983ded53228a15d6f1b65bd923')
        for path, expected in {
            'src/core/native_q8_float_controls.hpp': '2633f8441b8ddf65e0ca90b54e6fc4bff297d3a98ebdc9a9d0c7918a7b3307ae',
            'src/kernels/vulkan/native_q8_stages.comp': 'c23db4f75b3b09a5bf5675f5a91753578255ecc0614f0822de46d72052044256',
        }.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)

    def test_observation_cannot_be_promoted_to_strict_pass(self):
        compare = module('compare_native_q8_stages')
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory);f = d / 'fixture';native = d / 'native';obs = d / 'obs'
            f.write_bytes(struct.pack('<6I', 0x3151384e, 32, 32, 2, 0, 0) + bytes(32 * 4 * 3))
            parts = [bytes(36), bytes(128), bytes(36), bytes([0xa5] * 36), bytes([0xa5] * 36)]
            payload = b''.join(struct.pack('<I', len(x)) + x for x in parts)
            native.write_bytes(struct.pack('<I', 0x3152384e) + payload)
            legacy = compare.compare(f, native, native)
            self.assertEqual(compare.comparison_exit(legacy), 0)
            self.assertNotIn('observation_only', legacy)
            self.assertNotIn('fp32_denorm_mode', legacy)
            obs.write_bytes(struct.pack('<II', 0x3252384e, 1) + payload)
            report = compare.compare(f, native, obs)
            self.assertTrue(report['exact'])
            self.assertTrue(report['observation_only'])
            self.assertFalse(report['qualified'])
            self.assertEqual(report['fp32_denorm_mode'], 'unspecified')
            self.assertEqual(compare.comparison_exit(report), 2)
            # Outer-view sentinel failure must survive even if all stage bytes match.
            obs.write_bytes(struct.pack('<II', 0x3252384e, 0) + payload)
            report = compare.compare(f, native, obs)
            self.assertTrue(report['observation_view_sentinel_failure'])
            self.assertEqual(compare.comparison_exit(report), 1)
            changed = bytearray(payload);changed[4] ^= 1
            obs.write_bytes(struct.pack('<II', 0x3252384e, 1) + changed)
            report = compare.compare(f, native, obs)
            self.assertEqual(report['sections']['input_q8']['different_bytes'], 1)
            self.assertEqual(compare.comparison_exit(report), 1)
            with self.assertRaises(ValueError):compare.compare(f, obs, native)

    def test_native_conversion_holes_are_not_qualified(self):
        compare = module('compare_native_q8_stages')
        self.assertEqual(compare.contract_exceptions([0] * 32, bytes(36)), [])
        holes = compare.contract_exceptions([1] + [0] * 31, bytes(36))
        self.assertEqual(holes[0]['nan_to_int8_lanes'], list(range(1, 32)))
        holes = compare.contract_exceptions([0x7fc12345] + [0] * 31, bytes(36))
        self.assertEqual(holes[0]['nan_to_int8_lanes'], [0])
        holes = compare.contract_exceptions([0] * 32, struct.pack('<HH', 0, 0x7e00) + bytes(32))
        self.assertTrue(holes[0]['nan_half_metadata'])

    def test_comparator_is_exact_and_rejects_sentinels_and_truncation(self):
        compare = module('compare_native_q8_stages')
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            fixture = d / 'fixture'
            fixture.write_bytes(struct.pack('<6I', 0x3151384e, 32, 32, 2, 0, 0) + bytes(32 * 4 * 3))
            sections = [bytes(36), bytes(128), bytes(36), bytes([0xa5] * 36), bytes([0xa5] * 36)]
            def write(path, parts):
                path.write_bytes(struct.pack('<I', 0x3152384e) + b''.join(struct.pack('<I', len(x)) + x for x in parts))
            a, b = d / 'a', d / 'b'
            write(a, sections);write(b, sections)
            self.assertTrue(compare.compare(fixture, a, b)['exact'])
            changed = list(sections);changed[0] = bytes([1]) + changed[0][1:];write(b, changed)
            report = compare.compare(fixture, a, b)
            self.assertFalse(report['exact'])
            self.assertEqual(report['sections']['input_q8']['different_bytes'], 1)
            changed = list(sections);changed[-1] = bytes(36);write(a, changed);write(b, changed)
            self.assertFalse(compare.compare(fixture, a, b)['exact'])
            b.write_bytes(b.read_bytes()[:-1])
            with self.assertRaises(ValueError):compare.compare(fixture, a, b)


if __name__ == '__main__':
    unittest.main()
