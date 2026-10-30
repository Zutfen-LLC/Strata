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

    def test_consumer_uses_native_entry_without_new_kernel_or_requantization(self):
        text = (ROOT / 'tests/kernels/native_q8_consumer.cpp').read_text()
        self.assertEqual(text.count('native_expert_grouped('), 1)
        self.assertNotIn('quantize_q8_1_rows(', text)
        self.assertNotIn('__global__', text)
        self.assertNotIn('iq_set_old_kernels(', text)
        self.assertIn('read(gate.data(),scratch.p', text)
        self.assertIn('bits(v[i])!=bits(v[per_entry+i])', text)

    def test_consumer_preflight_hashes_roles_and_exact_difference(self):
        consumer = module('native_q8_consumer')
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory);f = d / 'fixture';n = d / 'native';o = d / 'observation'
            # Synthetic parser data only; never claimed as GPU capture evidence.
            inputs = [0] * 832
            inputs[160:192] = range(1, 33)
            f.write_bytes(struct.pack('<8I', 0x3151384e, 64, 832, 4, 2, 5, 5, 9) +
                          struct.pack('<' + 'I' * 832, *inputs) + bytes(832 * 8))
            packed = bytearray(936);packed[3 * 36 + 1] = 0x3c;packed[3 * 36 + 4] = 127
            packed[184:216] = bytes([127] * 32)
            fused = bytearray([0xa5] * 936);fused[4 * 36:18 * 36] = bytes(14 * 36)
            parts = [bytes(packed), bytes(3328), bytes(936), bytes(fused), bytes([0xa5] * 936)]
            def write(path, observation, sections):
                header = struct.pack('<II', 0x3252384e, 1) if observation else struct.pack('<I', 0x3152384e)
                path.write_bytes(header + b''.join(struct.pack('<I', len(x)) + x for x in sections))
            write(n, False, parts)
            changed = bytearray(packed);changed[184:216] = bytes(32)
            observed = [bytes(changed)] + parts[1:];write(o, True, observed)
            report = consumer.validate(f, n, o)
            self.assertFalse(report['qualified'])
            self.assertEqual(report['sections']['input_q8']['different_bytes'], 32)
            self.assertEqual(consumer.checked_bytes(f, hashlib.sha256(f.read_bytes()).hexdigest()), f.read_bytes())
            with self.assertRaises(ValueError):consumer.checked_bytes(f, '0' * 64)
            with self.assertRaises(ValueError):consumer.checked_bytes(f, 'bad hash')
            changed[216] = 1;write(o, True, [bytes(changed)] + parts[1:])
            with self.assertRaises(ValueError):consumer.validate(f, n, o)
            write(o, False, observed)
            with self.assertRaises(ValueError):consumer.validate(f, n, o)
            write(o, True, [observed[0], bytes(936)] + observed[2:])
            with self.assertRaises(ValueError):consumer.validate(f, n, o)

    def test_native_hq_capture_requires_retained_output_and_weights(self):
        consumer = module('native_q8_consumer')
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory);raw = d / 'native-consumer.raw';weights = d / 'native-consumer.raw.weights.bin'
            hq = d / 'native-consumer.raw.hq.bin'
            raw.write_bytes(b'original raw output');weights.write_bytes(b'original weights');hq.write_bytes(bytes(range(72)))
            expected = consumer.validate_hq_capture(d, raw.read_bytes(), weights.read_bytes())
            self.assertEqual(expected, hashlib.sha256(hq.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):consumer.validate_hq_capture(d, b'different oracle', weights.read_bytes())
            with self.assertRaises(ValueError):consumer.validate_hq_capture(d, raw.read_bytes(), b'different weights')
            for length in (0, 71, 73, 256):
                hq.write_bytes(bytes(length))
                with self.assertRaises(ValueError):consumer.validate_hq_capture(d, raw.read_bytes(), weights.read_bytes())
        source = (ROOT / 'tests/kernels/native_q8_consumer.cpp').read_text()
        self.assertIn('static_cast<const uint8_t*>(scratch.p)+hq_offset', source)
        self.assertLess(source.index('checked(cudaDeviceSynchronize())'), source.index('Bytes hq(c::HQ_BYTES)'))
        self.assertIn('word(raw,0x3143384e);save(raw,gate);save(raw,up);save(raw,output)', source)

    def test_gu_oracle_and_raw_bit_comparison(self):
        gu = module('native_q8_gu')
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory);oracle = d / 'oracle';result = d / 'result'
            values = (0x3f800000, 0xbf800000, 0, 0x80000000) * 16
            out = values * 8
            def section(v):return struct.pack('<I', len(v)) + struct.pack('<' + 'I' * len(v), *v)
            oracle.write_bytes(struct.pack('<I', 0x3143384e) + section(values) + section(values) + section(out))
            expected = gu.validate_oracle(oracle)
            result.write_bytes(struct.pack('<II', 0x3147384e, 1) + section(values) + section(values))
            report = gu.compare(expected, result)
            self.assertTrue(report['exact']);self.assertFalse(report['qualified'])
            self.assertFalse(report['quantizer_qualified']);self.assertEqual(report['original_comparator_exit'], 1)
            altered = list(values);altered[2] = 0x80000000
            result.write_bytes(struct.pack('<II', 0x3147384e, 1) + section(altered) + section(values))
            report = gu.compare(expected, result)
            self.assertFalse(report['exact']);self.assertEqual(report['stages']['gate']['different_bits'], 1)
            self.assertEqual(report['stages']['gate']['differences'][0]['native_bits'], '00000000')
            self.assertEqual(report['stages']['gate']['differences'][0]['vulkan_bits'], '80000000')
            result.write_bytes(struct.pack('<II', 0x3147384e, 0) + section(values) + section(values))
            self.assertFalse(gu.compare(expected, result)['exact'])
            oracle.write_bytes(struct.pack('<I', 0x3143384e) + section(altered) + section(values) + section(out))
            with self.assertRaises(ValueError):gu.validate_oracle(oracle)
            result.write_bytes(struct.pack('<II', 0x3147384e, 1) + section(values[:32]))
            with self.assertRaises(ValueError):gu.compare(expected, result)

    def test_gu_native_receipt_binds_branch_and_artifacts(self):
        gu = module('native_q8_gu')
        hashes = {name: str(i) * 64 for i, name in enumerate(('fixture', 'native', 'observation', 'oracle', 'weights'))}
        receipt = {'source_head': 'a' * 40, 'consumer_sha256': 'b' * 64, 'actual_child_exit': 0,
                   'native_consumer_only': True, 'finite_synthetic_weights': True, 'qualified': False,
                   'original_comparator_exit': 1, 'native_environment': {'STRATA_KQ_SPLIT': None},
                   'sha256': {'fixture': hashes['fixture'], 'native': hashes['native'], 'observation': hashes['observation'],
                              'native-consumer.raw': hashes['oracle'], 'native-consumer.raw.weights.bin': hashes['weights']}}
        gu.validate_receipt(receipt, hashes)
        receipt['native_environment']['STRATA_KQ_SPLIT'] = '1'
        with self.assertRaises(ValueError):gu.validate_receipt(receipt, hashes)
        receipt['native_environment']['STRATA_KQ_SPLIT'] = None
        receipt['sha256']['native-consumer.raw'] = 'f' * 64
        with self.assertRaises(ValueError):gu.validate_receipt(receipt, hashes)
        with self.assertRaises(ValueError):gu.validate_receipt([], hashes)

    def test_gu_null_archive_receipt_requires_explicit_bound_identity(self):
        gu = module('native_q8_gu')
        hashes = {name: str(i) * 64 for i, name in enumerate(('fixture', 'native', 'observation', 'oracle', 'weights', 'native-receipt'))}
        receipt = {'source_head': None, 'consumer_sha256': 'b' * 64, 'actual_child_exit': 0,
                   'native_consumer_only': True, 'finite_synthetic_weights': True, 'qualified': False,
                   'original_comparator_exit': 1, 'native_environment': {'STRATA_KQ_SPLIT': None},
                   'sha256': {'fixture': hashes['fixture'], 'native': hashes['native'], 'observation': hashes['observation'],
                              'native-consumer.raw': hashes['oracle'], 'native-consumer.raw.weights.bin': hashes['weights']}}
        identity = {'source_head': 'a' * 40, 'source_archive_sha256': 'c' * 64,
                    'consumer_sha256': 'b' * 64, 'native_receipt_sha256': hashes['native-receipt']}
        with self.assertRaises(ValueError):gu.validate_receipt(receipt, hashes)
        gu.validate_receipt(receipt, hashes, identity)
        self.assertIsNone(receipt['source_head'])  # Original receipt is never repaired in place.
        for field in identity:
            bad = dict(identity);bad[field] = None
            with self.assertRaises(ValueError):gu.validate_receipt(receipt, hashes, bad)
        for field in ('consumer_sha256', 'native_receipt_sha256'):
            bad = dict(identity);bad[field] = 'f' * 64
            with self.assertRaises(ValueError):gu.validate_receipt(receipt, hashes, bad)
        missing_source = dict(receipt);del missing_source['source_head']
        with self.assertRaises(ValueError):gu.validate_receipt(missing_source, hashes, identity)
        receipt['source_head'] = 'd' * 40
        with self.assertRaises(ValueError):gu.validate_receipt(receipt, hashes, identity)
        gu.validate_receipt(receipt, hashes)
        receipt['source_head'] = 'malformed'
        with self.assertRaises(ValueError):gu.validate_receipt(receipt, hashes, identity)

    def test_down_receipt_requires_actual_verified_hq_with_archive_identity(self):
        down = module('native_q8_down')
        hashes = {name: str(i) * 64 for i, name in enumerate(('fixture', 'native', 'observation', 'oracle', 'weights', 'native-receipt', 'hq'))}
        receipt = {'source_head': None, 'consumer_sha256': 'b' * 64, 'actual_child_exit': 0,
                   'native_consumer_only': True, 'finite_synthetic_weights': True, 'qualified': False,
                   'original_comparator_exit': 1, 'native_environment': {'STRATA_KQ_SPLIT': None},
                   'sha256': {'fixture': hashes['fixture'], 'native': hashes['native'], 'observation': hashes['observation'],
                              'native-consumer.raw': hashes['oracle'], 'native-consumer.raw.weights.bin': hashes['weights'],
                              'native-consumer.raw.hq.bin': hashes['hq']}, 'hq_reuse_verified': True,
                   'retained_oracle_sha256': hashes['oracle'], 'retained_weights_sha256': hashes['weights']}
        identity = {'source_head': 'a' * 40, 'source_archive_sha256': 'c' * 64,
                    'consumer_sha256': 'b' * 64, 'native_receipt_sha256': hashes['native-receipt']}
        down.validate_receipt(receipt, hashes, identity)
        for key in ('hq_reuse_verified', 'retained_oracle_sha256', 'retained_weights_sha256'):
            bad = dict(receipt);bad[key] = False
            with self.assertRaises(ValueError):down.validate_receipt(bad, hashes, identity)
        receipt['sha256']['native-consumer.raw.hq.bin'] = 'f' * 64
        with self.assertRaises(ValueError):down.validate_receipt(receipt, hashes, identity)
        down.validate_hq(bytes(72))
        for n in (0, 71, 73):
            with self.assertRaises(ValueError):down.validate_hq(bytes(n))
        bad_hq = bytearray(72);bad_hq[37] = 0x7c
        with self.assertRaises(ValueError):down.validate_hq(bad_hq)

    def test_down_scatter_raw_bits_and_untouched_destinations(self):
        down = module('native_q8_down')
        # Deliberately distinct synthetic entries make a wrong entry lookup or
        # destination reorder observable in this parser/comparator test.
        output = [0x3f800000, 0, 0x80000000, 0xbf800000] * 64 + [0x40000000, 0x80000000, 0, 0xc0000000] * 64
        expected = [output + [0xa5a5a5a5] * 512,
                    [0xa5a5a5a5] * 512 + output[256:] + [0xa5a5a5a5] * 256,
                    [0xa5a5a5a5] * 1024]
        def packed(sections, status=1):
            return struct.pack('<II', 0x3144384e, status) + b''.join(
                struct.pack('<I', len(v)) + struct.pack('<' + 'I' * len(v), *v) for v in sections)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'down.raw';path.write_bytes(packed(expected))
            oracle = [[], [], output];report = down.compare(oracle, path)
            self.assertTrue(report['exact']);self.assertFalse(report['qualified']);self.assertFalse(report['quantizer_qualified'])
            expected[1][512] = output[0]  # Wrong hidden entry for destination 2.
            path.write_bytes(packed(expected));report = down.compare(oracle, path)
            self.assertFalse(report['exact']);self.assertEqual(report['stages']['sparse_scatter']['different_bits'], 1)
            expected[1][512] = output[256];expected[2][0] = 0
            path.write_bytes(packed(expected));report = down.compare(oracle, path)
            self.assertFalse(report['exact']);self.assertFalse(report['sentinels_ok']);self.assertTrue(report['view_sentinels_ok'])
            expected[2][0] = 0xa5a5a5a5;expected[0][1] = 0x80000000
            path.write_bytes(packed(expected));self.assertFalse(down.compare(oracle, path)['exact'])
            path.write_bytes(packed(expected, 0));self.assertFalse(down.compare(oracle, path)['sentinels_ok'])
            path.write_bytes(packed(expected)[:-1])
            with self.assertRaises(ValueError):down.compare(oracle, path)
        shader = (ROOT / 'src/kernels/vulkan/native_q8_down.comp').read_text()
        self.assertIn('q8_row_dot(base,uint(e),nb,tid)', shader)
        self.assertIn('uint(m[p.mb+33+e])*p.h+row', shader)
        self.assertNotIn('atomic', shader)
        self.assertIn('#include "native_q8_dot.glsl"', shader)
        self.assertIn('#include "native_q8_dot.glsl"', (ROOT / 'src/kernels/vulkan/native_q8_gu.comp').read_text())

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

class ComposedTools(unittest.TestCase):
    def test_composed_boundary_and_failure_reporting(self):
        composed = module('native_q8_composed')
        oracle = [[0x3f800000] * 64, [0x40000000] * 64, [0x40400000] * 512]
        hq = bytes(72)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'composed.raw'
            def capture(parts, sentinels=1):
                raw.write_bytes(struct.pack('<2I', 0x3150384e, sentinels) + b''.join(
                    struct.pack('<I', len(part)) + struct.pack('<' + 'I' * len(part), *part) for part in parts))
            parts = [oracle[0][:], oracle[1][:], [0] * 18, oracle[2][:]]
            capture(parts)
            exact = composed.compare(oracle, hq, raw)
            self.assertTrue(exact['exact'])
            self.assertFalse(exact['qualified'])
            self.assertFalse(exact['quantizer_qualified'])
            self.assertEqual(exact['original_comparator_exit'], 1)
            self.assertEqual(exact['fp32_denorm_mode'], 'unspecified')
            # A single code byte is a failure even if final output still matches.
            parts[2][1] = 1
            capture(parts)
            mismatch = composed.compare(oracle, hq, raw)
            self.assertFalse(mismatch['exact'])
            self.assertEqual(mismatch['stages']['hidden_q8_1']['different_bytes'], 1)
            self.assertEqual(mismatch['stages']['hidden_q8_1']['differences'][0]['kind'], 'code')
            parts[2][1] = 0
            parts[2][0] = 0x7c00
            capture(parts)
            metadata = composed.compare(oracle, hq, raw)['stages']['hidden_q8_1']
            self.assertEqual(metadata['nonfinite_metadata_offsets'], [0])
            self.assertEqual(metadata['differences'][0]['kind'], 'metadata')
            parts[2][0] = 0
            parts[3][0] = 0x80000000
            oracle[2][0] = 0  # isolate the sign bit of zero
            capture(parts)
            self.assertFalse(composed.compare(oracle, hq, raw)['exact'])
            parts[3][0] = 0x7f800000
            capture(parts)
            self.assertEqual(composed.compare(oracle, hq, raw)['stages']['down']['nonfinite_indices'], [0])
            capture(parts, 0)
            self.assertFalse(composed.compare(oracle, hq, raw)['sentinels_ok'])
            raw.write_bytes(raw.read_bytes()[:-1])
            with self.assertRaises(ValueError):composed.compare(oracle, hq, raw)

    def test_composition_explicit_selection_and_no_intermediate_host_upload(self):
        import subprocess
        result = subprocess.run(['python3', str(ROOT / 'tools/native_q8_composed.py')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--observe-unspecified-fp32-denorm', result.stderr)
        source = (ROOT / 'tests/kernels/native_q8_composed_component.cpp').read_text()
        begin = source.index('strata_vk_q8_gu_run(')
        end = source.index('Bytes hq(72)', begin)
        self.assertNotIn('ggml_backend_tensor_set', source[begin:end])
        self.assertNotIn('ggml_backend_tensor_get', source[begin:end])
        self.assertIn('strata_vk_native_stages_run(activation.get(),2,g,u,h,q', source[begin:end])
        self.assertIn('strata_vk_q8_down_run(down.get(),w,q,o', source[begin:end])


class GeometryTools(unittest.TestCase):
    def test_production_sections_roles_and_exact_failures(self):
        geometry = module('native_q8_geometry')
        sections = [[0] * n if i in (0, 4) else [0x3f800000] * n for i, n in enumerate(geometry.COUNTS)]
        def raw(parts, observation=False, status=1):
            return struct.pack('<II', 0x324d384e if observation else 0x314d384e, status) + b''.join(
                struct.pack('<I', len(part)) + struct.pack('<' + 'I' * len(part), *part) for part in parts)
        with tempfile.TemporaryDirectory() as directory:
            n, v = Path(directory) / 'native', Path(directory) / 'vulkan'
            n.write_bytes(raw(sections));v.write_bytes(raw(sections, True))
            report = geometry.compare(n, v)
            self.assertTrue(report['exact']);self.assertFalse(report['qualified']);self.assertFalse(report['quantizer_qualified'])
            self.assertEqual(report['stages']['input_q8']['compared'], 8640)
            self.assertEqual(report['stages']['hidden_q8']['compared'], 2160)
            self.assertEqual(report['stages']['gate']['compared'], 1920)
            self.assertEqual(report['stages']['output']['compared'], 7680)
            sections[4][1] = 1;v.write_bytes(raw(sections, True))
            report = geometry.compare(n, v)
            self.assertFalse(report['exact']);self.assertEqual(report['stages']['hidden_q8']['different'], 1)
            self.assertEqual(report['stages']['hidden_q8']['differences'][0]['kind'], 'code')
            sections[4][0] = 0x7c00;v.write_bytes(raw(sections, True))
            self.assertEqual(geometry.compare(n, v)['stages']['hidden_q8']['vulkan_nonfinite'], [0])
            sections[4][0] = 0;sections[4][1] = 0
            sections[5][0] = 0;n.write_bytes(raw(sections))
            sections[5][0] = 0x80000000;v.write_bytes(raw(sections, True))
            self.assertEqual(geometry.compare(n, v)['stages']['output']['different'], 1)
            sections[5][0] = 0xa5a5a5a5;v.write_bytes(raw(sections, True))
            self.assertFalse(geometry.compare(n, v)['vulkan_sentinels_ok'])
            with self.assertRaises(ValueError):geometry.capture(n, True)
            v.write_bytes(v.read_bytes()[:-1])
            with self.assertRaises(ValueError):geometry.compare(n, v)

    def test_production_dispatch_and_narrow_weight_bound(self):
        geometry = module('native_q8_geometry')
        dispatch = dict(h=2560, f=640, entries=3, groups=3, grid_groups=2, split_off_q8=True,
                        kq_split=False, gu_split_q8=False, grouped_v1=False, old_kernels=False, exp_phase=0,
                        gate_up='native_gu_kernel<8>', activation='swiglu_q8_1_entries_kernel', down='native_down_kernel<8>')
        geometry.validate_dispatch(dispatch)
        for key, bad in (('kq_split', True), ('grouped_v1', True), ('gate_up', 'native_gu_swiglu_kernel<8>'), ('split_off_q8', 1)):
            changed = dict(dispatch);changed[key] = bad
            with self.assertRaises(ValueError):geometry.validate_dispatch(changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights';path.write_bytes(bytes(geometry.WEIGHT_BYTES))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):geometry.preflight.checked_bytes(path, digest)
            self.assertEqual(len(geometry.preflight.checked_bytes(path, digest, max_bytes=geometry.WEIGHT_BYTES)), 5222400)
            path.write_bytes(path.read_bytes() + b'x')
            with self.assertRaises(ValueError):geometry.preflight.checked_bytes(path, hashlib.sha256(path.read_bytes()).hexdigest(), max_bytes=geometry.WEIGHT_BYTES)


if __name__ == '__main__':
    unittest.main()
