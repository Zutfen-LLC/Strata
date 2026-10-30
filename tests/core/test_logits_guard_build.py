"""Pin gates, external-source immutability, and integer-only shader build checks. No device access."""
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
GGML = Path(sys.argv.pop(1)).resolve()
BUILD = Path(sys.argv.pop(1)).resolve()
CMAKE = sys.argv.pop(1)
NATIVE_STAGES = len(sys.argv) > 1 and sys.argv.pop(1).upper() in ("ON", "TRUE", "1")
FILES = ('src/ggml-vulkan/ggml-vulkan.cpp', 'src/ggml-vulkan/ggml-vulkan-common.h',
         'src/ggml-vulkan/ggml-vulkan-types.h', 'src/ggml-vulkan/ggml-vulkan-push-constants.h',
         'src/ggml-backend-impl.h', 'src/ggml-impl.h', 'include/ggml.h',
         'src/ggml-vulkan/ggml-vulkan-buffers.cpp', 'include/ggml-backend.h', 'include/ggml-alloc.h')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GuardBuild(unittest.TestCase):
    def gate(self, source, directory):
        script = directory / 'check.cmake'
        script.write_text(f'include("{ROOT / "cmake/strata_logits_guard.cmake"}")\n'
                          f'strata_logits_guard_validate("{source}")\n')
        return subprocess.run([CMAKE, '-P', str(script)], text=True, capture_output=True)

    def test_pin_rejects_each_modified_or_missing_private_file(self):
        before = {name: digest(GGML / name) for name in FILES}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(self.gate(GGML, directory).returncode, 0)
            candidate = directory / 'ggml'
            for name in FILES:
                path = candidate / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(GGML / name, path)
            for name in FILES:
                path = candidate / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n// incompatible private source\n')
                result = self.gate(candidate, directory)
                self.assertNotEqual(result.returncode, 0, name)
                self.assertIn('incompatible pinned source', result.stderr)
                self.assertIn(name, result.stderr)
                path.unlink()
                self.assertNotEqual(self.gate(candidate, directory).returncode, 0, name)
                path.write_bytes(original)
            self.assertEqual(self.gate(candidate, directory).returncode, 0)
        self.assertEqual(before, {name: digest(GGML / name) for name in FILES})

    def test_copy_is_exact_original_plus_local_includes(self):
        original = (GGML / FILES[0]).read_bytes()
        copy = (BUILD / 'ggml-vulkan-guard.cpp').read_bytes()
        suffix = b'\n#include "strata_vk_logits_guard.inc"\n'
        if NATIVE_STAGES:
            suffix += b'#include "strata_vk_native_stages.inc"\n'
        if NATIVE_STAGES:
            # Exactly these five native-only hunks; all remaining backend bytes
            # and the acceleration assignment/disable controls stay unchanged.
            replacements = {
                'vk_device ggml_vk_get_device(size_t idx) {': '#include "native_q8_vk_device.hpp"\nvk_device ggml_vk_get_device(size_t idx) {',
                '        vk_device device = std::make_shared<vk_device_struct>();': '        auto strata_q8_device = std::make_shared<StrataVkQ8Device>();\n        vk_device device = strata_q8_device;',
                '        device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;': '        strata_q8_device->strata_q8_dot.requested = device->integer_dot_product;\n        device->integer_dot_product = device->integer_dot_product && shader_integer_dot_product_props.integerDotProduct4x8BitPackedSignedAccelerated;',
                '        if (device->integer_dot_product) {\n            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;': '        if (strata_q8_device->strata_q8_dot.requested) {\n            last_struct->pNext = (VkBaseOutStructure *)&shader_integer_dot_product_features;',
                '        device->device = device->physical_device.createDevice(device_create_info);': '        device->device = device->physical_device.createDevice(device_create_info);\n        strata_q8_device->strata_q8_dot.logical_device_created(shader_integer_dot_product_features.shaderIntegerDotProduct != VK_FALSE);',
            }
            for before, after in replacements.items():
                self.assertEqual(original.count(before.encode()), 1)
                original = original.replace(before.encode(), after.encode())
        self.assertEqual(copy, original + suffix)

    def test_native_dot_patch_rejects_wrong_pin_and_ambiguous_anchors(self):
        if not NATIVE_STAGES:self.skipTest('native stages disabled')
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            for text in ('nothing', 'anchor anchor'):
                script = d / 'anchors.cmake'
                script.write_text(f'include("{ROOT / "cmake/strata_native_q8_integer_dot.cmake"}")\n'
                                  f'set(candidate "{text}")\nstrata_native_q8_dot_replace(candidate "anchor" "replacement")\n')
                result = subprocess.run([CMAKE, '-P', str(script)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('exactly one pinned patch anchor', result.stderr)
            script = d / 'pin.cmake'
            script.write_text(f'include("{ROOT / "cmake/strata_native_q8_integer_dot.cmake"}")\n'
                              'strata_native_q8_dot_patch("wrong source" result)\n')
            result = subprocess.run([CMAKE, '-P', str(script)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('incompatible pinned backend source', result.stderr)

    def test_default_guard_generation_remains_exact_without_native_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            d = Path(temporary)
            candidate = d / 'ggml'
            for name in FILES:
                target = candidate / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(GGML / name, target)
            backend = candidate / 'src/ggml-vulkan'
            (backend / 'CMakeLists.txt').write_text('add_library(ggml-vulkan STATIC ggml-vulkan.cpp)\n')
            (d / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.20)\nproject(default_guard LANGUAGES CXX)\n'
                'set(STRATA_BUILD_TESTS OFF)\n'
                f'add_subdirectory("{backend}" backend)\n'
                f'include("{ROOT / "cmake/strata_logits_guard.cmake"}")\nstrata_enable_logits_guard()\n')
            result = subprocess.run([CMAKE, '-S', str(d), '-B', str(d / 'build')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = (GGML / FILES[0]).read_bytes() + b'\n#include "strata_vk_logits_guard.inc"\n'
            self.assertEqual((d / 'build/strata-logits-guard/ggml-vulkan-guard.cpp').read_bytes(), expected)

    def test_observation_helper_is_exact_pinned_copy_with_one_final_hook(self):
        if not NATIVE_STAGES:
            self.skipTest('standalone native stages disabled')
        original = (GGML / FILES[0]).read_text()
        begin = original.index('static void ggml_vk_create_pipeline_func(')
        end = original.index('\nvoid ggml_vk_destroy_pipeline(', begin)
        expected = original[begin:end].replace('static void ggml_vk_create_pipeline_func(',
            'static void strata_vk_observation_create_pipeline_func(', 1)
        expected = expected.replace('uint32_t required_subgroup_size) {',
            'uint32_t required_subgroup_size, const char* diagnostic_spv_path) {', 1)
        anchor = '    pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);'
        self.assertEqual(expected.count(anchor), 1)
        expected = expected.replace(anchor,
            '    native_q8_validate_observation_effective(static_cast<const uint32_t*>(spv_data), spv_size/4, '
            'shader_module_create_info.pCode, shader_module_create_info.codeSize/4, device->float_controls_rte_fp16, '
            'device->float_controls_denorm_preserve_fp16, diagnostic_spv_path);\n' + anchor)
        self.assertEqual((BUILD / 'strata_native_observe_pipeline.inc').read_text(),
            '// Generated from the exact pinned MIT-licensed GGML helper.\n' + expected)

    def test_quant_trace_is_original_plus_fixed_stores_and_rejects_wrong_base(self):
        if not NATIVE_STAGES:self.skipTest('native stages disabled')
        original = (ROOT / 'src/kernels/vulkan/native_q8_stages.comp').read_text()
        stores = (ROOT / 'src/kernels/vulkan/native_q8_quant_trace.glsl').read_text()
        anchor = '    float clamped=rounded>127.0?127.0:rounded< -127.0?-127.0:rounded;'
        self.assertEqual(original.count(anchor), 1)
        self.assertEqual((BUILD / 'native_q8_quant_trace.comp').read_text(), original.replace(anchor, stores + anchor))
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), 'c23db4f75b3b09a5bf5675f5a91753578255ecc0614f0822de46d72052044256')
        for field in ('maxima[lane]', 'd', 'ratio', 'rounded'):
            self.assertIn('=' + field + ';', stores)
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / 'trace.cmake'
            script.write_text(f'include("{ROOT / "cmake/strata_native_quant_trace.cmake"}")\n'
                              'strata_native_quant_trace_source("wrong base" "stores" result)\n')
            result = subprocess.run([CMAKE, '-P', str(script)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('incompatible original shader', result.stderr)

    def test_rn_variant_changes_only_quantizer_divisions(self):
        if not NATIVE_STAGES:self.skipTest('native stages disabled')
        original = (ROOT / 'src/kernels/vulkan/native_q8_stages.comp').read_text()
        division = (ROOT / 'src/kernels/vulkan/native_f32_div_rn.glsl').read_text()
        expected = original.replace('#version 450', '#version 450\n' + division + '\nfloat native_quant_div_rn(float a,float b) {return uintBitsToFloat(strata_f32_div_rn_bits(floatBitsToUint(a),floatBitsToUint(b)));}')
        expected = expected.replace('precise float scale=maxima[lane]/127.0;', 'precise float scale=native_quant_div_rn(maxima[lane],127.0);')
        expected = expected.replace('precise float ratio=xi/d;', 'precise float ratio=native_quant_div_rn(xi,d);')
        self.assertEqual((BUILD / 'native_q8_quant_rn.comp').read_text(), expected)
        self.assertIn('precise float silu=gate[p.gate_bias+i]/denominator;', expected)

    def test_shader_has_integer_classification_and_no_float_type(self):
        data = (BUILD / 'logits_nonfinite.spv').read_bytes()
        words = struct.unpack('<' + 'I' * (len(data) // 4), data)
        self.assertEqual(words[0], 0x07230203)
        ops = []
        offset = 5
        while offset < len(words):
            size, op = words[offset] >> 16, words[offset] & 0xffff
            self.assertGreater(size, 0)
            ops.append(op)
            offset += size
        self.assertEqual(offset, len(words))
        self.assertNotIn(22, ops)  # OpTypeFloat: no floating loads/arithmetic to optimize
        self.assertIn(199, ops)    # OpBitwiseAnd: exponent classification
        self.assertIn(197, ops)    # OpBitwiseOr: status reduction


if __name__ == '__main__':
    unittest.main()
