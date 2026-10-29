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
        self.assertEqual(copy, original + suffix)

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
