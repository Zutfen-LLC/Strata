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

    def test_copy_is_exact_original_plus_one_local_include(self):
        original = (GGML / FILES[0]).read_bytes()
        copy = (BUILD / 'ggml-vulkan-guard.cpp').read_bytes()
        self.assertEqual(copy, original + b'\n#include "strata_vk_logits_guard.inc"\n')

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
