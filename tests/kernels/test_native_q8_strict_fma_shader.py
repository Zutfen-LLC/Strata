"""Validate compiler-emitted strict modules and preservation of arithmetic/control order."""
from pathlib import Path
import re
import subprocess
import sys
import unittest

BUILD, DIS, VAL = Path(sys.argv.pop(1)), sys.argv.pop(1), sys.argv.pop(1)


def disassemble(path):
    subprocess.run([VAL, '--target-env', 'vulkan1.2', str(path)], check=True, capture_output=True)
    return subprocess.check_output([DIS, '--raw-id', str(path)], text=True)


def operations(assembly):
    # Intrinsic declarations can retain dead function-local variables/types.
    # Compare the actual arithmetic and ordering, not optimizer bookkeeping.
    selected = []
    for line in assembly.splitlines():
        match = re.search(r'\bOp(\w+)\b', line)
        if not match:
            continue
        op = match[1]
        if op == 'FmaKHR' or op == 'ExtInst' and ' Fma ' in line:
            selected.append('Fma')
        elif op.startswith(('FAdd', 'FMul', 'FSub', 'FDiv', 'SDot', 'ConvertSToF',
                            'BitwiseXor', 'ShiftRightLogical', 'ControlBarrier', 'LoopMerge')):
            selected.append(op)
    return selected


class StrictShader(unittest.TestCase):
    def test_strict_fma_and_unchanged_arithmetic_order(self):
        for stage in ('gu', 'down'):
            with self.subTest(stage=stage):
                legacy = disassemble(BUILD / f'native_q8_{stage}.spv')
                strict = disassemble(BUILD / f'native_q8_{stage}_strict_fma.spv')
                self.assertIn('OpCapability FMAKHR', strict)
                self.assertIn('OpExtension "SPV_KHR_fma"', strict)
                self.assertEqual(strict.count('OpFmaKHR'), 1)
                self.assertNotRegex(strict, r'OpExtInst .* Fma ')
                self.assertNotIn('FPFastMathMode', strict)
                self.assertNotIn('OpGroupNonUniform', strict)
                self.assertNotIn('FMAKHR', legacy)
                self.assertNotIn('SPV_KHR_fma', legacy)
                self.assertRegex(legacy, r'OpExtInst .* Fma ')
                self.assertEqual(operations(strict), operations(legacy))
                self.assertEqual(strict.count(' NoContraction'), legacy.count(' NoContraction'))
                # Scale FMUL remains separately rounded before strict accumulation.
                definitions = dict(re.findall(r'(%\d+) = (Op[^\n]+)', strict))
                fma = next(value.split() for value in definitions.values() if value.startswith('OpFmaKHR '))
                self.assertTrue(definitions[fma[2]].startswith('OpFMul '))
                self.assertIn(f'OpDecorate {fma[2]} NoContraction', strict)
                self.assertTrue(definitions[fma[3]].startswith('OpConvertSToF '))
                self.assertEqual(definitions[fma[1]], 'OpTypeFloat 32')


if __name__ == '__main__':
    unittest.main()
