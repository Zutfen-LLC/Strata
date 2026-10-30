"""Compile the actual router and inspect arithmetic in optimized SPIR-V.
This checks instruction constraints, not cross-device transcendental accuracy.
"""
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def inspect(assembly):
    assert 'LocalSize 32 1 1' in assembly
    assert not re.search(r'OpTypeFloat (?:16|64)\b|OpGroupNonUniform|FPFastMathMode| Fma |OpFma', assembly)
    precise = set(re.findall(r'OpDecorate (%\w+) NoContraction', assembly))
    arithmetic = re.findall(r'(%\w+) = (OpFAdd|OpFSub|OpFMul|OpFDiv) (.*)', assembly)
    assert arithmetic and all(i in precise for i, _, _ in arithmetic), 'uncontrolled FP arithmetic'
    divides = {i for i, op, _ in arithmetic if op == 'OpFDiv'}
    assert len(divides) == 2, 'requires full-softmax and selected-sum reciprocals'
    multiply_inputs = [args.split()[-2:] for _, op, args in arithmetic if op == 'OpFMul']
    assert all(any(d in args for args in multiply_inputs) for d in divides), 'reciprocals must multiply'
    assert len(re.findall(r'\bExp\b', assembly)) == 1
    assert 'OpAtomicOr' in assembly and 'OpIsNan' in assembly and 'OpIsInf' in assembly
    assert assembly.count('OpControlBarrier') >= 9 and assembly.count('OpBitwiseXor') == 4
    assert 'OpFOrdEqual' in assembly and 'OpSLessThan' in assembly, 'tie comparison missing'


def main():
    glslc, dis, val = sys.argv[1:]
    source = ROOT / 'src/kernels/vulkan/native_router.comp'
    with tempfile.TemporaryDirectory(prefix='native-router-spv-') as directory:
        output = pathlib.Path(directory) / 'router.spv'
        subprocess.run([glslc, '-fshader-stage=compute', '--target-env=vulkan1.2', '-O', str(source), '-o', str(output)], check=True)
        subprocess.run([val, '--target-env', 'vulkan1.2', str(output)], check=True)
        assembly = subprocess.check_output([dis, str(output)], text=True)
        inspect(assembly)
        # Negative control on the actual module text: one removed contraction
        # guard must fail the same admission assertions.
        mutated = re.sub(r'OpDecorate %\w+ NoContraction', '', assembly, count=1)
        try:
            inspect(mutated)
        except AssertionError:
            pass
        else:
            raise AssertionError('missing NoContraction mutation escaped')
    print('router optimized SPIR-V arithmetic/geometry checks passed; no GPU execution')


if __name__ == '__main__':
    main()
