"""Compile the real shader and inspect required arithmetic, no GPU claim."""
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    compiler, disassembler, validator = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        module = pathlib.Path(temp) / "shared.spv"
        subprocess.run([compiler, "-fshader-stage=compute", "--target-env=vulkan1.2", "-O",
                        str(ROOT / "src/kernels/vulkan/native_shared.comp"), "-o", str(module)], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.2", str(module)], check=True)
        asm = subprocess.check_output([disassembler, str(module)], text=True)
        for required in ('OpCapability FMAKHR', '"SPV_KHR_fma"', 'OpFmaKHR', 'OpSDot', 'NoContraction', 'OpControlBarrier'):
            assert required in asm, required
        assert ' Fma ' not in asm, 'ordinary GLSL Fma is not strict FMA'
        assert 'OpGroupNonUniform' not in asm, 'logical CUDA reductions must not depend on hardware subgroup width'
        assert 'LocalSize 256 1 1' in asm
        matrix = pathlib.Path(temp) / "indexer-bf16.spv"
        source = ROOT / "src/kernels/vulkan/native_shared.comp"
        subprocess.run([compiler, "-fshader-stage=compute", "--target-env=vulkan1.2", "-O",
                        "-DSTRATA_QSA_INDEXER_BF16_MATRIX=1", str(source), "-o", str(matrix)], check=True)
        subprocess.run([validator, "--target-env", "vulkan1.2", str(matrix)], check=True)
        matrix_asm = subprocess.check_output([disassembler, str(matrix)], text=True)
        fmas = re.findall(r"(%\w+) = OpFmaKHR %float (.*)", matrix_asm)
        assert len(fmas) == 3 and fmas[1][0] in fmas[2][1].split(), 'ordered BF16 FMA chain differs'
        assert 'NoContraction' in matrix_asm and 'OpGroupNonUniform' not in matrix_asm
        expanded = subprocess.check_output([compiler, '-E', '-fshader-stage=compute', '-DSTRATA_QSA_INDEXER_BF16_MATRIX=1', str(source)], text=True)
        compact = re.sub(r'\s+', '', expanded)
        assert 'w[p.wb+row*(p.ni/2)+pair]' in compact and 'output_values[p.ob+token*p.no+row]' in compact
    print('native shared shader compilation, validation and arithmetic checks passed')
