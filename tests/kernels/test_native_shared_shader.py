"""Compile the real shader and inspect required arithmetic, no GPU claim."""
import pathlib
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
    print('native shared shader compilation, validation and arithmetic checks passed')
