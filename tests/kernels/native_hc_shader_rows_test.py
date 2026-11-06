#!/usr/bin/env python3
"""CPU-only SPIR-V regression checks; not a substitute for wrapper dispatch tests."""
import contextlib
import os
import pathlib
import re
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[2]
glslc, val, dis = sys.argv[1:4]


def compile_shader(source, define, directory):
    output = pathlib.Path(directory) / (source + (define or '') + '.spv')
    command = [glslc, '-fshader-stage=compute', '--target-env=vulkan1.2']
    if define:
        command.append('-D' + define)
    subprocess.run(command + [str(root / 'src/kernels/vulkan' / (source + '.comp')),
                              '-o', str(output)], check=True)
    subprocess.run([val, '--target-env', 'vulkan1.2', str(output)], check=True)
    return subprocess.check_output([dis, str(output)], text=True)


def row_product(text, variable, constant, count=1):
    # Follow the generated SSA load into the multiply, rather than checking GLSL text.
    pattern = (r'(%\w+) = OpLoad %uint %' + variable + r'\s+'
               r'%\w+ = OpIMul %uint \1 %uint_' + str(constant) + r'\b')
    assert len(re.findall(pattern, text)) >= count, (variable, constant)


directory = (contextlib.nullcontext(tempfile.mkdtemp(prefix='strata-hc-spirv-'))
             if os.environ.get('STRATA_TEST_PRESERVE_MODEL_FIXTURE') == '1'
             else tempfile.TemporaryDirectory(prefix='strata-hc-spirv-'))
with directory as tmp:
    if os.environ.get('STRATA_TEST_PRESERVE_MODEL_FIXTURE') == '1':
        print(f'Preserving SPIR-V artifacts in {tmp}', flush=True)
    for define in ('', 'STRATA_HC_FINAL_WRITE'):
        shader = compile_shader('native_hc_pending', define, tmp)
        for stride in (10240, 2560, 4):
            row_product(shader, 'row', stride)
    for define in ('', 'STRATA_HC_HEAD_DIV_RN'):
        shader = compile_shader('native_hc_mix', define, tmp)
        row_product(shader, 'token', 10240, 2)
        row_product(shader, 'token', 2560)
        # Projection input uses token*width. Weight indexing is row*width,
        # with no token term in that address.
        assert re.search(r'(%\w+) = OpLoad %uint %token\s+(%\w+) = OpLoad %uint %width\s+%\w+ = OpIMul %uint \1 \2', shader)
        assert re.search(r'(%\w+) = OpLoad %uint %row\s+(%\w+) = OpLoad %uint %width\s+%\w+ = OpIMul %uint \1 \2', shader)
print('HC shader variants validated; SPIR-V row strides checked')
