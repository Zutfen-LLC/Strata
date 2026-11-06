"""Verify strict subgroup SPIR-V retains both ordered XOR addition steps."""
import re
import subprocess
import sys
from pathlib import Path

for stage in ('gu', 'down'):
    path = Path(sys.argv[1]) / f'native_q8_{stage}_subgroup_fma.spv'
    asm = subprocess.check_output([sys.argv[2], str(path)], text=True)
    assert 'OpFmaKHR' in asm and 'OpCapability GroupNonUniformShuffle' in asm
    assert 'OpGroupNonUniformFAdd' not in asm
    assert re.search(r'OpBitwiseXor %uint %\w+ %uint_16', asm)
    assert re.search(r'OpPhi %uint %uint_8', asm)
    pairs = re.findall(r'(%\w+) = OpGroupNonUniformShuffle %float %\w+ (%\w+) %\w+.*?(%\w+) = OpFAdd %float (%\w+) (%\w+)', asm, re.S)
    assert len(pairs) == 2, (stage, pairs)
    for index, (shuffle, current, result, lhs, rhs) in enumerate(pairs):
        assert (lhs, rhs) == ((shuffle, current) if index == 0 else (current, shuffle)), (stage, index)
        assert f'OpDecorate {result} NoContraction' in asm
print('strict subgroup shuffle and ordered FAdd validated')
