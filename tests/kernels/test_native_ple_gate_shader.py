#!/usr/bin/env python3
"""Check the compiled T1 gate's geometry, explicit FMA and protected arithmetic."""
import argparse
from pathlib import Path
import re
import struct
import subprocess
import tempfile

p=argparse.ArgumentParser()
p.add_argument('--shader',type=Path,required=True)
p.add_argument('--glslc',type=Path,required=True)
p.add_argument('--validator',type=Path,required=True)
p.add_argument('--disassembler',type=Path,required=True)
p.add_argument('--scratch',type=Path,required=True)
a=p.parse_args()
with tempfile.TemporaryDirectory(dir=a.scratch) as t:
    spv=Path(t)/'gate.spv';asm=Path(t)/'gate.spvasm'
    subprocess.run([str(a.glslc),'--target-env=vulkan1.2','-O','-fshader-stage=compute',str(a.shader),'-o',str(spv)],check=True)
    subprocess.run([str(a.validator),'--target-env','vulkan1.2',str(spv)],check=True)
    subprocess.run([str(a.disassembler),str(spv),'-o',str(asm)],check=True)
    data=spv.read_bytes();text=asm.read_text()
    if 'LocalSize 512 1 1' not in text or struct.pack('<I',0x3ca1e89b) not in data:
        raise ValueError('native gate workgroup or scale constant differs')
    if text.count('OpFmaKHR')!=1 or text.count('OpControlBarrier')<7 or 'OpGroupNonUniform' in text:
        raise ValueError('gate explicit FMA or XOR reduction geometry differs')
    operations=re.findall(r'^\s*(%\w+)\s*=\s*Op(FMul|FAdd|FDiv)\b',text,re.M)
    if sum(k=='FMul' for _,k in operations)<2 or sum(k=='FAdd' for _,k in operations)<4 or sum(k=='FDiv' for _,k in operations)!=1:
        raise ValueError('gate operation sequence differs')
    decorations=set(re.findall(r'OpDecorate\s+(%\w+)\s+NoContraction',text))
    if any(name not in decorations for name,_ in operations):
        raise ValueError('unprotected gate arithmetic')
    print('Optimized SPIR-V gate geometry, protected arithmetic, explicit FMA and barrier structure pass; intrinsic parity is unqualified.')
