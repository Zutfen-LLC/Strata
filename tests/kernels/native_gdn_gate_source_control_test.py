#!/usr/bin/env python3
"""Source/host control only; not a shader or vendor arithmetic oracle."""
import math
import pathlib
import struct

root = pathlib.Path(__file__).resolve().parents[2]
source = (root / 'src/kernels/vulkan/native_gdn_gate_epilogue.glsl').read_text()
for required in ('v=ck(a+dt[p.off[3]+r])', 'if(v>20.0)sp=v;',
                 'strata_llvm_log1p(e,wide,kind)', 'ck(sp)*ssa[p.off[4]+r]',
                 'float e=exp(-a)', 'float b=1.0/den',
                 'atomicOr(status[p.off[8]],2u)', 'atomicOr(status[p.off[8]],4u)',
                 'atomicOr(status[p.off[8]],8u)'):
    assert required in source, required

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def bits(x):
    return struct.unpack('<f', struct.pack('<I', x))[0]

# Choose the projection after adding dt so that the branch is exercised on
# either side of, and exactly at, the representable threshold.
for v, branch in [(bits(0x419fffff), False), (20.0, False), (bits(0x41a00001), True)]:
    a, dt = f32(v - 1.0), 1.0
    actual = f32(a + dt)
    assert actual == v and (actual > 20.0) == branch
    sp = actual if branch else math.log1p(math.exp(actual))
    assert f32(sp * -2.0) < 0 and f32(sp * 2.0) > 0
for a in (-100.0, -40.0, 0.0, 40.0, 100.0):
    beta = 1.0 / (1.0 + math.exp(-a))
    assert 0.0 <= beta <= 1.0
assert f32(1.0 / (1.0 + math.exp(-100.0))) == 1.0
print('GDN gate source and host controls passed; no device arithmetic')
