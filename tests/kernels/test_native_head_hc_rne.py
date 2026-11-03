"""Frozen operands and strict RNE division for the new head contract."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import compare_native_head_hc_contract as c
assert c.rne_div(0x33c013a7,0x3fffffff)==0x334013a8
assert c.n.add(0x3f800000,0x3f7ffffe)==0x3fffffff
assert c.m.inside(0x334013a9,c.m.quotient_bounds(0x33c013a7,0x3fffffff))
assert c.rne_div(0x33c013a7,0x3fffffff)!=0x334013a9
assert c.rne_div(0xb3c013a7,0x3fffffff)==0xb34013a8
assert c.rne_div(0,0x3f800000)==0 and c.rne_div(0x80000000,0x3fffffff)==0x80000000
assert c.rne_div(0x3f800000,0x40000000)==0x3f000000
for numerator,denominator in [(0x7f800000,0x3f800000),(1,0x3f800000),(0x3f800000,0)]:
 try:c.rne_div(numerator,denominator)
 except ValueError:pass
 else:raise AssertionError('unsupported RNE operands accepted')
# The separately versioned head-only candidate admits only the observed finite
# negative overflow class; the shared HC method keeps its previous refusal.
for v in [0xc2b20000,0xc2b20001,0xc2c803e5,0xff7fffff]:
 assert c.m.head_silu_overflow_candidate(v)
 assert c.m.head_silu_overflow_controls(v,0x7f800000,0x80000000)
 assert c.m.activation_bounds(v,'vulkan',True,True)==(0x80000000,0x80000000)
 assert c.rne_div(v,0x7f800000)==0x80000000
 try:c.m.activation_bounds(v,'vulkan',True)
 except ValueError:pass
 else:raise AssertionError('shared SiLU overflow admission changed')
 for e,a in [(0x7f7fffff,0x80000000),(0xff800000,0x80000000),(0x7fc00000,0x80000000),(0x7f800000,0),(0x7f800000,1)]:
  try:c.m.head_silu_overflow_controls(v,e,a)
  except ValueError:pass
  else:raise AssertionError('invalid overflow evidence accepted')
for v in [0xc2b1ffff,0x42b20000,0x80000000]:assert not c.m.head_silu_overflow_candidate(v)
for v in [0xff800000,0x7f800000,0xffc00000,0x80000001]:
 try:c.m.head_silu_overflow_candidate(v)
 except ValueError:pass
 else:raise AssertionError('invalid overflow input accepted')
for v in [0xc2b1ffff,0x42b20000,0xff800000,0xffc00000]:
 try:c.rne_div(v,0x7f800000)
 except ValueError:pass
 else:raise AssertionError('out-of-scope infinite denominator accepted')
print('retained RNE and head-only negative overflow boundary/nonfinite refusals passed')
