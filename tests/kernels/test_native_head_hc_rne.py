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
print('retained, negative, signed-zero and unsupported RNE cases passed')
