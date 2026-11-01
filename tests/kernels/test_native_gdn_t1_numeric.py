import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_gdn_t1_numeric as G
import native_hc_numeric as N
# Independently pinned RNE rsqrt(128) and rsqrt(1), plus gross wrong common results.
assert G.rsqrt_ok(0x3db504f3,N.bits(128.))
assert G.rsqrt_ok(0x3f800000,N.bits(1.))
assert not G.rsqrt_ok(0x3db50500,N.bits(128.))
assert not G.rsqrt_ok(0x3f800010,N.bits(1.))
for wrong in (0x7f800000,1):
 try:G.rsqrt_ok(wrong,N.bits(1.))
 except ValueError:pass
 else:raise AssertionError('nonfinite/subnormal accepted')
print('GDN primitive bound acceptance/rejection independent vectors pass')
