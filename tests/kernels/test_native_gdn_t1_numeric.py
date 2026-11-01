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

# Independent input/output bit vectors for actual underflow and sigmoid tails.
# exp(-138.10023498535156) < half the least F32 subnormal, so RNE is +0.
for backend in ('native','vulkan'):
 assert G.decay_exp_ok(N.bits(-138.10023498535156),0,backend)
 assert G.decay_exp_ok(N.bits(-128.),0,backend)
 assert G.decay_exp_ok(N.bits(-104.),0,backend)
 assert G.decay_exp_ok(N.bits(-88.),0,backend) # permitted FTZ, not a source clamp
 assert G.decay_exp_ok(N.bits(0.),0x3f800000,backend)
 assert not G.decay_exp_ok(N.bits(-1.),0,backend)
 assert not G.decay_exp_ok(N.bits(-138.10023498535156),0x00800000,backend)
 assert G.M.sigmoid_controls(N.bits(104.),0,0x3f800000,backend)
 assert G.M.sigmoid_controls(N.bits(-89.),0x7f800000,0,backend)
 for v,e in ((1.,0x3f800000),(-88.,1),(-138.,0x7f800000)):
  try:ok=G.decay_exp_ok(N.bits(v),e,backend)
  except ValueError:ok=False
  assert not ok
 for z,e,g in ((-88.,0x7f800000,0),(-89.,0x7f800000,1),(104.,0,0x3f7fffff)):
  try:ok=G.M.sigmoid_controls(N.bits(z),e,g,backend)
  except ValueError:ok=False
  assert not ok
print('GDN exp underflow/sigmoid tail acceptance and material-error refusals pass')
