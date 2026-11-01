from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_gdn_conv_t1_numeric as C
N=C.N
for b in ('native','vulkan'):
 for v,e,a in [(0.,0x3f800000,0),(104.,0,N.bits(104.)),(-89.,0x7f800000,0x80000000)]:assert C.silu_controls(N.bits(v),e,a,b)
 for v,e,a in [(0.,0x3f800000,1),(104.,0,N.bits(103.)),(-89.,0x7f800000,0),(-88.,0x7f800000,0x80000000)]:
  try:ok=C.silu_controls(N.bits(v),e,a,b)
  except ValueError:ok=False
  assert not ok,'wrong primitive/transition/sign accepted'
assert C.G.rsqrt_ok(N.bits(1.),N.bits(1.))
assert not C.G.rsqrt_ok(N.bits(2.),N.bits(1.))
print('GDN conv independent SiLU zero/tails/sign/transition and wrong-rsqrt refusals pass')
