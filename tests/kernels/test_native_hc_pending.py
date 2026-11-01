import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools'))
import native_hc_pending_numeric as p
import native_hc_numeric as n
class Pending(unittest.TestCase):
 def args(self):
  # Independent integer/dyadic expected vectors; no control helper generates outputs.
  x=[0x3f800000]*10240;bo=[0x3e800000]*2560;inj=[0]*4;gates=[0x3f800000]*4;exp=[0x3f800000]*4;residual=[0x3fa00000]*10240
  return x,bo,inj,gates,exp,residual
 def test_pinned_identity_both_backends(self):
  for backend in ['native','vulkan']:self.assertTrue(p.pending_checks(*self.args(),backend))
 def test_identically_wrong_backends_refuse(self):
  args=list(self.args());args[-1]=[0x3fa00001]*10240
  for backend in ['native','vulkan']:
   with self.assertRaisesRegex(ValueError,'exact FMA'):p.pending_checks(*args,backend)
 def test_bad_gate_and_exp_refuse(self):
  for at,bits in [(3,0x40000000),(4,0x40000000)]:
   args=list(self.args());args[at]=[bits]*4
   with self.assertRaises(ValueError):p.pending_checks(*args,'vulkan')
 def test_independent_fused_cancellation(self):
  self.assertEqual(n.fma(0x3f800001,0x3f7ffffe,0xbf800000),0xa8800000)
  self.assertNotEqual(n.add(n.mul(0x3f800001,0x3f7ffffe),0xbf800000),0xa8800000)
 def test_underflow_nonfinite_shape_refuse(self):
  for b in [1,0x7f800000,0x7fc00000]:
   args=list(self.args());args[0]=args[0][:];args[0][0]=b
   with self.assertRaises(ValueError):p.pending_checks(*args,'native')
  args=list(self.args());args[1]=[]
  with self.assertRaises(ValueError):p.pending_checks(*args,'native')
 def test_version_and_size_refuse(self):
  for raw in [b'',b'HCPEND1\0',b'HCPEND2\0'+bytes(p.SIZE-8)]:
   with self.assertRaises(ValueError):p.parse(raw)
if __name__=='__main__':unittest.main()
