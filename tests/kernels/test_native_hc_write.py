import pathlib,sys,unittest,struct
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools'))
import native_hc_write_numeric as w
import native_hc_numeric as n
class Write(unittest.TestCase):
 def args(self):return [0x3f800000]*10240,[0x3e800000]*2560,[0x80000000]*4,[0x3f800000]*4,[0x3fa00000]*10240
 def test_independent_identity_both(self):
  for backend in ['native','vulkan']:self.assertTrue(w.controls(*self.args(),backend))
 def test_identically_wrong_outputs_rejected(self):
  a=list(self.args());a[-1]=[0x3fa00001]*10240
  for backend in ['native','vulkan']:
   with self.assertRaisesRegex(ValueError,'exact local FMA'):w.controls(*a,backend)
 def test_wrong_gate_refused(self):
  a=list(self.args());a[3]=[0x40000000]*4
  with self.assertRaisesRegex(ValueError,'gate intrinsic'):w.controls(*a,'vulkan')
 def test_independent_signed_zero_and_cancellation(self):
  self.assertEqual(n.fma(0x3e800000,0x80000000,0),0)
  self.assertEqual(n.fma(0x3f800001,0x3f7ffffe,0xbf800000),0xa8800000)
  self.assertNotEqual(n.add(n.mul(0x3f800001,0x3f7ffffe),0xbf800000),0xa8800000)
 def test_exact_layout(self):
  r=[n.bits(float(c+1)) for c in range(4) for d in range(2560)];b=[n.bits(float(d%17)/32) for d in range(2560)];out=[n.bits(float(c+1)+float(d%17)/32) for c in range(4) for d in range(2560)]
  self.assertTrue(w.controls(r,b,[0]*4,[0x3f800000]*4,out,'native'))
 def test_nonfinite_domain_and_shape(self):
  for bad in [1,0x7f800000,0x7fc00000]:
   a=list(self.args());a[0]=a[0][:];a[0][0]=bad
   with self.assertRaises(ValueError):w.controls(*a,'native')
  with self.assertRaises(ValueError):w.parse(b'HCWRITE1'+bytes(51216-1))
 def test_derived_gate_domain_edges(self):
  limit=0x42aeac4c
  for backend in ['native','vulkan']:
   for sign in [0,0x80000000]:
    w.gate_bounds(n.mul(limit|sign,0x40800000),backend)
    with self.assertRaises(ValueError):w.gate_bounds(n.mul((limit+1)|sign,0x40800000),backend)
 def test_exact_fixture_roundtrip_and_bad_injection(self):
  r,b,i,_,_=self.args();raw=b'HCWRITE1'+struct.pack('<12804I',*(r+b+i));self.assertEqual(w.parse(raw),(r,b,i))
  raw=raw[:-16]+struct.pack('<4f',400.,0.,0.,0.)
  with self.assertRaises(ValueError):w.parse(raw)
if __name__=='__main__':unittest.main()
