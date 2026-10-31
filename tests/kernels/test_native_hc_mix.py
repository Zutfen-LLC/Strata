"""Independent pinned source-order controls; no runtime-generated expected captures."""
import sys,unittest,struct,tempfile,subprocess,re,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_hc_mix_numeric as m
import native_hc_numeric as n
class MixTests(unittest.TestCase):
 def test_warp_chunk_order_pinned(self):
  for width in [320,10240]:
   x=[0]*width;w=[0]*width
   for j,v in [(0,100000000.),(8,1.),(256,-100000000.)]:x[j]=n.bits(v);w[j]=0x3f80
   # Native lane 0 cancels its two chunks before XOR with lane 1.
   self.assertEqual(m.dot(x,w,0,width,width==320),0x3f800000)
   serial=n.add(n.add(n.bits(100000000.),0x3f800000),n.bits(-100000000.))
   self.assertEqual(serial,0)
 def test_eight_fma_association(self):
  x=[0]*320;w=[0]*320
  for j,v in enumerate([100000000.,1.,-100000000.,1.]):x[j]=n.bits(v);w[j]=0x3f80
  self.assertEqual(m.dot(x,w,0,320,True),0x3f800000)
 def test_row_and_last_column(self):
  x=[0]*320;w=[0]*640;x[319]=0x3e800000;w[639]=0x4000
  self.assertEqual(m.dot(x,w,0,320,True),0)
  self.assertEqual(m.dot(x,w,1,320,True),0x3f000000)
 def test_stream_sum_pinned(self):
  x=[0]*10240;g=[0x3f800000]*10240
  for c,v in enumerate([100000000.,1.,-100000000.,1.]):x[c*2560]=n.bits(v)
  actual=m.mixed_exact(x,g);self.assertEqual(actual[0],0x3e800000);self.assertTrue(all(v==0 for v in actual[1:]))
 def test_zero_signed_inputs(self):
  self.assertEqual(m.dot([0x80000000]*320,[0]*320,0,320,True),0)
  self.assertEqual(m.activation_bounds(0,'native',True),(0,0))
 def test_exp_bounds_reject_common_corruption(self):
  for backend in ['native','vulkan']:
   for x in [-16.,-1.,0.,1.,16.]:
    v=n.bits(x);bounds=m.exp_bounds(v,backend)
    self.assertTrue(m.inside(n.bits(float(m.decimal_exp(v))),bounds))
    self.assertFalse(m.inside(n.bits(2*float(m.decimal_exp(v))),bounds))
 def test_range_nonfinite_underflow(self):
  for v in [n.bits(16.01),0x7f800000,0x7fc00000,1]:
   with self.assertRaises(ValueError):m.exp_bounds(v,'vulkan')
 def test_material_division_error_rejected(self):
  b=m.quotient_bounds(0x3f800000,0x40000000)
  self.assertTrue(m.inside(0x3f000000,b));self.assertFalse(m.inside(0x3f000020,b))
 def test_dense_pinned_vector(self):
  self.assertEqual(m.dot([0x3e000000]*320,[0x3e00]*320,0,320,True),0x40a00000)
  self.assertEqual(m.dot([0x3e000000]*10240,[0x3e00]*10240,0,10240),0x43200000)
 def test_identical_bad_outputs_rejected_both_backends(self):
  from generate_native_hc_mix import generate
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'fixture').write_bytes(generate('zero'))
   (p/'low.f32le').write_bytes(struct.pack('<320f',*([1.]*320)))
   (p/'mixed.f32le').write_bytes(bytes(2560*4))
   for backend in ['native','vulkan']:
    with self.assertRaisesRegex(ValueError,'down SiLU bound'):m.check(p/'fixture',p,backend)
 def test_native_wrapper_headers_signature_and_host_refusals(self):
  root=Path(__file__).resolve().parents[2];text=(root/'src/kernels/cuda/fused_gr.cu').read_text()
  self.assertIn('#include <stdexcept>',text)
  # Compile the actual declaration and source precondition independently of
  # CUDA. This catches host API/header drift; it does not claim nvcc coverage.
  begin=text.rindex('void fused_gr_normalized_probe(');end=text.index('    GrMulti m{};',begin)
  validation=text[begin:end]+'}\n'
  includes='\n'.join(re.findall(r'^#include <(?:atomic|type_traits|cmath|cstdio|cstdlib|cstring|random|string|vector|stdexcept)>$',text,re.M))
  code=includes+'\n#include "strata/kernels/fused_gr.hpp"\nnamespace strata::kernels {\n'+validation+'}\n'
  code+='\nusing Fn=void(*)(const float*,const uint16_t*,const uint16_t*,const float*,float*,float*,float*,int,void*);\nstatic_assert(std::is_same_v<decltype(&strata::kernels::fused_gr_normalized_probe),Fn>);\nint main(){float x=1;uint16_t w=0;\n for(int variant:{0,1,2,3,4})for(int missing=-1;missing<7;++missing){\n  bool refused=false;\n  try{strata::kernels::fused_gr_normalized_probe(missing==0?nullptr:&x,missing==1?nullptr:&w,missing==2?nullptr:&w,missing==3?nullptr:&x,missing==4?nullptr:&x,missing==5?nullptr:&x,missing==6?nullptr:&x,variant,nullptr);}catch(const std::invalid_argument&){refused=true;}\n  if(refused!=(missing>=0||(variant!=1&&variant!=3)))return 1;\n }return 0;}\n'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'control.cpp').write_text(code)
   subprocess.run([shutil.which('c++') or 'c++','-std=c++17','-I'+str(root/'include'),str(p/'control.cpp'),'-o',str(p/'control')],check=True,capture_output=True)
   subprocess.run([str(p/'control')],check=True)
 def test_bad_shape_version(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fixture';p.write_bytes(b'HCMIX02\0')
   with self.assertRaises(ValueError):m.read_fixture(p)
if __name__=='__main__':unittest.main()
