"""Primitive bounds only; source-order state/output control is the independent C++ checker.
No blanket cross-backend output tolerance or full-model qualification.
"""
import json,struct,argparse
from fractions import Fraction
from pathlib import Path
import native_hc_numeric as N
import native_hc_mix_numeric as M
COUNTS=(786432,10240,48,48,6144,128)
def words(path,n):
 raw=Path(path).read_bytes();N.need(len(raw)==n*4,'GDN exact result shape');return list(struct.unpack('<'+'I'*n,raw))
def rsqrt_ok(actual,arg):
 N.normal(actual);N.need(0<actual<0x7f800000,'GDN positive rsqrt')
 rn=N.rsqrt_nearest(arg)
 return abs(N.rational(actual)-N.rational(rn))<=Fraction(5,2)*max(N.ulp(actual),N.ulp(rn))
def decay_exp_ok(v,e,backend):
 N.normal(v);N.need(N.value(v)<=0,'GDN nonpositive decay log')
 # Recurrence arithmetic remains normal/zero, so a preserved subnormal exp
 # refuses; legal FTZ zero is admitted only inside the underflow allowance.
 N.normal(e);N.need(0<=e<=0x3f800000,'GDN decay result in [0,1]')
 if M.in_activation_domain(v):return M.inside(e,M.exp_bounds(v,backend))
 return M.sigmoid_exp_allowed(v^0x80000000,e,backend)

def check(fixture,result,backend,norm_argument="norm-arg.f32le"):
 N.need(backend in ('native','vulkan'),'GDN backend');data=Path(fixture).read_bytes();N.need(data[:8]==struct.pack('<II',0x314e4447,1) and len(data)==8+4*sum(COUNTS),'GDN fixed T1 fixture')
 a=[];at=8
 for n in COUNTS:a.append(list(struct.unpack_from('<'+'I'*n,data,at)));at+=n*4
 for row in a:
  for x in row:N.normal(x)
 root=Path(result);ex=words(root/'decay-exp.f32le',48);se=words(root/'sigmoid-exp.f32le',6144);sig=words(root/'sigmoid.f32le',6144);rs=words(root/'rs.f32le',49);args=words(root/norm_argument,48)
 for v,e in zip(a[2],ex):N.need(decay_exp_ok(v,e,backend),'GDN exp(gate) intrinsic/underflow')
 for v,e,g in zip(a[4],se,sig):
  M.sigmoid_controls(v,e,g,backend)
 for r,arg in zip(rs,args+[N.bits(128.)]):N.need(rsqrt_ok(r,arg),'GDN source-argument rsqrt bound')
 return {'contract':'native-gdn-t1-exp-tails-v2','backend':backend,'primitive_intervals':True,'required_separate_exact_local_control':True,'qualified':False,'full_model_parity':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);p.add_argument('--backend',choices=['native','vulkan'],required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result,a.backend),indent=2))
