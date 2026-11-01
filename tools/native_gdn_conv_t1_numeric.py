"""Existing primitive contracts; no blanket convolution/h output epsilon."""
from pathlib import Path
from fractions import Fraction
import struct,json,argparse
import native_hc_numeric as N
import native_hc_mix_numeric as M
import native_gdn_t1_numeric as G
COUNTS=(30720,10240,40960)
def words(path,count):
 raw=Path(path).read_bytes();N.need(len(raw)==count*4,'GDN conv exact shape');return list(struct.unpack('<'+'I'*count,raw))
def silu_controls(v,e,a,backend):
 N.normal(v);N.normal(a);kind=M.sigmoid_class(v)
 N.need(M.sigmoid_exp_allowed(v,e,backend),'GDN conv SiLU exp intrinsic')
 if kind=='normal':N.need(M.inside(a,M.quotient_bounds(v,N.add(0x3f800000,e))),'GDN conv own-exp divide')
 elif kind=='positive_tail':N.need(a==v,'GDN conv denominator-one original identity')
 else:N.need(a==0x80000000,'GDN conv original negative/Inf signed-zero identity')
 return True

def check(fixture,result,backend):
 raw=Path(fixture).read_bytes();N.need(raw[:8]==struct.pack('<II',0x31564347,1) and len(raw)==327688,'GDN conv fixed fixture')
 for x in struct.unpack('<'+'I'*sum(COUNTS),raw[8:]):N.normal(x)
 root=Path(result);pre=words(root/'pre.f32le',10240);ex=words(root/'exp.f32le',10240);silu=words(root/'silu.f32le',10240)
 for p,e,a in zip(pre,ex,silu):silu_controls(p,e,a,backend)
 rs=words(root/'rs.f32le',32);args=words(root/'norm-arg.f32le',32)
 for r,a in zip(rs,args):N.need(G.rsqrt_ok(r,a),'GDN conv exact-argument rsqrt bound')
 return {'contract':'native-gdn-conv-t1-v1','backend':backend,'pointwise_intrinsics':True,'separate_exact_local_control_required':True,'qualified':False,'full_model_parity':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);p.add_argument('--backend',choices=['native','vulkan'],required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result,a.backend),indent=2))
