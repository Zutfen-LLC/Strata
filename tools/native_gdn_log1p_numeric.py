"""Predeclared dependency gate, not a runtime log1p implementation.

Target <= one adjacent F32 step from correctly rounded log1p. Vulkan's
primitive log allowance alone is weaker; passing it does not imply this gate.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import struct,json,argparse
import native_hc_numeric as N
COUNT=256
FIELDS=('result','compensated','u','log','numerator','denominator')
def q(b):
 a=b&0x7fffffff
 N.need(a<0x7f800000,'log1p nonfinite output')
 if a<0x800000:r=Fraction(a,1<<149)
 else:r=Fraction((a&0x7fffff)|0x800000)*Fraction(2)**((a>>23)-150)
 return -r if b>>31 else r
def nearest(v):
 if not v:return 0
 sign=0x80000000 if v<0 else 0;v=abs(v);seed=N.bits(float(v))
 return sign|min(range(max(0,seed-3),min(0x7f800000,seed+4)),key=lambda b:(abs(q(b)-v),b&1))
def ideal_log(b,oneplus=False,precision=120):
 with localcontext() as c:
  c.prec=precision;v=q(b);d=Decimal(v.numerator)/Decimal(v.denominator)
  return Fraction((1+d).ln() if oneplus else d.ln())
def reference(b):
 if not b&0x7fffffff:return b
 a=nearest(ideal_log(b,True));N.need(a==nearest(ideal_log(b,True,180)),'log1p independent precision convergence');return a
def domain(b):
 N.normal(b);N.need((b&0x7fffffff)==0 or 0x800000<=b<=0x4e000000,'log1p input domain');return b
def ulp_at(v):
 a=abs(v);b=nearest(a)&0x7fffffff
 # Binade-aware spacing from exact value, not from observed result.
 if a<Fraction(1,1<<126):return Fraction(1,1<<149)
 e=(b>>23)-127
 while a<Fraction(2)**e:e-=1
 while a>=Fraction(2)**(e+1):e+=1
 return Fraction(2)**(e-23)
def check_values(xs,values,backend,require_target=True):
 N.need(backend in ('native','vulkan') and len(xs)==COUNT and set(values)==set(FIELDS),'log1p shape/backend')
 N.need(all(len(v)==COUNT for v in values.values()),'log1p output shape')
 worst=0;diagnostic=0
 for i,x in enumerate(xs):
  domain(x);r,c,u,l,n,d=[values[k][i] for k in FIELDS]
  for b in (r,c,u,l,n,d):q(b)
  expected=reference(x);distance=abs(r-expected) if r>>31==expected>>31 else 0xffffffff
  worst=max(worst,distance);diagnostic=max(diagnostic,abs(c-expected) if c>>31==expected>>31 else 0xffffffff)
  if require_target:N.need(distance<=1 and ((x&0x7fffffff)!=0 or r==x),f'log1p target >1 adjacent F32 step at {i}')
  N.need(u==N.add(0x3f800000,x),'log1p rounded u')
  if u==0x3f800000:
   N.need(c==x and l==n==d==0,'log1p tiny branch identity')
  else:
   N.need(d==N.add(u,0xbf800000) and n==N.mul(l,x),'log1p exact multiplication/subtraction')
   exact_log=ideal_log(u)
   if backend=='vulkan' and Fraction(1,2)<=q(u)<=2:
    N.need(abs(q(l)-exact_log)<Fraction(1,1<<21),'log1p Vulkan near-one log absolute bound')
   else:N.need(abs(q(l)-exact_log)<= (3 if backend=='vulkan' else 1)*ulp_at(exact_log),'log1p log intrinsic bound')
   quotient=q(n)/q(d)
   N.need(abs(q(c)-quotient)<=Fraction(5,2)*ulp_at(quotient),'log1p observed division bound')
  if backend=='vulkan':N.need(r==c,'log1p Vulkan compensated target identity')
 return {'contract':'native-gdn-log1p-dependency-v1','backend':backend,'max_target_adjacent_steps':worst,'max_compensated_diagnostic_steps':diagnostic,'target_checked':require_target,'qualified':False,'model_integration':False}
def words(path):
 raw=Path(path).read_bytes();N.need(len(raw)==COUNT*4,'log1p exact field length');return list(struct.unpack('<256I',raw))
def check(fixture,result,backend):
 raw=Path(fixture).read_bytes();N.need(len(raw)==1036 and raw[:12]==struct.pack('<III',0x31504c47,1,COUNT),'log1p fixture identity/shape')
 xs=list(struct.unpack('<256I',raw[12:]));root=Path(result)
 N.need((root/'fixture.bin').read_bytes()==raw,'log1p actual fixture binding')
 return check_values(xs,{k:words(root/(k+'.f32le')) for k in FIELDS},backend)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);p.add_argument('--backend',choices=['native','vulkan'],required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result,a.backend),indent=2))
