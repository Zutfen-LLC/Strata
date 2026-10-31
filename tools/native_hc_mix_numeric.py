"""Standalone T1 HC mixing control: exact source dots, bounded activations.
No norm/injection replacement or full-model acceptance is implied.
"""
import argparse,json,math,struct,hashlib
from pathlib import Path
from decimal import Decimal,localcontext
from fractions import Fraction
import native_hc_numeric as n
D,R,H=10240,320,2560
NAMES=('low-pre','low','low-exp','up-pre','gates','up-exp','mixed')
def read_fixture(path):
 raw=Path(path).read_bytes();n.need(len(raw)==13148168 and raw[:8]==b'HCMIX01\0','fixed T1 fixture/version')
 x=list(struct.unpack_from('<10240I',raw,8));down=list(struct.unpack_from('<3276800H',raw,40968));up=list(struct.unpack_from('<3276800H',raw,6594568))
 for v in x:n.normal(v)
 for v in down+up:n.normal(v<<16)
 return x,down,up

def dot(x,w,row,width,up=False):
 lanes=[0]*32;start=row*width
 for lane in range(32):
  first=True
  for chunk in range(lane,width//8,32):
   weights=w[start+chunk*8:start+chunk*8+8]
   # An all +0 BF16 chunk yields +0 with RNE fma and a +0 seed,
   # including signed-zero inputs. Domain rejects underflow elsewhere.
   d=0
   if any(weights):
    for k,v in enumerate(weights):d=n.fma(v<<16,x[chunk*8+k],d)
   if up and first:lanes[lane]=d
   else:lanes[lane]=n.add(lanes[lane],d)
   first=False
 for shift in (16,8,4,2,1):
  old=lanes[:];lanes=[n.add(v,old[i^shift]) for i,v in enumerate(old)]
 return lanes[0]

def decimal_exp(v):
 n.need(abs(n.value(v))<=16,'activation range [-16,16]')
 with localcontext() as ctx:
  ctx.prec=100;q=n.rational(v);return (Decimal(q.numerator)/Decimal(q.denominator)).exp()

def exp_bounds(v,backend):
 q=decimal_exp(v);center=n.bits(float(q));x=abs(n.rational(v))
 # GLSL absolute error measured in local ULP against exact mathematical exp;
 # CUDA number is observed error versus correctly rounded RNE, not a guarantee.
 radius=Fraction(3)+2*x if backend=='vulkan' else Fraction(2+math.floor(Fraction(1173,1000)*x))
 admitted=[]
 for b in range(center-40,center+41):
  error=abs(Fraction(Decimal.from_float(n.value(b)))-Fraction(q)) if backend=='vulkan' else abs(n.rational(b)-n.rational(center))
  if error<=radius*max(n.ulp(b),n.ulp(center)):admitted.append(b)
 n.need(admitted,'empty exp interval');return min(admitted),max(admitted)

def quotient_bounds(top,den):
 q=n.rational(top)/n.rational(den)
 if not q:return (top&0x80000000,)*2
 sign=0x80000000 if q<0 else 0;c=n.nearest(abs(q));admitted=[]
 for u in range(max(0x800000,c-4),min(0x7f800000,c+5)):
  if abs(n.rational(u)-abs(q))<=Fraction(5,2)*max(n.ulp(u),n.ulp(c)):admitted.append(u|sign)
 n.need(admitted,'empty division interval');return min(admitted,key=n.value),max(admitted,key=n.value)

def activation_bounds(v,backend,silu):
 lo,hi=exp_bounds(v^0x80000000,backend);vals=[];top=v if silu else 0x3f800000
 for e in (lo,hi):vals.extend(quotient_bounds(top,n.add(0x3f800000,e)))
 return min(vals,key=n.value),max(vals,key=n.value)

def inside(v,bounds):
 n.normal(v)
 if (bounds[0]&0x7fffffff)==0 and bounds[0]==bounds[1]:return v==bounds[0]
 return n.value(bounds[0])<=n.value(v)<=n.value(bounds[1])

def mixed_exact(x,g):
 out=[]
 for d in range(H):
  s=0
  for c in range(4):s=n.add(s,n.mul(x[c*H+d],g[c*H+d]))
  out.append(n.mul(s,0x3e800000))
 return out

def check(fixture,directory,backend):
 x,down,up=read_fixture(fixture);directory=Path(directory)
 def words(name,count):
  raw=(directory/(name+'.f32le')).read_bytes();n.need(len(raw)==count*4,'output shape '+name);v=list(struct.unpack('<'+str(count)+'I',raw))
  for b in v:n.normal(b)
  return v
 low=words('low',R);mixed=words('mixed',H)
 pre=[n.mul(dot(x,down,row,D),0x3e800000) for row in range(R)]
 n.need(all(inside(v,activation_bounds(p,backend,True)) for p,v in zip(pre,low)),'down SiLU bound')
 upre=[dot(low,up,row,R,True) for row in range(D)]
 n.need(all(abs(n.value(v))<=16 for v in upre),'up activation range')
 if backend=='vulkan':
  n.need(words('low-pre',R)==pre and words('up-pre',D)==upre,'exact own-input source projection')
  for count,ps,act,en,silu in [(R,pre,low,'low-exp',True),(D,upre,words('gates',D),'up-exp',False)]:
   es=words(en,count)
   for p,a,e in zip(ps,act,es):
    n.need(inside(e,exp_bounds(p^0x80000000,backend)),'exp bound')
    n.need(inside(a,quotient_bounds(p if silu else 0x3f800000,n.add(0x3f800000,e))),'own-exp activation association')
  n.need(mixed==mixed_exact(x,words('gates',D)),'exact own-gate mixed association')
 else:
  # Actual existing native kernels expose low/mixed only. Bound gates from
  # actual native low; never attribute up with the other backend's input.
  gb=[activation_bounds(v,backend,False) for v in upre]
  for d,v in enumerate(mixed):
   low_sum=high_sum=0
   for c in range(4):
    at=c*H+d;terms=[n.mul(x[at],b) for b in gb[at]]
    low_sum=n.add(low_sum,min(terms,key=n.value));high_sum=n.add(high_sum,max(terms,key=n.value))
   n.need(inside(v,(n.mul(low_sum,0x3e800000),n.mul(high_sum,0x3e800000))),'native mixed activation propagation')
 return {'backend':backend,'fixture_sha256':hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),'source_order_and_intrinsic_contract':True,'qualified':False,'full_model_parity':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--directory',required=True);p.add_argument('--backend',choices=['native','vulkan'],required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.directory,a.backend),indent=2))
