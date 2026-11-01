"""Standalone T1 HC mixing control: exact source dots, bounded activations.
No norm/injection replacement or full-model acceptance is implied.
"""
import argparse,json,math,struct,hashlib
from functools import lru_cache
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

# Domain follows the documented exp/division error policy and F32 normal limits,
# not any fixture extrema. See derive_activation_limit_bits and the contract.
ACTIVATION_LIMIT_BITS=0x42aeac4c
MIN_NORMAL_BITS=0x00800000
MIN_QUOTIENT_GUARD_BITS=0x00800004
MAX_DIV_DEN_BITS=0x7e800000

def derive_activation_limit_bits():
 with localcontext() as ctx:
  ctx.prec=100;u=Decimal(2)**-23;m=Decimal(2)**-126;qguard=m+4*(Decimal(2)**-149)
  def supports(bits):
   a=Decimal.from_float(n.value(bits));r=3+2*a
   alpha=(r*u*(1+u/2)+u/2)/(1-r*u)
   minimum=(-a).exp()*(1-alpha)
   maximum_den=(a.exp()*(1+alpha)+1)*(1+u/2)
   return minimum>=m and maximum_den*qguard<=1
  lo,hi=0,n.bits(128.)
  while lo<hi:
   mid=(lo+hi+1)//2
   if supports(mid):lo=mid
   else:hi=mid-1
  return lo

def in_activation_domain(v):
 n.normal(v);return (v&0x7fffffff)<=ACTIVATION_LIMIT_BITS

@lru_cache(maxsize=32768)
def decimal_exp(v):
 n.need(in_activation_domain(v),'activation outside derived normal exp/div domain')
 with localcontext() as ctx:
  ctx.prec=100;q=n.rational(v);return (Decimal(q.numerator)/Decimal(q.denominator)).exp()

def positive_rational(u):
 # Used to enumerate/refuse near-boundary subnormals, never to accept them.
 return Fraction(u,1<<149) if u<MIN_NORMAL_BITS else n.rational(u)

def positive_ulp(u):return Fraction(1,1<<149) if u<MIN_NORMAL_BITS else n.ulp(u)

@lru_cache(maxsize=32768)
def exp_bounds(v,backend):
 n.need(backend in ['native','vulkan'],'backend contract')
 q=decimal_exp(v);center=n.bits(float(q));x=abs(n.rational(v))
 radius=Fraction(3)+2*x if backend=='vulkan' else Fraction(2+math.floor(Fraction(1173,1000)*x))
 # Local ULP can double across a binade. Enumerate a complete neighborhood;
 # the old fixed +/-40 window could truncate intervals near binades.
 window=math.ceil(2*radius)+4;admitted=[]
 for b in range(max(1,center-window),min(0x7f800000,center+window+1)):
  error=abs(positive_rational(b)-Fraction(q)) if backend=='vulkan' else abs(positive_rational(b)-positive_rational(center))
  if error<=radius*max(positive_ulp(b),positive_ulp(center)):admitted.append(b)
 n.need(admitted and min(admitted)>=MIN_NORMAL_BITS,'exp underflow/subnormal domain')
 n.need(max(admitted)<0x7f800000,'exp overflow/nonfinite domain');return min(admitted),max(admitted)

def quotient_guard(top,den):
 n.normal(top);n.normal(den)
 n.need(0x3f800000<=den<=MAX_DIV_DEN_BITS,'activation denominator outside guaranteed division range')
 if top&0x7fffffff:
  threshold=n.mul(MIN_QUOTIENT_GUARD_BITS,den)
  n.need(n.rational(top&0x7fffffff)>=n.rational(threshold),'activation quotient may underflow')

@lru_cache(maxsize=65536)
def quotient_bounds(top,den):
 quotient_guard(top,den)
 if not top&0x7fffffff:return (top,)*2
 # Preserve local identities, including natural positive saturation when the
 # unchanged rounded denominator becomes one. No clamp/exp shortcut is added.
 if den==0x3f800000:return (top,)*2
 q=n.rational(top)/n.rational(den);sign=0x80000000 if q<0 else 0;c=n.nearest(abs(q));admitted=[]
 for u in range(max(1,c-10),min(0x7f800000,c+11)):
  if abs(positive_rational(u)-abs(q))<=Fraction(5,2)*max(positive_ulp(u),positive_ulp(c)):admitted.append(u|sign)
 n.need(admitted and all((u&0x7fffffff)>=MIN_NORMAL_BITS for u in admitted),'division underflow/subnormal domain')
 return min(admitted,key=n.value),max(admitted,key=n.value)

@lru_cache(maxsize=32768)
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
 n.need(all(in_activation_domain(v) for v in upre),'up activation outside derived domain')
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
 return {'contract':'native-hc-mix-normal-v2','backend':backend,'fixture_sha256':hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),'source_order_and_intrinsic_contract':True,'qualified':False,'full_model_parity':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--directory',required=True);p.add_argument('--backend',choices=['native','vulkan'],required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.directory,a.backend),indent=2))
