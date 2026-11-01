"""Native QSA local controls plus the existing rope_parity row-amplitude gate.
No final-logit epsilon or universal intrinsic-equivalence claim.
"""
from pathlib import Path
from decimal import Decimal,localcontext
from fractions import Fraction
import math,struct,json,argparse
import native_hc_numeric as N
import native_gdn_t1_numeric as G

def load(fixture):
 raw=Path(fixture).read_bytes();N.need(len(raw)>=24,'QSA fixture header');magic,version,cols,rows,stride,theta=struct.unpack_from('<6I',raw)
 N.need(magic==0x3152514e and version==1 and cols in (128,256) and 1<=rows<=24 and stride in (cols,2*cols),'QSA fixed geometry')
 N.need(len(raw)==24+(rows*stride+cols+rows)*4,'QSA exact fixture span');at=24;x=list(struct.unpack_from('<'+str(rows*stride)+'I',raw,at));at+=rows*stride*4;gamma=list(struct.unpack_from('<'+str(cols)+'I',raw,at));at+=cols*4;positions=struct.unpack_from('<'+str(rows)+'i',raw,at);N.need(all(0<=p<=255 for p in positions),'QSA text context domain')
 for r in range(rows):
  for v in x[r*stride:r*stride+cols]:N.normal(v)
 for v in gamma:N.normal(v)
 return cols,rows,stride,theta,x,gamma,positions

def words(p,n):
 raw=Path(p).read_bytes();N.need(len(raw)==n*4,'QSA result span');return list(struct.unpack('<'+str(n)+'I',raw))

def pow_ok(theta,pair,actual):
 # Native --use_fast_math Pow derives from exp2f(y*__log2f(x)). For
 # x in [.5,2], CUDA documents 2^-22 absolute log2 error and 2 ULP exp2f.
 # Include RNE product error; the 0.5 ULP on exp2 is conservative policy slack.
 N.normal(actual);N.need(0<N.value(actual)<=1,'positive bounded frequency')
 if pair==0:return actual==0x3f800000
 with localcontext() as c:
  c.prec=90;x=Decimal(N.rational(theta).numerator)/Decimal(N.rational(theta).denominator);log=x.ln()/Decimal(2).ln();y=Decimal(pair)*log
  rounded=N.bits(float(y));rounding=Decimal(N.ulp(rounded).numerator)/Decimal(N.ulp(rounded).denominator)/2
  delta=Decimal(pair)/(Decimal(2)**22)+rounding
  lo=((y-delta)*Decimal(2).ln()).exp();hi=((y+delta)*Decimal(2).ln()).exp()
  slack=Decimal(N.ulp(actual).numerator)/Decimal(N.ulp(actual).denominator)*Decimal('2.5');a=Decimal(N.rational(actual).numerator)/Decimal(N.rational(actual).denominator)
  return lo-slack<=a<=hi+slack

def check(fixture,directory):
 cols,rows,stride,theta,x,gamma,positions=load(fixture);root=Path(directory);N.need((root/'fixture.bin').read_bytes()==Path(fixture).read_bytes(),'actual QSA fixture binding');diag=words(root/'diag.f32le',rows*130);normalized=words(root/'normalized.f32le',rows*cols);rotated=words(root/'rotated.f32le',rows*cols);max_table=0.
 for r in range(rows):
  a=[N.mul(x[r*stride+i],x[r*stride+i]) if i<cols else 0 for i in range(256)]
  for offset in (16,8,4,2,1):a=[N.add(a[i],a[i^offset]) for i in range(256)]
  b=[a[i*32] if i<8 else 0 for i in range(32)]
  for offset in (16,8,4,2,1):b=[N.add(b[i],b[i^offset]) for i in range(32)]
  mean=N.mul(b[0],N.bits(1./cols));arg=N.add(mean,N.EPS);N.need(diag[r*130]==arg and G.rsqrt_ok(diag[r*130+1],arg),'QSA exact reduction/mean/epsilon and rsqrt contract')
  n=[N.mul(N.mul(diag[r*130+1],x[r*stride+i]),gamma[i]) for i in range(cols)];N.need(normalized[r*cols:(r+1)*cols]==n,'QSA exact normalization multiplication order')
  row=rotated[r*cols:(r+1)*cols];N.need(row[64:]==n[64:],'QSA nonrotated suffix changed')
  amplitude=max(max(abs(N.value(v)) for v in n),1e-30)
  for pair in range(32):
   freq,angle,co,si=(diag[r*130+offset+pair] for offset in (2,34,66,98));N.need(pow_ok(theta,pair,freq),'QSA native-derived fast-Pow interval');N.need(angle==N.mul(N.bits(float(positions[r])),freq),'QSA exact position-frequency multiplication')
   for v in (co,si):N.normal(v)
   t=N.value(angle);limit=2**-21.41 if abs(t)<=math.pi else 3e-3
   N.need(abs(N.value(co)-math.cos(t))<=limit and abs(N.value(si)-math.sin(t))<=limit,'QSA native small-angle trig / existing parity unit-basis gate')
   if positions[r]==0:N.need(co==0x3f800000 and si==0,'QSA zero-position identity')
   left=N.fma(n[pair],co,N.mul(n[pair+32],si)^0x80000000);right=N.fma(n[pair],si,N.mul(n[pair+32],co));N.need(row[pair]==left and row[pair+32]==right,'QSA rounded bs/bc then explicit FMA')
   # Existing native rope_parity check5 uses the ideal FP64 table, with
   # absolute error normalized by max row input amplitude (floor1e-30).
   # That handles cancellation without a relative-to-near-zero criterion.
   ideal=positions[r]*math.pow(1e7,-2.*pair/64);c,s=math.cos(ideal),math.sin(ideal);u,v=N.value(n[pair]),N.value(n[pair+32]);err=max(abs(N.value(row[pair])-(u*c-v*s)),abs(N.value(row[pair+32])-(u*s+v*c)))/amplitude;N.need(err<=3e-3,'existing native RoPE table/parity row-amplitude gate');max_table=max(max_table,err)
 return {'contract':'native-qsa-norm-rope-T1-v1','exact_own_input_operations':True,'rsqrt_and_native_derived_pow':True,'native_rope_parity_row_max':max_table,'qualified':False,'full_model_parity':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result),indent=2))
