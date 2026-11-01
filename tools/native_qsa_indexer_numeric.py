"""Native append exact state/rounding/order controls; existing QSA intrinsic gate.
No new tolerance, final-logit criterion or native-window qualification.
"""
from pathlib import Path
import struct,json,tempfile,argparse,math
import native_hc_numeric as N
import native_qsa_norm_rope_numeric as Q

def load(path):
 b=Path(path).read_bytes();N.need(len(b)>=28,'indexer header');magic,version,cells,base,first,steps,theta=struct.unpack_from('<7I',b)
 N.need(magic==0x3149514e and version==1 and 1<=cells<=256 and 1<=steps<=8 and first+steps<=cells and base%4==0 and base+cells<=256,'indexer T1 text geometry')
 count=512+(cells//4+1)*128+1;N.need(len(b)==28+(steps*128+128+count)*4,'indexer fixture span');w=list(struct.unpack_from('<'+str((len(b)-28)//4)+'I',b,28));raw=w[:steps*128];gamma=w[steps*128:steps*128+128];before=w[steps*128+128:]
 N.need(all(abs(N.value(v))<65520 and math.isfinite(N.value(v)) for v in raw),'indexer finite half-roundable raw')
 return cells,base,first,steps,theta,raw,gamma,before

def half(v):
 # Python's independently implemented IEEE binary16 pack uses RNE, including
 # halfway zero/subnormal cases; F32 expansion is exact.
 return N.bits(struct.unpack('<e',struct.pack('<e',N.value(v)))[0])

def check(fixture,result):
 root=Path(result);N.need((root/'fixture.bin').read_bytes()==Path(fixture).read_bytes(),'indexer actual fixture identity');cells,base,first,steps,theta,raw,gamma,before=load(fixture);count=len(before);states=Q.words(root/'state.u32le',steps*count);diags=Q.words(root/'diag.f32le',steps*386);complete=0;max_rope=0.
 with tempfile.TemporaryDirectory(prefix='qsa-indexer-local-') as tmp:
  tmp=Path(tmp)
  for step in range(steps):
   pos=first+step;d=diags[step*386:(step+1)*386];incoming=[half(v) for v in raw[step*128:(step+1)*128]];N.need(d[:128]==incoming,'indexer exact RNE F16 incoming');after=before.copy()
   if pos%4<3:after[(pos%4)*128:(pos%4+1)*128]=incoming
   if pos==0 or pos%4==3:
    complete+=1;mean=[]
    for c in range(128):
     s=incoming[c] if pos==0 else before[c]
     for j in range(1,4):s=N.add(s,incoming[c] if pos==0 or j==3 else before[j*128+c])
     mean.append(N.fma(N.bits(.25),s,0))
    N.need(d[128:256]==mean,'indexer ordered mean');absolute=0 if pos==0 else base+4*(pos//4);diag=d[256:];norm=[N.mul(N.mul(diag[1],v),gamma[c]) for c,v in enumerate(mean)];rot=norm.copy()
    for c in range(32):rot[c]=N.fma(norm[c],diag[66+c],N.mul(norm[c+32],diag[98+c])^0x80000000);rot[c+32]=N.fma(norm[c],diag[98+c],N.mul(norm[c+32],diag[66+c]))
    # Reuse the frozen QSA reduction/rsqrt/pow/trig/native RoPE checks on the
    # actual pooled mean and candidate row. The transform changes no values.
    row=states[step*count+512+(pos//4)*128:step*count+512+(pos//4+1)*128]
    N.need(row==rot,'indexer exact own rotation association')
    qb=struct.pack('<6I',0x3152514e,1,128,1,128,theta)+struct.pack('<256I',*(mean+gamma))+struct.pack('<i',absolute);(tmp/'fixture.bin').write_bytes(qb)
    for name,values in [('diag.f32le',diag),('normalized.f32le',norm),('rotated.f32le',row)]: (tmp/name).write_bytes(struct.pack('<'+str(len(values))+'I',*values))
    report=Q.check(tmp/'fixture.bin',tmp);max_rope=max(max_rope,report['native_rope_parity_row_max'])
    after[512+(pos//4)*128:512+(pos//4+1)*128]=row
    if pos==0:after[384:512]=row
    else:after[512+(pos//4+1)*128:512+(pos//4+2)*128]=before[384:512];after[-1]=absolute
   else:N.need(all(v==0 for v in d[128:]),'indexer inactive diagnostics initialized')
   actual=states[step*count:(step+1)*count];N.need(actual==after,'indexer exact tail/dead/spare/block/inactive-state preservation');before=actual
 return {'contract':'native-qsa-indexer-T1-v1','exact_rounding_state_order':True,'completed_or_spare_rows':complete,'existing_qsa_intrinsic_controls':True,'native_rope_parity_row_max':max_rope,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result),indent=2))
