"""QSA local primitive/merge/gate controls, reusing frozen native exp/sigmoid bounds.
The independent C++ source-order control is also mandatory: native chunk logits
are derived controls, while original launcher partials/attention/gated values are
actual outputs. No final-output epsilon or general native attention parity.
"""
from pathlib import Path
import struct,json,argparse
import native_hc_numeric as N
import native_hc_mix_numeric as M
import native_gdn_t1_numeric as G
CANARY=0x4acaf00d
COUNTS=lambda cells:[((cells+3)//4)*1024]*2+[6144,(cells+3)//4,2051,4,12288,204400,101376,6984,6144,12288,6144]
NAMES=['scratch','chunk','merge','attn','gate','gated']
def words(p,n):
 b=Path(p).read_bytes();N.need(len(b)==4*n,'QSA attention exact span '+str(p));return list(struct.unpack('<'+str(n)+'I',b))
def load(path):
 b=Path(path).read_bytes();N.need(len(b)>=12,'QSA attention fixture header');magic,version,cells=struct.unpack_from('<3I',b);N.need(magic==0x3141514e and version==1 and 1<=cells<=4096,'QSA FP16 T1 fixture geometry');counts=COUNTS(cells);N.need(len(b)==12+4*sum(counts[:7]),'QSA attention fixture span');v=list(struct.unpack_from('<'+str(sum(counts[:7]))+'I',b,12));inputs=[]
 for n in counts[:7]:inputs.append(v[:n]);v=v[n:]
 pos,n,nb,width=inputs[5];N.need(pos<cells and n==pos+1 and nb==n//4 and width==min(n,2051),'native step');N.need(all(i<n and (j==0 or i>inputs[4][j-1]) for j,i in enumerate(inputs[4][:width])),'native selected IDs');N.need(all(p==0xffffffff or p<counts[3] for p in inputs[3]),'native page table bounds');return cells,inputs

def sum32(a):
 for offset in (16,8,4,2,1):a=[N.add(v,a[i^offset]) for i,v in enumerate(a)]
 return a[0]

def quotient(a,b):
 if a&0x7fffffff==0:return a
 q=N.rational(a)/N.rational(b);return N.nearest(abs(q))|(0x80000000 if q<0 else 0)

def check(fixture,result,backend):
 N.need(backend in ('native','vulkan'),'QSA backend contract');root=Path(result);N.need((root/'fixture.bin').read_bytes()==Path(fixture).read_bytes(),'actual QSA input identity');cells,ins=load(fixture);counts=COUNTS(cells);scratch,chunk,merge,attn,gate,gated=[words(root/(name+'.f32le'),n) for name,n in zip(NAMES,counts[7:])];width=ins[5][3];exp_checks=0
 for kv in range(2):
  for c in range(33):
   here=min(64,width-c*64);slot=kv*33+c
   for h in range(12):
    head=kv*12+h;m=scratch[202752+slot*12+h];L=scratch[203544+slot*12+h]
    if here<=0:N.need(m==0xff7fffff and L==0,'inactive chunk metadata');continue
    ea=[];eb=[]
    for lane in range(32):
     for at,output in [(lane,ea),(lane+32,eb)]:
      i=head*2112+c*64+at;e=chunk[50688+i]
      live=at<here and ins[3][ins[4][c*64+at]//4]!=0xffffffff
      if live:
       delta=N.add(chunk[i],m^0x80000000);N.need(G.decay_exp_ok(delta,e,backend),'QSA chunk __expf/exp intrinsic');exp_checks+=1
      else:N.need(e==0 and chunk[i]==0xff7fffff,'masked score/exp identity')
      output.append(e)
    N.need(L==sum32([N.add(a,b) for a,b in zip(ea,eb)]),'QSA exact paired-exp/XOR sum')
 for head in range(24):
  kv,hl=divmod(head,12);M_actual=merge[792+head];L=0
  maxima=[scratch[202752+(kv*33+c)*12+hl] for c in range(33)]
  N.need(N.value(M_actual)==max(map(N.value,maxima)),'QSA merge maximum')
  for c,m in enumerate(maxima):
   i=head*33+c;slot=kv*33+c
   if m==0xff7fffff:N.need(merge[i]==CANARY,'inactive merge control changed');continue
   e=merge[i];delta=N.add(m,M_actual^0x80000000);N.need(G.decay_exp_ok(delta,e,backend),'QSA merge __expf/exp intrinsic');exp_checks+=1;L=N.fma(scratch[203544+slot*12+hl],e,L)
  N.need(merge[816+head]==L,'QSA exact merge L chronology')
  for d in range(256):
   i=head*256+d;acc=merge[840+i];N.need(attn[i]==(quotient(acc,L) if N.value(L)>0 else 0),'QSA native RNE merge division');raw=ins[6][head*512+256+d];M.sigmoid_controls(raw,gate[i],gate[6144+i],backend);N.need(gated[i]==N.mul(attn[i],gate[6144+i]),'QSA exact own-sigmoid gate product')
 return {'contract':'native-QSA-FP16-attention-T1-v1','exp_checks':exp_checks,'frozen_exp_sigmoid_controls':True,'exact_merge_and_own_gate':True,'native_logits_direct_snapshot':backend!='native','source_order_checker_also_required':True,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);p.add_argument('--backend',required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result,a.backend),indent=2))
