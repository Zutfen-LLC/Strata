"""Independent exact FP32 source-order scores and cell-policy selection oracle.

No intrinsic tolerance: score operations are FP32 RNE multiplication, FMA and
addition; selection is integer weighted radix in production and cell-sort here.
"""
from pathlib import Path
import struct,json,argparse
import native_hc_numeric as N
CAP=2051
CANARY=0x4acaf00d

def words(path,n):
 b=Path(path).read_bytes();N.need(len(b)==4*n,'QSA exact result span '+str(path));return list(struct.unpack('<'+str(n)+'I',b))
def load(path):
 b=Path(path).read_bytes();N.need(len(b)>=36,'QSA fixture header');h=struct.unpack_from('<9I',b);magic,version,cells,compute,cap,pos,n,nb,width=h
 N.need(magic==0x3153514e and version==1 and 1<=cells<=4096 and compute<=1 and cap==CAP and pos<cells and n==pos+1 and nb==n//4 and width==min(n,CAP),'QSA native T1 geometry')
 blocks=cells//4+2;count=blocks*128+128+512+blocks;N.need(len(b)==36+count*4,'QSA fixture span');w=list(struct.unpack_from('<'+str(count)+'I',b,36));pool=w[:blocks*128];dead=w[blocks*128:blocks*128+128];q=w[blocks*128+128:-blocks];seed=w[-blocks:]
 return compute,n,nb,width,pool,dead,q,seed

def key(b):
 # Independent IEEE word policy; normalizes either signed zero, NaNs lowest.
 a=b&0x7fffffff
 if a>0x7f800000:return 0
 if a==0:b=0
 return (~b)&0xffffffff if b>>31 else b|0x80000000

def expected(f):
 compute,n,nb,width,pool,dead,q,seed=load(f);scores=seed.copy()
 if compute:
  for x in pool[:nb*128]+dead+q:N.normal(x)
  for block in range(nb+1):
   k=dead if block==nb else pool[block*128:(block+1)*128];score=0
   for h in range(4):
    lanes=[]
    for lane in range(32):
     j=lane*4;v=N.mul(k[j],q[h*128+j])
     for c in range(1,4):v=N.fma(k[j+c],q[h*128+j+c],v)
     lanes.append(v)
    for offset in (16,8,4,2,1):lanes=[N.add(v,lanes[i^offset]) for i,v in enumerate(lanes)]
    score=N.add(score,lanes[0] if N.value(lanes[0])>0 else 0)
   if block==nb and n%4:score=N.add(score,N.bits(1e9))
   scores[block]=score
 cells=list(range(n))
 if n>width:cells=sorted(sorted(cells,key=lambda i:(-key(scores[i//4]),i))[:width])
 return scores,cells+[CANARY]*(CAP-len(cells))

def check(fixture,result):
 root=Path(result);N.need((root/'fixture.bin').read_bytes()==Path(fixture).read_bytes(),'QSA actual input identity');scores,ids=expected(fixture)
 N.need(words(root/'scores.f32le',len(scores))==scores,'QSA native score FP32/FMA/XOR/head/tail/inactive order differs')
 N.need(words(root/'selected.i32le',CAP)==ids,'QSA weighted threshold/ties/ascending/inactive selection differs')
 return {'contract':'native-qsa-score-select-T1-v1','exact_source_order_and_cell_policy':True,'competitive_selection':load(fixture)[1]>CAP,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result),indent=2))
