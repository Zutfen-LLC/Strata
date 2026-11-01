"""Same frozen log1p gate, new LLVM candidate; original Metal STOP is unchanged."""
from pathlib import Path
import struct,json,argparse
import native_gdn_log1p_numeric as G
def check(fixture,result):
 raw=Path(fixture).read_bytes();G.N.need(len(raw)==1036 and raw[:12]==struct.pack('<III',0x31504c47,1,256),'LLVM fixture');root=Path(result);G.N.need((root/'fixture.bin').read_bytes()==raw,'LLVM actual input identity');xs=struct.unpack('<256I',raw[12:]);ys=G.words(root/'result.f32le');worst=0;different=0
 for i,(x,y) in enumerate(zip(xs,ys)):
  G.domain(x);G.q(y);r=G.reference(x);d=abs(y-r) if y>>31==r>>31 else 0xffffffff;worst=max(worst,d);different+=y!=r
  G.N.need(d<=1 and ((x&0x7fffffff)!=0 or y==x),f'LLVM unchanged one-step target at {i}')
 return {'contract':'native-gdn-log1p-dependency-v1','candidate':'LLVM-libc-accurate-nonFMA-RNE','max_adjacent_steps':worst,'different_from_RNE_words':different,'exact_upstream_and_FP64_control_required':True,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);a=p.parse_args();print(json.dumps(check(a.fixture,a.result),indent=2))
