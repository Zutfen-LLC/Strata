"""Exercise frozen fixture and source-order controls through the real host entrypoints."""
from pathlib import Path
import hashlib,struct,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
import native_head_hc_numeric as c,native_hc_numeric as n,native_hc_mix_numeric as m
tool=Path(sys.argv[1]).resolve();component=Path(sys.argv[2]).resolve()
subprocess.run([str(tool),'--contract'],cwd='/',check=True,capture_output=True);subprocess.run([str(component),'--contract'],cwd='/',check=True,capture_output=True)
with tempfile.TemporaryDirectory(prefix='head-fixtures-') as tmp:
 d=Path(tmp)
 for name,digest in c.PINS.items():
  p=d/(name+'.bin');subprocess.run([str(tool),'--make',name,str(p)],cwd='/',check=True);assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
  subprocess.run([str(tool),'--check',str(p)],cwd='/',check=True);assert subprocess.run([str(tool),'--make',name,str(p)],cwd='/',capture_output=True).returncode
  if name=='cancel':raw=p.read_bytes()
  if name=='dense':dense=p.read_bytes()
  if name=='tails':tails=p.read_bytes()
 bad=d/'bad';bad.write_bytes(b'bad');assert subprocess.run([str(component),'--device','0','--expect-device','unused','--fixture',str(bad),'--output',str(d/'out')],cwd='/',capture_output=True).returncode and not (d/'out').exists()
 tx=struct.unpack_from('<10240I',tails,8);tw=struct.unpack_from('<10240I',tails,8+40960);tt,weighted=n.source_inputs((1,tx,tw,[0]*40960));trs=[n.rsqrt_nearest(n.add(n.nearest(n.rational(v)/2560),n.EPS)) for v in tt];tn=[n.mul(v,trs[i//2560]) for i,v in enumerate(weighted)]
 pre=n.mul(m.dot(tn,[0x3d00]*10240,0,10240),0x3e800000);low=m.activation_bounds(pre,'native',True)
 assert m.sigmoid_class(n.mul(low[0],0x40000000))=='positive_tail'
 assert m.sigmoid_class(n.mul(low[0],0xc0000000))=='overflow_tail'
 dx=struct.unpack_from('<10240I',dense,8);dw=struct.unpack_from('<10240I',dense,8+40960);dt,_=n.source_inputs((1,dx,dw,[0]*40960));assert dt==[0x4b0d61ac,0x4b0fd36b,0x4b03feec,0x4b04852c]
 for stream in range(4):
  serial=0
  for value in dx[stream*2560:(stream+1)*2560]:serial=n.add(serial,n.mul(value,value))
  assert serial!=dt[stream]
 x=struct.unpack_from('<10240I',raw,8);w=struct.unpack_from('<10240I',raw,8+40960);tot,weighted=n.source_inputs((1,x,w,[0]*40960));rs=[n.rsqrt_nearest(n.add(n.nearest(n.rational(v)/2560),n.EPS)) for v in tot];xn=[n.mul(v,rs[i//2560]) for i,v in enumerate(weighted)]
 weights=[0x3f80 if i in (0,8,256) else 0 for i in range(10240)]
 assert m.dot(xn,weights,0,10240)==0x34c013a7
 assert n.add(n.add(xn[0],xn[8]),xn[256])==0
print('Five frozen fixtures, startup/invalid-input contracts and independent cancellation control pass; no GPU')
