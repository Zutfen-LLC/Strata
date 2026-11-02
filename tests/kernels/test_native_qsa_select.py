from pathlib import Path
import sys,subprocess,tempfile,struct
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_qsa_select_numeric as Q

def refuse(f,out):
 try:Q.check(f,out)
 except ValueError:return
 raise AssertionError('incorrect source-order score or selected IDs accepted')
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for mode in range(8):
  f=root/f'{mode}.bin';out=root/f'{mode}.out';subprocess.run([sys.argv[1],'--synthetic',str(mode),str(f)],check=True);subprocess.run([sys.argv[1],'--oracle',str(f),str(out)],check=True);Q.check(f,out)
  for name in ('scores.f32le','selected.i32le'):
   p=out/name;original=p.read_bytes();bad=bytearray(original);bad[0]^=1;p.write_bytes(bad);refuse(f,out);p.write_bytes(original)
  # Both backends agreeing on incorrect output is insufficient. Alter an
  # inactive score or cell too: these bytes belong to the original candidate.
  p=out/'scores.f32le';b=p.read_bytes();bad=bytearray(b);bad[-1]^=1;p.write_bytes(bad);refuse(f,out);p.write_bytes(b)
 # Independent pinned controls, not generated from the production helpers.
 assert Q.key(0)==Q.key(0x80000000) and Q.key(0x7fc00001)==0
 f=root/'3.bin';s,ids=Q.expected(f);assert ids==list(range(2051))
 s,ids=Q.expected(root/'4.bin');assert ids==list(range(2050))+[2052]
 s,ids=Q.expected(root/'7.bin');assert ids==list(range(2051)) # 1e9 tail not infinity
 print('eight independent FP32/policy cases pass; altered scores/IDs/inactive bytes reject, including common wrong outputs')
