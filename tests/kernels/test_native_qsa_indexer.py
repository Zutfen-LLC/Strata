from pathlib import Path
import json,sys,tempfile,subprocess,struct
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_qsa_indexer_numeric as I

def refused(f,root):
 try:I.check(f,root)
 except ValueError:return
 raise AssertionError('incorrect state/primitive accepted')
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for mode in range(5):
  f=root/f'{mode}.bin';out=root/f'{mode}.out';subprocess.run([sys.argv[1],'--synthetic',str(mode),str(f)],check=True);subprocess.run([sys.argv[1],'--oracle',str(f),str(out)],check=True);report=I.check(f,out);assert report['exact_rounding_state_order']
  # Neither backend agreeing on wrong bytes nor untouched state corruption is
  # sufficient: exact local operation and state rules remain mandatory.
  p=out/'state.u32le';b=p.read_bytes();bad=bytearray(b);bad[512*4]^=1;p.write_bytes(bad);refused(f,out);p.write_bytes(b)
  p=out/'diag.f32le';b=p.read_bytes();bad=bytearray(b);bad[0]^=1;p.write_bytes(bad);refused(f,out);p.write_bytes(b)
  if mode==0:
   p=out/'state.u32le';b=p.read_bytes();bad=bytearray(b);bad[128*4]^=1;p.write_bytes(bad);refused(f,out);p.write_bytes(b)
 print('five deterministic native-indexer local oracles pass; altered conversion/pool/inactive state rejects')
