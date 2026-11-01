from pathlib import Path
import sys,subprocess,tempfile,struct
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_qsa_attention_numeric as A

def refused(f,d):
 try:A.check(f,d,'vulkan')
 except ValueError:return
 raise AssertionError('wrong common intrinsic/output accepted')
with tempfile.TemporaryDirectory() as t:
 root=Path(t)
 for mode in range(8):
  f=root/(str(mode)+'.bin');out=root/(str(mode)+'.out');subprocess.run([sys.argv[1],'--synthetic',str(mode),str(f)],check=True);subprocess.run([sys.argv[1],'--oracle',str(f),str(out)],check=True);A.check(f,out,'native');A.check(f,out,'vulkan')
  # Independent checker must reject invalid intrinsic controls even when both
  # backends agree. The C++ exact source-order checker remains mandatory too.
  for name,index,value in [('gate',6144,0x40000000),('attn',0,0x3f800000)]:
   p=out/(name+'.f32le');old=p.read_bytes();bad=bytearray(old);struct.pack_into('<I',bad,index*4,value);p.write_bytes(bad);refused(f,out);p.write_bytes(old)
 print('eight local attention fixtures pass frozen primitives/RNE merge/gate; wrong common sigmoid/output rejects')
