"""Frozen source controls reject common wrong outputs and exercise both widths."""
from pathlib import Path
import sys,tempfile,subprocess,struct
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import native_qsa_norm_rope_numeric as Q
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for width in (128,256):
  for mode in range(5):
   fixture=root/f'{width}-{mode}.bin';out=root/f'{width}-{mode}';subprocess.run([sys.argv[1],'--synthetic',str(mode),str(width),'1',str(fixture)],check=True);subprocess.run([sys.argv[1],'--oracle',str(fixture),str(out)],check=True);subprocess.run([sys.argv[1],'--check',str(fixture),str(out)],check=True);Q.check(fixture,out)
   data=(out/'normalized.f32le').read_bytes();wrong=bytearray(data);wrong[:4]=struct.pack('<f',1000.);(out/'normalized.f32le').write_bytes(wrong)
   try:Q.check(fixture,out)
   except ValueError:pass
   else:raise AssertionError('identical incorrect normalization accepted')
   (out/'normalized.f32le').write_bytes(data);data=(out/'diag.f32le').read_bytes();wrong=bytearray(data);wrong[66*4:67*4]=struct.pack('<f',.9);(out/'diag.f32le').write_bytes(wrong)
   try:Q.check(fixture,out)
   except ValueError:pass
   else:raise AssertionError('materially wrong cosine accepted')
print('ten deterministic width/stride/zero/sparse/dense/magnitude/boundary source-oracle cases and common normalization/trig errors checked; no GPU')
