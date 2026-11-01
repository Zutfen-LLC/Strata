from pathlib import Path
import re,struct,hashlib,json,sys,subprocess,tempfile,unittest
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'tools'));import native_llvm_log1p_numeric as L
class Test(unittest.TestCase):
 def test_source_constants(self):
  root=repo/'third_party/llvm-libc-log1p';manifest=json.loads((root/'provenance.json').read_text())
  for n,h in manifest['files'].items():self.assertEqual(hashlib.sha256((root/n).read_bytes()).hexdigest(),h)
  shader=(repo/'src/kernels/vulkan/native_llvm_log1p.glsl').read_text();table=(root/'upstream/acosh_float_constants.h').read_text()
  for name in ('R_LOG','C_LOG','LOG_R'):
   original=re.search(name+r'\[64\]\s*=\s*\{(.*?)\};',table,re.S)[1].split(',');expected=[struct.unpack('<Q',struct.pack('<d',float.fromhex(x.strip()) if 'p' in x.lower() else float(x)))[0] for x in original if x.strip()]
   body=re.search('llvm_'+name+r'\[64\].*?\((.*?)\);',shader,re.S)[1];got=[int(b,16)<<32|int(a,16) for a,b in re.findall(r'uvec2\(0x([0-9a-f]+)u,0x([0-9a-f]+)u\)',body)];self.assertEqual(got,expected)
 def test_heldout_and_rejection(self):
  control=Path(sys.argv[1]).resolve()
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for mode in range(4):
    fixture=root/(str(mode)+'.bin');out=root/str(mode);subprocess.run([str(control),'--synthetic',str(mode),str(fixture)],check=True);subprocess.run([str(control),'--oracle',str(fixture),str(out)],check=True);subprocess.run([str(control),'--check',str(fixture),str(out)],check=True);r=L.check(fixture,out);self.assertEqual(r['max_adjacent_steps'],0)
    p=out/'result.f32le';raw=bytearray(p.read_bytes());u=struct.unpack_from('<I',raw)[0];struct.pack_into('<I',raw,0,u+3);p.write_bytes(raw)
    with self.assertRaises(ValueError):L.check(fixture,out)
if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
