from pathlib import Path
import sys,unittest,tempfile,subprocess,struct
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'tools'));import native_gdn_ab_numeric as G
class Tests(unittest.TestCase):
 def test_controls(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);control=Path(sys.argv[1]).resolve()
   for mode in range(4):
    f=root/(str(mode)+'.bin');out=root/str(mode);subprocess.run([str(control),'--synthetic',str(mode),str(f)],check=True);subprocess.run([str(control),'--oracle',str(f),str(out)],check=True)
    for backend in ('native','vulkan'):
     subprocess.run([str(control),'--check',str(f),str(out),backend],check=True);G.check(f,out,backend)
    p=out/'gate.f32le';saved=p.read_bytes();bad=bytearray(saved);u=struct.unpack_from('<I',bad)[0];struct.pack_into('<I',bad,0,u^1);p.write_bytes(bad)
    with self.assertRaises(ValueError):G.check(f,out,'vulkan')
    p.write_bytes(saved);bad=bytearray(f.read_bytes());struct.pack_into('<I',bad,16,8);wrong=root/(str(mode)+'-T8.bin');wrong.write_bytes(bad)
    with self.assertRaises(ValueError):G.check(wrong,out,'vulkan')
    self.assertNotEqual(subprocess.run([str(control),'--check',str(wrong),str(out),'vulkan'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode,0)
 def test_branch_and_rejection(self):
  N=G.N
  G.softplus_controls(N.bits(90),0,N.bits(90),'vulkan')
  with self.assertRaises(ValueError):G.softplus_controls(N.bits(20),0,N.bits(20),'vulkan')
  with self.assertRaises(ValueError):G.softplus_controls(N.bits(90),0x7f800000,N.bits(90),'native')
  G.softplus_controls(N.bits(-140),0,0,'native')
  with self.assertRaises(ValueError):G.softplus_controls(N.bits(-2),0,0,'vulkan')
  with self.assertRaises(ValueError):G.softplus_controls(N.bits(-88),1,1,'vulkan')
  with self.assertRaises(ValueError):G.M.sigmoid_controls(N.bits(-88),0x7f800000,0,'native')
if __name__=='__main__':unittest.main(argv=[sys.argv[0]])
