import pathlib,sys,unittest,struct,tempfile,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools'))
import extract_native_hc_write as e
from compare_layer0_observer import COUNTS
class Extract(unittest.TestCase):
 def trace(self,root,change=None):
  fields={i:bytes(n*4) for i,n in enumerate(COUNTS)}
  fields[6]=struct.pack('<10i',*range(10));fields[18]=struct.pack('<i',1);fields[9]=b''
  # Deliberately distinct previous GDN BO and actual combined FFN block.
  fields[22]=struct.pack('<2560f',*([99.]*2560));fields[10]=struct.pack('<2560f',*([.25]*2560));fields[3]=struct.pack('<10240f',*([1.]*10240))
  if change:change(fields)
  raw=b'STRL004\0'+b'a'*64+b''.join(struct.pack('<II',k,len(v)//4)+v for k,v in fields.items())+struct.pack('<II',0xffffffff,0)
  p=root/'trace';p.write_bytes(raw);return p,hashlib.sha256(raw).hexdigest()
 def test_actual_fields_and_exclusive_output(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=pathlib.Path(tmp);p,h=self.trace(r);before=p.read_bytes();out=r/'out';receipt=e.extract(p,h,'a'*64,out)
   raw=(out/'fixture.bin').read_bytes();self.assertEqual(len(raw),51224);self.assertEqual(struct.unpack_from('<f',raw,8+40960)[0],.25);self.assertEqual(receipt['fields']['block']['field'],10);self.assertEqual(p.read_bytes(),before)
   with self.assertRaises(FileExistsError):e.extract(p,h,'a'*64,out)
 def test_bad_binding_digest_and_missing_boundary(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=pathlib.Path(tmp);p,h=self.trace(r)
   for a,b in [('0'*64,'a'*64),(h,'b'*64)]:
    with self.assertRaises(ValueError):e.extract(p,a,b,r/'bad')
   p,h=self.trace(r,lambda f:f.update({10:b''}))
   with self.assertRaises(ValueError):e.extract(p,h,'a'*64,r/'bad')
 def test_nonfinite_and_unsupported_domain(self):
  for value in [float('inf'),400.]:
   with tempfile.TemporaryDirectory() as tmp:
    r=pathlib.Path(tmp);p,h=self.trace(r,lambda f:f.update({11:struct.pack('<4f',value,0,0,0)}))
    with self.assertRaises(ValueError):e.extract(p,h,'a'*64,r/'bad')
if __name__=='__main__':unittest.main()
