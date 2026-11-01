"""Host producer/consumer contract regression, independent of experimental kernels."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import component_output_contract as C


class ProducerContract(unittest.TestCase):
    def test_deployed_output_names_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            producer = root / 'producer.py'
            # A separate deployed producer is the source of the contract. The
            # consumer must not substitute a different component's output name.
            producer.write_text('''import json, pathlib, sys
out=pathlib.Path(sys.argv[1]); out.mkdir()
fields={'h.f32le':16,'next-history.f32le':48}
(out/'contract.json').write_text(json.dumps({'qualified':False,'fields':fields}))
for name,size in fields.items(): (out/name).write_bytes(bytes(size))
''')
            out = root / 'output'
            subprocess.run([sys.executable, '-I', '-S', '-B', str(producer), str(out)],
                           cwd='/', check=True, capture_output=True)
            contract = out / 'contract.json'
            self.assertEqual(C.require(out, contract), {'h.f32le':16, 'next-history.f32le':48})
            (out/'h.f32le').rename(out/'y.f32le')
            with self.assertRaisesRegex(ValueError, 'h.f32le'): C.require(out, contract)
            (out/'y.f32le').rename(out/'h.f32le')
            (out/'h.f32le').write_bytes(bytes(12))
            with self.assertRaises(ValueError): C.require(out, contract)
            (out/'h.f32le').unlink()
            (out/'h.f32le').symlink_to(out/'next-history.f32le')
            with self.assertRaises(ValueError): C.require(out, contract)

    def test_invalid_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); contract=root/'contract.json'
            for fields in [{}, {'../outside':4}, {'h.f32le':True}, {'h.f32le':0}, {'h.f32le':(32<<20)+1}]:
                contract.write_text(json.dumps({'qualified':False,'fields':fields}))
                with self.assertRaises(ValueError): C.require(root,contract)
            contract.write_text(json.dumps({'qualified':True,'fields':{'h.f32le':4}}))
            with self.assertRaises(ValueError): C.require(root,contract)


if __name__ == '__main__': unittest.main()
