import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('counts',root/'tools/score_remote_route_counts.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Counts(unittest.TestCase):
    def test_cpp_capture_and_score(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);exe=d/'test';out=d/'counts'
            subprocess.run(['g++','-std=c++17','-pthread','-I'+str(root/'include'),str(root/'tests/worker/remote_route_counts_test.cpp'),'-o',str(exe)],check=True)
            subprocess.run([str(exe),str(out)],check=True)
            layers,experts,generated,rows=m.read(out)
            self.assertEqual((layers,experts,generated),(2,16,213))
            score=m.score(rows,{(0,2),(1,1)})
            self.assertEqual((score['selected_entries'],score['nonempty_calls']),(5,3))
            self.assertEqual(score['slice_histogram'],{1:1,2:2})
            raw=out.read_text()
            for bad in (raw.replace('complete=1','complete=0'),raw.replace('calls=3','calls=4'),
                        raw.replace('finish=stop','finish=trace'),raw+'0 1 1 1 1\n'):
                out.write_text(bad)
                with self.assertRaises(ValueError):m.read(out)
            overflow=d/'overflow'
            subprocess.run([str(exe),str(overflow),'overflow'],check=True)
            with self.assertRaises(ValueError):m.read(overflow)
            threaded=d/'threaded';subprocess.run([str(exe),str(threaded),'threads'],check=True)
            self.assertEqual(m.read(threaded)[3][0][2],4000)

if __name__=='__main__':unittest.main()
