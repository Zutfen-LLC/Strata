import importlib.util
from pathlib import Path
import sys
import json
import struct
import subprocess
import tempfile
import unittest
import numpy as np

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('score',root/'tools/score_remote_prompt_placement.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Score(unittest.TestCase):
    def test_cli_on_sealed_trace(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trace';plan=Path(d)/'plan';plan.write_text('0 1\n0 2\n')
            with path.open('wb') as f:
                f.write(b'STRFT01\0')
                def rec(kind,window,layer,tb,rows,cols,values):
                    payload=struct.pack('<'+('i' if kind in (0,1,4,9,10) else 'f')*len(values),*values)
                    f.write(struct.pack('<8q',kind,window,0,layer,tb,rows,cols,len(payload)));f.write(payload)
                rec(0,-1,-1,0,1,2,[10,20]);rec(10,-1,-1,0,1,9,[1,1,1,1,1,1,4,2,1]);rec(1,0,-1,0,1,2,[10,20])
                rec(2,0,0,0,2,1,[0,0]);rec(3,0,0,0,2,4,[0]*8)
                rec(4,0,0,0,2,2,[1,2,1,3]);rec(5,0,0,0,2,2,[.5]*4);rec(6,0,0,0,2,1,[0,0])
                rec(9,0,0,0,1,2,[3,3]);rec(9,0,0,1,1,2,[3,3])
                rec(12,0,0,0,1,1,[0]);rec(12,0,0,1,1,1,[0])
                rec(7,0,0,1,1,1,[0]);rec(8,0,0,1,1,1,[0]);rec(11,0,-1,0,1,2,[0,0])
            result=subprocess.run([sys.executable,str(root/'tools/score_remote_prompt_placement.py'),str(path),str(plan)],
                                  check=True,capture_output=True,text=True)
            score=json.loads(result.stdout)[str(plan)]
            self.assertEqual((score['selected_entries'],score['nonempty_calls']), (3,2))
    def test_window_slice_and_no_empty_calls(self):
        ids=np.array([[1,2,3],[4,1,5],[1,3,6]],dtype=np.int32)
        records={(0,0,7,4,0,3,3):ids,
                 (0,0,7,9,0,2,3):np.array([[3,3,0],[3,3,3]]),
                 (0,0,7,9,2,1,3):np.array([[3,3,3]])}
        r=m.score(records,{(7,1),(7,2)})
        self.assertEqual((r['selected_entries'],r['nonempty_calls']), (4,2))
        self.assertEqual(r['slice_histogram'],{1:1,3:1})
        self.assertEqual(r['entries_per_call'],2)
        with self.assertRaises(ValueError):m.score(records,{(7,3)})
        self.assertEqual(m.score(records,{(7,99)})['nonempty_calls'],0)
    def test_assignments_reject_duplicates_and_range(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_text('0 1\n1 2 # comment\n')
            self.assertEqual(m.assignments(p,2,3),{(0,1),(1,2)})
            for content in ('0 1\n0 1\n','2 1\n','-1 2\n','1 3\n',''):
                p.write_text(content)
                with self.assertRaises(ValueError):m.assignments(p,2,3)
    def test_bad_slice_fails(self):
        r={(0,0,0,4,0,1,2):np.array([[1,2]]),(0,0,0,9,1,1,2):np.array([[3,3]])}
        with self.assertRaises(ValueError):m.score(r,{(0,1)})

if __name__=='__main__':unittest.main()
