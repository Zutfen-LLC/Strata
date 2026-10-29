import array
import copy
import json
from pathlib import Path
import tempfile
import unittest
import executed_logits_v1 as E


class ExecutedLogits(unittest.TestCase):
    def fixture(self,root,size=0,count=2):
        root.mkdir();frames=[]
        for i,token in enumerate([0,2,1][:count+1]):
            name=f'pos-{2+i}.f32';(root/name).write_bytes(array.array('f',[0.,1.,2.] if i==0 else [0.,2.,1.]).tobytes())
            frames.append(dict(position=2+i,token=token,role='final_prompt' if i==0 else 'terminal' if i==count else 'generated',file=name))
        result=dict(schema=1,format='executed_heads_f32le_v1',head_policy='prompt_head_elision',
            chunk_policy='layer_major_v1' if size else 'serial_t1',prompt_chunk_size=size,vocabulary=3,
            prompt_tokens=[0,1,0],output_count=count,stop='max_new',frames=frames)
        (root/'manifest.json').write_text(json.dumps(result));return result

    def test_final_generated_terminal_and_fixed_comparator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root/'a');self.fixture(root/'b',2)
            result=E.compare(root/'a',root/'b')
            self.assertTrue(result['numeric_pass']);self.assertIs(result['limits'],E.C.LIMITS)
            self.assertEqual([f['role'] for f in result['frames']],['final_prompt','generated','terminal'])
            (root/'b'/'pos-4.f32').write_bytes(array.array('f',[100.,2.,1.]).tobytes())
            self.assertFalse(E.compare(root/'a',root/'b')['numeric_pass'])

    def test_zero_and_one_generated_frame(self):
        for count in (0,1):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.fixture(root/'a',count=count)
                manifest,rows=E.load(root/'a',0)
                self.assertEqual(len(rows),count+1)
                self.assertEqual(manifest['frames'][-1]['role'],'terminal' if count else 'final_prompt')

    def test_missing_extra_duplicate_frames_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'a';original=self.fixture(root,2)
            mutations=[]
            for key,value in [('prompt_chunk_size',True),('prompt_chunk_size',3),('chunk_policy','serial_t1'),('head_policy','eager_diagnostic')]:
                mutations.append(dict(original,**{key:value}))
            mutations.extend([dict(original,frames=original['frames'][:-1]),dict(original,frames=original['frames']+[original['frames'][-1]])])
            for key,value in [('position',2),('token',0),('file','pos-2.f32'),('role','final_prompt')]:
                bad=copy.deepcopy(original);bad['frames'][1][key]=value;mutations.append(bad)
            for bad in mutations:
                (root/'manifest.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):E.load(root,2)
            (root/'manifest.json').write_text(json.dumps(original))
            with self.assertRaises(ValueError):E.load(root,0)
            (root/'pos-0.f32').write_bytes(array.array('f',[0.,0.,0.]).tobytes())
            with self.assertRaises(ValueError):E.load(root,2)
            (root/'pos-0.f32').unlink();(root/'pos-3.f32').unlink()
            with self.assertRaises(FileNotFoundError):E.load(root,2)

    def test_incomplete_or_duplicate_manifest_and_cross_policy_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root/'a');manifest=self.fixture(root/'b',2)
            manifest['prompt_tokens'][0]=1;(root/'b'/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):E.compare(root/'a',root/'b')
            (root/'b'/'manifest.json').write_text('{"schema":1,"schema":1}')
            with self.assertRaises(ValueError):E.load(root/'b',2)
            (root/'b'/'manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):E.load(root/'b',2)
