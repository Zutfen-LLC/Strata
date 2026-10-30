import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('compare',Path(__file__).resolve().parents[2]/'tools/compare_layer0_observer.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def image(change=None,missing=(),binding=b'a'*64,reduced=False,legacy=False):
    raw=(b'STRL001\0' if legacy else b'STRL002\0')+binding
    for i,n in enumerate(c.COUNTS[:-1] if legacy else c.COUNTS):
        if i in missing:raw+=struct.pack('<II',i,0);continue
        data=bytes(n*4)
        if i==6:data=struct.pack('<10i',*range(10))
        if i==15:data=struct.pack('<i',248045)
        if i==18:data=struct.pack('<i',int(reduced))
        if change and i==change[0]:data=change[1]+data[4:]
        raw+=struct.pack('<II',i,n)+data
    return raw+struct.pack('<II',0xffffffff,0)

class Test(unittest.TestCase):
    def test_protocol(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'native',Path(t)/'portable';a.write_bytes(image(missing=(16,12,13)));b.write_bytes(image())
            r=c.compare(a,b);self.assertTrue(r['exact_observed_boundaries']);self.assertFalse(r['stages'][-1]['available'])
            b.write_bytes(image(change=(5,struct.pack('<f',1))))
            r=c.compare(a,b);self.assertEqual(r['first_difference'],'router_scores');self.assertEqual(next(s for s in r['stages'] if s['boundary']=='router_scores')['attribution'],'local_boundary_candidate')
            b.write_bytes(image(change=(4,struct.pack('<f',1))));r=c.compare(a,b)
            self.assertEqual(r['first_difference'],'moe_input')
            for bad in [image()[:-8],image(missing=(5,)),image(binding=b'b'*64),image(change=(3,struct.pack('<I',0x7fc00000))),b'bad']:
                b.write_bytes(bad)
                with self.assertRaises((ValueError,UnicodeDecodeError)):c.compare(a,b)
    def test_route_association_and_bounds(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a',Path(t)/'b';a.write_bytes(image());b.write_bytes(image(change=(6,struct.pack('<i',11))))
            result=c.compare(a,b)
            self.assertEqual(result['first_difference'],'router_ids')
            rows=next(s for s in result['stages'] if s['boundary']=='expert_rows')
            self.assertFalse(rows['comparable']);self.assertIsNone(rows['different_words'])
            raw=image();bad_count=raw[:76]+struct.pack('<I',c.COUNTS[0]+1)+raw[80:];b.write_bytes(bad_count)
            with self.assertRaises(ValueError):c.read(b)
            b.write_bytes(raw+struct.pack('<II',0,0))
            with self.assertRaises(ValueError):c.read(b)
            with b.open('wb') as f:f.truncate(c.LIMIT+1)
            with self.assertRaises(ValueError):c.read(b)
    def test_reduced_helper_unavailability(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'native',Path(t)/'portable'
            a.write_bytes(image(missing=(9,16),reduced=True));b.write_bytes(image())
            r=c.compare(a,b)
            self.assertTrue(r['native_helper_reduction']);self.assertFalse(r['complete_unweighted_rows_available'])
            self.assertTrue(r['exact_observed_boundaries']);self.assertFalse(r['combine_inputs_observed_and_equal'])
            b.write_bytes(image(change=(10,struct.pack('<f',1))))
            r=c.compare(a,b);self.assertEqual(r['first_difference'],'combined_output')
            combine=next(s for s in r['stages'] if s['boundary']=='combined_output')
            self.assertEqual(combine['attribution'],'input_boundary_unavailable')
            self.assertIn('expert_rows',combine['unavailable_inputs'])
            for invalid in (image(missing=(9,)),image(reduced=True),image(missing=(18,))):
                a.write_bytes(invalid)
                with self.assertRaises(ValueError):c.read(a)
            a.write_bytes(image(legacy=True));b.write_bytes(image())
            self.assertTrue(c.compare(a,b)['exact_observed_boundaries'])
    def test_downstream_attribution(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a',Path(t)/'b';a.write_bytes(image());raw=image(change=(4,struct.pack('<f',1)))
            # Change both normalized input and router scores; equal downstream operands are not inferred.
            offset=72+sum(8+n*4 for n in c.COUNTS[:5])+8
            raw=raw[:offset]+struct.pack('<f',2)+raw[offset+4:];b.write_bytes(raw)
            r=c.compare(a,b);scores=next(s for s in r['stages'] if s['boundary']=='router_scores')
            self.assertEqual(scores['attribution'],'upstream_difference_present')
if __name__=='__main__':unittest.main()
