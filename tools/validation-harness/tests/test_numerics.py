import array
import math
import pathlib
import sys
import tempfile
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import compare_logits


class NumericalTests(unittest.TestCase):
    def test_equal_and_small_perturbation(self):
        a=[-3.,0.,1.,5.]
        self.assertTrue(compare_logits.compare(a,a)['numeric_pass'])
        row=compare_logits.compare(a,[-3.,0.,1.00001,5.])
        self.assertTrue(row['numeric_pass']);self.assertTrue(row['top1_equal'])
        self.assertGreater(row['max_abs_logit'],0)

    def test_large_change_and_nonfinite_are_refused(self):
        self.assertFalse(compare_logits.compare([0.,1.],[1.,0.])['numeric_pass'])
        for value in (math.nan,math.inf,-math.inf):
            with self.assertRaises(ValueError):compare_logits.compare([0.,1.],[0.,value])
        with self.assertRaises(ValueError):compare_logits.compare([1.],[1.,2.])

    def test_raw_shape_and_finiteness(self):
        with tempfile.TemporaryDirectory() as directory:
            path=pathlib.Path(directory)/'row';path.write_bytes(b'x')
            with self.assertRaises(ValueError):compare_logits.read(path)
            values=array.array('f',[0.,1.])
            if sys.byteorder!='little':values.byteswap()
            path.write_bytes(values.tobytes());self.assertEqual(list(compare_logits.read(path)),[0.,1.])


if __name__=='__main__':unittest.main()
