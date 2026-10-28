"""Synthetic math controls for the private retained-row diagnostic; no model data."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('diagnostic', Path(__file__).with_name('portable-model-logit-diagnostic-v1.py'))
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class Diagnostics(unittest.TestCase):
    def test_identity(self):
        r = diag.analyze([1., 2., 3.], [1., 2., 3.])
        self.assertTrue(r['original_gate']['numeric_pass'])
        self.assertEqual(r['diagnostics']['centered_relative_l2'], 0)
        self.assertAlmostEqual(r['diagnostics']['correlation'], 1)

    def test_translation_preserves_probabilities_but_fails_frozen_gate(self):
        r = diag.analyze([-2., 0., 2.], [8., 10., 12.])
        self.assertFalse(r['original_gate']['numeric_pass'])
        d = r['diagnostics']
        self.assertEqual(d['mean_logit_delta'], 10)
        self.assertEqual(d['centered_delta_rms'], 0)
        self.assertEqual(r['original_gate']['max_abs_probability'], 0)
        self.assertAlmostEqual(d['affine_candidate_from_reference']['intercept'], 10)

    def test_affine_and_structured_residual(self):
        d = diag.analyze([-2., 0., 2.], [-3., 1., 5.])['diagnostics']
        self.assertAlmostEqual(d['affine_candidate_from_reference']['slope'], 2)
        self.assertAlmostEqual(d['affine_candidate_from_reference']['residual_rms'], 0)
        d = diag.analyze([-2., 0., 2.], [-2., 1., 2.])['diagnostics']
        self.assertGreater(d['affine_candidate_from_reference']['residual_rms'], 0)

    def test_degenerate_ties_and_one_element(self):
        d = diag.analyze([1., 1.], [2., 2.])['diagnostics']
        self.assertIsNone(d['correlation'])
        self.assertIsNone(d['affine_candidate_from_reference']['slope'])
        self.assertEqual(d['reference_top']['tokens'][0]['id'], 0)
        d = diag.analyze([1.], [2.])['diagnostics']
        self.assertIsNone(d['reference_top']['winning_logit_margin'])

    def test_invalid(self):
        for a, b in (([], []), ([1.], [1., 2.]), ([1.], [math.nan])):
            with self.assertRaises(ValueError):
                diag.analyze(a, b)


if __name__ == '__main__':
    unittest.main()
