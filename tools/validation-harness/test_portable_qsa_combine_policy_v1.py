import argparse
import unittest
from unittest import mock

import portable_quality_v1 as P
import portable_resident_v1 as R


class CombinePolicy(unittest.TestCase):
    def test_flags_and_strict_boolean_binding(self):
        parser=argparse.ArgumentParser()
        P.residency_arguments(parser)
        valid=[P.QSA_COMBINE_FLAG,'--qsa-production','--qsa-reuse-attention']
        flags=P.residency_flags(parser.parse_args(valid))
        self.assertTrue(P.combine_argv(flags))
        with self.assertRaises(ValueError):P.residency_flags(parser.parse_args([P.QSA_COMBINE_FLAG]))
        for command in (flags+[P.QSA_COMBINE_FLAG], [P.QSA_COMBINE_FLAG,'--qsa-production'],
                        [P.QSA_COMBINE_FLAG,'--qsa-reuse-attention']):
            with self.assertRaises(ValueError):P.combine_argv(command)
        for enabled in (False,True):
            P.combine_binding({'qsa_device_combine':enabled},enabled)
            for value in (1,0,'true',None,not enabled):
                with self.assertRaises(ValueError):P.combine_binding({'qsa_device_combine':value},enabled)
        P.combine_binding({},False)
        with self.assertRaises(ValueError):P.combine_binding({},True)

    def test_resident_rejects_missing_or_foreign_runtime_policy(self):
        for enabled,value in ((True,None),(True,False),(True,1),(False,True)):
            runner=object.__new__(R.Resident)
            runner.child=mock.Mock();runner.request_seconds=1
            runner.head_policy='eager_diagnostic';runner.prompt_chunk_size=0
            runner.qsa_device_combine=enabled
            done=dict(done=1,prompt_count=1,output_count=0)
            if value is not None:done['qsa_device_combine']=value
            runner.row=mock.Mock(side_effect=[dict(position=0,token=5,generated=False,greedy_next=7),
                                            dict(stop='max_new',position=1),done])
            with self.assertRaisesRegex(ValueError,'QSA device combine policy mismatch'):
                runner.request([5],0,1)
