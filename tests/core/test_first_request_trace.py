"""Host-only fixture and stop-placement checks; GPU execution is a separate check."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import os

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('compare_trace', ROOT / 'tools/compare_first_request_trace.py')
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def fixture(path, delta=0.0, route=False, window_ids=(10, 20), truncate=False):
    with open(path, 'wb') as f:
        f.write(b'STRFT01\0')
        def rec(kind, window, layer, rows, cols, values, tb=0):
            integer = kind in (0, 1, 4, 9, 10)
            payload = struct.pack('<'+('i' if integer else 'f')*len(values), *values)
            f.write(struct.pack('<8q', kind, window, 0, layer, tb, rows, cols, len(payload)))
            f.write(payload)
        rec(0, -1, -1, 1, 2, [10, 20])
        rec(10, -1, -1, 1, 9, [256, 64, 2, 1024, 2, 2, 2, 2, 2])
        rec(1, 0, -1, 1, 2, window_ids)
        for layer in range(2):
            for kind in (2, 3, 4, 5, 6):
                values = [1, 2, 3, 4]
                if kind == 6:
                    values[-1] += delta*(10**layer)
                if kind == 4 and route and layer == 1:
                    values[-1] = 5
                rec(kind, 0, layer, 2, 2, values)
            rec(9, 0, layer, 2, 2, [0, 3, 4, 0])
            rec(12, 0, layer, 2, 2, [1, 2, 3, 4])
            rec(7, 0, layer, 1, 4, [1, 2, 3, 4], tb=1)
            rec(8, 0, layer, 1, 2, [1, 2], tb=1)
        if not truncate:
            rec(11, 0, -1, 1, 2, [0.1, 0.2+delta*100])


class TraceTest(unittest.TestCase):
    def test_writer_disabled_active_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d)/'writer'
            subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I'+str(ROOT/'include'),
                            str(ROOT/'tests/core/first_request_trace_test.cpp'), '-o', str(binary)], check=True)
            env = dict(os.environ)
            env.pop('STRATA_FIRST_REQUEST_TRACE', None)
            target = str(Path(d)/'trace')
            subprocess.run([str(binary), target], env=env, check=True)
            env['STRATA_FIRST_REQUEST_TRACE'] = target
            subprocess.run([str(binary), target], env=env, check=True)

    def test_progression_and_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            fixture(a)
            fixture(b, delta=0.01, route=True)
            report = trace.compare(trace.read(a), trace.read(b))
            self.assertEqual(report['earliest_difference']['stage'], 'post_moe')
            self.assertEqual(report['earliest_difference']['layer'], 0)
            self.assertEqual(report['first_routing_difference']['layer'], 1)
            self.assertGreater(report['maximum_by_stage']['post_moe']['maxabs'], .09)
            self.assertIsNone(trace.compare(trace.read(a), trace.read(a))['earliest_difference'])
            fixture(b, window_ids=(10, 21))
            with self.assertRaisesRegex(ValueError, 'IDs differ'):
                trace.compare(trace.read(a), trace.read(b))
            fixture(b, truncate=True)
            with self.assertRaisesRegex(ValueError, 'unsealed'):
                trace.read(b)
            fixture(b)
            b.write_bytes(b.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'truncated'):
                trace.read(b)

    def test_source_stop_alignment(self):
        source = (ROOT/'src/program/generate.cpp').read_text()
        dump = source.index('// Diagnostics (as in generate)')
        stop = source.index('strata::core::first_request_trace().finish(', dump)
        commit = source.index('if (!ver.commit(a + 1, err))', stop)
        self.assertLess(source.index('if (!ver.run(T, window.data()', dump-2000), dump)
        block = source[stop:commit]
        self.assertIn('finish = "trace";', block)
        self.assertIn('break;', block)
        self.assertNotIn('max_new=', block)
        # The original window execution and request binding stay ahead of the stop.
        self.assertIn('const int T = (int) std::min<int64_t>(S, b - q);', source)
        verify = (ROOT/'src/core/verify.cpp').read_text()
        copy = verify.index('first_trace_copy(l, 4')
        self.assertLess(verify.index('if (remote_opt_) remote_opt_->combine('), copy)
        self.assertLess(copy, verify.index('stamp(l, 24, grp);'))
        self.assertLess(verify.index('const cudaError_t se = cudaStreamSynchronize(cs_);',
                                     verify.index('bool Verifier::run(')),
                        verify.index('if (!first_trace_flush(T, err))'))


if __name__ == '__main__':
    unittest.main()
