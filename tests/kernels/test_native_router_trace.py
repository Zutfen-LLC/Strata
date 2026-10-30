import hashlib
import importlib.util
import json
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F

ROOT = pathlib.Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


GEN = module('trace_source', ROOT/'tools/native_router_trace_source.py')
REPLAY = module('trace_replay', ROOT/'tools/replay_native_router_trace.py')


def synthetic_trace(change=False, reciprocal_offset=0):
    # Artificial intrinsic outputs for replay testing, not GPU/native results.
    trace = [0]*26
    trace[0] = 0x41000000
    for i in range(10):
        trace[i+1] = REPLAY.round_rne(F(16-i, 16))
    if change:
        trace[4] += 1
    trace[11] = REPLAY.round_rne(F(1, 1 << 35))
    exp = [trace[11]]*512
    for i in range(10):
        exp[7+i*32] = trace[i+1]
    lanes = []
    for lane in range(32):
        total = 0
        for i in range(16):
            total = REPLAY.add(total, exp[lane+i*32])
        lanes.append(total)
    trace[12] = REPLAY.xor_sum(lanes)
    # RNE reciprocals are convenient fabricated test data only. The product
    # replay accepts captured reciprocal bits and never replaces the intrinsic.
    trace[13] = REPLAY.round_rne(1/REPLAY.exact(trace[12])) + reciprocal_offset
    total = 0
    for i in range(10):
        trace[14+i] = REPLAY.multiply(trace[i+1], trace[13])
        total = REPLAY.add(total, trace[14+i])
    trace[24] = total
    trace[25] = REPLAY.round_rne(1/REPLAY.exact(total))
    ids = list(range(10))*8
    output = [0x3dcccccd]*80
    for rank in range(10):
        ids[20+rank] = 7+rank*32
        output[20+rank] = REPLAY.multiply(trace[14+rank], trace[25])
    return trace, ids, output


class TraceTests(unittest.TestCase):
    def test_source_correspondence(self):
        for kind, path in [('cuda', 'src/kernels/cuda/native_router.cu'), ('vulkan', 'src/kernels/vulkan/native_router.comp')]:
            source = (ROOT/path).read_text()
            generated = GEN.generate(kind, source)
            self.assertEqual(GEN.restore(kind, generated), GEN.cuda_body(source) if kind == 'cuda' else source)
            self.assertIn('trace[25]', generated)
            with self.assertRaises(ValueError):
                GEN.generate(kind, source+'\n')

    def test_exact_rne(self):
        self.assertEqual(REPLAY.round_rne(F(1)+F(1, 1 << 24)), 0x3f800000)
        self.assertEqual(REPLAY.round_rne(F(1)+F(3, 1 << 24)), 0x3f800002)
        self.assertEqual(REPLAY.round_rne(F(1, 1 << 150)), 0)
        self.assertEqual(REPLAY.round_rne(F(3, 1 << 150)), 2)
        self.assertEqual(REPLAY.add(0x3f000000, 0x33000000), 0x3f000000)

    def test_replay_and_mutations(self):
        trace, ids, output = synthetic_trace()
        self.assertTrue(REPLAY.replay(trace, ids, output)['arithmetic_matches'])
        changed, ci, co = synthetic_trace(True)
        self.assertTrue(REPLAY.replay(changed, ci, co)['arithmetic_matches'])
        self.assertTrue(REPLAY.classify(trace, changed)['exp'][3]['different'])
        changed, ci, co = synthetic_trace(reciprocal_offset=1)
        self.assertTrue(REPLAY.replay(changed, ci, co)['arithmetic_matches'])
        category = REPLAY.classify(trace, changed)
        self.assertTrue(category['full_reciprocal']['same_input'])
        self.assertTrue(category['full_reciprocal']['different_output'])
        self.assertFalse(any(x['different'] for x in category['exp']))
        for slot in (12, 13, 14, 24, 25):
            mutated = trace.copy()
            mutated[slot] += 1
            self.assertFalse(REPLAY.replay(mutated, ids, output)['arithmetic_matches'], slot)
        mutated = output.copy()
        mutated[20] ^= 1
        self.assertFalse(REPLAY.replay(trace, ids, mutated)['arithmetic_matches'])
        swapped = ids.copy()
        swapped[20], swapped[21] = swapped[21], swapped[20]
        self.assertFalse(REPLAY.replay(trace, swapped, output)['arithmetic_matches'])
        with self.assertRaises(ValueError):
            REPLAY.replay(trace[:-1], ids, output)
        with self.assertRaises(ValueError):
            REPLAY.replay([0xa5a5a5a5]*26, ids, output)

    def test_completion_and_process_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp)
            with self.assertRaises(OSError):
                REPLAY.completion(path)
            (path/'trace.complete.json').write_text(json.dumps(dict(schema=1, tokens=8, row=2, words=26, fixture_exact=True, trace_guards_ok=True)))
            (path/'capture.rc').write_text('1\n')
            with self.assertRaisesRegex(ValueError, 'process'):
                REPLAY.completion(path)
            (path/'capture.rc').write_text('0\n')
            REPLAY.completion(path)
            with self.assertRaises(OSError):
                REPLAY.completion(path, 'vulkan')
            (path/'module-validation.rc').write_text('1\n')
            with self.assertRaisesRegex(ValueError, 'module'):
                REPLAY.completion(path, 'vulkan')
            (path/'module-validation.rc').write_text('0\n')
            REPLAY.completion(path, 'vulkan')
            (path/'trace.complete.json').write_text(json.dumps(dict(schema=1, tokens=8, row=2, words=26, fixture_exact=True, trace_guards_ok=False)))
            with self.assertRaisesRegex(ValueError, 'guard'):
                REPLAY.completion(path)

    def test_unchanged_output_admission(self):
        _, ids, output = synthetic_trace()
        packed = lambda values: struct.pack('<'+'I'*len(values), *values)
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp)
            (path/'report.json').write_text(json.dumps(dict(tokens=8, experts=512, k=10, qualified=False, sentinels_ok=True, valid_output=True)))
            (path/'ids.i32').write_bytes(packed(ids))
            (path/'weights.f32').write_bytes(packed(output))
            (path/'view-0.bin').write_bytes(b'synthetic input view')
            for name, values, prefix in [('view-1.bin', ids, 5), ('view-2.bin', output, 7)]:
                (path/name).write_bytes(packed([0xa5a5a5a5]*prefix+values+[0xa5a5a5a5]*9))
            hashes = [hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ('weights.f32', 'view-0.bin', 'ids.i32')]
            self.assertTrue(REPLAY.receipt(path, *hashes))
            raw = bytearray((path/'weights.f32').read_bytes())
            raw[0] ^= 1
            (path/'weights.f32').write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'instrumentation attribution refused'):
                REPLAY.receipt(path, *hashes)
            (path/'weights.f32').write_bytes(packed(output))
            raw = bytearray((path/'view-2.bin').read_bytes())
            raw[0] ^= 1
            (path/'view-2.bin').write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'guard'):
                REPLAY.receipt(path, *hashes)


def shader_test(glslc, dis, val):
    inspect = module('shader_inspect', ROOT/'tests/kernels/test_native_router_shader.py').inspect
    source = (ROOT/'src/kernels/vulkan/native_router.comp').read_text()
    with tempfile.TemporaryDirectory() as temporary:
        path = pathlib.Path(temporary)
        assemblies = []
        for name, text in [('baseline', source), ('trace', GEN.generate('vulkan', source))]:
            comp, spv = path/(name+'.comp'), path/(name+'.spv')
            comp.write_text(text)
            subprocess.run([glslc, '-fshader-stage=compute', '--target-env=vulkan1.2', '-O', str(comp), '-o', str(spv)], check=True)
            subprocess.run([val, '--target-env', 'vulkan1.2', str(spv)], check=True)
            assembly = subprocess.check_output([dis, str(spv)], text=True)
            inspect(assembly)
            assemblies.append(assembly)
        for opcode in ('OpFAdd', 'OpFSub', 'OpFMul', 'OpFDiv', 'NoContraction'):
            assert assemblies[0].count(opcode) == assemblies[1].count(opcode), opcode
        assert re.search(r'OpTypeArray %float %\w+', assemblies[1])
        assert 'Offset 4' in assemblies[1]
    print('trace/baseline optimized shader arithmetic counts and controls match; no GPU execution')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--shader':
        shader_test(*sys.argv[2:])
    else:
        unittest.main()
