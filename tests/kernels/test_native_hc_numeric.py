"""Pinned independent C++ vectors and predicate tests using explicit mock receipts."""
import hashlib, json, math, struct, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import native_hc_numeric as n
from compare_native_hc_numeric import evaluate


def fixture(seed=17, cancellation=False):
    x = [0] * n.D
    w = [0x3f800000 if cancellation else 0x3f940000] * n.D
    wi = [0] * (4 * n.D)
    state = seed
    for c in range(4):
        for k in range(4):
            state = (1664525 * state + 1013904223) & 0xffffffff
            x[c * n.H + k] = 0x3e000000 + (state & 0x1ffffff)
    if cancellation:
        x = [0] * n.D
        x[0], x[1] = n.bits(1.25), n.bits(-1.25)
        wi[0] = wi[1] = 0x3f800000
    else:
        for row in range(4):
            for c in range(4):
                for k in range(4):
                    wi[row * n.D + c * n.H + k] = n.bits((row + c + k + 1) / 32)
    raw = struct.pack('<8I', 0x3143484e, 1, n.H, 4, 1, n.EPS, 30, 0)
    raw += struct.pack('<10240I', *x) + struct.pack('<10240I', *w)
    raw += struct.pack('<40960H', *[u >> 16 for u in wi])
    return raw


def write_results(root, data):
    f = n.parse_fixture(data)
    sums, weighted = n.source_inputs(f)
    rs = [n.rsqrt_nearest(n.add(n.nearest(n.rational(s) / n.H), n.EPS)) for s in sums]
    for backend in ('native', 'vulkan'):
        d = root / backend
        d.mkdir()
        (d / 'fixture.bin').write_bytes(data)
        values = rs if backend == 'native' else [u + 1 for u in rs]
        xn, inject = n.downstream(f, weighted, values)
        for name, words in [('rs', values), ('normalized', xn), ('inject', inject)]:
            (d / (name + '.f32le')).write_bytes(struct.pack('<' + str(len(words)) + 'I', *words))
        receipt = {'tokens': 1, 'actual_fused_gr_launcher': True, 'variant': 3} if backend == 'native' else {
            'tokens': 1, 'guards': True, 'finite_normal_domain': True, 'repeat_exact': True}
        (d / 'report.json').write_text(json.dumps(receipt))
    return f, weighted


class Test(unittest.TestCase):
    def test_exact_primitives_and_domain(self):
        self.assertEqual(n.round_dyadic((1 << 24) + 1, -24), 0x3f800000)
        self.assertEqual(n.round_dyadic((1 << 24) + 3, -24), 0x3f800002)
        # Fused cancellation retains -2^-46; a rounded product followed by add loses it.
        self.assertEqual(n.fma(0x3f800001, 0x3f7ffffe, 0xbf800000), 0xa8800000)
        self.assertEqual(n.add(0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(n.mul(0x80000000, 0x3f800000), 0x80000000)
        for u in (1, 0x7f800000, 0x7fc00001):
            with self.assertRaises(ValueError): n.normal(u)
        with self.assertRaises(ValueError): n.mul(0x800000, 0x800000)
        with self.assertRaises(ValueError): n.mul(0x7f7fffff, 0x7f7fffff)

    def test_held_out_cases_accept_and_reject(self):
        for seed in (17, 91):
            with tempfile.TemporaryDirectory() as temp:
                p = Path(temp)
                write_results(p, fixture(seed))
                r = evaluate(p / 'native', p / 'vulkan')
                self.assertTrue(r['accepted_component_boundaries'])
                self.assertFalse(r['exact_diagnostic']['exact_captured_boundaries'])
                self.assertFalse(r['qualified'])
                # Material normalization error cannot hide behind a broad output epsilon.
                path = p / 'vulkan/rs.f32le'
                old = path.read_bytes()
                words = list(struct.unpack('<4I', old)); words[0] += 128
                path.write_bytes(struct.pack('<4I', *words))
                r = evaluate(p / 'native', p / 'vulkan')
                self.assertFalse(r['accepted_component_boundaries'])
                self.assertTrue(any(v['check'] == 'rs_interval' for v in r['violations']))
                path.write_bytes(old)
                path = p / 'vulkan/normalized.f32le'
                words = list(struct.unpack('<10240I', path.read_bytes())); words[0] ^= 1
                path.write_bytes(struct.pack('<10240I', *words))
                r = evaluate(p / 'native', p / 'vulkan')
                self.assertFalse(r['accepted_component_boundaries'])
                self.assertIsNone(r['checks'][1]['injection_order_exact'])

    def test_independently_pinned_dense_multitoken_vectors(self):
        # Expected words/hashes come from native_hc_numeric_vector_oracle.cpp,
        # compiled independently with std::fma and no Python validator helpers.
        state = 0x1539
        def next_value():
            nonlocal state
            state = (1664525 * state + 1013904223) & 0xffffffff
            return state
        def f32(u): return struct.unpack('<f', struct.pack('<I', u))[0]
        def word(v): return struct.unpack('<I', struct.pack('<f', v))[0]
        x = []
        for token in range(3):
            for j in range(10240):
                u = next_value()
                x.append(word(math.ldexp(f32((0x3e000000 + (u & 0x3ffffff)) | (u & 0x80000000)), token - 1)))
        w = [word((64 + next_value() % 67) / 128) for _ in range(10240)]
        wi = [word((next_value() % 129 - 64) / 256) >> 16 for _ in range(40960)]
        payload = struct.pack('<30720I', *x) + struct.pack('<10240I', *w) + struct.pack('<40960H', *wi)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), '6251865ba51915d11753304de06645349bcdc29b54897d01bde8ba5cb91bce72')
        data = struct.pack('<8I', 0x3143484e, 1, 2560, 4, 3, 0x358637bd, 30, 0) + payload
        f = n.parse_fixture(data)
        sums, weighted = n.source_inputs(f)
        self.assertEqual(sums, [0x4785e944, 0x477ff44c, 0x476f80d6, 0x47765ea2,
                               0x487411d0, 0x4870de28, 0x486f55d0, 0x4871d772,
                               0x497ec314, 0x497dc2c5, 0x4960d476, 0x49870b5e])
        def digest(a): return hashlib.sha256(struct.pack('<' + str(len(a)) + 'I', *a)).hexdigest()
        self.assertEqual(digest(weighted), 'a7bec4f78e854117c8ffa9f698fcd4811b298d909ad3c9c4e07f8d37fd2526da')
        # RS values are fixed powers of two; this isolates indexing/dot ordering,
        # without sharing rsqrt or interval calculations with the test oracle.
        rs = [0x3f000000, 0x3f800000, 0x40000000, 0x40800000,
              0x3e800000, 0x3f000000, 0x3f800000, 0x40000000,
              0x3e000000, 0x3e800000, 0x3f000000, 0x3f800000]
        xn, inject = n.downstream(f, weighted, rs)
        self.assertEqual(digest(xn), '3c0fb798eccc5d0fb20e419799d7240970e750f230987faae9ff337430c0c835')
        self.assertEqual(inject, [0x4232cb61, 0xc260bb0e, 0x43671c86, 0x417cb2ec,
                                 0x3fb813c0, 0xc350b1e2, 0x42b8da62, 0x42bb6fab,
                                 0xc2fb010e, 0x42a77eda, 0xc2049b43, 0xc2d358ed])

    def test_independent_interval_edges(self):
        for backend in ('native', 'vulkan'):
            # S=0 fixes denominator=epsilon and RN(rsqrt)=1000. Adjacent
            # results have constant ULP: radius 2.5 accepts +/-2, rejects +/-3.
            for step in range(-3, 4):
                self.assertEqual(n.rs_allowed(0x447a0000 + step, 0, backend), abs(step) <= 2)
            # S=2560-10*2^-12: exact mean=1-2^-20. Adding encoded epsilon
            # rounds to 1. ULP below 1 is half ULP at 1; the v1 radius
            # therefore includes five downward steps but only two upward.
            for u, accepted in ((0x3f7ffffa, False), (0x3f7ffffb, True),
                                (0x3f800002, True), (0x3f800003, False)):
                self.assertEqual(n.rs_allowed(u, 0x451ffff6, backend), accepted)
        # S=2560 gives exact mean=1. Native denominator RN(1+eps) has
        # rsqrt center 1-8 small steps. Vulkan's admitted division neighbors
        # give denominators 1+[6..10] large steps and rsqrt centers 1-[6..10]
        # small steps. Applying the intrinsic radius pins these union edges.
        for backend, lo, hi in (('native', 0x3f7ffff6, 0x3f7ffffa),
                                ('vulkan', 0x3f7ffff4, 0x3f7ffffc)):
            for u, accepted in ((lo-1, False), (lo, True), (hi, True), (hi+1, False)):
                self.assertEqual(n.rs_allowed(u, 0x45200000, backend), accepted)

    def test_identical_wrong_outputs_do_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); f, weighted = write_results(p, fixture(91))
            rs = list(struct.unpack('<4I', (p / 'native/rs.f32le').read_bytes()))
            rs[0] += 128
            xn, inject = n.downstream(f, weighted, rs)
            # Both backends agree perfectly, including consistent downstream
            # computation from the same wrong RS. Absolute RS checks reject both.
            for backend in ('native', 'vulkan'):
                for name, words in [('rs', rs), ('normalized', xn), ('inject', inject)]:
                    (p / backend / (name+'.f32le')).write_bytes(struct.pack('<'+str(len(words))+'I', *words))
            result = evaluate(p / 'native', p / 'vulkan')
            self.assertTrue(result['exact_diagnostic']['exact_captured_boundaries'])
            self.assertFalse(result['accepted_component_boundaries'])
            self.assertEqual({(v['backend'], v['check']) for v in result['violations']},
                             {('native', 'rs_interval'), ('vulkan', 'rs_interval')})

    def test_near_zero_has_no_relative_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); write_results(p, fixture(cancellation=True))
            r = evaluate(p / 'native', p / 'vulkan')
            self.assertTrue(r['accepted_component_boundaries'])
            self.assertEqual(r['injection_absolute_bounds'][0]['absolute_error'], 0)
            self.assertGreater(r['injection_absolute_bounds'][0]['rounding_bound'], 0)
            # Even a normal tiny error inside the conservative forward bound fails exact local dot order.
            path = p / 'vulkan/inject.f32le'; words = list(struct.unpack('<4I', path.read_bytes()))
            words[0] = 0x00800000; path.write_bytes(struct.pack('<4I', *words))
            r = evaluate(p / 'native', p / 'vulkan')
            self.assertFalse(r['accepted_component_boundaries'])
            self.assertTrue(any(v['check'] == 'ordered_dot' for v in r['violations']))
            path.write_bytes(struct.pack('<4I', 0x7fc00000, 0, 0, 0))
            with self.assertRaises(ValueError): evaluate(p / 'native', p / 'vulkan')
            path = p / 'vulkan/fixture.bin'; path.write_bytes(b'wrong')
            with self.assertRaises(ValueError): evaluate(p / 'native', p / 'vulkan')

if __name__ == '__main__': unittest.main()
