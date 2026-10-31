"""Independent validator controls; these receipts are mocks, never GPU evidence."""
import json, struct, sys, tempfile, unittest
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
