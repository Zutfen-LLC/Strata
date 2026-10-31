"""Fixed HC offline acceptance: exact basic FP32 operations, bounded native intrinsics.

This is a bounded validator, not a runtime kernel or software-rsqrt replacement.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import math, struct

H, C, D = 2560, 4, 10240
EPS = 0x358637bd
U = Fraction(1, 1 << 24)
GAMMA53 = 53 * U / (1 - 53 * U)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def value(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]


def normal(u):
    a = u & 0x7fffffff
    need(a == 0 or 0x800000 <= a < 0x7f800000, 'nonfinite/subnormal domain')
    return u


@lru_cache(maxsize=65536)
def dyadic(u):
    normal(u)
    a = u & 0x7fffffff
    if a == 0:
        return 0, 0
    return (-1 if u >> 31 else 1) * ((a & 0x7fffff) | 0x800000), (a >> 23) - 150


def rational(u):
    n, e = dyadic(u)
    return Fraction(n << e) if e >= 0 else Fraction(n, 1 << -e)


def round_dyadic(n, e, negative_zero=False):
    if not n:
        return 0x80000000 if negative_zero else 0
    sign = 0x80000000 if n < 0 else 0
    n = abs(n)
    exponent = e + n.bit_length() - 1
    need(-126 <= exponent <= 127, 'reference underflow/overflow unsupported')
    shift = n.bit_length() - 24
    if shift > 0:
        q, remainder = divmod(n, 1 << shift)
        half = 1 << (shift - 1)
        q += remainder > half or (remainder == half and q & 1)
    else:
        q = n << -shift
    if q == 1 << 24:
        q >>= 1
        exponent += 1
    need(exponent <= 127, 'reference overflow')
    return sign | ((exponent + 127) << 23) | (q & 0x7fffff)


def mul(a, b):
    x, e = dyadic(a)
    y, f = dyadic(b)
    return round_dyadic(x * y, e + f, bool((a ^ b) >> 31))


def fma(a, b, c):
    x, e = dyadic(a)
    y, f = dyadic(b)
    z, g = dyadic(c)
    e += f
    k = min(e, g)
    n = ((x * y) << (e - k)) + (z << (g - k))
    return round_dyadic(n, k, not x * y and not z and bool((a ^ b) >> 31) and bool(c >> 31))


def add(a, b):
    return fma(0x3f800000, a, b)


def nearest(q):
    """Exact rational distance selects nearest F32; float only seeds the search."""
    if not q:
        return 0
    need(q >= Fraction(1, 1 << 126), 'reference division underflow unsupported')
    u = bits(float(q))
    need(0 <= u < 0x7f800000, 'positive normal reference required')
    choices = range(max(0x800000, u - 3), min(0x7f800000, u + 4))
    result = min(choices, key=lambda b: (abs(rational(b) - q), b & 1))
    normal(result)
    need(not q or result, 'reference underflow unsupported')
    return result


def ulp(u):
    _, e = dyadic(u)
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def rsqrt_nearest(d):
    with localcontext() as ctx:
        ctx.prec = 100
        q = rational(d)
        exact = Decimal(q.numerator) / Decimal(q.denominator)
        root = Decimal(1) / exact.sqrt()
        u = bits(float(root))
        return min(range(u - 3, u + 4), key=lambda b: (abs(Decimal.from_float(value(b)) - root), b & 1))


def rs_allowed(actual, total, backend):
    normal(actual)
    need(0 < actual < 0x7f800000, 'RS must be positive normal')
    q = rational(total) / H
    center = nearest(q)
    means = [center]
    if backend == 'vulkan' and total:
        # GLSL division: <=2.5 ULP from the exact quotient. Include binade edges.
        means = [m for m in range(max(0x800000, center - 6), min(0x7f800000, center + 7))
                 if abs(rational(m) - q) <= Fraction(5, 2) * max(ulp(m), ulp(center))]
    for mean in means:
        denominator = add(mean, EPS)
        rn = rsqrt_nearest(denominator)
        # 2 vendor ULP + at most 0.5 ULP for the rounded reference center.
        if abs(rational(actual) - rational(rn)) <= Fraction(5, 2) * max(ulp(actual), ulp(rn)):
            return True
    return False


def parse_fixture(data):
    header = struct.unpack_from('<8I', data)
    magic, version, h, c, t, eps, typ, flags = header
    need((magic, version, h, c, eps, typ, flags) == (0x3143484e, 1, H, C, EPS, 30, 0) and 1 <= t <= 8,
         'selected HC fixture contract')
    need(len(data) == 32 + t * D * 4 + D * 4 + C * D * 2, 'fixture size')
    x = struct.unpack_from('<' + str(t * D) + 'I', data, 32)
    w = struct.unpack_from('<10240I', data, 32 + t * D * 4)
    wi = [u << 16 for u in struct.unpack_from('<40960H', data, 32 + (t + 1) * D * 4)]
    for u in (*x, *w, *wi):
        normal(u)
    return t, x, w, wi


def source_inputs(fixture):
    t, x, w, wi = fixture
    totals = []
    weighted = [mul(u, w[i % D]) for i, u in enumerate(x)]
    for token in range(t):
        partial = [[0] * 256 for _ in range(C)]
        for tid in range(256):
            for j in range(tid * 4, D, 1024):
                r = x[token * D + j:token * D + j + 4]
                sq = mul(r[1], r[1])
                for k in (0, 2, 3):
                    sq = fma(r[k], r[k], sq)
                partial[j // H][tid] = add(partial[j // H][tid], sq)
        for offset in (16, 8, 4, 2, 1):
            partial = [[add(row[i], row[i ^ offset]) for i in range(256)] for row in partial]
        for row in partial:
            total = 0
            for warp in range(8):
                total = add(total, row[warp * 32])
            totals.append(total)
    return totals, weighted


def downstream(fixture, weighted, rs):
    t, _, _, wi = fixture
    xn = [mul(u, rs[i // D * C + i % D // H]) for i, u in enumerate(weighted)]
    inject = []
    for token in range(t):
        for row in range(C):
            lanes = []
            for lane in range(32):
                acc = 0
                for chunk in range(lane, D // 8, 32):
                    v = 0
                    for k in range(8):
                        j = chunk * 8 + k
                        v = fma(wi[row * D + j], xn[token * D + j], v)
                    acc = add(acc, v)
                lanes.append(acc)
            for offset in (16, 8, 4, 2, 1):
                lanes = [add(lanes[i], lanes[i ^ offset]) for i in range(32)]
            inject.append(lanes[0])
    return xn, inject


def abs_product_sum(weights, terms):
    pairs = []
    for w, (n, e) in zip(weights, terms):
        a, f = dyadic(w)
        if a and n:
            pairs.append((abs(a * n), e + f))
    if not pairs:
        return Fraction(0)
    base = min(e for _, e in pairs)
    n = sum(n << (e - base) for n, e in pairs)
    return Fraction(n << base) if base >= 0 else Fraction(n, 1 << -base)


def propagation(fixture, xn, inject):
    t, _, _, wi = fixture
    rows = []
    for token in range(t):
        a = xn['native'][token * D:(token + 1) * D]
        b = xn['vulkan'][token * D:(token + 1) * D]
        delta = []
        for u, v in zip(a, b):
            n, e = dyadic(u)
            m, f = dyadic(v)
            k = min(e, f)
            delta.append(((n << (e - k)) - (m << (f - k)), k))
        for row in range(C):
            weights = wi[row * D:(row + 1) * D]
            dependency = abs_product_sum(weights, delta)
            rounding = GAMMA53 * (abs_product_sum(weights, map(dyadic, a)) + abs_product_sum(weights, map(dyadic, b)))
            bound = dependency + rounding
            error = abs(rational(inject['native'][token * C + row]) - rational(inject['vulkan'][token * C + row]))
            need(error <= bound, 'injection absolute propagation bound')
            rows.append({'token': token, 'row': row, 'absolute_error': float(error),
                         'dependency_bound': float(dependency), 'rounding_bound': float(rounding),
                         'absolute_bound': float(bound)})
    return rows
