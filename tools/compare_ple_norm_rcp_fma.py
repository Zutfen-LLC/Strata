"""Offline candidate controls. The existing guarded runner authenticates runs.

Native outputs are measurements, not fitted acceptance bounds. No CUDA
reciprocal/rsqrt equivalence or whole-model acceptance follows from this check.
"""
import hashlib
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path
import native_hc_numeric as fp


def row_sum(values):
    partial = [0] * 1024
    for lane in range(1024):
        for col in range(lane, 2560, 1024):
            partial[lane] = fp.fma(values[col], values[col], partial[lane])
    for offset in (16, 8, 4, 2, 1):
        previous = partial[:]
        partial = [fp.add(previous[i], previous[i ^ offset]) for i in range(1024)]
    partial = [partial[32*i] for i in range(32)]
    for offset in (16, 8, 4, 2, 1):
        previous = partial[:]
        partial = [fp.add(previous[i], previous[i ^ offset]) for i in range(32)]
    return partial[0]


def controls(x, gamma, output, diagnostic):
    if tuple(map(len, (x, gamma, output, diagnostic))) != (10240, 10240, 10240, 20):
        raise ValueError('candidate input/output geometry differs')
    for value in (*x, *gamma, *output, *diagnostic):
        fp.normal(value)
    for row in range(4):
        total, reciprocal, shifted, scale, mean = diagnostic[row*5:row*5+5]
        if total != row_sum(x[row*2560:(row+1)*2560]):
            raise ValueError('source-order sum differs')
        exact = Fraction(1, 2560)
        center = fp.nearest(exact)
        if reciprocal >> 31 or not reciprocal or abs(fp.rational(reciprocal)-exact) > Fraction(5, 2)*max(fp.ulp(reciprocal), fp.ulp(center)):
            raise ValueError('reciprocal outside existing Vulkan division bound')
        if shifted != fp.fma(total, reciprocal, fp.EPS):
            raise ValueError('sum * reciprocal + epsilon FMA differs')
        if mean != fp.mul(total, reciprocal):
            raise ValueError('diagnostic mean differs')
        reference = fp.rsqrt_nearest(shifted)
        if scale >> 31 or not scale or abs(fp.rational(scale)-fp.rational(reference)) > Fraction(5, 2)*max(fp.ulp(scale), fp.ulp(reference)):
            raise ValueError('rsqrt outside existing Vulkan bound')
        for col in range(row*2560, (row+1)*2560):
            if output[col] != fp.mul(fp.mul(scale, x[col]), gamma[col]):
                raise ValueError('weighted output association differs')


def read(path, count, pin=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular evidence file required: '+str(path))
    raw = path.read_bytes()
    if len(raw) != count*4 or (pin is not None and hashlib.sha256(raw).hexdigest() != pin):
        raise ValueError('evidence size or digest differs: '+str(path))
    return raw


def compare(case):
    fixture, candidate = Path(case['fixture']), Path(case['candidate'])
    snapshots = {}
    for name, count, pin in [('input.f32le', 10240, case['input_sha256']), ('gamma.f32le', 10240, case['gamma_sha256'])]:
        snapshots[fixture/name] = read(fixture/name, count, pin)
    native = read(case['native_output'], 10240, case['native_sha256'])
    snapshots[Path(case['native_output'])] = native
    for name, count in [('output.f32le', 10240), ('diagnostic.f32le', 20)]:
        snapshots[candidate/name] = read(candidate/name, count)
    report = candidate/'report.json'
    snapshots[report] = report.read_bytes()
    if json.loads(snapshots[report]) != {'actual_vulkan': True, 'immutable_inputs_canaries_repeat': True, 'rcp_fma_observation': True, 'qualified': False}:
        raise ValueError('candidate report identity differs')
    words = lambda raw: struct.unpack('<'+'I'*(len(raw)//4), raw)
    output = snapshots[candidate/'output.f32le']
    controls(words(snapshots[fixture/'input.f32le']), words(snapshots[fixture/'gamma.f32le']), words(output), words(snapshots[candidate/'diagnostic.f32le']))
    a, b = words(native), words(output)
    for value in a:
        fp.normal(value)
    indices = [i for i in range(10240) if a[i] != b[i]]
    order = lambda u: (~u)&0xffffffff if u&0x80000000 else u|0x80000000
    result = {'different_words': len(indices), 'max_ulp': max((abs(order(a[i])-order(b[i])) for i in indices), default=0),
              'max_abs': max((abs(fp.value(a[i])-fp.value(b[i])) for i in indices), default=0),
              'own_controls_pass': True, 'diagnostic_only': True, 'qualified': False, 'native_intrinsic_equivalence': False}
    if any(path.read_bytes() != raw for path, raw in snapshots.items()):
        raise ValueError('evidence changed during comparison')
    return result


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: compare_ple_norm_rcp_fma.py CASE_CONFIG_JSON')
    print(json.dumps(compare(json.loads(Path(sys.argv[1]).read_text())), sort_keys=True, allow_nan=False))
