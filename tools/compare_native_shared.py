#!/usr/bin/env python3
"""Compare paired native shared-expert receipts without qualifying intrinsics.

Exact checks are conditional on identical consumed inputs. Exp/fast division
remain observations; this tool never certifies universal cross-vendor identity.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

NAMES = ('x', 'wg', 'wu', 'wd', 'ws', 'qx', 'gate', 'up', 'h', 'qh', 'down', 'raw', 'sigmoid', 'out', 'dummy')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def counts(h, f, t):
    require(32 <= h <= 8192 and 32 <= f <= 8192 and h % 32 == 0 and f % 32 == 0 and 1 <= t <= 8,
            'unsupported shared geometry')
    sizes = (h*t, h//32*f*34//4, h//32*f*34//4, f//32*h*34//4, h//2,
             h//32*9*t, f*t, f*t, f*t, f//32*9*t, h*t, t, t, h*t, 1)
    require(sum((n+32)*4 for n in sizes) <= 256*1024*1024, 'shared fixture exceeds budget')
    return dict(zip(NAMES, sizes))


def fixture(path):
    data = Path(path).read_bytes()
    require(len(data) >= 28, 'truncated shared header')
    magic, version, h, f, t, typ, binding = struct.unpack_from('<7I', data)
    require((magic, version, typ, binding) == (0x3153484e, 1, 8, 1), 'shared fixture header/type/binding')
    sizes = counts(h, f, t)
    inputs, offset = {}, 28
    for name in NAMES[:5]:
        end = offset + sizes[name]*4
        inputs[name] = data[offset:end]
        offset = end
    require(len(data) == offset, 'shared fixture length')
    return (h, f, t), sizes, inputs, hashlib.sha256(data).hexdigest()


def load(directory, geometry, sizes, inputs):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    h, f, t = geometry
    require(all(report.get(k) is True for k in ('observation_only', 'sentinels_ok', 'finite')), 'receipt flags failed')
    require(report.get('qualified') is False, 'receipt must remain unqualified')
    require(tuple(report.get(k) for k in ('h', 'f', 'tokens')) == (h, f, t), 'receipt geometry mismatch')
    images = {}
    for name, size in sizes.items():
        data = (directory/(name+'.bin')).read_bytes()
        require(len(data) == size*4, name+' capture length')
        require(name not in inputs or data == inputs[name], name+' captured input differs from fixture')
        images[name] = data
    for name in ('gate', 'up', 'h', 'down', 'raw', 'sigmoid', 'out'):
        require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', images[name])), name+' nonfinite capture')
    require(images['dummy'] == struct.pack('<I', 0xa5a5a5a5), 'dummy modified')
    return images


def consumed_q8(data):
    # Q8_0 dot reads ds.x plus qs. ds.y is an unused sum-half and is reported separately.
    require(len(data) % 36 == 0, 'Q8_1 capture length')
    return b''.join(data[i:i+2]+data[i+4:i+36] for i in range(0, len(data), 36))


def difference(a, b):
    left = struct.unpack('<'+str(len(a)//4)+'I', a)
    right = struct.unpack('<'+str(len(b)//4)+'I', b)
    indices = [i for i, (x, y) in enumerate(zip(left, right)) if x != y]
    return {'exact': not indices, 'different_words': len(indices), 'first_word': indices[0] if indices else None}


def compare(path, native, vulkan):
    geometry, sizes, inputs, digest = fixture(path)
    a, b = load(native, geometry, sizes, inputs), load(vulkan, geometry, sizes, inputs)
    stages = {name: difference(a[name], b[name]) for name in NAMES[5:-1]}
    qx_equal = consumed_q8(a['qx']) == consumed_q8(b['qx'])
    qh_equal = consumed_q8(a['qh']) == consumed_q8(b['qh'])
    failures = []
    if not qx_equal:
        failures.append('input quantization consumed scale/codes differ')
    if qx_equal:
        for name in ('gate', 'up'):
            if not stages[name]['exact']:
                failures.append(name+' dense MMVQ differs with identical consumed input')
    if not stages['raw']['exact']:
        failures.append('BF16/F32 scalar dot differs on original identical activations')
    if stages['h']['exact'] and not qh_equal:
        failures.append('SwiGLU quantization differs on identical intermediate')
    if qh_equal and not stages['down']['exact']:
        failures.append('down dense MMVQ differs with identical consumed input')
    if stages['down']['exact'] and stages['sigmoid']['exact'] and not stages['out']['exact']:
        failures.append('final multiplication differs on identical down/sigmoid inputs')
    same_projections = stages['gate']['exact'] and stages['up']['exact']
    return {'fixture_sha256': digest, 'geometry': dict(zip(('h', 'f', 'tokens'), geometry)),
            'observation_only': True, 'qualified': False,
            'exact_arithmetic_checks_passed': not failures, 'exact_arithmetic_failures': failures,
            'qx_consumed_equal': qx_equal, 'qh_consumed_equal': qh_equal, 'stages': stages,
            'swiglu_same_input': same_projections, 'sigmoid_same_input': stages['raw']['exact'],
            'first_differing_stage': next((name for name in NAMES[5:-1] if not stages[name]['exact']), None),
            'limitation': 'Exp and CUDA fast division are cross-vendor observations; no approximate-unit emulation or universal bit-identity claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--native', required=True)
    parser.add_argument('--vulkan', required=True)
    args = parser.parse_args()
    result = compare(args.fixture, args.native, args.vulkan)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['exact_arithmetic_checks_passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as error:
        raise SystemExit('shared comparison refused: '+str(error))
