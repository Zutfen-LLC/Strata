#!/usr/bin/env python3
"""Exact consumed-boundary comparison for one retained native Q8 layer request."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

SUFFIXES = ('.input.q8_1', '.gate.f32', '.up.f32', '.hidden.q8_1', '.output.f32')

def read(prefix, suffix):
    p = Path(str(prefix) + suffix)
    size = p.stat().st_size
    if not 0 < size <= (1 << 20):
        raise ValueError('boundary size outside one-layer bound: ' + suffix)
    return p.read_bytes()

def compare(native, vulkan):
    a, b = read(native, '.metadata'), read(vulkan, '.metadata')
    if a != b:
        raise ValueError('request/plan/group/entry metadata mismatch')
    lines = a.decode('ascii').splitlines()
    if len(lines) != 8:
        raise ValueError('invalid one-layer metadata')
    header = lines[0].split()
    tokens = int(header[header.index('T') + 1])
    groups, slots, starts, dst, tok, original = ([int(v) for v in line.split()] for line in lines[2:])
    entries = len(dst)
    if not (1 <= tokens <= 8 and 1 <= len(groups) <= 14 and len(slots) == len(groups)
            and len(starts) == len(groups)+1 and starts[0] == 0 and starts[-1] == entries
            and starts == sorted(starts) and 1 <= entries <= 64 and dst == list(range(entries))
            and len(tok) == len(original) == entries and len(set(original)) == entries
            and all(0 <= v < tokens for v in tok)):
        raise ValueError('invalid group/entry geometry')
    expected_sizes = (tokens*2880, entries*2560, entries*2560, entries*720, entries*10240)
    report = {'exact_consumed_boundaries' : True, 'qualified': False,
              'quantizer_qualified': False, 'observation_only': True,
              'historical_omnibus_failure_unchanged': True, 'stages': {}}
    for suffix, expected_size in zip(SUFFIXES, expected_sizes):
        a, b = read(native, suffix), read(vulkan, suffix)
        if len(a) != expected_size or len(a) != len(b):
            raise ValueError('boundary length mismatch: ' + suffix)
        packed = suffix.endswith('q8_1')
        if len(a) % (36 if packed else 4):
            raise ValueError('boundary element size mismatch: ' + suffix)
        invalid = []
        for name, data in (('native', a), ('vulkan', b)):
            if packed:
                bad = any((struct.unpack_from('<H', data, i + j)[0] & 0x7c00) == 0x7c00
                          for i in range(0, len(data), 36) for j in (0, 2))
            else:
                bad = any((x[0] & 0x7f800000) == 0x7f800000 for x in struct.iter_unpack('<I', data))
            if bad:
                invalid.append(name)
        av, bv = (a, b) if packed else (list(struct.unpack('<' + 'I' * (len(a)//4), a)), list(struct.unpack('<' + 'I' * (len(b)//4), b)))
        different = [i for i, (x, y) in enumerate(zip(av, bv)) if x != y]
        report['stages'][suffix] = {'compared': len(av), 'unit': 'byte' if packed else 'raw_f32_bits',
            'different': len(different), 'first': different[0] if different else None,
            'nonfinite': invalid, 'native_sha256': hashlib.sha256(a).hexdigest(),
            'vulkan_sha256': hashlib.sha256(b).hexdigest()}
        if different or invalid:
            report['exact_consumed_boundaries'] = False
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('native', type=Path)
    p.add_argument('vulkan', type=Path)
    p.add_argument('report', type=Path)
    a = p.parse_args()
    result = compare(a.native, a.vulkan)
    with a.report.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    return 0 if result['exact_consumed_boundaries'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
