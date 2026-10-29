#!/usr/bin/env python3
"""Exact bounded stage comparison. Any byte difference returns failure, no tolerance.
NaN-to-integer codes have no portable native contract; their differences are
reported, never normalized or presented as qualified success.
"""
import argparse
import json
import math
from pathlib import Path
import struct

NAMES = ('input_q8', 'separate_hidden', 'separate_q8', 'fused_q8', 'empty_q8')


def read_words(data, offset, count):
    size = count * 4
    if offset + size > len(data):
        raise ValueError('truncated capture')
    return struct.unpack_from('<' + 'I' * count, data, offset), offset + size


def fixture(path):
    data = path.read_bytes()
    (magic, ff, n, ns), off = read_words(data, 0, 4)
    if magic != 0x3151384e or not (0 < n <= 1048576 and ff and ff % 32 == 0 and n % ff == 0 and 0 < ns <= 15):
        raise ValueError('invalid fixture')
    starts, off = read_words(data, off, ns)
    if any(s > n // ff or (i and s < starts[i - 1]) for i, s in enumerate(starts)):
        raise ValueError('invalid intervals')
    arrays = []
    for _ in range(3):
        words, off = read_words(data, off, n)
        arrays.append(words)
    if off != len(data):
        raise ValueError('trailing fixture')
    return ff, n, starts, arrays


def capture(path, n):
    data = path.read_bytes()
    (magic,), off = read_words(data, 0, 1)
    if magic not in (0x3152384e, 0x3252384e):
        raise ValueError('invalid result')
    observation = magic == 0x3252384e
    sentinel_ok = True
    if observation:
        (flag,), off = read_words(data, off, 1)
        if flag not in (0, 1):
            raise ValueError('invalid observation sentinel status')
        sentinel_ok = bool(flag)
    result = {'observation_only': observation, 'sentinels_ok': sentinel_ok}
    for name in NAMES:
        (size,), off = read_words(data, off, 1)
        expected = n * 4 if name == 'separate_hidden' else n // 32 * 36
        if size != expected or off + size > len(data):
            raise ValueError('invalid section size')
        result[name] = data[off:off + size]
        off += size
    if off != len(data):
        raise ValueError('trailing result')
    return result


def differences(a, b):
    return [{'offset': i, 'native': x, 'vulkan': y} for i, (x, y) in enumerate(zip(a, b)) if x != y]


def contract_exceptions(raw, packed):
    result = []
    for block in range(len(raw) // 32):
        values = [struct.unpack('<f', struct.pack('<I', x))[0] for x in raw[block * 32:(block + 1) * 32]]
        numeric = [abs(x) for x in values if not math.isnan(x)]
        maximum = max(numeric) if numeric else math.nan
        # Clamp happens after F32 division; include scale-underflow conversion holes.
        scale = min(maximum / 127.0, 65504.0)
        scale = struct.unpack('<f', struct.pack('<f', scale))[0]
        lanes = [i for i, x in enumerate(values) if math.isnan(x) or
                 (maximum != 0.0 and scale == 0.0 and x == 0.0)]
        ds = struct.unpack_from('<HH', packed, block * 36)
        nan_metadata = any((h & 0x7c00) == 0x7c00 and (h & 0x3ff) for h in ds)
        if lanes or nan_metadata:
            result.append({'block': block, 'nan_to_int8_lanes': lanes,
                           'nan_half_metadata': nan_metadata})
    return result


def compare(fixture_path, native_path, vulkan_path):
    ff, n, starts, arrays = fixture(fixture_path)
    native, vulkan = capture(native_path, n), capture(vulkan_path, n)
    if native['observation_only']:
        raise ValueError('native oracle must not be an observation capture')
    report = {'exact': True, 'elements': n, 'n_ff': ff, 'starts': starts, 'sections': {}, 'internal_paths': {}}
    for name in NAMES:
        a, b = native[name], vulkan[name]
        diffs = differences(a, b)
        item = {'different_bytes': len(diffs), 'differences': diffs}
        if name == 'separate_hidden':
            words_a, words_b = struct.unpack('<' + 'I' * n, a), struct.unpack('<' + 'I' * n, b)
            item['different_values'] = []
            maximum = 0.0
            for i, (x, y) in enumerate(zip(words_a, words_b)):
                if x == y:
                    continue
                fx, fy = struct.unpack('<f', struct.pack('<I', x))[0], struct.unpack('<f', struct.pack('<I', y))[0]
                if math.isfinite(fx) and math.isfinite(fy):
                    maximum = max(maximum, abs(fx - fy))
                item['different_values'].append({'index': i, 'native_bits': hex(x), 'vulkan_bits': hex(y),
                    'gate_bits': hex(arrays[1][i]), 'up_bits': hex(arrays[2][i])})
            item['max_finite_abs_difference'] = maximum
        else:
            item['different_blocks'] = sorted({d['offset'] // 36 for d in diffs})
            item['different_ds_bytes'] = sum(d['offset'] % 36 < 4 for d in diffs)
            item['different_code_bytes'] = sum(d['offset'] % 36 >= 4 for d in diffs)
        report['sections'][name] = item
        if diffs:
            report['exact'] = False
    lo, hi = starts[0] * ff // 32 * 36, starts[-1] * ff // 32 * 36
    for label, result in [('native', native), ('vulkan', vulkan)]:
        active_diffs = differences(result['separate_q8'][lo:hi], result['fused_q8'][lo:hi])
        report['internal_paths'][label] = {'active_separate_vs_fused_different_bytes': len(active_diffs),
            'differences': active_diffs}
        if active_diffs:
            report['exact'] = False
        inactive = result['fused_q8'][:lo] + result['fused_q8'][hi:]
        if any(b != 0xa5 for b in inactive + result['empty_q8']):
            report['exact'] = False
            report['internal_paths'][label]['sentinel_failure'] = True
    hidden_words = struct.unpack('<' + 'I' * n, native['separate_hidden'])
    report['native_contract_exceptions'] = {
        'input_q8': contract_exceptions(arrays[0], native['input_q8']),
        'separate_q8': contract_exceptions(hidden_words, native['separate_q8'])}
    if not vulkan['sentinels_ok']:
        report['exact'] = False
        report['observation_view_sentinel_failure'] = True
    if vulkan['observation_only']:
        report['observation_only'] = True
        report['fp32_denorm_mode'] = 'unspecified'
    report['qualified'] = False  # a synthetic fixture never qualifies product integration
    report['fixture_contract_defined'] = not any(report['native_contract_exceptions'].values())
    report['limits'] = ('Exact bytes are required, including NaNs. Native NaN-to-int8 conversion and NaN payloads '
                        'are not a cross-backend contract. See raw differences; no mismatch is waived. '
                        'One fixture does not qualify every exp/division input or hardware.')
    return report


def comparison_exit(report):
    if not report['exact']:
        return 1
    return 2 if report.get('observation_only', False) or not report['fixture_contract_defined'] else 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--vulkan', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    report = compare(a.fixture, a.native, a.vulkan)
    a.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    summary = {'exact': report['exact'], 'different_bytes': {
        k: v['different_bytes'] for k, v in report['sections'].items()}}
    if report.get('observation_only', False):
        summary.update(qualified=False, fp32_denorm_mode='unspecified')
    print(json.dumps(summary))
    # Even exact bytes in a native conversion hole must not resemble qualification.
    raise SystemExit(comparison_exit(report))
