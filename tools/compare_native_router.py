#!/usr/bin/env python3
"""Exact paired receipt comparison; any changed ID or weight bit fails."""
import argparse
import hashlib
import json
import math
import pathlib
import struct


def require(condition, message="invalid router receipt"):
    if not condition:
        raise AssertionError(message)


def compare(native, vulkan):
    reports = [json.loads((path / 'report.json').read_text()) for path in (native, vulkan)]
    require(reports[0]['tokens'] == reports[1]['tokens'])
    tokens = reports[0]['tokens']
    require(1 <= tokens <= 8)
    for report in reports:
        require(report['sentinels_ok'] and report['valid_output'])
        require(report['qualified'] is False and report['experts'] == 512 and report['k'] == 10)
    result = {'qualified': False, 'tokens': tokens}
    for name in ('ids.i32', 'weights.f32'):
        a, b = [(path / name).read_bytes() for path in (native, vulkan)]
        require(len(a) == len(b) == tokens * 10 * 4)
        wa, wb = [struct.unpack('<' + 'I' * (len(x) // 4), x) for x in (a, b)]
        for data, words in ((a, wa), (b, wb)):
            if name == 'ids.i32':
                require(all(x < 512 for x in words))
                require(all(len(set(words[t*10:t*10+10])) == 10 for t in range(tokens)))
            else:
                require(all(math.isfinite(x) and x >= 0 for x in struct.unpack('<' + 'f' * (len(data)//4), data)))
        result[name] = {'different_bits': sum(x != y for x, y in zip(wa, wb)),
                        'native_sha256': hashlib.sha256(a).hexdigest(),
                        'vulkan_sha256': hashlib.sha256(b).hexdigest()}
    # Identical complete logits view proves input identity, including offsets.
    a, b = [(path / 'view-0.bin').read_bytes() for path in (native, vulkan)]
    require(len(a) == len(b) == (3 + tokens * 512 + 9) * 4 and a == b, 'different input payloads')
    result['input_view_sha256'] = hashlib.sha256(a).hexdigest()
    result['exact'] = all(result[name]['different_bits'] == 0 for name in ('ids.i32', 'weights.f32'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=pathlib.Path)
    parser.add_argument('vulkan', type=pathlib.Path)
    args = parser.parse_args()
    result = compare(args.native, args.vulkan)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['exact'] else 1)
