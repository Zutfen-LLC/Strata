"""Read the six GGUF shards and check all ordinary-QSA indexer BF16 roles."""
from array import array
import hashlib
import json
from pathlib import Path
import sys
from gguf_reader import GGUFFile

ROWS = {'indexer.k_proj.weight': 128, 'indexer.q_proj.weight': 512}
NAMES = {f'blk.{layer}.{name}': rows for layer in range(3, 48, 4) for name, rows in ROWS.items()}


def inspect_role(path, start, size):
    classes = dict(zero=0, normal=0, subnormal=0, nonfinite=0)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(start)
        remaining = size
        while remaining:
            data = stream.read(min(remaining, 1 << 20))
            if not data or len(data) % 2:
                raise ValueError('short or unaligned BF16 role')
            remaining -= len(data)
            digest.update(data)
            words = array('H', data)
            if sys.byteorder != 'little':
                words.byteswap()
            for word in words:
                exp, fraction = word & 0x7f80, word & 0x7f
                if exp == 0x7f80:
                    classes['nonfinite'] += 1
                elif exp:
                    classes['normal'] += 1
                elif fraction:
                    classes['subnormal'] += 1
                else:
                    classes['zero'] += 1
    return classes, digest.hexdigest()


def scan(paths):
    if len(paths) != 6:
        raise ValueError('exactly six GGUF shards are required')
    paths = [Path(path).resolve(strict=True) for path in paths]
    if len(set(paths)) != 6:
        raise ValueError('duplicate GGUF shard path')
    files = [GGUFFile(path) for path in paths]
    expected_total = files[0].metadata.get('split.tensors.count')
    for number, f in enumerate(files):
        if (f.metadata.get('split.count') != 6 or f.metadata.get('split.no') != number
                or not isinstance(expected_total, int) or f.metadata.get('split.tensors.count') != expected_total):
            raise ValueError('invalid or unordered GGUF split metadata')
    if sum(len(f.tensors) for f in files) != expected_total:
        raise ValueError('incomplete GGUF split')
    if files[0].metadata.get('general.architecture') != 'qwen4exp':
        raise ValueError('unsupported GGUF architecture')
    roles = {}
    for f in files:
        for tensor in f.tensors:
            if tensor.name not in NAMES:
                continue
            if tensor.name in roles:
                raise ValueError('duplicate role ' + tensor.name)
            rows = NAMES[tensor.name]
            size = rows * 2560 * 2
            if tensor.type_id != 30 or tensor.shape != [2560, rows] or tensor.expected_bytes() != size:
                raise ValueError('unsupported role type or shape: ' + tensor.name)
            start = f.data_start + tensor.offset
            if tensor.offset < 0 or start + size > f.path.stat().st_size:
                raise ValueError('role outside GGUF: ' + tensor.name)
            classes, digest = inspect_role(f.path, start, size)
            roles[tensor.name] = {'type': 'BF16', 'shape': tensor.shape, 'bytes': size,
                                  'sha256': digest, 'classes': classes}
    if set(roles) != set(NAMES):
        raise ValueError('missing indexer roles: ' + ', '.join(sorted(set(NAMES) - set(roles))))
    supported = all(not (r['classes']['nonfinite'] or r['classes']['subnormal']) for r in roles.values())
    return {'roles': dict(sorted(roles.items())), 'supported': supported, 'qualified': False}


if __name__ == '__main__':
    try:
        report = scan(sys.argv[1:])
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report['supported'] else 1)
    except (OSError, ValueError, KeyError) as exc:
        print('QSA indexer BF16 role preflight: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
