"""Inspect ordinary GDN output Q8_0 roles without extracting weights."""
from pathlib import Path
import json
import sys

from gguf_reader import GGUFFile
from native_qsa_wo_role_preflight import inspect_role

ROWS = 2560
BLOCKS = ROWS * 192
BYTES = BLOCKS * 34
NAMES = {f'blk.{layer}.ssm_out.weight' for layer in range(48) if layer % 4 != 3 and layer != 1}


def scan(paths):
    if len(paths) != 6:
        raise ValueError('exactly six GGUF shards are required')
    paths = [Path(p).resolve(strict=True) for p in paths]
    if len(set(paths)) != 6:
        raise ValueError('duplicate GGUF shard path')
    files = [GGUFFile(p) for p in paths]
    split, expected_total = [], None
    for f in files:
        count, number, total = (f.metadata.get(k) for k in ('split.count', 'split.no', 'split.tensors.count'))
        if count != 6 or not isinstance(number, int) or not 0 <= number < 6 or not isinstance(total, int):
            raise ValueError('invalid GGUF split metadata')
        if expected_total is None:
            expected_total = total
        if total != expected_total:
            raise ValueError('GGUF split tensor totals differ')
        split.append(number)
    if split != list(range(6)) or sum(len(f.tensors) for f in files) != expected_total:
        raise ValueError('incomplete or unordered GGUF split')
    if files[0].metadata.get('general.architecture') != 'qwen4exp':
        raise ValueError('unsupported GGUF architecture')
    roles = {}
    for f in files:
        for tensor in f.tensors:
            if tensor.name not in NAMES:
                continue
            if tensor.name in roles:
                raise ValueError('duplicate role ' + tensor.name)
            if tensor.type_id != 8 or tensor.shape != [6144, 2560] or tensor.expected_bytes() != BYTES:
                raise ValueError('unsupported role type or shape: ' + tensor.name)
            start = f.data_start + tensor.offset
            if tensor.offset < 0 or start + BYTES > f.path.stat().st_size:
                raise ValueError('role outside GGUF: ' + tensor.name)
            counts, digest = inspect_role(f.path, start, BLOCKS)
            roles[tensor.name] = {'type': 'Q8_0', 'shape': tensor.shape, 'bytes': BYTES,
                                  'sha256': digest, 'scale_classes': counts}
    if set(roles) != NAMES:
        raise ValueError('missing GDN output roles: ' + ', '.join(sorted(NAMES - set(roles))))
    return {'roles': dict(sorted(roles.items())),
            'supported': all(not row['scale_classes']['nonfinite'] for row in roles.values()),
            'qualified': False}


if __name__ == '__main__':
    try:
        report = scan(sys.argv[1:])
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report['supported'] else 1)
    except (OSError, ValueError, KeyError) as error:
        print('GDN output role preflight: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
