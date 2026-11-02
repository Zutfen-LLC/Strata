"""Exercise the real role scanner against an ordered six-shard GGUF fixture."""
from pathlib import Path
import struct
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tools'))
import native_qsa_indexer_bf16_role_preflight as scanner

assert len(scanner.NAMES) == 24
with tempfile.TemporaryDirectory(prefix='indexer-bf16-role-') as tmp:
    paths, starts = [], []
    names = list(scanner.NAMES.items())
    for index in range(6):
        path = Path(tmp) / f'{index}.gguf'
        paths.append(path)
        text = lambda s: struct.pack('<Q', len(s)) + s.encode()
        items = names[index * 4:index * 4 + 4]
        header = bytearray(struct.pack('<IIQQ', 0x46554747, 3, len(items), 4 if index == 0 else 3))
        if index == 0:
            header += text('general.architecture') + struct.pack('<I', 8) + text('qwen4exp')
        for key, typ, value in [('split.count', 2, 6), ('split.no', 2, index), ('split.tensors.count', 5, 24)]:
            header += text(key) + struct.pack('<I', typ) + struct.pack('<H' if typ == 2 else '<i', value)
        offsets, offset = [], 0
        for name, rows in items:
            offsets.append(offset)
            header += text(name) + struct.pack('<IQQIQ', 2, 2560, rows, 30, offset)
            offset += rows * 2560 * 2
        header += bytes((-len(header)) % 32)
        starts.append(len(header) + offsets[0])
        with path.open('wb') as stream:
            stream.write(header)
            stream.truncate(len(header) + offset)
    report = scanner.scan(paths)
    assert report['supported'] and not report['qualified'] and len(report['roles']) == 24
    assert all(role['classes']['zero'] == scanner.NAMES[name] * 2560 for name, role in report['roles'].items())
    original = report['roles'][names[0][0]]['sha256']
    with paths[0].open('r+b') as stream:
        stream.seek(starts[0]); stream.write(b'\x01\x00')
    report = scanner.scan(paths)
    assert not report['supported'] and report['roles'][names[0][0]]['classes']['subnormal'] == 1
    assert report['roles'][names[0][0]]['sha256'] != original
    with paths[0].open('r+b') as stream:
        stream.seek(starts[0]); stream.write(b'\x80\x7f')
    assert scanner.scan(paths)['roles'][names[0][0]]['classes']['nonfinite'] == 1
    try:
        scanner.scan(paths[::-1])
    except ValueError:
        pass
    else:
        raise AssertionError('unordered shards accepted')
print('24 BF16 roles, identities, domain refusals and six-shard framing pass')
