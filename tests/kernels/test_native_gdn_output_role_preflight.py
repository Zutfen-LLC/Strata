"""Check ordinary GDN output role framing and scale-domain admission."""
from pathlib import Path
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tools'))
spec = importlib.util.spec_from_file_location('gdn_out_roles', root / 'tools/native_gdn_output_role_preflight.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert len(m.NAMES) == 35 and 'blk.1.ssm_out.weight' not in m.NAMES
m.NAMES = {'blk.0.ssm_out.weight', 'blk.2.ssm_out.weight'}
with tempfile.TemporaryDirectory(prefix='gdn-out-roles-') as tmp:
    paths, starts, type_offsets = [], [], []
    names = sorted(m.NAMES)
    for index in range(6):
        path = Path(tmp) / f'{index}.gguf'
        paths.append(path)
        text = lambda s: struct.pack('<Q', len(s)) + s.encode()
        count = int(index < 2)
        header = bytearray(struct.pack('<IIQQ', 0x46554747, 3, count, 4 if index == 0 else 3))
        if index == 0:
            header += text('general.architecture') + struct.pack('<I', 8) + text('qwen4exp')
        for key, typ, value in [('split.count', 2, 6), ('split.no', 2, index), ('split.tensors.count', 5, 2)]:
            header += text(key) + struct.pack('<I', typ) + struct.pack('<H' if typ == 2 else '<i', value)
        if count:
            type_offsets.append(len(header) + len(text(names[index])) + 4 + 8 + 8)
            header += text(names[index]) + struct.pack('<IQQIQ', 2, 6144, 2560, 8, 0)
        header += bytes((-len(header)) % 32)
        starts.append(len(header))
        with path.open('wb') as f:
            f.write(header)
            f.truncate(len(header) + (2560 * 192 * 34 if count else 0))
    report = m.scan(paths)
    assert report['supported'] and set(report['roles']) == m.NAMES
    assert all(row['scale_classes']['zero'] == 2560 * 192 for row in report['roles'].values())
    if len(sys.argv) > 1:
        packed, output = Path(tmp) / 'packed', Path(tmp) / 'output'
        # A literal row exercises both four-byte fragments: 2*1 + (-1)*1 = 1.
        with paths[0].open('r+b') as f:
            f.seek(starts[0]); f.write(b'\x00\x3c\x02\x00\x00\x00\xff')
        image = bytearray(6912); image[:2] = b'\x00\x3c'; image[4] = image[8] = 1
        expected = bytearray(10240); expected[:4] = struct.pack('<f', 1.0)
        packed.write_bytes(image); output.write_bytes(expected)
        command = [str(Path(sys.argv[1]).resolve()), *(str(p) for p in paths), '0', str(packed), str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0 and json.loads(result.stdout)['exact'] is True, result.stderr
        image[:2] = b'\x01\x00'; packed.write_bytes(image)
        expected[:4] = struct.pack('<f', 2.0 ** -24); output.write_bytes(expected)
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0 and json.loads(result.stdout)['exact'] is True, result.stderr
        image[:2] = b'\x00\x7c'; packed.write_bytes(image)
        assert subprocess.run(command, capture_output=True, timeout=120).returncode
        image[:2] = b'\x01\x00'; packed.write_bytes(image)
        altered = bytearray(output.read_bytes()); altered[0] = 1; output.write_bytes(altered)
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode and 'projected byte mismatch' in result.stderr
        command[7] = '1'
        assert subprocess.run(command, capture_output=True, timeout=120).returncode
    with paths[0].open('r+b') as f:
        f.seek(type_offsets[0]); f.write(struct.pack('<I', 30))
    try:
        m.scan(paths)
    except ValueError as error:
        assert 'unsupported role type or shape' in str(error)
    else:
        raise AssertionError('BF16 output role accepted')
    with paths[0].open('r+b') as f:
        f.seek(type_offsets[0]); f.write(struct.pack('<I', 8))
    with paths[1].open('r+b') as f:
        f.seek(starts[1]); f.write(b'\x01\x00')
    assert m.scan(paths)['roles'][names[1]]['scale_classes']['subnormal'] == 1
    with paths[1].open('r+b') as f:
        f.seek(starts[1]); f.write(b'\x00\x7c')
    assert not m.scan(paths)['supported']
    try:
        m.scan(paths[::-1])
    except ValueError:
        pass
    else:
        raise AssertionError('unordered shards accepted')
print('GDN output role framing and finite-scale admission passed')
