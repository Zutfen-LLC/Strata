from pathlib import Path
import subprocess
import hashlib
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import native_qsa_projection_q8_numeric as N

FIXTURE_HASHES = [
    '0d9098e4e6802a38c768751d0961966c21c17686bba0dfc52d2c021b7e949413',
    'a4edc6eb16a60b47a49e0f5d4b9dfbc9299c445b94a4737568d0cee19e8be5e0',
    'f970f19c01da8e815227df7bc14129f14ac58c0df16ec85f44b66cf544c07688',
    '24add7ab099be5fde0072f44f68b2f6228ea967f165fa33b8f3135e42b14758a',
    '4203c42967b515536d69ecafcae8e6fd5fd498b90011ecffbae0cdd1bb60b055',
]

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for mode in range(5):
        fixture = root / f'fixture{mode}.bin'
        subprocess.run([sys.argv[1], '--synthetic', str(mode), str(fixture)], check=True)
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == FIXTURE_HASHES[mode]
        control = root / f'host{mode}.bin'
        subprocess.run([sys.argv[1], '--host-control', str(fixture), str(control)], check=True)
        for side in ('native', 'vulkan'):
            d = root / f'{side}{mode}'
            d.mkdir()
            (d / 'fixture.bin').write_bytes(fixture.read_bytes())
            (d / 'qx.bin').write_bytes(control.read_bytes())
        native, vulkan = root / f'native{mode}', root / f'vulkan{mode}'
        assert N.compare(fixture, native, vulkan)['exact']
        data = bytearray(control.read_bytes())
        data[8 * 36] ^= 1
        (vulkan / 'qx.bin').write_bytes(data)
        result = N.compare(fixture, native, vulkan)
        assert not result['exact'] and result['differences'][0]['byte'] == 8 * 36
        common = bytearray(control.read_bytes())
        common[2 * 36 + 4] ^= 1
        (native / 'qx.bin').write_bytes(common)
        (vulkan / 'qx.bin').write_bytes(common)
        try:
            N.compare(fixture, native, vulkan)
        except ValueError:
            pass
        else:
            raise AssertionError('identical but wrong pinned outputs accepted')
        (vulkan / 'qx.bin').write_bytes(bytes(N.SIZE - 1))
        try:
            N.compare(fixture, native, vulkan)
        except ValueError:
            pass
        else:
            raise AssertionError('truncated packed output accepted')
    print('exact packed-byte comparator rejects differences and truncated output')
