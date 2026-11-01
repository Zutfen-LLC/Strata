"""Exact packed-byte gate for the bounded projection-input Q8_1 experiment."""
from pathlib import Path
import argparse
import json

SIZE = 6144 // 32 * 36


def compare(fixture, native, vulkan):
    fixture = Path(fixture).read_bytes()
    roots = [Path(native), Path(vulkan)]
    for root in roots:
        if (root / 'fixture.bin').read_bytes() != fixture:
            raise ValueError('projection q8: fixture identity mismatch')
    packed = [(root / 'qx.bin').read_bytes() for root in roots]
    if any(len(data) != SIZE for data in packed):
        raise ValueError('projection q8: packed output size')
    def pin(d, s, codes):
        return d.to_bytes(2, 'little') + s.to_bytes(2, 'little') + bytes(c % 256 for c in codes) + bytes(32 - len(codes))
    pins = [pin(0, 0, []), pin(0, 0x8000, []),
            pin(0x3c00, 0x57f0, [127, 1, -1, 2, -2]),
            pin(0x3c00, 0x57f0, [127, 0]),
            pin(0x7bff, 0x4000, [127, 0, -127, 0]),
            pin(0x7bff, 0x7bff, [127]), pin(0x3c00, 0xd7f0, [-127, 0]),
            pin(0x3c00, 0x57f1, [127])]
    for data in packed:
        if any(data[i * 36:(i + 1) * 36] != expected for i, expected in enumerate(pins)):
            raise ValueError('projection q8: independent packed control failed')
    differences = [{'byte': i, 'block': i // 36, 'block_byte': i % 36,
                    'native': a, 'vulkan': b}
                   for i, (a, b) in enumerate(zip(*packed)) if a != b]
    return {'compared_bytes': SIZE, 'different_bytes': len(differences),
            'differences': differences, 'exact': not differences,
            'qualified': False, 'model_integration': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'native', 'vulkan'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    result = compare(a.fixture, a.native, a.vulkan)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['exact'] else 1)
