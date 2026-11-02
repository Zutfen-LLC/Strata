"""Exact packed-byte gate for the bounded projection-input Q8_1 experiment."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile

SIZE = 6144 // 32 * 36
FROZEN_FIXTURES = frozenset({
    '0d9098e4e6802a38c768751d0961966c21c17686bba0dfc52d2c021b7e949413',
    'a4edc6eb16a60b47a49e0f5d4b9dfbc9299c445b94a4737568d0cee19e8be5e0',
    'f970f19c01da8e815227df7bc14129f14ac58c0df16ec85f44b66cf544c07688',
    '24add7ab099be5fde0072f44f68b2f6228ea967f165fa33b8f3135e42b14758a',
    '4203c42967b515536d69ecafcae8e6fd5fd498b90011ecffbae0cdd1bb60b055',
})


def compare(fixture, native, vulkan, validator, validator_sha256, fixture_sha256):
    fixture = Path(fixture).read_bytes()
    digest = hashlib.sha256(fixture).hexdigest()
    if fixture_sha256 not in FROZEN_FIXTURES or digest != fixture_sha256:
        raise ValueError('projection q8: fixture does not match a frozen case')
    validator = Path(validator).resolve(strict=True)
    if hashlib.sha256(validator.read_bytes()).hexdigest() != validator_sha256:
        raise ValueError('projection q8: validator identity mismatch')
    # Validate the exact bytes being compared, from an independent temporary copy.
    # The package binds the validator digest to the compiled control executable.
    with tempfile.TemporaryDirectory(prefix='projection-q8-validate-') as tmp:
        candidate = Path(tmp) / 'fixture.bin'
        candidate.write_bytes(fixture)
        result = subprocess.run([str(validator), '--validate', str(candidate)],
                                cwd=tmp, capture_output=True, text=True, timeout=10)
        if result.returncode != 0 or result.stdout or result.stderr:
            raise ValueError('projection q8: compiled input validation failed')
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
            'fixture_sha256': digest, 'validator_sha256': validator_sha256,
            'qualified': False, 'model_integration': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'fixture-sha256', 'validator', 'validator-sha256', 'native', 'vulkan'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    result = compare(a.fixture, a.native, a.vulkan, a.validator, a.validator_sha256, a.fixture_sha256)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['exact'] else 1)
