#!/usr/bin/env python3
"""Compare the unchanged native and candidate Vulkan component outputs exactly."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--control-sha256', required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--vulkan', type=Path, required=True)
    a = p.parse_args()
    control = a.control.resolve(strict=True)
    if sha(control) != a.control_sha256:
        raise ValueError('control executable hash mismatch')
    with tempfile.TemporaryDirectory(prefix='indexer-bf16-compare-') as tmp:
        contract = Path(tmp) / 'contract.json'
        subprocess.run([str(control), '--contract', str(a.fixture.resolve()), str(contract)],
                       cwd='/', check=True, capture_output=True)
        if json.loads(contract.read_text()) != {'fields': {'key.f32le': 512, 'query.f32le': 2048}, 'qualified': False}:
            raise ValueError('unexpected fixture contract')
    for directory, flag in ((a.native, 'actual_native_launcher'), (a.vulkan, 'actual_vulkan_producer')):
        if json.loads((directory / 'report.json').read_text()) != {
                flag: True, 'immutable_canary_repeat': True, 'qualified': False}:
            raise ValueError('producer report mismatch')
        for role, length in (('key', 512), ('query', 2048)):
            data = (directory / (role + '.f32le')).read_bytes()
            if len(data) != length or data != (directory / (role + '-repeat.f32le')).read_bytes():
                raise ValueError('output size or repeat mismatch: ' + role)
        subprocess.run([str(control), '--check', str(a.fixture.resolve()),
                        str((directory / 'key.f32le').resolve()), str((directory / 'query.f32le').resolve())],
                       cwd='/', check=True, capture_output=True)
    for role in ('key', 'query'):
        if (a.native / (role + '.f32le')).read_bytes() != (a.vulkan / (role + '.f32le')).read_bytes():
            raise ValueError('native/Vulkan byte mismatch: ' + role)
    print(json.dumps({'exact': True, 'bytes_compared': 2560, 'qualified': False}, sort_keys=True))


if __name__ == '__main__':
    main()
