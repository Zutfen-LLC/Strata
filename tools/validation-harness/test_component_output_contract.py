"""Actual convolution producer output, including the recurrence-y filename trap."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import component_output_contract as C

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    contract = root / 'contract.json'
    subprocess.run([sys.argv[1], '--contract', str(contract)], check=True)
    expected = {'next-history.f32le': 122880, 'h.f32le': 40960, 'pre.f32le': 40960,
                'exp.f32le': 40960, 'silu.f32le': 40960, 'rs.f32le': 128, 'norm-arg.f32le': 128}
    assert json.loads(contract.read_text())['fields'] == expected
    for name, size in expected.items():
        (root / name).write_bytes(bytes(size))
    assert C.require(root, contract) == expected
    (root / 'h.f32le').rename(root / 'y.f32le')
    try:
        C.require(root, contract)
    except ValueError as error:
        assert 'h.f32le' in str(error)
    else:
        raise AssertionError('recurrence y silently substituted for convolution h')
    (root / 'y.f32le').rename(root / 'h.f32le')
    (root / 'rs.f32le').write_bytes(bytes(124))
    try:
        C.require(root, contract)
    except ValueError:
        pass
    else:
        raise AssertionError('truncated primitive field accepted')
    (root / 'rs.f32le').unlink()
    (root / 'rs.f32le').symlink_to(root / 'norm-arg.f32le')
    try:
        C.require(root, contract)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpected output symlink accepted')
print('compiled producer h/history contract, recurrence-y substitution, truncated and linked outputs checked; no GPU')
