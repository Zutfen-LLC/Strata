"""Exercise the actual compiled producer/control contract from another cwd."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from component_output_contract import require

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    fixture = root / 'fixture.bin'
    subprocess.run([sys.argv[1], '--synthetic', '2', str(fixture)], cwd=root, check=True)
    for i, program in enumerate(sys.argv[1:]):
        out = root / f'contract-{i}.json'
        r = subprocess.run([program, '--contract', str(fixture), str(out)], cwd=root,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert not r.stdout and not r.stderr
    truncated = root / 'truncated.bin'
    truncated.write_bytes(fixture.read_bytes()[:-1])
    for i, program in enumerate(sys.argv[1:]):
        invalid = root / f'invalid-{i}.json'
        r = subprocess.run([program, '--contract', str(truncated), str(invalid)], cwd=root,
                           capture_output=True, text=True)
        assert r.returncode != 0 and not invalid.exists()
    assert (root / 'contract-0.json').read_bytes() == (root / 'contract-1.json').read_bytes()
    fields = json.loads((root / 'contract-0.json').read_text())['fields']
    assert fields == {'qx.bin': 6912}
    (root / 'qx.bin').write_bytes(bytes(6912))
    require(root, root / 'contract-0.json')
    (root / 'qx.bin').rename(root / 'wrong.bin')
    try:
        require(root, root / 'contract-0.json')
    except ValueError:
        pass
    else:
        raise AssertionError('wrong output name accepted')
    print('actual producer and control contracts agree from unrelated cwd')
