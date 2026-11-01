"""Consume the existing compiled producer fields/byte-size contract; no runner."""
import json
from pathlib import Path


def require(directory, contract):
    data = json.loads(Path(contract).read_text())
    fields = data.get('fields')
    if data.get('qualified') is not False or not isinstance(fields, dict) or not fields:
        raise ValueError('invalid producer output contract')
    for name, size in fields.items():
        if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..') or
                type(size) is not int or not 0 < size <= 32 << 20):
            raise ValueError('invalid producer field/span')
        path = Path(directory) / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
            raise ValueError('producer output absent/invalid: ' + name)
    return fields
