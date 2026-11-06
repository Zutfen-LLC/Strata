#!/usr/bin/env python3
"""Compare one distinct-expert T2/K10 CPU record with retained Vulkan stages."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import compare_native_q8_layer as native


def read_exact(path, size):
    data = Path(path).read_bytes()
    if len(data) != size:
        raise ValueError(f'wrong size: {path}')
    return data


def q8_fields(cpu, vulkan):
    if len(cpu) % 34 or len(vulkan) != len(cpu) // 34 * 36:
        raise ValueError('packed Q8 shape mismatch')
    # Q8_1 contains an additional sum half; Q8_0 does not consume it.
    return (b''.join(cpu[i:i+2] + cpu[i+2:i+34] for i in range(0, len(cpu), 34)),
            b''.join(vulkan[i:i+2] + vulkan[i+4:i+36] for i in range(0, len(vulkan), 36)))


def compare(a, b, unit):
    if len(a) != len(b) or len(a) % unit:
        raise ValueError('stage shape mismatch')
    diffs = [i//unit for i in range(0, len(a), unit) if a[i:i+unit] != b[i:i+unit]]
    return {'equal': not diffs, 'different': len(diffs), 'first': diffs[0] if diffs else None,
            'cpu_sha256': hashlib.sha256(a).hexdigest(), 'vulkan_sha256': hashlib.sha256(b).hexdigest()}


def run(cpu, vk):
    raw = native.read(vk, '.metadata')
    tokens, entries, _ = native.parse_metadata(raw)
    normal, _ = native.normalize_metadata_header(raw)
    lines = normal.decode('ascii').splitlines()
    original = list(map(int, lines[7].split()))
    if tokens != 2 or entries != 20 or sorted(original) != list(range(20)):
        raise ValueError('expected full original T2/K10 association')
    def stage(suffix, size):
        return read_exact(str(vk)+suffix, size)
    vi = stage('.input.q8_1', 2*80*36)
    vg = stage('.gate.f32', 20*640*4)
    vu = stage('.up.f32', 20*640*4)
    vh = stage('.hidden.q8_1', 20*20*36)
    vo = stage('.output.f32', 20*2560*4)
    result = {'qualified': False, 'entries': [], 'input': [], 'union': None}
    for t in range(2):
        packed = read_exact(cpu/f'entry-{t*10}.input.q8_0', 80*34)
        a,b = q8_fields(packed, vi[t*80*36:(t+1)*80*36])
        result['input'].append(compare(a,b,1))
    for j, i in enumerate(original):
        row = {'original_entry': i, 'token': i//10}
        for name, src, width in [('gate',vg,640),('up',vu,640),('row',vo,2560)]:
            cpu_data=read_exact(cpu/f'entry-{i}.{name}.f32',width*4)
            row[name]=compare(cpu_data,src[j*width*4:(j+1)*width*4],4)
        hidden=read_exact(cpu/f'entry-{i}.hidden.q8_0',20*34)
        a,b=q8_fields(hidden,vh[j*20*36:(j+1)*20*36])
        row['hidden_packed']=compare(a,b,1)
        result['entries'].append(row)
    result['union']=compare(read_exact(cpu/'union.f32',2*2560*4),
                            read_exact(cpu/'vulkan-union.f32',2*2560*4),4)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cpu',type=Path);p.add_argument('vulkan',type=Path)
    args=p.parse_args()
    print(json.dumps(run(args.cpu,args.vulkan),indent=2))

if __name__ == '__main__':
    main()
