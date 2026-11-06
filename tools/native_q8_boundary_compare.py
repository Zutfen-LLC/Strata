#!/usr/bin/env python3
"""Compare raw Q8_1 blocks from one named CUDA producer and Vulkan observation."""
import argparse
import json
from pathlib import Path

NAMES = ('control', 'signed_subnormal', 'normal_boundary', 'rounding')
def read(path):
    data = path.read_bytes()
    if len(data) != 144:
        raise ValueError(f'{path}: expected exactly four 36-byte Q8_1 blocks')
    return [{'scale_bits_le': data[i:i+2].hex(), 'sum_bits_le': data[i+2:i+4].hex(),
             'quantized_bytes': data[i+4:i+36].hex()} for i in range(0, 144, 36)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda', type=Path, required=True)
    p.add_argument('--vulkan', type=Path, required=True)
    p.add_argument('--producer', choices=('expert','gdn-projection'), required=True)
    a=p.parse_args(); c,v=read(a.cuda),read(a.vulkan)
    rows=[{'block': n, 'cuda': x, 'vulkan': y, 'equal': x==y} for n,x,y in zip(NAMES,c,v)]
    print(json.dumps({'producer':a.producer, 'qualified':False, 'fp32_denorm':'unspecified',
                      'all_equal':all(r['equal'] for r in rows), 'blocks':rows}, indent=2))
    return 0 if all(r['equal'] for r in rows) else 1
if __name__=='__main__': raise SystemExit(main())
