#!/usr/bin/env python3
"""Create synthetic inputs for the existing HC mixer component, not model data."""
import argparse,hashlib,struct
from pathlib import Path
# Inputs are four times the target where representable; each row selects one
# BF16 1.0. The final case reaches the largest finite down dot without overflow.
CASES={
 'tail': (0xc3c803e5,0xc3b20000,0xc3b20001,0xc3c80000,0xc4000000,0xff7fffff),
 'transition': (0xc3b1ffff,0xc3b00000,0xc3b170a4,0xc3aeac4d),
 'near': (0xc3b1ffff,),
 'neg88': (0xc3b00000,),
 'neg8872': (0xc3b170a4,),
 'beyond': (0xc3aeac4d,),
 'positive': (0x43c80000,),
}
D=10240

def generate(kind,destination):
 if kind not in CASES:raise ValueError('unsupported fixture kind')
 p=Path(destination)
 if p.exists() or p.is_symlink():raise ValueError('fresh fixture required')
 data=bytearray(13148168);data[:8]=b'HCMIX01\0'
 for row,bits in enumerate(CASES[kind]):
  struct.pack_into('<I',data,8+row*4,bits)
  struct.pack_into('<H',data,40968+(row*D+row)*2,0x3f80)
 with p.open('xb') as f:f.write(data)
 return hashlib.sha256(data).hexdigest()

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=sorted(CASES));parser.add_argument('output');args=parser.parse_args()
 print(generate(args.kind,args.output))
