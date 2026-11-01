"""Deterministic synthetic final-write operands only; no weights/model access."""
import argparse,struct
from pathlib import Path
def generate(case):
 if case not in ['zero','cancel','sparse']:raise ValueError('write synthetic case')
 b=[(d%7-3)/128 for d in range(2560)];i=[-8.,-1.,0.,4.];r=[(d%17-8)/32 for d in range(10240)]
 if case=='zero':r=[0.]*10240;b=[0.]*2560;i=[0.]*4
 if case=='cancel':i=[-0.]*4;r=[-b[d%2560] for d in range(10240)]
 return b'HCWRITE1'+struct.pack('<12804f',*(r+b+i))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['zero','cancel','sparse'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.output.open('xb') as f:f.write(generate(a.case))
