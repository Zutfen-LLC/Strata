"""Deterministic public synthetic operands, no model data."""
import argparse,struct
from pathlib import Path

def generate(case):
 x=[((i*73%257)-128)/128 for i in range(10240)];down=[0]*3276800;up=[0]*3276800
 if case=='zero':x=[0.]*10240
 elif case not in ['sparse','dense']:raise ValueError('case')
 if case=='dense':
  # Small exactly representable BF16 weights keep all activation preinputs
  # inside the separately checked [-16,16] contract.
  down=[(0x3900 if (i*13%7)<3 else 0xb900) for i in range(3276800)]
  up=[(0x3b00 if (i*17%11)<5 else 0xbb00) for i in range(3276800)]
 elif case=='sparse':
  # Nonzero entries cover lane/chunk/tile/stream/row boundaries.
  for r in range(320):
   for j in [0,7,8,255,256,2559,2560,5119,5120,7679,7680,10239]:down[r*10240+j]=(0x3e00 if (r+j)%2 else 0xbe00)
  for r in range(10240):
   for j in [0,7,8,255,256,319]:up[r*320+j]=(0x3e00 if (r+j)%2 else 0xbe00)
 return b'HCMIX01\0'+struct.pack('<10240f',*x)+struct.pack('<3276800H',*down)+struct.pack('<3276800H',*up)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['zero','sparse','dense'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
 with Path(a.output).open('xb') as f:f.write(generate(a.case))
