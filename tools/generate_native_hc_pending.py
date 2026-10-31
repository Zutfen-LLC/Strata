"""Deterministic public synthetic roles; no model extraction or GPU use."""
import argparse,struct
from pathlib import Path

def generate(kind):
 if kind not in ['zero','cancel','sparse']:raise ValueError('synthetic pending case')
 bo=[(d%7-3)/128 for d in range(2560)];inj=[-8.,-1.,0.,4.];x=[(i%17-8)/32 for i in range(10240)]
 if kind=='zero':x=[0.]*10240;bo=[0.]*2560;inj=[0.]*4
 if kind=='cancel':inj=[0.]*4;x=[-bo[i%2560] for i in range(10240)]
 w=[.75+(i%13)/32 for i in range(10240)]
 def bf16(v):
  b=struct.unpack('<I',struct.pack('<f',v))[0];return (b+0x7fff+((b>>16)&1))>>16
 wi=[bf16(((i*31)%97-48)/512) for i in range(40960)]
 down=bytearray(6553600);up=bytearray(6553600)
 for row in range(320):struct.pack_into('<H',down,(row*10240+(row*31)%10240)*2,bf16((row%7-3)/32))
 for row in range(10240):struct.pack_into('<H',up,(row*320+(row*7)%320)*2,bf16((row%5-2)/32))
 return b'HCPEND1\0'+struct.pack('<10240f',*x)+struct.pack('<10240f',*w)+struct.pack('<40960H',*wi)+struct.pack('<2560f',*bo)+struct.pack('<4f',*inj)+down+up
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',choices=['zero','cancel','sparse'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.output.open('xb') as f:f.write(generate(a.case))
