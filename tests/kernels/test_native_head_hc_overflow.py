"""Head-only negative overflow through the real numerical checker, no GPU."""
import struct,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
import native_hc_mix_numeric as m
D,R=10240,320
with tempfile.TemporaryDirectory(prefix='head-negative-overflow-') as temp:
 d=Path(temp);fixture=d/'fixture.bin';raw=bytearray(13148168);raw[:8]=b'HCMIX01\0'
 # Independently chosen exact dot: 1 * -400 / 4 = -100, row 74.
 struct.pack_into('<I',raw,8,0x3f800000)
 struct.pack_into('<H',raw,40968+74*D*2,0xc3c8)
 fixture.write_bytes(raw)
 native=d/'native';vk=d/'vulkan';native.mkdir();vk.mkdir()
 low=[0]*R;low[74]=0x80000000
 mixed=[0]*2560;mixed[0]=0x3e000000
 def write(directory,name,values):
  (directory/(name+'.f32le')).write_bytes(struct.pack('<'+str(len(values))+'I',*values))
 for directory in (native,vk):
  write(directory,'low',low);write(directory,'mixed',mixed)
 pre=[0]*R;pre[74]=0xc2c80000
 exp=[0x3f800000]*R;exp[74]=0x7f800000
 for name,values in [('low-pre',pre),('low-exp',exp),('up-pre',[0]*D),('up-exp',[0x3f800000]*D),('gates',[0x3f000000]*D)]:write(vk,name,values)
 for backend,directory in [('native',native),('vulkan',vk)]:
  assert m.check(fixture,directory,backend,head_silu_overflow=True)['source_order_and_intrinsic_contract']
  try:m.check(fixture,directory,backend)
  except ValueError:pass
  else:raise AssertionError('shared HC checker accepted head-only overflow')
 for name,bad in [('low-exp',0x7f7fffff),('low',0)]:
  original=(vk/(name+'.f32le')).read_bytes();wrong=bytearray(original);struct.pack_into('<I',wrong,74*4,bad);(vk/(name+'.f32le')).write_bytes(wrong)
  try:m.check(fixture,vk,'vulkan',head_silu_overflow=True)
  except ValueError:pass
  else:raise AssertionError('invalid overflow '+name+' accepted')
  (vk/(name+'.f32le')).write_bytes(original)
print('Real head numerical checker accepts exact signed overflow and rejects shared/altered cases; constructed inputs only')
