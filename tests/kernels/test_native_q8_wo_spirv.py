"""Validate WO module structure without executing a device."""
from pathlib import Path
import re,subprocess,sys,tempfile
shader,compiler,validator,disassembler=map(Path,sys.argv[1:])
with tempfile.TemporaryDirectory(prefix='q8-wo-spirv-') as t:
 module=Path(t)/'wo.spv';out=Path(t)/'wo.spvasm'
 subprocess.run([str(compiler),'-fshader-stage=compute','--target-env=vulkan1.2','-O',str(shader),'-o',str(module)],check=True)
 subprocess.run([str(validator),'--target-env','vulkan1.2',str(module)],check=True)
 subprocess.run([str(disassembler),str(module),'-o',str(out)],check=True)
 text=out.read_text();lines=text.splitlines()
 assert 'LocalSize 128 1 1' in text and 'OpCapability FMAKHR' in text and 'SPV_KHR_fma' in text
 dots=re.findall(r'^\s*(%\w+) = OpSDot %\w+ %\w+ %\w+ PackedVectorFormat4x8Bit$',text,re.M);assert len(dots)==2
 adds=[re.search(r'^\s*(%\w+) = OpIAdd %int ('+re.escape(dots[0])+r') ('+re.escape(dots[1])+r')$',line) for line in lines]
 dot_sum=next((m.group(1) for m in adds if m),None);assert dot_sum
 cast=re.search(r'^\s*(%\w+) = OpConvertSToF %float '+re.escape(dot_sum)+r'$',text,re.M);assert cast
 scale=re.findall(r'^\s*(%\w+) = OpFMul %float %\w+ %\w+$',text,re.M);assert len(scale)==1
 fma=re.findall(r'^\s*%\w+ = OpFmaKHR %float (%\w+) (%\w+) (%\w+)$',text,re.M);assert len(fma)==1 and fma[0][:2]==(scale[0],cast.group(1))
 fadds=re.findall(r'^\s*(%\w+) = OpFAdd %float (%\w+) (%\w+)$',text,re.M);assert len(fadds)==4
 assert fadds[1][1]==fadds[0][0] and fadds[2][1]==fadds[1][0]
 for ident in [scale[0]]+[x[0] for x in fadds]:assert f'OpDecorate {ident} NoContraction' in text
 for n in ('uint_192','uint_6528','uint_34','uint_32','uint_64','uint_96','uint_16'):
  assert re.search(r'%'+n+r' = OpConstant %uint ',text)
 assert 'OpBitwiseXor' in text and 'OpShiftRightLogical' in text
 print('WO SPIR-V signed dots, scale, strict FMA, ordered four-warp adds and XOR reduction verified; no device')
