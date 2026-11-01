"""Inspect compiled arithmetic, not a native numerical qualification."""
import pathlib,subprocess,sys,tempfile
root=pathlib.Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d)/'hc.spv';subprocess.run([sys.argv[1],'-fshader-stage=compute','--target-env=vulkan1.2','-O',str(root/'src/kernels/vulkan/native_hc.comp'),'-o',str(p)],check=True)
 subprocess.run([sys.argv[3],'--target-env','vulkan1.2',str(p)],check=True);s=subprocess.check_output([sys.argv[2],str(p)],text=True)
 for text in ('OpCapability FMAKHR','"SPV_KHR_fma"','OpFmaKHR','NoContraction','OpControlBarrier','InverseSqrt','LocalSize 256 1 1'):assert text in s,text
 assert ' Fma ' not in s and 'OpGroupNonUniform' not in s
 precise={line.split()[1] for line in s.splitlines() if 'OpDecorate' in line and 'NoContraction' in line}
 for line in s.splitlines():
  if any(op in line for op in ('= OpFAdd ','= OpFMul ','= OpFDiv ')):assert line.split()[0] in precise,line
 # Trace the actual compiled first square operands through loads/access chains.
 defs={line.split('=',1)[0].strip():line.split('=',1)[1].split() for line in s.splitlines() if '=' in line}
 def lane(value):
  load=defs[value];assert load[:2]==['OpLoad','%float'],load
  access=defs[load[2]];assert access[0]=='OpAccessChain',access
  const=defs[access[-1]];assert const[0]=='OpConstant',const
  return int(const[-1])
 mul=next((key,op) for key,op in defs.items() if op[0]=='OpFMul')
 fmas=[(key,op) for key,op in defs.items() if op[0]=='OpFmaKHR']
 assert lane(mul[1][2])==lane(mul[1][3])==1,'native first rounded square must be y*y'
 prior=mul[0]
 for (key,op),expected in zip(fmas[:3],(0,2,3)):
  assert lane(op[2])==lane(op[3])==expected and op[4]==prior,(expected,op)
  prior=key
 print('native HC strict source-order shader compile/validation passed; rsqrt equivalence unqualified')
