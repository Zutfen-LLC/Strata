from pathlib import Path
import importlib.util,io,sys
p=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('wo_extract',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lines=['code for sm_75\n','Function : native_small_mmvq_kernel_wrong_arch\n','/*0000*/ EXIT;\n','code for sm_86\n','Function : native_small_mmvq_kernel_small\n','/*0000*/ EXIT;\n','Function : native_small_mmvq_kernel_target\n','/*0008*/ FFMA R0,R1,R2,R3;\n','Function : unrelated\n','PRIVATE\n']
def demangle(n):return 'void native_small_mmvq_kernel<(anonymous namespace)::Q80Block, 8, '+('false' if n.endswith('target') else 'true')+'>(void)'
out=io.StringIO();records=m.select(lines,out,demangle);assert len(records)==1 and records[0]['instructions']==1 and 'PRIVATE' not in out.getvalue() and 'small\n' not in out.getvalue()
for invalid in (lines[:3],lines[:-4],lines+lines[-4:-2]):
 try:m.select(invalid,io.StringIO(),demangle)
 except ValueError:continue
 raise AssertionError('missing or duplicate selected kernel accepted')
print('selected native kernel extraction and refusal controls pass; synthetic SASS only')
