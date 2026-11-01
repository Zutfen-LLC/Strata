"""Compile actual shared-array declaration branches; host syntax/layout only."""
from pathlib import Path
import os,re,shutil,subprocess,tempfile
repo=Path(__file__).resolve().parents[1]
source=(repo/'src/kernels/cuda/iq_kernels.cu').read_text()
block=re.search(r'#if defined\(__HIPCC__\)\n#define STRATA_SHARED_ALIGN16_U32.*?#endif',source,re.S).group()
declarations=re.findall(r'STRATA_SHARED_ALIGN16_INT4\(s_u, (ED_NC \* 2 \* ED_[XH]B)\);',source)
assert declarations==['ED_NC * 2 * ED_XB','ED_NC * 2 * ED_HB']
assert not re.search(r'__shared__\s+alignas\(16\)\s+int4',source.replace(block,''))
compiler=shutil.which(os.environ.get('CXX','c++'));assert compiler,'host C++ compiler required'
with tempfile.TemporaryDirectory(prefix='shared-int4-') as tmp:
 root=Path(tmp)
 for hip in [False,True]:
  for expression in declarations:
   prefix=('#define __HIPCC__ 1\n' if hip else '')+'#define __shared__\nstruct int4 {int x,y,z,w;};\n#define ED_NC 4\n#define ED_XB 8\n#define ED_HB 16\n'
   text=prefix+block+'\n'+f'STRATA_SHARED_ALIGN16_INT4(s_u, {expression});\n'+f'static_assert(sizeof(s_u)==({expression})*16);\nstatic_assert(__alignof__(s_u)>=16);\n'
   cpp=root/'check.cpp';cpp.write_text(text)
   result=subprocess.run([compiler,'-std=c++17','-Werror','-fsyntax-only',str(cpp)],capture_output=True,text=True)
   assert result.returncode==0,result.stderr
   if hip:
    cpp.write_text(text.replace('__attribute__((aligned(16)))',''))
    result=subprocess.run([compiler,'-std=c++17','-fsyntax-only',str(cpp)],capture_output=True,text=True)
    assert result.returncode!=0,'missing HIP alignment not detected'
print('PASS actual HIP/CUDA declaration branches and both array extents, 16-byte storage/size, missing-alignment rejection; host compiler only, not HIP/CUDA kernel coverage')
