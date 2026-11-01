"""Test-only: retain upstream algorithm bodies; replace includes with RNE shims."""
from pathlib import Path
import sys,re,json,hashlib
repo=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);root=repo/'third_party/llvm-libc-log1p'
for n,h in json.loads((root/'provenance.json').read_text())['files'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
body='#include "native_llvm_log1p_shim.hpp"\n'
for n in ('acosh_float_constants.h','acoshf_utils.h','log1pf.h'):body+=re.sub(r'^#include[^\n]*','',(root/'upstream'/n).read_text(),flags=re.M)+'\n'
(out/'strata_llvm_log1p_oracle.hpp').write_text(body)
s=(repo/'src/kernels/vulkan/native_llvm_log1p.glsl').read_text();a,b=s.split('double llvm_ma',1)
a=re.sub(r'uvec2\[\d+\]\(','{',a);a=a.replace('));\n',')};\n')
s=(a+'double llvm_ma'+b).replace('out double wide','double& wide').replace('out uint path','uint& path').replace('1.0LF','1.0')
(out/'strata_llvm_log1p_adapter_body.hpp').write_text(s)
