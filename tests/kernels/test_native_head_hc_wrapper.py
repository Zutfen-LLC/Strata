"""Compile the actual native wrapper against an explicitly simulated CUDA provider."""
from pathlib import Path
import json,os,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];tool=Path(sys.argv[1]).resolve();cxx=os.environ.get('CXX','c++');ggml=Path(sys.argv[2])
with tempfile.TemporaryDirectory(prefix='head-wrapper-') as tmp:
 d=Path(tmp)
 (d/'cuda_runtime.h').write_text('''#pragma once
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;typedef void* cudaStream_t;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;struct cudaDeviceProp{char name[256]="signature-fixture GPU";};
inline const char* cudaGetErrorString(int){return "fake";}inline int cudaMalloc(void**p,size_t n){*p=malloc(n);return *p?0:1;}inline int cudaFree(void*p){free(p);return 0;}inline int cudaMemset(void*p,int v,size_t n){memset(p,v,n);return 0;}inline int cudaMemcpy(void*d,const void*s,size_t n,int){memcpy(d,s,n);return 0;}inline int cudaSetDevice(int n){return n==0?0:1;}inline int cudaGetDeviceProperties(cudaDeviceProp*p,int){*p=cudaDeviceProp{};return 0;}inline int cudaStreamCreate(void**s){*s=malloc(1);return 0;}inline int cudaStreamDestroy(void*s){free(s);return 0;}inline int cudaGetLastError(){return 0;}inline int cudaStreamSynchronize(void*s){return s?0:1;}
''')
 (d/'fake.cpp').write_text('''#include "strata/kernels/fused_gr.hpp"
#include <cstring>
#include <stdexcept>
namespace strata::kernels {bool fused_gr_supported(int64_t a,int64_t b,int64_t c){return a==2560&&b==4&&c==320;}void fused_gr_check(){} int fused_gr_variant(){return 3;} bool fused_gr_read_multi(const FusedGrArgs*a,int t,float*x,void*s,unsigned long long*,int){if(!a||t!=1||!x||!s||a->apply||a->w_inject||a->inject_out||!a->R||!a->w_norm||!a->w_down||!a->w_up||!a->lo||!a->rs||!a->mixed)throw std::runtime_error("fake bad native call");memset(x,0,40960);memset(a->rs,0,16);memset(a->lo,0,1280);memset(a->mixed,0,10240);return false;}}
''')
 exe=d/'oracle';subprocess.run([cxx,'-std=c++17','-x','c++','-I'+str(d),'-I'+str(root/'include'),'-I'+str(ggml/'include'),str(root/'tests/kernels/native_head_hc_reference.cu'),str(d/'fake.cpp'),'-o',str(exe)],check=True,capture_output=True)
 if os.environ.get('HEAD_HC_FAKE_NATIVE_OUT'):
  import shutil;shutil.copyfile(exe,os.environ['HEAD_HC_FAKE_NATIVE_OUT'])
 subprocess.run([str(exe),'--contract'],cwd='/',check=True,capture_output=True)
 for case in ('zero','sparse','dense','cancel','tails'):
  f=d/(case+'.bin');o=d/(case+'-out');subprocess.run([str(tool),'--make',case,str(f)],cwd='/',check=True)
  cmd=[str(exe),'--device','0','--expect-device','signature-fixture GPU','--fixture',str(f),'--output',str(o)]
  subprocess.run(cmd,cwd='/',check=True,capture_output=True);r=json.loads((o/'report.json').read_text());assert r['native_launcher'] and r['repeat'] and r['canaries'] and not r['qualified']
  assert (o/'fixture.bin').read_bytes()==f.read_bytes()
  assert subprocess.run(cmd,cwd='/',capture_output=True).returncode!=0
 mode=d/'mode-out';env=dict(os.environ,STRATA_GR_NORM_FAST='1');assert subprocess.run([str(exe),'--device','0','--expect-device','signature-fixture GPU','--fixture',str(d/'zero.bin'),'--output',str(mode)],cwd='/',env=env,capture_output=True).returncode!=0 and not mode.exists()
 bad=d/'bad';bad.write_bytes(b'bad');assert subprocess.run([str(exe),'--device','0','--expect-device','signature-fixture GPU','--fixture',str(bad),'--output',str(d/'bad-out')],cwd='/',capture_output=True).returncode!=0
print('Actual native wrapper five cases, device/stream/no-injection, repeat, input/canaries and refusal pass with simulated provider; no GPU')
