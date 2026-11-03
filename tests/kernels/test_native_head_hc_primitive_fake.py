"""Exercise probe host/CLI with a fake CUDA provider and rewritten launch syntax."""
from pathlib import Path
import json,os,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];source=root/'tests/kernels/native_head_hc_primitive_probe.cu';checker=root/'tools/native_head_hc_primitive_report.py'
with tempfile.TemporaryDirectory(prefix='head-primitive-fake-') as tmp:
 d=Path(tmp)
 (d/'cuda_runtime.h').write_text('''#pragma once
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#define __global__
typedef int cudaError_t;typedef void* cudaStream_t;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;struct cudaDeviceProp{char name[256]="signature-fixture GPU";};struct FakeDim{unsigned x=0;};static FakeDim blockIdx,threadIdx;
inline const char* cudaGetErrorString(int){return "fake";}inline int cudaMalloc(void**p,size_t n){*p=malloc(n);return *p?0:1;}inline int cudaFree(void*p){free(p);return 0;}inline int cudaMemcpyAsync(void*d,const void*s,size_t n,int,void* stream){if(!stream)return 1;memcpy(d,s,n);return 0;}inline int cudaSetDevice(int n){return n==0?0:1;}inline int cudaGetDeviceProperties(cudaDeviceProp*p,int){*p=cudaDeviceProp{};return 0;}inline int cudaStreamCreate(void**s){*s=malloc(1);return *s?0:1;}inline int cudaStreamDestroy(void*s){free(s);return 0;}inline int cudaGetLastError(){return 0;}inline int cudaStreamSynchronize(void*s){return s?0:1;}
inline float __uint_as_float(uint32_t x){float f;memcpy(&f,&x,4);return f;}inline uint32_t __float_as_uint(float f){uint32_t x;memcpy(&x,&f,4);return x;}inline float __expf(float x){return std::exp(x);}inline float rsqrtf(float x){return 1.0f/std::sqrt(x);}
''')
 text=source.read_text();needle='native_head_hc_primitive_observation<<<1,1,0,stream.value>>>'
 assert text.count(needle)==1;text=text.replace(needle,'native_head_hc_primitive_observation');fake=d/'probe.cpp';fake.write_text(text);exe=d/'probe'
 subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-I'+str(d),str(fake),'-o',str(exe)],check=True,capture_output=True)
 p=subprocess.run([str(exe),'--contract'],cwd='/',capture_output=True,text=True);assert p.returncode==0 and p.stdout.strip()=='native_head_hc_primitive_probe_v1 fixed_inputs observation_only'
 out=d/'out';cmd=[str(exe),'--device','0','--expect-device','signature-fixture GPU','--output',str(out)];p=subprocess.run(cmd,cwd='/',capture_output=True,text=True);assert p.returncode==0,p.stderr
 p=subprocess.run([sys.executable,'-I','-S','-B',str(checker),str(out)],cwd='/',capture_output=True,text=True);assert p.returncode==0,p.stderr;assert json.loads(p.stdout)['comparisons']['fixed_division_matches_rne']
 assert subprocess.run(cmd,cwd='/',capture_output=True).returncode!=0
 assert subprocess.run([str(exe),'--device','0','--expect-device','','--output',str(d/'empty')],cwd='/',capture_output=True).returncode!=0 and not (d/'empty').exists()
print('Probe host CLI, fixed inputs, repeat/canaries and report pass with fake provider and rewritten launch; CUDA not compiled or executed')
