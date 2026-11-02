"""Real producer CLIs with a host CUDA provider; no GPU parity is claimed."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[2]
control=Path(sys.argv[1]).resolve();vulkan=Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory(prefix='indexer-bf16-contract-') as tmp:
 d=Path(tmp);fixture=d/'fixture';subprocess.run([str(control),'--write',str(fixture)],cwd='/',check=True)
 (d/'cuda_runtime.h').write_text('''#pragma once\n#include <cstddef>\nstruct CUstream_st;using cudaStream_t=CUstream_st*;using cudaError_t=int;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;struct cudaDeviceProp{char name[256];};const char* cudaGetErrorString(cudaError_t);cudaError_t cudaMalloc(void**,size_t);cudaError_t cudaFree(void*);cudaError_t cudaMemcpy(void*,const void*,size_t,int);cudaError_t cudaSetDevice(int);cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);cudaError_t cudaGetLastError();cudaError_t cudaStreamCreate(cudaStream_t*);cudaError_t cudaStreamDestroy(cudaStream_t);cudaError_t cudaStreamSynchronize(cudaStream_t);\n''')
 (d/'provider.cpp').write_text(r'''#include "cuda_runtime.h"
#include "strata/kernels/bf16_gemv.hpp"
#include "native_qsa_indexer_bf16.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
struct CUstream_st{int marker;};static cudaStream_t active=nullptr;
const char* cudaGetErrorString(cudaError_t){return "provider failure";}
cudaError_t cudaMalloc(void**p,size_t n){*p=std::malloc(n);return *p?0:1;}cudaError_t cudaFree(void*p){std::free(p);return 0;}
cudaError_t cudaMemcpy(void*a,const void*b,size_t n,int){std::memcpy(a,b,n);return 0;}cudaError_t cudaSetDevice(int n){return n?1:0;}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*p,int){std::strcpy(p->name,"signature-fixture GPU");return 0;}
cudaError_t cudaGetLastError(){return 0;}cudaError_t cudaStreamCreate(cudaStream_t*s){if(active)return 1;active=*s=new CUstream_st{42};return 0;}
cudaError_t cudaStreamDestroy(cudaStream_t s){if(s!=active)return 1;delete s;active=nullptr;return 0;}cudaError_t cudaStreamSynchronize(cudaStream_t s){return s==active?0:1;}
namespace strata::kernels {void bf16_gemv_fp32_mmvf(const float*x,const uint16_t*w,float*y,int64_t ni,int64_t no,void*s){
 if(!x||!w||!y||ni!=2560||(no!=128&&no!=512)||s!=active||(reinterpret_cast<uintptr_t>(x)&7)||(reinterpret_cast<uintptr_t>(w)&3)||(reinterpret_cast<uintptr_t>(y)&3))throw std::runtime_error("native launcher precondition");
 auto v=core::portable_detail::qsa_indexer_bf16_control(x,w,size_t(no));std::memcpy(y,v.data(),v.size()*4);}}
''')
 native=d/'native';subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(d),'-I'+str(ROOT/'include'),'-I'+str(ROOT/'src/core'),'-I'+str(ROOT/'tests/kernels'),'-x','c++',str(ROOT/'tests/kernels/native_qsa_indexer_bf16_reference.cu'),str(d/'provider.cpp'),'-o',str(native)],check=True)
 for name,exe in [('input',control),('native',native),('vulkan',vulkan)]:
  out=d/(name+'.json');p=subprocess.run([str(exe),'--contract',str(fixture),str(out)],cwd='/',capture_output=True,text=True);assert p.returncode==0 and not p.stdout and not p.stderr,p.stderr
  assert json.loads(out.read_text())=={'fields':{'key.f32le':512,'query.f32le':2048},'qualified':False}
 output=d/'native-out';p=subprocess.run([str(native),'--device','0','--expect-device','signature-fixture GPU','--fixture',str(fixture),'--output',str(output)],cwd='/',capture_output=True,text=True);assert p.returncode==0,p.stderr
 subprocess.run([str(control),'--check',str(fixture),str(output/'key.f32le'),str(output/'query.f32le')],cwd='/',check=True)
 fake=d/'fake-vulkan';shutil.copytree(output,fake)
 (fake/'report.json').write_text(json.dumps({'actual_vulkan_producer':True,'immutable_canary_repeat':True,'qualified':False}))
 compare=[sys.executable,str(ROOT/'tools/native_qsa_indexer_bf16_compare.py'),'--control',str(control),'--control-sha256',hashlib.sha256(control.read_bytes()).hexdigest(),'--fixture',str(fixture),'--native',str(output),'--vulkan',str(fake)]
 passed=subprocess.run(compare,cwd='/',capture_output=True,text=True);assert passed.returncode==0,passed.stderr
 assert json.loads(passed.stdout)=={'bytes_compared':2560,'exact':True,'qualified':False}
 changed=bytearray((fake/'query.f32le').read_bytes());changed[0]^=1
 (fake/'query.f32le').write_bytes(changed);(fake/'query-repeat.f32le').write_bytes(changed)
 assert subprocess.run(compare,cwd='/',capture_output=True).returncode!=0
 shutil.copyfile(output/'query.f32le',fake/'query.f32le');shutil.copyfile(output/'query-repeat.f32le',fake/'query-repeat.f32le')
 wrong=compare.copy();wrong[wrong.index('--control-sha256')+1]='0'*64
 assert subprocess.run(wrong,cwd='/',capture_output=True).returncode!=0
 bad=fixture/'key.bf16le';data=bytearray(bad.read_bytes());data[0:2]=b'\x01\x00';bad.write_bytes(data)
 assert subprocess.run([str(vulkan),'--contract',str(fixture),str(d/'bad.json')],cwd='/',capture_output=True).returncode!=0
 assert subprocess.run(compare,cwd='/',capture_output=True).returncode!=0
 print('actual native and Vulkan startup contracts and native CLI pass with host CUDA provider; no GPU')
