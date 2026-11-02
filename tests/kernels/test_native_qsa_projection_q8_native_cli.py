"""Run the actual native producer CLI with a host CUDA provider fixture.

The real launcher precondition functions are extracted from native_mmvq.cu and
its public declaration is compiled directly. Host quantization simulates output;
this test checks API usage and cleanup, not native numerical parity.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/kernels/cuda/native_mmvq.cu'

def build(destination, replay=False):
    destination = Path(destination)
    header = destination / 'cuda_runtime.h'
    header.write_text(r'''#pragma once
#include <cstddef>
struct CUstream_st;
using cudaStream_t = CUstream_st*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
struct cudaDeviceProp { char name[256]; int major; int minor; };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaMalloc(void**,size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*,const void*,size_t,int);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);
cudaError_t cudaGetLastError();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
''')
    source = SOURCE.read_text()
    start = source.index('void validate_shape(')
    end = source.index('void launch_check()', start)
    preconditions = source[start:end]
    assert 'validate_stream' in preconditions and 'if (!stream)' in preconditions
    assert 'validate_stream(stream);' in source[source.index('void native_quantize_q8_1('):source.index('void native_swiglu_quantize_q8_1(', source.index('void native_quantize_q8_1('))]
    provider = destination / 'provider.cpp'
    provider.write_text(r'''#include "cuda_runtime.h"
#include "native_qsa_projection_q8_fixture.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <cstdint>
struct CUstream_st { int marker; };
static cudaStream_t active=nullptr;
static int launches=0;
static void log(const char* s){std::ofstream(std::getenv("Q8_TEST_TRACE"),std::ios::app)<<s<<'\n';}
const char* cudaGetErrorString(cudaError_t){return "provider failure";}
cudaError_t cudaMalloc(void** p,size_t n){*p=std::malloc(n);return *p?0:1;}
cudaError_t cudaFree(void* p){std::free(p);return 0;}
cudaError_t cudaMemcpy(void* a,const void* b,size_t n,int){std::memcpy(a,b,n);return 0;}
cudaError_t cudaSetDevice(int n){return n==0?0:1;}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p,int){std::strcpy(p->name,"signature-fixture GPU");p->major=8;p->minor=6;return 0;}
cudaError_t cudaGetLastError(){return 0;}
cudaError_t cudaDeviceSynchronize(){return 0;}
cudaError_t cudaStreamCreate(cudaStream_t* s){if(active)return 1;active=*s=new CUstream_st{42};log("create");return 0;}
cudaError_t cudaStreamDestroy(cudaStream_t s){if(!s||s!=active)return 1;log("destroy");delete s;active=nullptr;return 0;}
cudaError_t cudaStreamSynchronize(cudaStream_t s){if(!s||s!=active)return 1;log("sync");return 0;}
constexpr int Q8K=32,MAX_NCOLS=8;
''' + preconditions + r'''
namespace strata::kernels {
void native_quantize_q8_1(const float* x,void* q,int n,int cols,void* stream){
    validate_shape(n,cols);validate_pointer(x);validate_pointer(q);validate_stream(stream);
    if(stream!=active || n!=6144 || cols!=1)throw std::runtime_error("wrong owned stream or shape");
    ++launches;log("launch");if(std::getenv("Q8_TEST_FAIL_LAUNCH"))throw std::runtime_error("injected launch failure");std::vector<float> input(x,x+n);auto bytes=native_q8_fixture::quantize(input);std::memcpy(q,bytes.data(),bytes.size());
}}
''')
    output = destination / 'native-producer-provider-test'
    subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-fno-fast-math',
                    '-ffp-contract=off', *(['-DSTRATA_PROJECTION_Q8_REPLAY=1'] if replay else []), '-I'+str(destination), '-I'+str(ROOT/'include'),
                    '-I'+str(ROOT/'tests/kernels'), '-x', 'c++',
                    str(ROOT/'tests/kernels/native_qsa_projection_q8_reference.cu'),
                    str(provider), '-o', str(output)], check=True)
    return output

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exe = build(root)
        fixture = root / 'fixture.bin'
        subprocess.run([sys.argv[1], '--synthetic', '0', str(fixture)], check=True)
        trace = root / 'calls.txt'
        result = subprocess.run([str(exe), '--device', '0', '--expect-device', 'signature-fixture GPU',
                                 '--fixture', str(fixture), '--output', str(root/'out')],
                                cwd=root, env=dict(os.environ, Q8_TEST_TRACE=str(trace)),
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert trace.read_text().splitlines() == ['create', 'launch', 'sync', 'launch', 'sync', 'destroy']
        assert (root/'out/qx.bin').stat().st_size == 6912
        failure_trace = root / 'failure-calls.txt'
        result = subprocess.run([str(exe), '--device', '0', '--expect-device', 'signature-fixture GPU',
                                 '--fixture', str(fixture), '--output', str(root/'failure-out')],
                                cwd=root, env=dict(os.environ, Q8_TEST_TRACE=str(failure_trace),
                                                   Q8_TEST_FAIL_LAUNCH='1'),
                                capture_output=True, text=True)
        assert result.returncode != 0 and 'injected launch failure' in result.stderr
        assert failure_trace.read_text().splitlines() == ['create', 'launch', 'destroy']
        print('actual native producer CLI uses and releases one stream through both launcher calls; host provider only')
