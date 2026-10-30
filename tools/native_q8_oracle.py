#!/usr/bin/env python3
"""Generate a bounded CUDA oracle using verbatim current native stage definitions.
No GPU execution, install or model access. Compile generated file with nvcc.
"""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('q8_1_store', 'quantize_q8_1_kernel', 'swiglu_entries_kernel', 'swiglu_q8_1_entries_kernel')


def definition(text, name):
    # Unique definition, preserving qualifiers and the complete brace-balanced body.
    import re
    pattern = re.compile(r'^__(?:device|global)__[^\n]*\b' + name + r'\(', re.M)
    hits = list(pattern.finditer(text))
    if len(hits) != 1:
        raise ValueError(f'expected exactly one native definition of {name}')
    start = hits[0].start()
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
        end += 1
    return text[start:end]


def generate():
    source = ROOT / 'src/kernels/cuda/iq_kernels.cu'
    text = source.read_text()
    prefix = '''// Generated from native iq_kernels.cu; do not edit this oracle.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "strata/kernels/q8_1_finite.hpp"
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"
#include "native_q8_fixture.hpp"
using namespace strata::kernels;
using namespace native_q8_fixture;
'''
    return (prefix + '// Native source SHA256: ' + hashlib.sha256(source.read_bytes()).hexdigest() + '\n'
            + '\n\n'.join(definition(text, n) for n in NAMES) + '\n' + HARNESS)


HARNESS = r'''
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Allocation {void* p=nullptr; explicit Allocation(size_t n){checked(cudaMalloc(&p,n));}
    ~Allocation(){if(p)cudaFree(p);} Allocation(const Allocation&)=delete;};
int main(int argc,char** argv){try {
    need(argc==9 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output",
        "use --device INDEX --expect-device SUBSTRING --fixture FILE --output FILE");
    std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"bad device");
    checked(cudaSetDevice(std::stoi(index)));cudaDeviceProp props{};checked(cudaGetDeviceProperties(&props,std::stoi(index)));
    need(std::string(argv[4]).size() && std::string(props.name).find(argv[4])!=std::string::npos,"unexpected CUDA device");
    auto f=read_fixture(argv[6]);const size_t n=f.x.size(),nb=n/32*36;
    Allocation x(n*4),g(n*4),u(n*4),h(n*4),q(nb),starts(std::max(size_t(2),f.starts.size())*4),ng(4);
    checked(cudaMemcpy(x.p,f.x.data(),n*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(g.p,f.gate.data(),n*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(u.p,f.up.data(),n*4,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(starts.p,f.starts.data(),f.starts.size()*4,cudaMemcpyHostToDevice));
    int groups=int(f.starts.size()-1);checked(cudaMemcpy(ng.p,&groups,4,cudaMemcpyHostToDevice));
    auto result=blank(n);const unsigned grid=unsigned((n+255)/256);
    auto read=[&](Bytes& b,void* p){checked(cudaGetLastError());checked(cudaDeviceSynchronize());
        checked(cudaMemcpy(b.data(),p,b.size(),cudaMemcpyDeviceToHost));};
    quantize_q8_1_kernel<<<grid,256>>>((float*)x.p,(block_q8_1*)q.p,n);read(result.input,q.p);
    swiglu_entries_kernel<<<grid,256>>>((float*)g.p,(float*)u.p,(float*)h.p,n);read(result.hidden,h.p);
    quantize_q8_1_kernel<<<grid,256>>>((float*)h.p,(block_q8_1*)q.p,n);read(result.separate,q.p);
    checked(cudaMemset(q.p,0xa5,nb));
    swiglu_q8_1_entries_kernel<<<grid,256>>>((float*)g.p,(float*)u.p,(int32_t*)starts.p,(int32_t*)ng.p,f.n_ff,(block_q8_1*)q.p);
    read(result.fused,q.p);
    int32_t empty[2]={3,3};groups=1;checked(cudaMemcpy(starts.p,empty,8,cudaMemcpyHostToDevice));
    checked(cudaMemcpy(ng.p,&groups,4,cudaMemcpyHostToDevice));checked(cudaMemset(q.p,0xa5,nb));
    swiglu_q8_1_entries_kernel<<<grid,256>>>((float*)g.p,(float*)u.p,(int32_t*)starts.p,(int32_t*)ng.p,f.n_ff,(block_q8_1*)q.p);
    read(result.empty,q.p);for(uint8_t b:result.empty)need(b==0xa5,"empty native interval overwritten");
    write_result(argv[8],result);std::printf("native CUDA oracle device=%s n=%zu; capture only, not parity\n",props.name,n);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native oracle: %s\n",e.what());return 1;}}
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', type=Path, required=True)
    args = parser.parse_args()
    args.generate.write_text(generate())
