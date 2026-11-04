#include "portable_native_ple_conv.hpp"
#include "ggml-cpu.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace strata::core::portable_detail;
namespace strata::core::portable_detail {
struct StrataVkNativePleConv {};
bool fail_conv = false;
StrataVkNativePleConv* strata_vk_native_ple_conv_create(ggml_backend_t,const char*,size_t) { return new StrataVkNativePleConv; }
void strata_vk_native_ple_conv_destroy(StrataVkNativePleConv* p) noexcept { delete p; }
size_t strata_vk_native_ple_conv_bytes(const StrataVkNativePleConv*) { return 0; }
void strata_vk_native_ple_conv_run(StrataVkNativePleConv*,ggml_tensor* const t[10]) {
    if (fail_conv) throw std::runtime_error("injected refusal");
    std::vector<float> h(92160), n(10240), next(92160), output(10240, 1.0f);
    ggml_backend_tensor_get(t[0],h.data(),0,h.size()*4);
    ggml_backend_tensor_get(t[1],n.data(),0,n.size()*4);
    for(size_t c=0;c<n.size();++c){for(size_t j=0;j<8;++j)next[c*9+j]=h[c*9+j+1];next[c*9+8]=n[c];}
    ggml_backend_tensor_set(t[5],next.data(),0,next.size()*4);
    for(size_t i=6;i<10;++i)ggml_backend_tensor_set(t[i],output.data(),0,output.size()*4);
}
}
int main() {
    auto backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    std::vector<uint16_t> weights(40960,0x3c00);
    PortableNativePleConv owner(backend.get(),weights.data(),weights.size()*2,"",2u<<20);
    auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
    ggml_tensor* originals[4]; ggml_tensor* copies[4];
    for(size_t i=0;i<4;++i){originals[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,i?10240:92160);copies[i]=ggml_cpy(ctx.get(),originals[i],owner.input(i));}
    owner.bind(copies,originals);
    auto inputs=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
    std::vector<float> history(92160), norm(10240,4), hidden(10240,1), gated(10240,2);
    for(size_t c=0;c<10240;++c)for(size_t j=0;j<9;++j)history[c*9+j]=float(j);
    std::vector<float>* values[]={&history,&norm,&hidden,&gated};
    for(size_t i=0;i<4;++i){ggml_backend_tensor_set(originals[i],values[i]->data(),0,values[i]->size()*4);ggml_backend_tensor_set(owner.input(i),values[i]->data(),0,values[i]->size()*4);}
    owner.run(copies,originals);
    assert(owner.result()!=originals[2] && owner.history()!=owner.input(0));
    std::vector<float> next(92160),unchanged(92160);
    ggml_backend_tensor_get(owner.history(),next.data(),0,next.size()*4);
    ggml_backend_tensor_get(owner.input(0),unchanged.data(),0,unchanged.size()*4);
    assert(unchanged==history && next[0]==1 && next[7]==8 && next[8]==4);
    fail_conv=true; bool refused=false; try{owner.run(copies,originals);}catch(...){refused=true;} assert(refused);
    assert(!owner.ready()); fail_conv=false; owner.run(copies,originals); assert(owner.ready());
    auto* wrong=originals[0]; originals[0]=originals[1]; refused=false;
    try{owner.run(copies,originals);}catch(...){refused=true;} assert(refused && !owner.ready()); originals[0]=wrong;
    norm[0]=std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(owner.input(1),norm.data(),0,norm.size()*4);
    refused=false; try{owner.run(copies,originals);}catch(...){refused=true;} assert(refused && !owner.ready());
    std::cout << "PLE convolution candidate ownership, refusal and retry passed with host provider\n";
}
