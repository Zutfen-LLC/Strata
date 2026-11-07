#include "ggml-backend.h"
static int copy_computes = 0;
static ggml_status copy_test_compute(ggml_backend_t b, ggml_cgraph* g) { ++copy_computes; return ggml_backend_graph_compute(b,g); }
#define STRATA_PLE_CONV_COPY_TEST_COMPUTE copy_test_compute
#include "portable_native_ple_conv_copy_graph.hpp"
#undef STRATA_PLE_CONV_COPY_TEST_COMPUTE
#include "ggml-cpu.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace strata::core::portable_detail;
namespace strata::core::portable_detail {
struct StrataVkNativePleConv {size_t bytes;};
bool fail_conv = false;
bool corrupt_history = false, nonfinite_output = false;
size_t component_creates=0, dispatch_rows=0;
StrataVkNativePleConv* strata_vk_native_ple_conv_create(ggml_backend_t backend,const char*,size_t) {
    auto metadata=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({4096,nullptr,true}),ggml_free);
    ggml_new_tensor_1d(metadata.get(),GGML_TYPE_I32,1);
    auto bytes=ggml_backend_alloc_ctx_tensors_from_buft_size(metadata.get(),ggml_backend_get_default_buffer_type(backend));
    assert(bytes>0);++component_creates;return new StrataVkNativePleConv{bytes};
}
void strata_vk_native_ple_conv_destroy(StrataVkNativePleConv* p) noexcept { delete p; }
size_t strata_vk_native_ple_conv_bytes(const StrataVkNativePleConv* p) { return p->bytes; }
void strata_vk_native_ple_conv_run(StrataVkNativePleConv*,ggml_tensor* const t[10]) {
    if (fail_conv) throw std::runtime_error("injected refusal");
    const size_t rows=ggml_nbytes(t[1])/(10240*4); dispatch_rows=rows;
    std::vector<float> h(92160), n(10240*rows), x(n.size()), g(n.size()), next(92160);
    std::vector<float> conv(n.size()), result(n.size()), pre(n.size()), exponent(n.size());
    ggml_backend_tensor_get(t[0],h.data(),0,h.size()*4);
    ggml_backend_tensor_get(t[1],n.data(),0,n.size()*4);
    ggml_backend_tensor_get(t[3],x.data(),0,x.size()*4);
    ggml_backend_tensor_get(t[4],g.data(),0,g.size()*4);
    // Test weights are all exactly one. Model the shader's immutable-history indexing.
    for(size_t c=0;c<10240;++c){
        for(size_t r=0;r<rows;++r){size_t i=r*10240+c;
            pre[i]=((h[c*9+r]+h[c*9+r+3])+h[c*9+r+6])+n[i];
            exponent[i]=std::exp(-pre[i]);conv[i]=pre[i]/(1.f+exponent[i]);
            result[i]=x[i]+(g[i]+conv[i]);
        }
        for(size_t j=0;j<9-rows;++j)next[c*9+j]=h[c*9+j+rows];
        for(size_t r=0;r<rows;++r)next[c*9+9-rows+r]=n[r*10240+c];
    }
    if(corrupt_history)next[0]=77.f;
    if(nonfinite_output)result[0]=std::numeric_limits<float>::infinity();
    ggml_backend_tensor_set(t[5],next.data(),0,next.size()*4);
    std::vector<float>* out[]={&conv,&result,&pre,&exponent};
    for(size_t i=6;i<10;++i)ggml_backend_tensor_set(t[i],out[i-6]->data(),0,out[i-6]->size()*4);
}
}
int main() {
    auto backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    std::vector<uint16_t> weights(40960,0x3c00);
    const auto exact=PortableNativePleConv::estimate(backend.get());
    bool short_refused=false;
    try{PortableNativePleConv short_owner(backend.get(),weights.data(),weights.size()*2,1,"",exact-1);}catch(...){short_refused=true;}
    assert(short_refused && component_creates==0);
    PortableNativePleConv owner(backend.get(),weights.data(),weights.size()*2,1,"",exact);
    assert(component_creates==1 && owner.bytes()==exact);
    std::vector<float> f32(40960);
    f32[0]=1.0f;f32[1]=1.00048828125f;f32[2]=1.00146484375f;
    f32[3]=0x1p-24f;f32[4]=0x1p-25f;f32[5]=-0.0f;f32[6]=65504.f;
    const auto packed=PortableNativePleConv::native_weights(f32.data(),f32.size()*4,0);
    assert(packed[0]==0x3c00 && packed[1]==0x3c00 && packed[2]==0x3c02 &&
           packed[3]==0x0001 && packed[4]==0x0000 && packed[5]==0x8000 && packed[6]==0x7bff);
    for(float invalid:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),65520.f}){
        f32[0]=invalid;bool rejected=false;try{(void)PortableNativePleConv::native_weights(f32.data(),f32.size()*4,0);}catch(...){rejected=true;}assert(rejected);
    }
    weights[0]=0x7c00;bool invalid_half=false;
    try{(void)PortableNativePleConv::native_weights(weights.data(),weights.size()*2,1);}catch(...){invalid_half=true;}
    assert(invalid_half);weights[0]=0x3c00;
    auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
    ggml_tensor* originals[4]; ggml_tensor* copies[4]; ggml_tensor* stale[4];
    for(size_t i=0;i<4;++i){originals[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,i?10240:92160);ggml_set_output(originals[i]);stale[i]=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,i?10240:92160);copies[i]=ggml_cpy(ctx.get(),originals[i],owner.input(i));}
    owner.bind(copies,originals);
    auto inputs=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
    std::vector<float> history(92160), norm(10240,4), hidden(10240,1), gated(10240,2);
    for(size_t c=0;c<10240;++c)for(size_t j=0;j<9;++j)history[c*9+j]=float(j);
    std::vector<float>* values[]={&history,&norm,&hidden,&gated};
    for(size_t i=0;i<4;++i){ggml_backend_tensor_set(originals[i],values[i]->data(),0,values[i]->size()*4);ggml_backend_tensor_set(owner.input(i),values[i]->data(),0,values[i]->size()*4);}
    owner.run(copies,originals,hidden);
    assert(owner.result()!=originals[2] && owner.history()!=owner.input(0));
    std::vector<float> next(92160),unchanged(92160);
    ggml_backend_tensor_get(owner.history(),next.data(),0,next.size()*4);
    ggml_backend_tensor_get(owner.input(0),unchanged.data(),0,unchanged.size()*4);
    assert(unchanged==history && next[0]==1 && next[7]==8 && next[8]==4);
    // A post-schedule source rewrite and a real destination corruption cause
    // the same predicate to fail. Neither establishes the physical cause.
    const char* names[]={"history","normalized","hidden","gated"};
    for(size_t i=0;i<4;++i){
        auto altered=*values[i];altered[0]=88.f;
        ggml_backend_tensor_set(stale[i],altered.data(),0,altered.size()*4);
        auto* saved=copies[i]->src[0];copies[i]->src[0]=stale[i];
        bool caught=false;try{owner.run(copies,originals,hidden);}catch(const std::exception& e){auto s=std::string(e.what());caught=s.find(std::string("PLE conv: prefix copy differs: ")+names[i]+" byte ")!=std::string::npos&&s.find("logical=destination")!=std::string::npos;}
        if(i==2)assert(!caught&&owner.ready());else assert(caught&&!owner.ready());copies[i]->src[0]=saved;
        ggml_backend_tensor_set(owner.input(i),altered.data(),0,altered.size()*4);
        caught=false;try{owner.run(copies,originals,hidden);}catch(const std::exception& e){auto s=std::string(e.what());caught=s.find(std::string("PLE conv: prefix copy differs: ")+names[i]+" byte ")!=std::string::npos&&s.find(i==2?"expected=request-residual":"logical=scheduled-source")!=std::string::npos;}
        assert(caught&&!owner.ready());
        ggml_backend_tensor_set(owner.input(i),values[i]->data(),0,values[i]->size()*4);
        owner.run(copies,originals,hidden);assert(owner.ready());
    }
    auto wrong_hidden=hidden;wrong_hidden[0]=88.f;
    bool wrong_residual=false;try{owner.run(copies,originals,wrong_hidden);}catch(const std::exception& e){wrong_residual=std::string(e.what()).find("expected=request-residual")!=std::string::npos;}
    assert(wrong_residual&&!owner.ready());
    wrong_hidden.pop_back();wrong_residual=false;
    try{owner.run(copies,originals,wrong_hidden);}catch(const std::exception& e){wrong_residual=std::string(e.what()).find("hidden residual span")!=std::string::npos;}
    assert(wrong_residual&&!owner.ready());owner.run(copies,originals,hidden);
    fail_conv=true; bool refused=false; try{owner.run(copies,originals,hidden);}catch(...){refused=true;} assert(refused);
    assert(!owner.ready()); fail_conv=false; owner.run(copies,originals,hidden); assert(owner.ready());
    corrupt_history=true; refused=false; try{owner.run(copies,originals,hidden);}catch(...){refused=true;} assert(refused && !owner.ready()); corrupt_history=false;
    nonfinite_output=true; refused=false; try{owner.run(copies,originals,hidden);}catch(...){refused=true;} assert(refused && !owner.ready()); nonfinite_output=false;
    auto* wrong=originals[0]; originals[0]=originals[1]; refused=false;
    try{owner.run(copies,originals,hidden);}catch(...){refused=true;} assert(refused && !owner.ready()); originals[0]=wrong;
    norm[0]=std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(originals[1],norm.data(),0,norm.size()*4);
    ggml_backend_tensor_set(owner.input(1),norm.data(),0,norm.size()*4);
    refused=false; try{owner.run(copies,originals,hidden);}catch(...){refused=true;} assert(refused && !owner.ready());
    {
        const auto budget=PortableNativePleConv::estimate(backend.get(),2);
        PortableNativePleConv pair(backend.get(),weights.data(),weights.size()*2,1,"",budget,2);
        auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
        ggml_tensor* src[4]; ggml_tensor* cp[4];
        for(size_t i=0;i<4;++i){src[i]=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,i?20480:92160);cp[i]=ggml_cpy(c.get(),src[i],pair.input(i));}
        pair.bind(cp,src);
        auto b=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),backend.get()),ggml_backend_buffer_free);
        std::vector<float> h(92160), n(20480), x(20480), g(20480);
        for(size_t k=0;k<10240;++k){for(size_t j=0;j<9;++j)h[k*9+j]=float(j+1)*0.01f+float(k%7)*0.001f; n[k]=0.11f; n[10240+k]=0.22f; x[k]=0.31f; x[10240+k]=0.42f; g[k]=0.53f; g[10240+k]=0.64f;}
        std::vector<float>* v[]={&h,&n,&x,&g};
        for(size_t i=0;i<4;++i){ggml_backend_tensor_set(src[i],v[i]->data(),0,v[i]->size()*4);ggml_backend_tensor_set(pair.input(i),v[i]->data(),0,v[i]->size()*4);}
        pair.run(cp,src,x);assert(pair.ready()&&dispatch_rows==2);
        std::vector<float> out(92160);ggml_backend_tensor_get(pair.history(),out.data(),0,out.size()*4);
        // Independent sequential T1 mutation, checking both intermediate rows and history.
        std::vector<float> actual_conv(20480), actual_result(20480);
        ggml_backend_tensor_get(pair.convolution(),actual_conv.data(),0,actual_conv.size()*4);
        ggml_backend_tensor_get(pair.result(),actual_result.data(),0,actual_result.size()*4);
        auto control=h;
        for(size_t r=0;r<2;++r) for(size_t k=0;k<10240;++k){
            auto* q=&control[k*9]; size_t i=r*10240+k;
            float sum=((q[0]+q[3])+q[6])+n[i];
            float activation=sum/(1.f+std::exp(-sum));
            assert(actual_conv[i]==activation && actual_result[i]==x[i]+(g[i]+activation));
            if(r==1 && k==0) assert(q[0]==h[1] && q[3]==h[4] && q[6]==h[7]);
            for(size_t j=0;j<8;++j) q[j]=q[j+1]; q[8]=n[i];
        }
        assert(out==control);
        assert(pair.calls()==1);
        n[10240]=std::numeric_limits<float>::quiet_NaN();
        ggml_backend_tensor_set(src[1],n.data(),0,n.size()*4);ggml_backend_tensor_set(pair.input(1),n.data(),0,n.size()*4);
        dispatch_rows=0;bool bad=false;try{pair.run(cp,src,x);}catch(...){bad=true;}assert(bad&&!pair.ready()&&dispatch_rows==0&&pair.calls()==1);
    }
    for(size_t rows: {size_t(1),size_t(2)}) {
        auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({16384,nullptr,true}),ggml_free);
        std::array<ggml_tensor*,4> src{};
        for(size_t i=0;i<4;++i)src[i]=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,i?10240*rows:92160);
        auto b=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(c.get(),backend.get()),ggml_backend_buffer_free);
        std::vector<float> h(92160),n(10240*rows),x(n.size()),g(n.size());
        for(size_t k=0;k<10240;++k){for(size_t j=0;j<9;++j)h[9*k+j]=0.01f*float(j+1);for(size_t r=0;r<rows;++r){n[r*10240+k]=0.1f+0.1f*r;x[r*10240+k]=0.3f+0.1f*r;g[r*10240+k]=0.5f+0.1f*r;}}
        std::vector<float>* v[]={&h,&n,&x,&g};for(size_t i=0;i<4;++i)ggml_backend_tensor_set(src[i],v[i]->data(),0,v[i]->size()*4);
        auto budget=PortableNativePleConvCopyGraph::estimate(backend.get(),rows);bool bad=false;
        try{PortableNativePleConvCopyGraph short_graph(backend.get(),src,weights.data(),weights.size()*2,1,"",budget-1,rows);}catch(...){bad=true;}assert(bad);
        PortableNativePleConvCopyGraph graph(backend.get(),src,weights.data(),weights.size()*2,1,"",budget,rows);
        assert(graph.bytes()<=budget&&graph.copy(1)->src[0]==src[1]);
        if(rows==2){
            auto* saved=src[1]->data;
            auto base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(src[1]->buffer));
            auto capacity=ggml_backend_buffer_get_size(src[1]->buffer);
            for(auto address : {uintptr_t(0),base+capacity+4,base+capacity-81916}){
                src[1]->data=reinterpret_cast<void*>(address); auto calls=graph.calls(); auto computes=copy_computes; dispatch_rows=0; bool caught=false;
                try{graph.run(src,x);}catch(...){caught=true;}
                if(!caught||graph.ready()||graph.calls()!=calls||dispatch_rows!=0||copy_computes!=computes) throw std::runtime_error("conv copy span refusal");
            }
            src[1]->data=reinterpret_cast<void*>(base+4);
            auto calls=graph.calls(); auto computes=copy_computes;
            graph.run(src,x);
            if(!graph.ready()||graph.calls()!=calls+1||copy_computes!=computes+1||dispatch_rows!=rows) throw std::runtime_error("conv copy valid offset");
            src[1]->data=saved;
        }
        auto initial_calls=graph.calls();
        graph.run(src,x);assert(graph.ready()&&graph.calls()==initial_calls+1&&dispatch_rows==rows);
        std::vector<float> next(92160),original(92160);ggml_backend_tensor_get(graph.history(),next.data(),0,next.size()*4);ggml_backend_tensor_get(src[0],original.data(),0,original.size()*4);
        assert(original==h);for(size_t r=0;r<rows;++r)assert(next[9-rows+r]==n[r*10240]);
        auto wrong=src;wrong[1]=src[2];bad=false;try{graph.run(wrong,x);}catch(...){bad=true;}assert(bad&&!graph.ready()&&graph.calls()==initial_calls+1);
        fail_conv=true;bad=false;try{graph.run(src,x);}catch(...){bad=true;}fail_conv=false;assert(bad&&!graph.ready()&&graph.calls()==initial_calls+1);
        graph.run(src,x);assert(graph.ready()&&graph.calls()==initial_calls+2);graph.abort();assert(!graph.ready());
        bad=false;try{(void)graph.history();}catch(...){bad=true;}assert(bad);
    }
    std::cout << "PLE convolution candidate ownership, refusal and retry passed with host provider\n";
}
