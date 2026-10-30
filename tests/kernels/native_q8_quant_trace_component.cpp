#include "native_q8_geometry_fixture.hpp"
#include "native_q8_quant_trace.hpp"
#include "strata_vk_native_stages.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
namespace geo=native_q8_fixture::geometry;
int main(int argc,char** argv){try {
    need(argc==10 && std::string(argv[1])=="--observe-unspecified-fp32-denorm","quant trace: explicit observation required");
    ++argv;
    need(std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output","quant trace: use tools/native_q8_quant_trace.py");
    const auto input=geo::read_fixture(argv[6]);const std::string index=argv[2];
    need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"quant trace: invalid device");
    const size_t device=std::stoull(index);need(device<size_t(ggml_backend_vk_get_device_count()),"quant trace: device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"quant trace: Vulkan init failed");
    const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && std::string(name).find(argv[4])!=std::string::npos,"quant trace: unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({8192,nullptr,true}),ggml_free);need(bool(ctx),"quant trace: metadata failed");
    auto make=[&](ggml_type type,size_t n,size_t prefix){auto* base=ggml_new_tensor_1d(ctx.get(),type,n+prefix+3);
        return std::pair{base,ggml_view_1d(ctx.get(),base,n,prefix*4)};};
    auto [ib,i]=make(GGML_TYPE_F32,geo::INPUT_FLOATS,1);
    // Existing stage bindings/shape are unchanged. Only the first 16 words of
    // this ordinary hidden plane are written; the rest must remain canary bytes.
    auto [hb,h]=make(GGML_TYPE_F32,geo::INPUT_FLOATS,3);
    auto [qb,q]=make(GGML_TYPE_I32,geo::INPUT_BYTES/4,5);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"quant trace: buffer failed");
    for(auto* base:{ib,hb,qb}){Bytes b(ggml_nbytes(base),0xa5);ggml_backend_tensor_set(base,b.data(),0,b.size());}
    ggml_backend_tensor_set(i,input.data(),0,input.size()*4);
    const std::string effective=std::string(argv[8])+".effective.spv";
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> component(
        strata_vk_native_stages_create_quant_trace_observation(backend.get(),effective.c_str()),strata_vk_native_stages_destroy);
    const int32_t starts[]={0,3};
    strata_vk_native_stages_run(component.get(),0,i,i,h,q,geo::H,starts,2); // exactly one quantization dispatch
    Bytes packed(geo::INPUT_BYTES),trace(NATIVE_Q8_TRACE_WORDS*4);
    ggml_backend_tensor_get(q,packed.data(),0,packed.size());ggml_backend_tensor_get(h,trace.data(),0,trace.size());
    bool sentinels=true;std::ofstream views(std::string(argv[8])+".views.bin",std::ios::binary);
    need(bool(views),"quant trace: views create failed");word(views,0x3156544e);
    size_t slot=0;
    for(auto* base:{ib,hb,qb}){Bytes b(ggml_nbytes(base));ggml_backend_tensor_get(base,b.data(),0,b.size());
        word(views,uint32_t(b.size()));views.write(reinterpret_cast<const char*>(b.data()),b.size());
        const size_t prefix[]={1,3,5},count[]={geo::INPUT_FLOATS,geo::INPUT_FLOATS,geo::INPUT_BYTES/4};
        for(size_t j=0;j<b.size();++j)if(j<prefix[slot]*4 || j>=(prefix[slot]+count[slot])*4 ||
            (slot==1 && j>=prefix[slot]*4+trace.size()))sentinels&=b[j]==0xa5;
        if(slot==0){const auto original=geo::bytes(input);sentinels&=std::equal(original.begin(),original.end(),b.begin()+4);}
        ++slot;}
    need(bool(views),"quant trace: views write failed");
    std::ofstream out(argv[8],std::ios::binary);need(bool(out),"quant trace: raw create failed");
    word(out,0x3154384e);word(out,sentinels?1:0);
    for(const auto* bytes:{&packed,&trace}){word(out,uint32_t(bytes->size()/4));out.write(reinterpret_cast<const char*>(bytes->data()),bytes->size());}
    need(bool(out),"quant trace: raw write failed");
    printf("quantization_trace_only=true observation_only=true qualified=false quantizer_qualified=false fp32_denorm_mode=unspecified cause_isa_proven=false device=%s sentinels=%d\n",name,int(sentinels));
    need(sentinels,"quant trace: sentinel/input mutation; raw evidence retained");return 0;
}catch(const std::exception& e){fprintf(stderr,"Q8 quant trace: %s\n",e.what());return 1;}}
