#include "native_q8_gu_fixture.hpp"
#include "strata_vk_q8_gu.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    need(argc==13 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--native" && std::string(argv[7])=="--observation" &&
        std::string(argv[9])=="--weights" && std::string(argv[11])=="--output",
        "GU: run through tools/native_q8_gu.py for hashes and native oracle comparison");
    auto native=consumer::read_capture(argv[6],false),observation=consumer::read_capture(argv[8],true);
    consumer::validate_pair(native,observation);const auto weights=gu::read_weights(argv[10]);
    Bytes paired(native.input.begin(),native.input.begin()+consumer::PREFIX_BYTES);
    paired.insert(paired.end(),observation.input.begin(),observation.input.begin()+consumer::PREFIX_BYTES);
    Q8GuLayout layout{256,32,2,2,1,{0},{0,2},{0,1}};(void)q8_gu_metadata(layout,weights.size());
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"GU: invalid device");
    const size_t device=std::stoull(index);need(device<size_t(ggml_backend_vk_get_device_count()),"GU: device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"GU: Vulkan init failed");const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && std::string(name).find(argv[4])!=std::string::npos,"GU: unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"GU: metadata allocation failed");
    auto make=[&](ggml_type type,size_t n,size_t prefix){auto* base=ggml_new_tensor_1d(ctx.get(),type,n+prefix+3);
        return std::pair{base,ggml_view_1d(ctx.get(),base,n,prefix*4)};};
    auto [wb,w]=make(GGML_TYPE_I32,weights.size()/4,3);auto [xb,x]=make(GGML_TYPE_I32,paired.size()/4,5);
    auto [gb,g]=make(GGML_TYPE_F32,64,7);auto [ub,u]=make(GGML_TYPE_F32,64,9);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"GU: tensor allocation failed");
    for(auto* base:{wb,xb,gb,ub}){Bytes bytes(ggml_nbytes(base),0xa5);ggml_backend_tensor_set(base,bytes.data(),0,bytes.size());}
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());ggml_backend_tensor_set(x,paired.data(),0,paired.size());
    const std::string effective=std::string(argv[12])+".effective.spv";
    std::unique_ptr<StrataVkQ8Gu,decltype(&strata_vk_q8_gu_destroy)> component(
        strata_vk_q8_gu_create(backend.get(),effective.c_str()),strata_vk_q8_gu_destroy);
    strata_vk_q8_gu_run(component.get(),w,x,g,u,layout);
    std::vector<float> gate(64),up(64);ggml_backend_tensor_get(g,gate.data(),0,gate.size()*4);ggml_backend_tensor_get(u,up.data(),0,up.size()*4);
    bool sentinels=true;std::ofstream views(std::string(argv[12])+".views.bin",std::ios::binary);
    need(bool(views),"GU: cannot create view receipt");word(views,0x3155474e);
    size_t slot=0;
    for(auto* base:{wb,xb,gb,ub}){Bytes bytes(ggml_nbytes(base));ggml_backend_tensor_get(base,bytes.data(),0,bytes.size());
        word(views,uint32_t(bytes.size()));views.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        const size_t prefixes[]={3,5,7,9},counts[]={weights.size()/4,paired.size()/4,64,64};
        for(size_t i=0;i<bytes.size();++i)if(i<prefixes[slot]*4 || i>=(prefixes[slot]+counts[slot])*4)sentinels&=bytes[i]==0xa5;
        if(slot<2){const auto& expected=slot==0?weights:paired;
            sentinels&=std::equal(expected.begin(),expected.end(),bytes.begin()+prefixes[slot]*4);}++slot;}
    need(bool(views),"GU: view receipt write failed");gu::write_result(argv[12],gate,up,sentinels);
    std::printf("native_gu_kernel_8_component=true qualified=false quantizer_qualified=false device=%s entries=2 h=256 f=32 sentinels=%d\n",name,int(sentinels));
    need(sentinels,"GU: sentinel/input mutation; raw receipts retained");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"Q8 GU component: %s\n",e.what());return 1;}}
