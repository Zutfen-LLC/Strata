#include "native_q8_geometry_fixture.hpp"
namespace geo=native_q8_fixture::geometry;
#include "strata_vk_q8_gu.hpp"
#include "strata_vk_q8_down.hpp"
#include "strata_vk_native_stages.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    if(argc==3 && std::string(argv[1])=="--make-fixture"){geo::write_fixture(argv[2]);return 0;}
    need(argc==12 && std::string(argv[1])=="--observe-unspecified-fp32-denorm",
        "compose: explicit observation selection required; strict path remains unchanged");
    ++argv; // remove the explicit observation selector for fixed-position arguments
    need(argc==12 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--weights" && std::string(argv[9])=="--output",
        "compose: run through tools/native_q8_geometry.py for hashes and native oracle comparison");
    const auto input=geo::read_fixture(argv[6]);const auto weights=geo::read_weights(argv[8]);
    Q8GuLayout layout{geo::H,geo::F,geo::ENTRIES,geo::TOKENS,geo::GRID_GROUPS,{0,0,0},geo::STARTS,geo::TOKENS_BY_ENTRY};
    (void)q8_gu_metadata(layout,weights.size());
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"compose: invalid device");
    const size_t device=std::stoull(index);need(device<size_t(ggml_backend_vk_get_device_count()),"compose: device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"compose: Vulkan init failed");const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && std::string(name).find(argv[4])!=std::string::npos,"compose: unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"compose: metadata allocation failed");
    auto make=[&](ggml_type type,size_t n,size_t prefix){auto* base=ggml_new_tensor_1d(ctx.get(),type,n+prefix+3);
        return std::pair{base,ggml_view_1d(ctx.get(),base,n,prefix*4)};};
    auto [wb,w]=make(GGML_TYPE_I32,weights.size()/4,3);auto [xb,x]=make(GGML_TYPE_I32,geo::INPUT_BYTES/4,5);
    auto [ib,i]=make(GGML_TYPE_F32,geo::INPUT_FLOATS,17);
    auto [gb,g]=make(GGML_TYPE_F32,geo::FLOATS,7);auto [ub,u]=make(GGML_TYPE_F32,geo::FLOATS,9);
    auto [hb,h]=make(GGML_TYPE_F32,geo::FLOATS,11);auto [qb,q]=make(GGML_TYPE_I32,geo::HQ_BYTES/4,13);
    auto [ob,o]=make(GGML_TYPE_F32,geo::OUTPUT_FLOATS,15);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"compose: tensor allocation failed");
    for(auto* base:{wb,xb,gb,ub,hb,qb,ob,ib}){Bytes bytes(ggml_nbytes(base),0xa5);ggml_backend_tensor_set(base,bytes.data(),0,bytes.size());}
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());ggml_backend_tensor_set(i,input.data(),0,input.size()*4);
    const std::string effective=std::string(argv[10])+".gu.effective.spv";
    const std::string activation_spv=std::string(argv[10])+".activation.effective.spv";
    const std::string down_spv=std::string(argv[10])+".down.effective.spv";
    // Refuse missing required observation controls before any GPU dispatch.
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)> activation(
        strata_vk_native_stages_create_observation(backend.get(),activation_spv.c_str()),strata_vk_native_stages_destroy);
    std::unique_ptr<StrataVkQ8Down,decltype(&strata_vk_q8_down_destroy)> down(
        strata_vk_q8_down_create(backend.get(),down_spv.c_str()),strata_vk_q8_down_destroy);
    std::unique_ptr<StrataVkQ8Gu,decltype(&strata_vk_q8_gu_destroy)> component(
        strata_vk_q8_gu_create(backend.get(),effective.c_str()),strata_vk_q8_gu_destroy);
    const int32_t input_starts[]={0,geo::TOKENS};
    strata_vk_native_stages_run(activation.get(),0,i,i,i,x,geo::H,input_starts,2);
    strata_vk_q8_gu_run(component.get(),w,x,g,u,layout);
    // These are the SAME device views: no gate/up or packed-hidden host upload.
    // Existing runners fence each dispatch; no new scheduling or arithmetic.
    strata_vk_native_stages_run(activation.get(),2,g,u,h,q,geo::F,layout.starts.data(),layout.starts.size());
    Q8DownLayout down_layout{geo::H,geo::F,geo::ENTRIES,geo::ENTRIES,geo::GRID_GROUPS,{geo::DOWN_OFFSET,geo::DOWN_OFFSET,geo::DOWN_OFFSET},geo::STARTS,geo::DESTINATIONS};
    strata_vk_q8_down_run(down.get(),w,q,o,down_layout);
    // Supplementary native separate activation diagnostic; never supplies q/down.
    strata_vk_native_stages_run(activation.get(),1,g,u,h,q,geo::F,layout.starts.data(),layout.starts.size());
    std::array<Bytes,6> sections;
    size_t section=0;
    for(auto* tensor:{x,g,u,h,q,o}){sections[section].resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor,sections[section].data(),0,sections[section].size());++section;}
    bool sentinels=true;std::ofstream views(std::string(argv[10])+".views.bin",std::ios::binary);
    need(bool(views),"compose: cannot create view receipt");word(views,0x3256474e);
    size_t slot=0;
    for(auto* base:{wb,xb,gb,ub,hb,qb,ob,ib}){Bytes bytes(ggml_nbytes(base));ggml_backend_tensor_get(base,bytes.data(),0,bytes.size());
        word(views,uint32_t(bytes.size()));views.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        const size_t prefixes[]={3,5,7,9,11,13,15,17},counts[]={weights.size()/4,geo::INPUT_BYTES/4,geo::FLOATS,geo::FLOATS,geo::FLOATS,geo::HQ_BYTES/4,geo::OUTPUT_FLOATS,geo::INPUT_FLOATS};
        for(size_t i=0;i<bytes.size();++i)if(i<prefixes[slot]*4 || i>=(prefixes[slot]+counts[slot])*4)sentinels&=bytes[i]==0xa5;
        if(slot==0)sentinels&=std::equal(weights.begin(),weights.end(),bytes.begin()+prefixes[slot]*4);
        if(slot==7){const auto original=geo::bytes(input);sentinels&=std::equal(original.begin(),original.end(),bytes.begin()+prefixes[slot]*4);}
        ++slot;}
    need(bool(views),"geometry: view receipt write failed");geo::write_result(argv[10],true,sentinels,sections);
    std::printf("composed_observation_only=true qualified=false quantizer_qualified=false fp32_denorm_mode=unspecified device=%s entries=3 groups=3 h=2560 f=640 sentinels=%d\n",name,int(sentinels));
    need(sentinels,"compose: sentinel/input mutation; raw receipts retained");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"Q8 composed component: %s\n",e.what());return 1;}}
