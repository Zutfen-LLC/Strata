#include "native_q8_down_fixture.hpp"
#include "strata_vk_q8_down.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_q8_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char** argv){try {
    need(argc==11 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--hq" && std::string(argv[7])=="--weights" && std::string(argv[9])=="--output",
        "down: use tools/native_q8_down.py for verified native hq, hashes and oracle comparison");
    const auto hq=down::read_hq(argv[6]),weights=gu::read_weights(argv[8]);
    Q8DownLayout l{256,32,2,4,1,{17408},{0,2},{0,1}};(void)q8_down_metadata(l,weights.size());
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"down: invalid device");
    const size_t device=std::stoull(index);need(device<size_t(ggml_backend_vk_get_device_count()),"down: device absent");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend),"down: Vulkan init failed");const char* name=ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
    need(std::string(argv[4]).size() && std::string(name).find(argv[4])!=std::string::npos,"down: unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"down: metadata allocation failed");
    auto make=[&](ggml_type type,size_t n,size_t prefix){auto* base=ggml_new_tensor_1d(ctx.get(),type,n+prefix+3);
        return std::pair{base,ggml_view_1d(ctx.get(),base,n,prefix*4)};};
    auto [wb,w]=make(GGML_TYPE_I32,weights.size()/4,3);auto [xb,x]=make(GGML_TYPE_I32,hq.size()/4,5);
    auto [ob,o]=make(GGML_TYPE_F32,1024,7);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"down: tensor allocation failed");
    for(auto* base:{wb,xb}){Bytes blank(ggml_nbytes(base),0xa5);ggml_backend_tensor_set(base,blank.data(),0,blank.size());}
    ggml_backend_tensor_set(w,weights.data(),0,weights.size());ggml_backend_tensor_set(x,hq.data(),0,hq.size());
    const std::string effective=std::string(argv[10])+".effective.spv";
    std::unique_ptr<StrataVkQ8Down,decltype(&strata_vk_q8_down_destroy)> component(
        strata_vk_q8_down_create(backend.get(),effective.c_str()),strata_vk_q8_down_destroy);
    std::array<std::vector<float>,3> results;bool sentinels=true;
    std::ofstream views(std::string(argv[10])+".views.bin",std::ios::binary);need(bool(views),"down: view receipt create failed");word(views,0x3156444e);
    for(size_t pass=0;pass<3;++pass){
        if(pass==1){l.starts={1,2};l.destination_ids={-1,2};}
        if(pass==2){l.weight_offsets.clear();l.starts={1};l.destination_ids={-1,-1};}
        Bytes blank(ggml_nbytes(ob),0xa5);ggml_backend_tensor_set(ob,blank.data(),0,blank.size());
        strata_vk_q8_down_run(component.get(),w,x,o,l);
        results[pass].resize(1024);ggml_backend_tensor_get(o,results[pass].data(),0,4096);
        size_t slot=0;
        for(auto* base:{wb,xb,ob}){Bytes bytes(ggml_nbytes(base));ggml_backend_tensor_get(base,bytes.data(),0,bytes.size());
            word(views,uint32_t(bytes.size()));views.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
            const size_t prefixes[]={3,5,7},counts[]={weights.size()/4,hq.size()/4,1024};
            for(size_t i=0;i<bytes.size();++i)if(i<prefixes[slot]*4 || i>=(prefixes[slot]+counts[slot])*4)sentinels&=bytes[i]==0xa5;
            if(slot<2){const auto& expected=slot==0?weights:hq;
                sentinels&=std::equal(expected.begin(),expected.end(),bytes.begin()+prefixes[slot]*4);}++slot;}
    }
    need(bool(views),"down: view receipt write failed");down::write_result(argv[10],results,sentinels);
    printf("native_down_kernel_8_component=true qualified=false quantizer_qualified=false device=%s h=256 f=32 entries=2 passes=3 sentinels=%d\n",name,int(sentinels));
    need(sentinels,"down: sentinel/input mutation; raw receipts retained");return 0;
}catch(const std::exception& e){fprintf(stderr,"native down component: %s\n",e.what());return 1;}}
