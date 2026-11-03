#include "native_qsa_wo_compose_fixture.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_qsa_wo_compose;
using namespace strata::core::portable_detail;
int main(int argc,char**argv){try{
    if(argc==5&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3],argv[4]);return 0;}
    need(argc==11&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&
         std::string(argv[5])=="--weights"&&std::string(argv[7])=="--fixture"&&std::string(argv[9])=="--output","composition: invalid Vulkan arguments");
    auto ww=weights(argv[6]);auto input=read_input(argv[8]);std::string idx=argv[2];
    need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"composition: invalid device");
    auto device=std::stoull(idx);need(device<size_t(ggml_backend_vk_get_device_count()),"composition: missing Vulkan device");
    std::filesystem::path dir=argv[10];need(std::filesystem::create_directory(dir),"composition: output exists");std::filesystem::copy_file(argv[8],dir/"fixture.bin");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend)&&std::string(argv[4]).size()&&std::string(ggml_backend_dev_description(ggml_backend_get_device(backend.get()))).find(argv[4])!=std::string::npos,"composition: unexpected Vulkan device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"composition: metadata");
    auto make=[&](ggml_type type,size_t n){auto*b=ggml_new_tensor_1d(ctx.get(),type,n+16);return std::pair{b,ggml_view_1d(ctx.get(),b,n,32)};};
    auto [xb,x]=make(GGML_TYPE_F32,q::elements);auto [wb,wt]=make(GGML_TYPE_I32,ww.size()/4);
    auto [qb,packed]=make(GGML_TYPE_I32,q::bytes/4);auto [sb,stale]=make(GGML_TYPE_I32,q::bytes/4);auto [yb,y]=make(GGML_TYPE_F32,rows);
    auto*shortq=ggml_view_1d(ctx.get(),qb,q::bytes/4-1,32);
    constexpr size_t budget=32u<<20;
    size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));
    need(bound<budget,"composition: tensor budget");
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)>quant(
        strata_vk_native_stages_create_quant_rcp_observation(backend.get(),(dir/"quant-effective.spv").c_str(),budget-bound),strata_vk_native_stages_destroy);
    need(bool(quant)&&strata_vk_native_stages_bytes(quant.get())<=budget-bound,"composition: quantizer owner budget");
    std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)>wo(
        strata_vk_q8_wo_create(backend.get(),(dir/"wo-effective.spv").c_str(),budget-strata_vk_native_stages_bytes(quant.get())),strata_vk_q8_wo_destroy);
    need(bool(wo)&&strata_vk_q8_wo_bytes(wo.get())<=budget-bound-strata_vk_native_stages_bytes(quant.get()),"composition: WO owner budget");
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
    need(bool(buffer)&&ggml_backend_buffer_get_size(buffer.get())<=budget-strata_vk_native_stages_bytes(quant.get())-strata_vk_q8_wo_bytes(wo.get()),"composition: actual buffer budget");
    constexpr uint32_t guard=0xa5a5a5a5;
    std::vector<uint32_t>xi(q::elements+16,guard),wi(ww.size()/4+16,guard),qi(q::bytes/4+16,guard),si(q::bytes/4+16,guard),yi(rows+16,guard);
    std::memcpy(xi.data()+8,input.data(),q::elements*4);std::memcpy(wi.data()+8,ww.data(),ww.size());std::fill(si.begin()+8,si.end()-8,0u);
    auto reset=[&](){for(auto pair:{std::make_pair(xb,&xi),std::make_pair(wb,&wi),std::make_pair(qb,&qi),std::make_pair(sb,&si),std::make_pair(yb,&yi)})ggml_backend_tensor_set(pair.first,pair.second->data(),0,pair.second->size()*4);};
    auto contents=[](ggml_tensor*t,size_t n){std::vector<uint32_t>v(n);ggml_backend_tensor_get(t,v.data(),0,n*4);return v;};
    const int32_t starts[]={0,1};
    auto quantify=[&](ggml_tensor*out){strata_vk_native_stages_run(quant.get(),0,x,x,x,out,q::elements,starts,2);};
    reset();bool refused=false;try{quantify(shortq);}catch(...){refused=true;}
    need(refused&&contents(qb,qi.size())==qi&&contents(xb,xi.size())==xi,"composition: quantizer invalid call committed");
    refused=false;try{strata_vk_q8_wo_run(wo.get(),wt,shortq,y,rows);}catch(...){refused=true;}
    need(refused&&contents(qb,qi.size())==qi&&contents(wb,wi.size())==wi&&contents(yb,yi.size())==yi,"composition: WO invalid call committed");
    Bytes first_q,first_y;
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
    Bytes first_direct;
#endif
    for(int pass=0;pass<2;++pass){
        reset();quantify(packed);
        auto qr=contents(qb,qi.size());Bytes packed_bytes(q::bytes);std::memcpy(packed_bytes.data(),qr.data()+8,packed_bytes.size());
        packed_domain(ww,packed_bytes);
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
        fixture_scale_words(argv[8],input,packed_bytes);
#endif
        for(size_t i=0;i<8;++i)need(qr[i]==guard&&qr[qr.size()-1-i]==guard,"composition: packed canary changed");
        // The same tensor written above is passed directly to the WO owner.
        strata_vk_q8_wo_run(wo.get(),wt,packed,y,rows);
        auto yr=contents(yb,yi.size());
        need(contents(xb,xi.size())==xi&&contents(wb,wi.size())==wi&&contents(qb,qi.size())==qr,"composition: input, weights or producer image mutated");
        for(size_t i=0;i<8;++i)need(yr[i]==guard&&yr[yr.size()-1-i]==guard,"composition: output canary changed");
        Bytes result(rows*4);std::memcpy(result.data(),yr.data()+8,result.size());
        for(size_t i=0;i<rows;++i){float v;std::memcpy(&v,result.data()+i*4,4);(void)w::checked(v);}
        ggml_backend_tensor_set(yb,yi.data(),0,yi.size()*4);strata_vk_q8_wo_run(wo.get(),wt,stale,y,rows);
        auto bad=contents(yb,yi.size());need(std::memcmp(bad.data()+8,result.data(),result.size())!=0,"composition: stale packed input did not change output");
        need(contents(sb,si.size())==si&&contents(xb,xi.size())==xi&&contents(wb,wi.size())==wi&&contents(qb,qi.size())==qr,"composition: stale test mutated inputs");
        for(size_t i=0;i<8;++i)need(bad[i]==guard&&bad[bad.size()-1-i]==guard,"composition: stale output canary changed");
#ifdef STRATA_GDN_OUTPUT_SCALE_SUITE
        auto direct=direct_packed();std::array<uint32_t,q::bytes/4+16> direct_image{};direct_image.fill(guard);std::memcpy(direct_image.data()+8,direct.data(),q::bytes);
        ggml_backend_tensor_set(sb,direct_image.data(),0,direct_image.size()*4);ggml_backend_tensor_set(yb,yi.data(),0,yi.size()*4);
        strata_vk_q8_wo_run(wo.get(),wt,stale,y,rows);bad=contents(yb,yi.size());
        Bytes direct_out(rows*4);std::memcpy(direct_out.data(),bad.data()+8,direct_out.size());direct_controls(direct_out);
        for(size_t i=0;i<8;++i)need(bad[i]==guard&&bad[bad.size()-1-i]==guard,"GDN output: direct canary changed");
        auto direct_after=contents(sb,si.size());need(std::equal(direct_after.begin(),direct_after.end(),direct_image.begin())&&contents(wb,wi.size())==wi,"GDN output: direct input or weights mutated");
        if(pass)need(direct_out==first_direct,"GDN output: direct repeat changed");else{first_direct=direct_out;save(dir/"direct-qx.bin",direct.data(),direct.size());save(dir/"direct-out.f32le",direct_out.data(),direct_out.size());}
#endif
        if(pass)need(packed_bytes==first_q&&result==first_y,"composition: reset/repeat changed");
        first_q=packed_bytes;first_y=result;
        save(dir/(pass?"qx-repeat.bin":"qx.bin"),packed_bytes.data(),packed_bytes.size());
        save(dir/(pass?"out-repeat.f32le":"out.f32le"),result.data(),result.size());
    }
    std::ofstream(dir/"report.json")<<"{\"actual_vulkan_quantizer_and_wo\":true,\"producer_buffer_consumed\":true,\"stale_changes_output\":true,\"invalid_noncommit_retry\":true,\"immutable_canary_repeat\":true,\"qualified\":false}\n";
    return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
