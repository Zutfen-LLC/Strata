#include "native_qsa_qkv_fixture.hpp"
#include "strata_vk_native_stages.hpp"
#include "strata_vk_q8_wo.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_qsa_qkv_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char**argv){try {
    if(argc==5&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3],argv[4]);return 0;}
    need(argc==11&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--weights"&&std::string(argv[7])=="--fixture"&&std::string(argv[9])=="--output","QSA QKV: invalid Vulkan arguments");
    auto input=read_input(argv[8]);std::array<Bytes,3>w;for(size_t i=0;i<3;++i)w[i]=read_weights(std::filesystem::path(argv[6])/(std::string(names[i])+".bin"),i);
    std::string idx=argv[2];need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"QSA QKV: invalid device");
    auto device=std::stoull(idx);need(device<size_t(ggml_backend_vk_get_device_count()),"QSA QKV: missing Vulkan device");
    std::filesystem::path dir=argv[10];need(std::filesystem::create_directory(dir),"QSA QKV: output exists");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend)&&std::string(argv[4]).size()&&std::string(ggml_backend_dev_description(ggml_backend_get_device(backend.get()))).find(argv[4])!=std::string::npos,"QSA QKV: unexpected Vulkan device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({32768,nullptr,true}),ggml_free);need(bool(ctx),"QSA QKV: metadata");
    auto make=[&](ggml_type type,size_t n){auto*b=ggml_new_tensor_1d(ctx.get(),type,n+16);return std::pair{b,ggml_view_1d(ctx.get(),b,n,32)};};
    auto [xb,x]=make(GGML_TYPE_F32,q::elements);auto [qb,packed]=make(GGML_TYPE_I32,q::bytes/4);auto [sb,stale]=make(GGML_TYPE_I32,q::bytes/4);
#ifdef STRATA_QKV_Q81_SCALE_SUITE
    auto [db,direct_tensor]=make(GGML_TYPE_I32,q::bytes/4);auto direct=direct_packed();std::array<Bytes,3>first_direct;
#endif
    auto*shortq=ggml_view_1d(ctx.get(),qb,q::bytes/4-1,32);std::array<ggml_tensor*,3>wb{},wt{},yb{},y{};
    for(size_t i=0;i<3;++i){auto a=make(GGML_TYPE_I32,w[i].size()/4);wb[i]=a.first;wt[i]=a.second;auto b=make(GGML_TYPE_F32,rows[i]);yb[i]=b.first;y[i]=b.second;}
    constexpr size_t budget=64u<<20;size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));need(bound<budget,"QSA QKV: tensor budget");
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)>quant(strata_vk_native_stages_create_quant_rcp_observation(backend.get(),(dir/"quant-effective.spv").c_str(),budget-bound),strata_vk_native_stages_destroy);
    std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)>owner(strata_vk_qsa_qkv_create(backend.get(),(dir/"qkv-effective.spv").c_str(),budget-strata_vk_native_stages_bytes(quant.get())),strata_vk_q8_wo_destroy);
    need(bound+strata_vk_native_stages_bytes(quant.get())+strata_vk_q8_wo_bytes(owner.get())<=budget,"QSA QKV: owner budget");
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"QSA QKV: allocation");
    constexpr uint32_t guard=0xa5a5a5a5;std::vector<uint32_t>xi(q::elements+16,guard),qi(q::bytes/4+16,guard),si(q::bytes/4+16,guard);std::memcpy(xi.data()+8,input.data(),q::elements*4);std::fill(si.begin()+8,si.end()-8,0u);
#ifdef STRATA_QKV_Q81_SCALE_SUITE
    std::vector<uint32_t>di(q::bytes/4+16,guard);std::memcpy(di.data()+8,direct.data(),direct.size());
#endif
    std::array<std::vector<uint32_t>,3>wi,yi;for(size_t i=0;i<3;++i){wi[i].assign(w[i].size()/4+16,guard);std::memcpy(wi[i].data()+8,w[i].data(),w[i].size());yi[i].assign(rows[i]+16,guard);}
    auto contents=[](ggml_tensor*t,size_t n){std::vector<uint32_t>v(n);ggml_backend_tensor_get(t,v.data(),0,n*4);return v;};
    auto reset=[&](){ggml_backend_tensor_set(xb,xi.data(),0,xi.size()*4);ggml_backend_tensor_set(qb,qi.data(),0,qi.size()*4);ggml_backend_tensor_set(sb,si.data(),0,si.size()*4);for(size_t i=0;i<3;++i){ggml_backend_tensor_set(wb[i],wi[i].data(),0,wi[i].size()*4);ggml_backend_tensor_set(yb[i],yi[i].data(),0,yi[i].size()*4);}};
    const int32_t starts[]={0,1};auto quantify=[&](ggml_tensor*out){strata_vk_native_stages_run(quant.get(),0,x,x,x,out,q::elements,starts,2);};
    reset();bool refused=false;try{quantify(shortq);}catch(...){refused=true;}need(refused&&contents(qb,qi.size())==qi,"QSA QKV: bad producer committed");
    refused=false;try{strata_vk_q8_wo_run(owner.get(),wt[0],shortq,y[0],rows[0]);}catch(...){refused=true;}need(refused&&contents(yb[0],yi[0].size())==yi[0],"QSA QKV: bad consumer committed");
    Bytes first_q;std::array<Bytes,3>first_y;
    for(int pass=0;pass<2;++pass){reset();quantify(packed);auto qr=contents(qb,qi.size());Bytes packed_bytes(q::bytes);std::memcpy(packed_bytes.data(),qr.data()+8,q::bytes);q::pinned(packed_bytes);packed_domain(packed_bytes);
#ifdef STRATA_QKV_Q81_SCALE_SUITE
        check_scale_words(packed_bytes,case_index(argv[8]));
#endif
        for(size_t j=0;j<8;++j)need(qr[j]==guard&&qr[qr.size()-1-j]==guard,"QSA QKV: packed canary");std::array<Bytes,3>output;
        for(size_t i=0;i<3;++i){strata_vk_q8_wo_run(owner.get(),wt[i],packed,y[i],rows[i]);auto result=contents(yb[i],yi[i].size());
            need(contents(xb,xi.size())==xi&&contents(wb[i],wi[i].size())==wi[i]&&contents(qb,qi.size())==qr,"QSA QKV: input changed");for(size_t j=0;j<8;++j)need(result[j]==guard&&result[result.size()-1-j]==guard,"QSA QKV: output canary");
            output[i].resize(rows[i]*4);std::memcpy(output[i].data(),result.data()+8,output[i].size());ggml_backend_tensor_set(yb[i],yi[i].data(),0,yi[i].size()*4);
            strata_vk_q8_wo_run(owner.get(),wt[i],stale,y[i],rows[i]);auto bad=contents(yb[i],yi[i].size());need(std::memcmp(bad.data()+8,output[i].data(),output[i].size())!=0,"QSA QKV: stale input not detected");
            need(contents(sb,si.size())==si&&contents(xb,xi.size())==xi&&contents(wb[i],wi[i].size())==wi[i]&&contents(qb,qi.size())==qr,"QSA QKV: stale test changed inputs");for(size_t j=0;j<8;++j)need(bad[j]==guard&&bad[bad.size()-1-j]==guard,"QSA QKV: stale output canary");
        }
        output_controls(output);if(pass)need(packed_bytes==first_q&&output==first_y,"QSA QKV: repeat changed");first_q=packed_bytes;first_y=output;
        save(dir/(pass?"qx-repeat.bin":"qx.bin"),packed_bytes.data(),packed_bytes.size());for(size_t i=0;i<3;++i)save(dir/(std::string(names[i])+(pass?"-repeat.f32le":".f32le")),output[i].data(),output[i].size());
#ifdef STRATA_QKV_Q81_SCALE_SUITE
        ggml_backend_tensor_set(db,di.data(),0,di.size()*4);std::array<Bytes,3>direct_output;
        for(size_t i=0;i<3;++i){ggml_backend_tensor_set(yb[i],yi[i].data(),0,yi[i].size()*4);strata_vk_q8_wo_run(owner.get(),wt[i],direct_tensor,y[i],rows[i]);auto result=contents(yb[i],yi[i].size());
            need(contents(db,di.size())==di&&contents(wb[i],wi[i].size())==wi[i],"QSA QKV: direct packed inputs changed");for(size_t j=0;j<8;++j)need(result[j]==guard&&result[result.size()-1-j]==guard,"QSA QKV: direct output canary");direct_output[i].resize(rows[i]*4);std::memcpy(direct_output[i].data(),result.data()+8,direct_output[i].size());}
        direct_controls(direct_output);if(pass)need(direct_output==first_direct,"QSA QKV: direct repeat changed");first_direct=direct_output;
        save(dir/(pass?"direct-qx-repeat.bin":"direct-qx.bin"),direct.data(),direct.size());for(size_t i=0;i<3;++i)save(dir/(std::string("direct-")+names[i]+(pass?"-repeat.f32le":".f32le")),direct_output[i].data(),direct_output[i].size());
#endif
    }
    std::ofstream report(dir/"report.json");report<<"{\"actual_vulkan_quantizer_and_mmvq\":true,\"one_packed_input\":true,\"stale_changes_output\":true,\"invalid_noncommit_retry\":true,\"immutable_canary_repeat\":true";
#ifdef STRATA_QKV_Q81_SCALE_SUITE
    report<<",\"direct_packed_controls\":true,\"all_case_scale_literals\":true";
#endif
    report<<",\"qualified\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
