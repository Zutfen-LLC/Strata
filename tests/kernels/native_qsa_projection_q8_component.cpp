#include "native_qsa_projection_q8_fixture.hpp"
#include "strata_vk_native_stages.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_projection_q8;
using namespace strata::core::portable_detail;
int main(int argc,char**argv){try{
    if(argc==4&&std::string(argv[1])=="--contract"){contract(read(argv[2]),argv[3]);return 0;}
    need(argc==9&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--fixture"&&std::string(argv[7])=="--output","projection q8: invalid arguments");
    auto input=read(argv[6]);std::string idx=argv[2];need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"projection q8: invalid device");size_t device=std::stoull(idx);
    need(device<size_t(ggml_backend_vk_get_device_count()),"projection q8: missing device");
    std::filesystem::path dir=argv[8];need(std::filesystem::create_directory(dir),"projection q8: output exists");std::filesystem::copy_file(argv[6],dir/"fixture.bin");trace(input,dir/"host-rne-trace.csv");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_vk_init(device),ggml_backend_free);
    need(bool(backend)&&std::string(argv[4]).size()&&std::string(ggml_backend_dev_description(ggml_backend_get_device(backend.get()))).find(argv[4])!=std::string::npos,"projection q8: unexpected device");
    std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"projection q8: metadata");
    auto make=[&](ggml_type type,size_t n){auto*b=ggml_new_tensor_1d(ctx.get(),type,n+16);return std::pair{b,ggml_view_1d(ctx.get(),b,n,32)};};
    auto [xb,x]=make(GGML_TYPE_F32,elements);auto [qb,q]=make(GGML_TYPE_I32,bytes/4);
    auto*shortq=ggml_view_1d(ctx.get(),qb,bytes/4-1,32);auto*alias=ggml_view_1d(ctx.get(),xb,bytes/4,32);alias->type=GGML_TYPE_I32;
    size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));need(bound<=1u<<20,"projection q8: tensor budget");
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer)&&ggml_backend_buffer_get_size(buffer.get())<=1u<<20,"projection q8: actual tensor budget");
    std::unique_ptr<StrataVkNativeStages,decltype(&strata_vk_native_stages_destroy)>stage(strata_vk_native_stages_create_quant_rn_observation(backend.get(),(dir/"effective.spv").c_str(),1u<<20),strata_vk_native_stages_destroy);need(bool(stage),"projection q8: RN stage");
    std::vector<uint32_t> xi(elements+16,canary),qi(bytes/4+16,canary);std::memcpy(xi.data()+8,input.data(),elements*4);
    auto reset=[&](){ggml_backend_tensor_set(xb,xi.data(),0,xi.size()*4);ggml_backend_tensor_set(qb,qi.data(),0,qi.size()*4);};
    auto contents=[&](ggml_tensor*t,size_t n){std::vector<uint32_t>v(n);ggml_backend_tensor_get(t,v.data(),0,n*4);return v;};
    const int32_t starts[]={0,1};auto run=[&](ggml_tensor*out){strata_vk_native_stages_run(stage.get(),0,x,x,x,out,elements,starts,2);};
    reset();for(auto*out:{shortq,alias}){bool refused=false;try{run(out);}catch(...){refused=true;}need(refused,"projection q8: invalid call accepted");need(contents(xb,xi.size())==xi&&contents(qb,qi.size())==qi,"projection q8: refusal mutated buffers");}
    Bytes first;for(int pass=0;pass<2;++pass){reset();run(q);auto a=contents(xb,xi.size()),b=contents(qb,qi.size());Bytes out(bytes);std::memcpy(out.data(),b.data()+8,bytes);save(dir/(pass?"qx-repeat.bin":"qx.bin"),out);need(a==xi,"projection q8: input mutated");for(size_t i=0;i<8;++i)need(b[i]==canary&&b[b.size()-1-i]==canary,"projection q8: canary changed");if(pass)need(out==first,"projection q8: reset/repeat mismatch");first=out;}
    pinned(first);std::ofstream(dir/"report.json")<<"{\"actual_vulkan\":true,\"mode\":0,\"rn_candidate\":true,\"immutable_canary_repeat_refusal\":true,\"qualified\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
