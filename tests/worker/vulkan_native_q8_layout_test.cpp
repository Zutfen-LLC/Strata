#include "vulkan_native_q8_layout.hpp"
#include <iostream>
#include <fstream>
#include "ggml-cpu.h"
using namespace strata::worker;
using namespace strata::worker::vulkan_detail;
int main(int argc,char** argv){try {
    require(native_q8_weight_bytes(24,128u<<20)==24*size_t(5222400),"packed slot accounting");
    for(auto [slots,budget]:{std::pair<size_t,size_t>{0,SIZE_MAX},{25,SIZE_MAX},{1,5222399}}){bool denied=false;try{native_q8_weight_bytes(slots,budget);}catch(const std::exception&){denied=true;}require(denied,"invalid resident weights accepted");}
    Backend backend(ggml_backend_cpu_init(),ggml_backend_free);
    Context storage(ggml_init({4096,nullptr,true}),ggml_free),views(ggml_init({4096,nullptr,true}),ggml_free);
    auto* base=ggml_new_tensor_1d(storage.get(),GGML_TYPE_I32,16);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(storage.get(),backend.get()),ggml_backend_buffer_free);
    auto* view=native_q8_view(views.get(),base,4);
    require(view->buffer==base->buffer && view->data==base->data && ggml_nbytes(view)==16,"post-allocation view lost backend binding");
    int32_t words[]={1,2,3,4},readback[4]{};ggml_backend_tensor_set(view,words,0,sizeof(words));ggml_backend_tensor_get(base,readback,0,sizeof(readback));
    require(std::equal(std::begin(words),std::end(words),std::begin(readback)),"view/base storage differs");
    GpuPlan p;p.n_layers=1;p.n_expert=32;
    for(int e=0;e<32;++e)p.assets.push_back({0,uint16_t(e),5222400,{}});
    InferRequest r;r.T=2;r.K=3;r.ids={3,1,3,1,2,3};r.mask={1,1,1,1,0,1};
    auto t=native_q8_groups(p,r);
    require(t.experts==std::vector<int32_t>({3,1}) && t.asset_slots==std::vector<int32_t>({3,1}) &&
        t.start==std::vector<int32_t>({0,3,5}) && t.tok==std::vector<int32_t>({0,0,1,0,1}) &&
        t.original==std::vector<int32_t>({0,2,5,1,3}) && t.dst==std::vector<int32_t>({0,1,2,3,4}),"native first-occurrence/stable-entry mapping");
    auto refused=[&](const InferRequest& q){try{native_q8_groups(p,q);}catch(const std::exception&){return;}throw std::runtime_error("unsupported call accepted");};
    auto q=r;q.mask[0]=2;refused(q);q=r;q.ids[0]=40;refused(q);q=r;q.mask.assign(6,0);refused(q);
    q.T=8;q.K=10;q.mask.assign(80,1);q.ids.assign(80,0);refused(q);
    q.T=2;q.K=10;q.mask.assign(20,1);q.ids.resize(20);for(int i=0;i<20;++i)q.ids[i]=i;
    require(native_q8_groups(p,q).experts.size()==20,"full 20-expert request refused");
    q.T=3;q.K=10;q.mask.assign(30,0);q.ids.resize(30);for(int i=0;i<30;++i){q.ids[i]=i;q.mask[i]=i<24;}
    require(native_q8_groups(p,q).experts.size()==24,"24-group boundary refused");
    q.mask[24]=1;refused(q);
    q.T=7;q.K=10;q.mask.assign(70,0);q.ids.assign(70,0);for(int i=0;i<64;++i)q.mask[i]=1;
    require(native_q8_groups(p,q).dst.size()==64,"64-entry boundary refused");
    if(argc==3 && std::string(argv[1])=="--metadata-fixture") {
        InferRequest emitted;emitted.T=2;emitted.K=10;emitted.layer=0;
        const std::string prefix=argv[2];
        for(bool legacy:{false,true}) {
            std::ofstream out(prefix+(legacy?".legacy":".numeric"),std::ios::binary);
            if(legacy)out<<"record "<<0<<" model "<<p.model_id<<" placement "<<p.placement_id.a<<' '<<p.placement_id.b<<" layer "<<emitted.layer<<" T "<<emitted.T<<" K "<<emitted.K<<'\n';
            else native_q8_metadata_header(out,0,p,emitted);
            out<<"00000000000000000000000000000000 00000000000000000000000000000000 00000000000000000000000000000000 00000000000000000000000000000000 \n";
            for(int row=0;row<6;++row){const int count=row==2?21:20;for(int i=0;i<count;++i)out<<(row==4?i/10:i)<<' ';out<<'\n';}
            require(bool(out),"metadata fixture write failed");
        }
    } else require(argc==1,"invalid metadata fixture arguments");
    std::cout<<"native grouping, entry restoration, capacities and refusal checks pass\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
