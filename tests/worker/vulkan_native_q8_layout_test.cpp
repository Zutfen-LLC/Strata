#include "vulkan_native_q8_layout.hpp"
#include "vulkan_expert_cache.hpp"
#include <iostream>
#include <fstream>
#include "ggml-cpu.h"
using namespace strata::worker;
using namespace strata::worker::vulkan_detail;
int main(int argc,char** argv){try {
    require(native_q8_weight_bytes(24,128u<<20)==24*size_t(5222400),"packed slot accounting");
    for(auto [slots,budget]:{std::pair<size_t,size_t>{0,SIZE_MAX},{SIZE_MAX,SIZE_MAX},{1,5222399}}){bool denied=false;try{native_q8_weight_bytes(slots,budget);}catch(const std::exception&){denied=true;}require(denied,"invalid resident weights accepted");}
    for(size_t slots:{25,205,206,822,823,960,1378}) {
        const auto layout=expert_block_layout(slots,5222400,1ull<<30);
        require(layout.slots_per_block==205 && layout.blocks==1+(slots-1)/205 &&
            layout.bytes==native_q8_weight_bytes(slots,SIZE_MAX),"whole-blob block/accounting boundary");
        require((slots-1)%205*size_t(5222400)+5222400<=1ull<<30,"last blob exceeds owner");
    }
    for(auto limits:{std::pair<size_t,size_t>{1,5222399},{SIZE_MAX,1ull<<30}}) {
        bool bad=false;try{expert_block_layout(limits.first,5222400,limits.second);}catch(const std::exception&){bad=true;}
        require(bad,"invalid backend block/overflow admitted");
    }
    Backend backend(ggml_backend_cpu_init(),ggml_backend_free);
    Context storage(ggml_init({4096,nullptr,true}),ggml_free),views(ggml_init({4096,nullptr,true}),ggml_free);
    auto* base=ggml_new_tensor_1d(storage.get(),GGML_TYPE_I32,16);
    Buffer buffer(ggml_backend_alloc_ctx_tensors(storage.get(),backend.get()),ggml_backend_buffer_free);
    auto* view=native_q8_view(views.get(),base,4);
    require(view->buffer==base->buffer && view->data==base->data && ggml_nbytes(view)==16,"post-allocation view lost backend binding");
    bool oversized=false;try{native_q8_view(views.get(),base,17);}catch(const std::exception&){oversized=true;}
    require(oversized,"oversized workspace view accepted");
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
    q.T=9;q.K=10;refused(q);q.T=8;q.K=11;refused(q);q.T=0;q.K=10;refused(q);
    q.T=2;q.K=10;q.mask.assign(20,1);q.ids.resize(20);for(int i=0;i<20;++i)q.ids[i]=i;
    require(native_q8_groups(p,q).experts.size()==20,"full 20-expert request refused");
    q.T=3;q.K=10;q.mask.assign(30,0);q.ids.resize(30);for(int i=0;i<30;++i){q.ids[i]=i;q.mask[i]=i<24;}
    require(native_q8_groups(p,q).experts.size()==24,"24-group boundary refused");
    q.mask[24]=1;require(native_q8_groups(p,q).experts.size()==25,"25-group boundary refused");
    p.n_expert=128;for(int e=32;e<128;++e)p.assets.push_back({0,uint16_t(e),5222400,{}});
    q.T=8;q.K=10;q.mask.assign(80,1);q.ids.resize(80);
    for(int i=0;i<80;++i)q.ids[i]=127-i;
    auto all=native_q8_groups(p,q);require(all.experts.size()==80 && all.asset_slots.front()==127 && all.asset_slots.back()==48 && all.start.back()==80,"80 distinct groups/high resident slot refused");
    // Native T8/K10 repeats assigned experts. Assert the complete stable
    // grouping and inverse reply association for full and interleaved masks.
    p.assets.resize(24);
    for(int n:{64,65,79,80})for(bool mixed:{false,true}) {
        q.T=8;q.K=10;q.mask.assign(80,0);q.ids.resize(80);
        for(int i=0;i<80;++i){q.ids[i]=(i*7+3)%24;q.mask[mixed?(i*37)%80:i]=i<n;}
        auto grouped=native_q8_groups(p,q);
        require(grouped.dst.size()==size_t(n),"entry capacity refused");
        std::vector<int32_t> expected_experts,expected_entries,expected_tokens,expected_starts;
        for(int i=0;i<80;++i)if(q.mask[i] && std::find(expected_experts.begin(),expected_experts.end(),q.ids[i])==expected_experts.end())expected_experts.push_back(q.ids[i]);
        for(auto expert:expected_experts){expected_starts.push_back(expected_entries.size());
            for(int i=0;i<80;++i)if(q.mask[i] && q.ids[i]==expert){expected_entries.push_back(i);expected_tokens.push_back(i/10);}}
        expected_starts.push_back(n);
        require(grouped.experts==expected_experts && grouped.asset_slots==expected_experts && grouped.original==expected_entries &&
            grouped.tok==expected_tokens && grouped.start==expected_starts,"stable grouped mapping changed");
        std::vector<int32_t> restored(80,-1);
        for(int i=0;i<n;++i){require(grouped.dst[i]==i,"packed destination changed");restored[grouped.original[i]]=i;}
        for(int i=0;i<80;++i)require(q.mask[i]?(restored[i]>=0 && grouped.original[restored[i]]==i):restored[i]==-1,"reply order restoration/padded entry");
    }
    q.ids[0]=24;q.mask[0]=1;refused(q); // assigned shape, missing resident asset
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
