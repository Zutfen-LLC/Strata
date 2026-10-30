// Exercise the real cache's workspace/views, ownership and reply restoration
// with CPU storage and component call recorders. No shader/numerical GPU proof.
#include "vulkan_native_q8_layout.hpp"
#include "vulkan_expert_cache.hpp"
#include "../../src/core/native_q8_address.hpp"
#include <set>
namespace {
int fail_allocation=0,allocation_attempts=0;
bool fail_upload=false;
std::set<ggml_backend_buffer_t> cache_buffers;
ggml_backend_buffer_t test_allocate(ggml_context* ctx,ggml_backend_t backend) {
    if(++allocation_attempts==fail_allocation)return nullptr;
    auto* result=ggml_backend_alloc_ctx_tensors(ctx,backend);if(result)cache_buffers.insert(result);return result;
}
void test_free(ggml_backend_buffer_t buffer) {cache_buffers.erase(buffer);ggml_backend_buffer_free(buffer);}
void test_upload(ggml_tensor* tensor,const void* data,size_t offset,size_t size) {
    if(fail_upload)throw std::runtime_error("injected upload failure");
    ggml_backend_tensor_set(tensor,data,offset,size);
}
}
#define ggml_backend_alloc_ctx_tensors test_allocate
#define ggml_backend_buffer_free test_free
#define ggml_backend_tensor_set test_upload
#include "vulkan_native_q8_cache.hpp"
#undef ggml_backend_alloc_ctx_tensors
#undef ggml_backend_buffer_free
#undef ggml_backend_tensor_set
#include "ggml-cpu.h"
#include <iostream>
#include <cstring>
#include <limits>
using namespace strata::worker;
using namespace strata::worker::vulkan_detail;
namespace {
int owners=0,gu_calls=0,down_calls=0,stage_calls=0,fail_create=0;
size_t metadata_bytes=0;
bool fail_down=false,check_weight_owners=false;
int address_features=3;
template<class F>void refuses(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid cache operation accepted");}
struct MetadataOwner {
    Context ctx{nullptr,ggml_free};Buffer buffer{nullptr,ggml_backend_buffer_free};ggml_tensor* metadata;
    MetadataOwner(ggml_backend_t backend,size_t words) {
        if(fail_create && owners==fail_create)throw std::runtime_error("injected component creation failure");
        ctx.reset(ggml_init({4096,nullptr,true}));metadata=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,words);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),backend));require(bool(buffer),"test metadata allocation failed");metadata_bytes+=bytes();++owners;
    }
    ~MetadataOwner(){metadata_bytes-=bytes();--owners;}
    size_t bytes()const{return ggml_backend_buffer_get_size(buffer.get());}
};
}
namespace strata::core::portable_detail {
size_t block_limit=2*size_t(5222400);
size_t strata_vk_q8_resident_block_limit(ggml_backend_t){q8_require_address_features(address_features&1,address_features&2);return block_limit;}
struct StrataVkNativeStages:MetadataOwner{using MetadataOwner::MetadataOwner;};
struct StrataVkQ8Gu:MetadataOwner{using MetadataOwner::MetadataOwner;};
struct StrataVkQ8Down:MetadataOwner{using MetadataOwner::MetadataOwner;};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rn_observation(ggml_backend_t b,const char*,size_t){return new StrataVkNativeStages(b,STRATA_Q8_STAGE_WORDS);}
StrataVkQ8Gu* strata_vk_q8_gu_create(ggml_backend_t b,const char*){return new StrataVkQ8Gu(b,q8_metadata_words);}
StrataVkQ8Down* strata_vk_q8_down_create(ggml_backend_t b,const char*){return new StrataVkQ8Down(b,q8_metadata_words);}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages* p){return p?p->bytes():0;}
size_t strata_vk_q8_gu_bytes(const StrataVkQ8Gu* p){return p?p->bytes():0;}
size_t strata_vk_q8_down_bytes(const StrataVkQ8Down* p){return p?p->bytes():0;}
void strata_vk_native_stages_destroy(StrataVkNativeStages* p)noexcept{delete p;}
const ggml_tensor* strata_vk_q8_gu_metadata_tensor(const StrataVkQ8Gu* p) {return p?p->metadata:nullptr;}
void strata_vk_q8_gu_destroy(StrataVkQ8Gu* p)noexcept{delete p;}
const ggml_tensor* strata_vk_q8_down_metadata_tensor(const StrataVkQ8Down* p) {return p?p->metadata:nullptr;}
void strata_vk_q8_down_destroy(StrataVkQ8Down* p)noexcept{delete p;}
void strata_vk_native_stages_run(StrataVkNativeStages*,uint32_t mode,const ggml_tensor* g,const ggml_tensor*,ggml_tensor*,ggml_tensor* q,uint32_t f,const int32_t* starts,size_t count) {
    ++stage_calls;require(mode==0 || mode==2,"unexpected cache stage");
    require(ggml_nelements(g)==int64_t(starts[count-1])*f,"stage workspace view size");
    require(ggml_nelements(q)==int64_t(starts[count-1])*(f/32)*9,"packed stage view size");
}
void strata_vk_q8_gu_run_blocks(StrataVkQ8Gu* s,const std::vector<const ggml_tensor*>& w,const ggml_tensor* x,ggml_tensor* g,ggml_tensor* u,const Q8GuLayout& l) {
    ++gu_calls;
    if(check_weight_owners)for(size_t group=0;group<w.size();++group) {
        uint32_t marker=0;ggml_backend_tensor_get(w[group],&marker,l.weight_offsets[group],4);
        require(marker==80-group,"GU owner/slot offset changed");
    }
    std::vector<size_t> sizes;for(auto* t:w)sizes.push_back(ggml_nbytes(t));const auto m=q8_gu_metadata(l,sizes);
    ggml_backend_tensor_set(s->metadata,m.data(),0,sizeof(m));
    require(ggml_nelements(x)==int64_t(l.tokens)*(l.h/32)*9 && ggml_nelements(g)==int64_t(l.entries)*l.f && ggml_nelements(u)==ggml_nelements(g),"GU workspace view size");
    // Marker is only an association probe, not a replacement arithmetic path.
    std::vector<float> values(ggml_nelements(g),1.f);ggml_backend_tensor_set(g,values.data(),0,ggml_nbytes(g));ggml_backend_tensor_set(u,values.data(),0,ggml_nbytes(u));
}
void strata_vk_q8_down_run_blocks(StrataVkQ8Down* s,const std::vector<const ggml_tensor*>& w,const ggml_tensor* q,ggml_tensor* o,const Q8DownLayout& l) {
    ++down_calls;
    if(check_weight_owners)for(size_t group=0;group<w.size();++group) {
        uint32_t marker=0;ggml_backend_tensor_get(w[group],&marker,l.weight_offsets[group],4);
        require(marker==80-group,"down owner/matrix offset changed");
    }
    std::vector<size_t> sizes;for(auto* t:w)sizes.push_back(ggml_nbytes(t));const auto m=q8_down_metadata(l,sizes);ggml_backend_tensor_set(s->metadata,m.data(),0,sizeof(m));
    require(ggml_nelements(q)==int64_t(l.entries)*(l.f/32)*9 && ggml_nelements(o)==int64_t(l.destinations)*l.h,"down workspace view size");
    std::vector<float> values(ggml_nelements(o));
    for(uint32_t e=0;e<l.entries;++e)std::fill_n(values.begin()+l.destination_ids[e]*l.h,l.h,float(e+1));
    if(fail_down)values.back()=std::numeric_limits<float>::quiet_NaN();
    ggml_backend_tensor_set(o,values.data(),0,ggml_nbytes(o));
}
}
int main(){try {
    Backend backend(ggml_backend_cpu_init(),ggml_backend_free);require(bool(backend),"CPU storage unavailable");
    size_t actual=0,scratch=0;
    for(int repeat=0;repeat<2;++repeat) {
        NativeQ8Cache cache(backend.get(),2,128u<<20,"call-recorder","",false);
        require(owners==3,"component ownership");
        const auto initial=cache.allocation_snapshot();
        require(initial.gpu_allocation_count==5 && initial.gpu_weight_bytes==2*size_t(5222400) && initial.gpu_allocated_bytes==cache.buffer_bytes() && initial.gpu_scratch_bytes==cache.scratch_bytes(),"actual allocation accounting");
        if(repeat)require(actual==cache.buffer_bytes() && scratch==cache.scratch_bytes(),"recreated allocations unstable");
        actual=cache.buffer_bytes();scratch=cache.scratch_bytes();
        // Measure the old/new aligned allocation sizes on this backend instead
        // of confusing raw payload growth with backend allocation growth.
        Context old(ggml_init({65536,nullptr,true}),ggml_free);
        ggml_new_tensor_1d(old.get(),GGML_TYPE_I32,2*5222400/4);
        ggml_new_tensor_1d(old.get(),GGML_TYPE_F32,8*H);ggml_new_tensor_1d(old.get(),GGML_TYPE_I32,8*(H/32)*9);
        for(int j=0;j<3;++j)ggml_new_tensor_1d(old.get(),GGML_TYPE_F32,64*F);
        ggml_new_tensor_1d(old.get(),GGML_TYPE_I32,64*(F/32)*9);ggml_new_tensor_1d(old.get(),GGML_TYPE_F32,64*H);
        size_t old_tensors=ggml_backend_alloc_ctx_tensors_from_buft_size(old.get(),ggml_backend_get_default_buffer_type(backend.get()));
        require(strata::core::portable_detail::q8_metadata_words==322 && scratch>=80*(3*F*4+(F/32)*36+H*4),"entry workspace undercharged");
        std::cout<<"host_storage_only=true tensor_and_metadata_bytes="<<actual<<" scratch_bytes="<<scratch<<" old_tensor_bytes="<<old_tensors<<" new_tensor_bytes="<<actual-metadata_bytes<<" component_metadata_bytes="<<metadata_bytes<<"\n";
        std::vector<uint8_t> blob(5222400);cache.upload(0,blob);cache.upload(1,blob);
        refuses([&]{cache.upload(0,blob);});refuses([&]{cache.upload(2,blob);});
        GpuPlan p;p.n_layers=1;p.n_expert=2;for(int e=0;e<2;++e)p.assets.push_back({0,uint16_t(e),5222400,{}});
        InferRequest r;r.T=8;r.K=10;r.session=4;r.sequence=9;r.ids.resize(80);r.mask.assign(80,1);r.activations.assign(8*H,1.f);
        for(int i=0;i<80;++i)r.ids[i]=(i*7+i/10)%2;
        for(int n:{64,65,79,80}) {
            for(int i=0;i<80;++i)r.mask[(i*37)%80]=i<n;
            const int gu_before=gu_calls,down_before=down_calls,stage_before=stage_calls;
            GpuExecutionTrace trace;const auto reply=cache.run(p,r,&trace);
            require(gu_calls==gu_before+1 && down_calls==down_before+1 && stage_calls==stage_before+2,"capacity call was split");
            require(reply.indices.size()==size_t(n) && reply.rows.size()==size_t(n)*H && reply.session==4 && reply.sequence==9,"reply shape/identity");
            size_t at=0;for(int i=0;i<80;++i)if(r.mask[i]) {
                require(reply.indices[at]==i,"reply order");const auto found=std::find(trace.original.begin(),trace.original.end(),i);
                const float marker=float(found-trace.original.begin()+1);
                for(size_t h=0;h<H;++h)require(reply.rows[at*H+h]==marker,"reply row association");
                ++at;
            }
            const auto after=cache.allocation_snapshot();require(after.gpu_allocated_bytes==initial.gpu_allocated_bytes && after.gpu_scratch_bytes==initial.gpu_scratch_bytes && after.gpu_allocation_count==5,"run allocation growth");
        }
        const auto prior=cache.run(p,r);auto retained=prior;
        fail_down=true;refuses([&]{retained=cache.run(p,r);});fail_down=false;
        require(retained.indices==prior.indices && retained.rows==prior.rows,"failed output published partial reply");
        auto single=r;single.T=1;single.ids.resize(10);single.mask.assign(10,1);single.activations.resize(H);
        require(cache.run(p,single).indices.size()==10,"existing T1 request refused");
        require(cache.run(p,r).indices.size()==80,"failed output poisoned next call");
    }
    require(owners==0,"cache leaked component owners");
    // This budget covers weights and the old 64-entry tensor workspace, but
    // cannot hold the enlarged workspace. Refusal happens before owners exist.
    refuses([&]{NativeQ8Cache cache(backend.get(),2,actual-200000,"too-small","",false);});require(owners==0,"budget refusal left partial state");
    fail_create=1;refuses([&]{NativeQ8Cache cache(backend.get(),2,128u<<20,"partial","",false);});fail_create=0;
    require(owners==0,"partial component creation leaked");
    {
        using namespace strata::core::portable_detail;
        NativeQ8Cache cache(backend.get(),81,600u<<20,"multiblock","",false);
        const auto initial=cache.allocation_snapshot();
        require(initial.gpu_allocation_count==45 && initial.gpu_scratch_bytes==scratch,"block/scratch allocation accounting");
        std::vector<uint8_t> blob(5222400);for(uint32_t slot=0;slot<81;++slot) {
            std::memcpy(blob.data(),&slot,4);std::memcpy(blob.data()+3481600,&slot,4);cache.upload(slot,blob);
        }
        GpuPlan p;p.n_layers=1;p.n_expert=81;for(int e=0;e<81;++e)p.assets.push_back({0,uint16_t(e),5222400,{}});
        InferRequest r;r.T=8;r.K=10;r.mask.assign(80,1);r.activations.assign(8*H,1.f);
        for(int e=0;e<80;++e)r.ids.push_back(80-e);
        const int before=gu_calls;GpuExecutionTrace trace;check_weight_owners=true;const auto reply=cache.run(p,r,&trace);check_weight_owners=false;
        require(reply.indices.size()==80 && trace.experts.size()==80 && trace.asset_slots.front()==80 && gu_calls==before+1,"80 distinct groups across blocks");
        require(cache.allocation_snapshot().gpu_allocated_bytes==initial.gpu_allocated_bytes,"multi-block run allocation growth");
    }
    require(owners==0 && metadata_bytes==0,"multi-block release leaked");
    require(cache_buffers.empty(),"resident/scratch buffers leaked");
    for(address_features=0;address_features<3;++address_features) {
        const auto attempts=allocation_attempts;
        refuses([&]{NativeQ8Cache unsupported(backend.get(),3,64u<<20,"unsupported","",false);});
        require(owners==0 && cache_buffers.empty() && allocation_attempts==attempts,"unsupported features allocated residency");
    }
    address_features=3;
    // Fail each of the two resident allocations and scratch after one success.
    for(int failure=1;failure<=3;++failure) {
        allocation_attempts=0;fail_allocation=failure;
        refuses([&]{NativeQ8Cache failed(backend.get(),3,64u<<20,"allocation-failure","",false);});
        require(cache_buffers.empty() && owners==0 && metadata_bytes==0,"partial allocation rollback leaked");
    }
    fail_allocation=0;
    {
        NativeQ8Cache failed(backend.get(),3,64u<<20,"upload-failure","",false);
        std::vector<uint8_t> blob(5222400);failed.upload(0,blob);fail_upload=true;
        refuses([&]{failed.upload(1,blob);});fail_upload=false;
        refuses([&]{failed.upload(2,blob);});
        GpuPlan p;InferRequest r;refuses([&]{failed.run(p,r);});
    }
    require(cache_buffers.empty() && owners==0,"upload failure ownership leaked");
    // Exact accounting is admission: one byte less refuses without owners.
    {NativeQ8Cache exact(backend.get(),2,actual,"exact","",false);require(exact.buffer_bytes()==actual,"exact budget changed");}
    refuses([&]{NativeQ8Cache short_budget(backend.get(),2,actual-1,"short","",false);});
    require(owners==0 && metadata_bytes==0,"exact refusal rollback");
    std::cout<<"native 80-entry cache call/size/order/accounting tests passed; no GPU numerical proof\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
