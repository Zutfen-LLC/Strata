#pragma once
#include "portable_layer_chunk_fixture.hpp"
#include "portable_ple_fixture.hpp"
#include <limits>
#include "../../src/core/portable_memory.hpp"
#include <set>

namespace gdn_shared_fixture {
using namespace strata::core;
using gdn_fixture::check;
struct BackingCapture {
    static inline decltype(ggml_backend_i::graph_compute) original=nullptr;
    static inline ggml_backend_buffer_t scratch=nullptr,state=nullptr;
    static inline ggml_tensor* state_tensor=nullptr;
    static inline std::vector<ggml_tensor*> persistent;
    static ggml_status compute(ggml_backend_t backend,ggml_cgraph* graph) {
        const auto result=original(backend,graph);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* t=ggml_graph_node(graph,i);
            if(t->op==GGML_OP_GATED_DELTA_NET) {
                auto* root=t;while(root->view_src)root=root->view_src;
                scratch=root->buffer;
                root=t->src[5];while(root->view_src)root=root->view_src;
                state=root->buffer;state_tensor=root;
                persistent.clear();
                std::set<ggml_tensor*> visited;
                const auto walk=[&](auto&& self,ggml_tensor* tensor)->void {
                    if(!tensor || !visited.insert(tensor).second)return;
                    if(tensor->op==GGML_OP_NONE && tensor->buffer==state && tensor->type==GGML_TYPE_F32)
                        persistent.push_back(tensor);
                    for(auto* src:tensor->src)self(self,src);
                };
                for(int j=0;j<ggml_graph_n_nodes(graph);++j)walk(walk,ggml_graph_node(graph,j));
                std::sort(persistent.begin(),persistent.end(),[](auto* a,auto* b){return ggml_nelements(a)<ggml_nelements(b);});
                break;
            }
        }
        return result;
    }
};
struct PlanFault {
    static inline decltype(ggml_backend_device_i::supports_op) original=nullptr;
    static bool supports(ggml_backend_dev_t device,const ggml_tensor* tensor) {
        return tensor->op!=GGML_OP_GATED_DELTA_NET && original(device,tensor);
    }
};
inline void run(PortableGdnOptions options, bool gpu=false) {
    std::string error;
    auto owner=PortableGdnScratch::create(options.execution,error);
    check(bool(owner),error);
    auto overflow=options.execution; overflow.graph_budget=SIZE_MAX;
    check(!PortableGdnScratch::create(overflow,error),"overflowing shared metadata budget rejected");
    options.scratch=owner;
    gdn_fixture::Fixture a_fixture,b_fixture;
    for(auto& v:b_fixture.views) v.name.replace(0,6,"blk.2.");
    // Change persistent weights, not merely the layer label.
    for(auto& v:b_fixture.views) if(v.name=="blk.2.ssm_conv1d.weight") {
        auto* bytes=b_fixture.data[10].data();
        for(size_t i=0;i<v.bytes/4;++i) { float x; std::memcpy(&x,bytes+4*i,4);x*=0.75f;std::memcpy(bytes+4*i,&x,4); }
    }
    auto make=[&](int layer,const std::vector<PortableTensorView>& views,PortableGdnOptions o) {
        auto result=PortableGdnLayer::create(layer,views,o,error);check(bool(result),error);return result;
    };
    auto a=make(0,a_fixture.views,options),b=make(2,b_fixture.views,options);
    const auto primed=owner->memory_usage();
    auto private_options=options;private_options.scratch.reset();
    auto pa=make(0,a_fixture.views,private_options),pb=make(2,b_fixture.views,private_options);
    check(a->scratch_owner()==owner.get() && b->scratch_owner()==owner.get() && !pa->scratch_owner(),"actual owner identity");
    auto total=[](const PortableMemoryUsage& m){return m.host_graph+m.device_graph;};
    const size_t private_buffers=total(pa->memory_usage())+total(pb->memory_usage());
    const size_t shared_buffers=total(a->memory_usage())+total(b->memory_usage())+total(primed);
    check(private_buffers>shared_buffers,"actual retained scheduler backing reduction");
    check(a->memory_usage().device_state==pa->memory_usage().device_state &&
          b->memory_usage().device_state==pb->memory_usage().device_state,"persistent state allocation unchanged");
    const PortableExpertRows provider=[](int64_t,const PortableRoute& r,std::vector<float>& rows,std::string& e) {
        return gdn_fixture::experts()(0,r,rows,e);
    };
    const PortableExpertRowsBatch batch=[&](int64_t l,const std::vector<PortableRoute>& routes,std::vector<float>& rows,std::string& e) {
        rows.clear();for(const auto& r:routes) {std::vector<float> one;if(!provider(l,r,one,e))return false;rows.insert(rows.end(),one.begin(),one.end());}return true;
    };
    std::map<const PortableGdnLayer*,std::vector<ggml_tensor*>> device_state;
    const auto observe=[&](PortableGdnLayer& layer,auto action) {
        auto* backend=options.execution.accelerator;
        const auto original=backend?backend->iface.graph_compute:nullptr;
        if(backend) {BackingCapture::original=original;backend->iface.graph_compute=BackingCapture::compute;}
        bool ok=false;
        try {ok=action();}catch(...){if(backend)backend->iface.graph_compute=original;throw;}
        if(backend) {backend->iface.graph_compute=original;if(options.resident_state)device_state[&layer]=BackingCapture::persistent;}
        return ok;
    };
    auto same=[&](const PortableGdnLayer& x,const PortableGdnLayer& y) {
        layer_chunk_fixture::same(layer_chunk_fixture::snapshot(x),layer_chunk_fixture::snapshot(y),true,"shared/private GDN");
        if(device_state.count(&x) && device_state.count(&y)) {
            const auto& a=device_state.at(&x);const auto& b=device_state.at(&y);
            check(a.size()==b.size() && a.size()>=2,"all resident state/history tensors captured");
            size_t bytes=0;
            for(size_t i=0;i<a.size();++i) {
                check(ggml_nbytes(a[i])==ggml_nbytes(b[i]),"resident state shape equality");
                std::vector<uint8_t> lhs(ggml_nbytes(a[i])),rhs(lhs.size());
                ggml_backend_tensor_get(a[i],lhs.data(),0,lhs.size());ggml_backend_tensor_get(b[i],rhs.data(),0,rhs.size());
                check(lhs==rhs,"all resident state/history bytes equal");bytes+=lhs.size();
            }
            check(bytes==3268608+(a.size()==3?9*10240*sizeof(float):0),"full recurrent/convolution/PLE state coverage");
        }
    };
    auto step=[&](PortableGdnLayer& x,PortableGdnLayer& y,int seed) {
        PortableRoute xr,yr;
        check(observe(x,[&]{return x.step(seed,a_fixture.input(seed),provider,xr,error);}),error);
        check(observe(y,[&]{return y.step(seed,a_fixture.input(seed),provider,yr,error);}),error);
        check(xr.ids==yr.ids && layer_chunk_fixture::exact(xr.weights,yr.weights),"byte-identical routing");same(x,y);
    };
    ggml_backend_buffer_t shared_backing=nullptr;
    std::set<ggml_backend_buffer_t> state_backings;
    for(int t=0;t<5;++t) {
        for(auto pair:{std::pair{a.get(),pa.get()},std::pair{b.get(),pb.get()}}) {
            auto* backend=options.execution.accelerator;
            PortableRoute r;
            const bool ok=observe(*pair.first,[&]{return pair.first->step(t,a_fixture.input(t+(pair.first==b.get()?31:0)),provider,r,error);});
            check(ok,error);
            if(backend) {
                check(BackingCapture::scratch!=nullptr,"actual scratch backing captured");
                if(!shared_backing)shared_backing=BackingCapture::scratch;
                check(shared_backing==BackingCapture::scratch,"alternating layers reuse identical backing buffer");
                state_backings.insert(BackingCapture::state);
            }
            check(observe(*pair.second,[&]{return pair.second->step(t,a_fixture.input(t+(pair.first==b.get()?31:0)),provider,r,error);}),error);
            same(*pair.first,*pair.second);
        }
    }
    if(options.resident_state && options.execution.accelerator)
        check(state_backings.size()==2 && !state_backings.count(shared_backing),"two persistent state backings distinct from shared scratch");
    if(options.policy==PortableGdnPolicy::eager_diagnostic)
        check(a->recurrent_state()!=b->recurrent_state() && a->recurrent_state().data()!=b->recurrent_state().data(),"different state/weights remain isolated");
    for(size_t T=2;T<=8;++T) {
        for(auto pair:{std::pair{a.get(),pa.get()},std::pair{b.get(),pb.get()}}) {
            std::vector<float> x;for(size_t i=0;i<T;++i) {auto row=a_fixture.input(int(i+T));x.insert(x.end(),row.begin(),row.end());}
            auto y=x;std::vector<PortableRoute> xr,yr;
            std::vector<int32_t> tokens(T,5);
            check(pair.first->prepare_chunk(T,error),error);
            check(pair.first->run_chunk(tokens,pair.first->tokens(),x,batch,xr,error),error);
            check(pair.second->run_chunk(tokens,pair.second->tokens(),y,batch,yr,error),error);
            check(layer_chunk_fixture::exact(x,y) && xr.size()==yr.size(),"byte-identical chunk/tail residuals");
            for(size_t i=0;i<xr.size();++i) check(xr[i].ids==yr[i].ids && layer_chunk_fixture::exact(xr[i].weights,yr[i].weights),"chunk routes exact");
            same(*pair.first,*pair.second);
        }
        step(*a,*pa,int(T));step(*b,*pb,int(T+41));
    }
    // The provider executes while the lease is held, even after readback.
    PortableRoute route;
    bool reentered=false;
    check(a->step(5,a_fixture.input(5),[&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e) {
        std::string rejected;
        reentered=!b->prepare_chunk(2,rejected) && rejected.find("already borrowed")!=std::string::npos;
        check(!a->reset(rejected) && rejected.find("already borrowed")!=std::string::npos,"same-layer reset rejects during lease");
        return provider(l,r,rows,e);
    },route,error),error);
    check(reentered,"provider cannot reenter another borrower");
    check(pa->step(5,a_fixture.input(5),provider,route,error),error);same(*a,*pa);
    {
        std::vector<float> rows=a_fixture.input(1),tail=a_fixture.input(2);
        rows.insert(rows.end(),tail.begin(),tail.end());auto reference=rows;
        std::vector<PortableRoute> got,expected;
        check(a->run_chunk({1,2},a->tokens(),rows,[&](int64_t l,const auto& routes,auto& out,auto& e) {
            std::string rejected;
            check(!b->prepare_chunk(2,rejected) && rejected.find("already borrowed")!=std::string::npos,
                  "chunk provider retains exclusive lease");
            return batch(l,routes,out,e);
        },got,error),error);
        check(pa->run_chunk({1,2},pa->tokens(),reference,batch,expected,error),error);
        check(layer_chunk_fixture::exact(rows,reference),"reentrant chunk exact output");same(*a,*pa);
        const auto before=layer_chunk_fixture::snapshot(*a);const auto uncommitted=rows;
        check(!a->run_chunk({1,2},a->tokens(),rows,[](int64_t,const auto&,auto&,auto&){return false;},got,error) &&
              a->poisoned() && got.empty() && rows==uncommitted,"chunk provider failure poisons without publication");
        layer_chunk_fixture::same(layer_chunk_fixture::snapshot(*a),before,true,"chunk failed provider");
        check(a->reset(error) && pa->reset(error),error);step(*a,*pa,0);step(*b,*pb,8);
    }
    for(int failure=0;failure<3;++failure) {
        const auto before=layer_chunk_fixture::snapshot(*a);
        check(!a->step(6,a_fixture.input(6),[failure](int64_t,const PortableRoute&,std::vector<float>& rows,std::string&) -> bool {
            if(failure==0)return false;if(failure==1)throw std::runtime_error("provider failure");
            rows.assign(25600,std::numeric_limits<float>::quiet_NaN());return true;
        },route,error),"provider failure rejected");
        layer_chunk_fixture::same(layer_chunk_fixture::snapshot(*a),before,true,"failed provider publication");
        check(a->reset(error) && pa->reset(error),error);step(*a,*pa,0);step(*b,*pb,11+failure);
    }
    check(!a->prepare_chunk(9,error),"unsupported shape rejects before execution");
    step(*a,*pa,3);
    if(!gpu && options.execution.accelerator) {
        auto* backend=options.execution.accelerator;
        for(size_t fail_at:{size_t(1),size_t(3)}) {
            check(a->reset(error) && pa->reset(error),error);
            const auto before=layer_chunk_fixture::snapshot(*a);
            layer_chunk_fixture::PostFault::original=backend->iface.graph_compute;
            layer_chunk_fixture::PostFault::calls=0;layer_chunk_fixture::PostFault::fail_at=fail_at;
            const auto saved=backend->iface.graph_compute;
            backend->iface.graph_compute=layer_chunk_fixture::PostFault::compute;
            const bool ok=a->step(5,a_fixture.input(5),provider,route,error);
            backend->iface.graph_compute=saved;
            check(!ok && a->poisoned()==(options.policy==PortableGdnPolicy::device_production),"shared compute failure preserves policy-specific poison");
            layer_chunk_fixture::same(layer_chunk_fixture::snapshot(*a),before,true,"compute failure publication");
            step(*b,*pb,12);check(a->reset(error),error);step(*a,*pa,0);
        }
        {
            const auto original_sync=backend->iface.synchronize;
            backend->iface.synchronize=[](ggml_backend_t){throw std::runtime_error("shared reset sync failure");};
            const bool ok=a->reset(error);
            backend->iface.synchronize=original_sync;
            check(!ok && a->poisoned() && error=="shared reset sync failure","failed shared reset remains poisoned");
            check(a->reset(error) && pa->reset(error),error);step(*a,*pa,0);
        }
        if(options.causal_chunk) {
            check(a->reset(error) && pa->reset(error),error);
            const auto before=layer_chunk_fixture::snapshot(*a);
            const auto original=backend->iface.graph_compute;
            BackingCapture::original=original;backend->iface.graph_compute=BackingCapture::compute;
            decltype(ggml_backend_buffer_i::set_tensor) original_set=nullptr;
            std::vector<float> rows=a_fixture.input(1),second=a_fixture.input(2);
            rows.insert(rows.end(),second.begin(),second.end());const auto saved=rows;
            std::vector<PortableRoute> routes;
            const bool ok=a->run_chunk({1,2},0,rows,[&](int64_t l,const auto& rr,auto& out,auto& e) {
                check(batch(l,rr,out,e),e);
                check(BackingCapture::state!=nullptr,"shared causal state captured");
                original_set=BackingCapture::state->iface.set_tensor;
                BackingCapture::state->iface.set_tensor=[](ggml_backend_buffer_t,ggml_tensor*,const void*,size_t,size_t){throw std::runtime_error("shared final commit failure");};
                return true;
            },routes,error);
            backend->iface.graph_compute=original;
            if(original_set)BackingCapture::state->iface.set_tensor=original_set;
            check(!ok && a->poisoned() && error=="shared final commit failure" && rows==saved && routes.empty(),
                  "shared failed final commit retains uncommitted rows/receipts");
            layer_chunk_fixture::same(layer_chunk_fixture::snapshot(*a),before,true,"commit failure publication");
            check(a->reset(error),error);step(*a,*pa,0);step(*b,*pb,14);
        }
        auto* device=ggml_backend_get_device(backend);
        PlanFault::original=device->iface.supports_op;
        device->iface.supports_op=PlanFault::supports;
        auto rejected=PortableGdnLayer::create(0,a_fixture.views,options,error);
        device->iface.supports_op=PlanFault::original;
        check(!rejected && error.find("declines")!=std::string::npos,"planning failure after persistent state binding");
        step(*b,*pb,13);
        auto denied_backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
        auto invalid=options;invalid.execution.accelerator=denied_backend.get();
        check(!PortableGdnLayer::create(0,a_fixture.views,invalid,error),"different backend cannot borrow device owner");
    }
    if(options.execution.request_accelerator) {
        const auto m=owner->memory_usage(),ma=a->memory_usage(),mb=b->memory_usage();
        const auto cost=[](const PortableMemoryUsage& x){return x.device_state+x.device_weights+x.device_graph+x.device_extra_bound+x.device_token_temporary_bound+x.device_shared_conversion_bound;};
        const size_t cap=cost(m)+cost(ma)+cost(mb);
        auto ledger=[] {PortableModelReport r;r.device_admission={0};r.device_scheduler={0};r.device_state={0};r.device_weights={0};r.device_workspace_bound={0};r.device_token_temporary_bound={0};r.device_shared_conversion_bound={0};return r;};
        auto admitted=ledger();
        portable_detail::account_device_memory(admitted,0,cap,m);
        portable_detail::account_device_memory(admitted,0,cap,ma);
        portable_detail::account_device_memory(admitted,0,cap,mb);
        // Conversion/temporary peaks deduplicate independently of shared backing.
        const size_t exact=admitted.device_admission[0];auto refused=ledger();bool denied=false;
        try {for(const auto& x:{m,ma,mb})portable_detail::account_device_memory(refused,0,exact-1,x);}catch(const std::exception&){denied=true;}
        check(denied,"one-byte-over shared plus persistent admission rejects");
        auto exact_ledger=ledger();for(const auto& x:{m,ma,mb})portable_detail::account_device_memory(exact_ledger,0,exact,x);
        check(exact_ledger.device_admission[0]==exact && exact_ledger.device_state[0]==ma.device_state+mb.device_state,
              "exact cap charges shared once and both states separately");
    }
    auto other=PortableGdnScratch::create(options.execution,error);check(bool(other),error);
    auto isolated_options=options;isolated_options.scratch=other;
    auto isolated=make(0,a_fixture.views,isolated_options);
    check(isolated->scratch_owner()!=a->scratch_owner(),"separate sessions isolate scratch");
    auto incompatible=options;incompatible.execution.threads++;
    check(!PortableGdnLayer::create(0,a_fixture.views,incompatible,error),"incompatible backend options rejected");
    const size_t final_private=total(pa->memory_usage())+total(pb->memory_usage());
    const size_t final_shared=total(a->memory_usage())+total(b->memory_usage())+total(owner->memory_usage());
    check(final_private>final_shared,"actual backing reduction survives all exercised shapes");
    const auto retained=owner->graph_bytes();
    b.reset();pb.reset();
    step(*a,*pa,7);check(owner->graph_bytes()>=retained,"destruction preserves owner high water");
    std::printf("[gdn-shared] production=%d batch=%d causal=%d private_scheduler_bytes=%zu shared_scheduler_bytes=%zu removed_bytes=%zu final_private_scheduler_bytes=%zu final_shared_scheduler_bytes=%zu retained_graph_bytes=%zu host_scheduler=%zu device_scheduler=%zu state_per_layer=%zu exact=1\n",
        int(options.policy==PortableGdnPolicy::device_production),int(options.batch_input_projections),int(options.causal_chunk),private_buffers,shared_buffers,private_buffers-shared_buffers,final_private,final_shared,owner->graph_bytes(),owner->memory_usage().host_graph,owner->memory_usage().device_graph,a->memory_usage().device_state);
    // Layer-1 PLE keeps history and predecessors local under the same owner.
    a.reset();pa.reset();isolated.reset();device_state.clear();
    ple_fixture::Fixture pf;
    auto ple_options=options;ple_options.ple_table=ple_fixture::small_table(error);check(bool(ple_options.ple_table),error);
    auto ple=make(1,pf.views,ple_options);ple_options.scratch.reset();
    auto ple_private=make(1,pf.views,ple_options);
    for(int token:{1,2,ple_options.ple_table->config().eos,3})step(*ple,*ple_private,token);
    for(size_t T=2;T<=8;++T) {
        std::vector<float> x;for(size_t i=0;i<T;++i){auto row=pf.input(int(i+T));x.insert(x.end(),row.begin(),row.end());}
        auto y=x;std::vector<PortableRoute> xr,yr;std::vector<int32_t> tokens(T,5);
        check(ple->run_chunk(tokens,ple->tokens(),x,batch,xr,error),error);
        check(ple_private->run_chunk(tokens,ple_private->tokens(),y,batch,yr,error),error);
        check(layer_chunk_fixture::exact(x,y),"PLE chunks exact");same(*ple,*ple_private);
    }
    check(ple->reset(error) && ple_private->reset(error),error);step(*ple,*ple_private,5);
}
}
