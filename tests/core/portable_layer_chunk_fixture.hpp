#pragma once
#include "strata/core/portable_gdn.hpp"
#include "strata/core/portable_qsa.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>

namespace layer_chunk_fixture {
using namespace strata::core;
inline void check(bool ok,const std::string& error) { if(!ok)throw std::runtime_error(error); }
template<class T> void append(std::vector<uint8_t>& out,const std::vector<T>& values) {
    if(!values.empty()) { const auto* p=reinterpret_cast<const uint8_t*>(values.data()); out.insert(out.end(),p,p+values.size()*sizeof(T)); }
}
struct Snapshot { uint64_t tokens; std::vector<float> last; std::vector<uint8_t> history; };
inline Snapshot snapshot(const PortableGdnLayer& layer,bool = false) {
    Snapshot s{layer.tokens(),layer.last_residual(),{}};
    const auto& previous=layer.token_history();
    append(s.history,std::vector<int32_t>(previous.begin(),previous.end()));
    if(layer.policy()==PortableGdnPolicy::eager_diagnostic) {
        append(s.history,layer.recurrent_state());append(s.history,layer.convolution_history());
        append(s.history,layer.ple_convolution_history());
    }
    return s;
}
inline Snapshot snapshot(const PortableQsaLayer& layer,bool device = false) {
    Snapshot s{layer.tokens(),layer.last_residual(),{}};
    PortableQsaState state;std::string error;
    if(layer.inspect_state(state,error)) {
        if(device) { append(s.history,state.keys);append(s.history,state.values);append(s.history,state.pooled);
            append(s.history,state.spare);append(s.history,state.tail);append(s.history,state.bias);append(s.history,state.mapping); }
    } else {
        append(s.history,layer.keys());append(s.history,layer.values());
        append(s.history,layer.pooled_keys());append(s.history,layer.indexer_tail());
    }
    return s;
}
inline bool exact(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
inline void same(const Snapshot& a,const Snapshot& b,bool last=true) {
    check(a.tokens==b.tokens&&a.history==b.history&&(!last||exact(a.last,b.last)),"chunk committed snapshot differs");
}
inline bool step(PortableGdnLayer& l,int32_t token,uint64_t,const std::vector<float>& x,
                 const PortableExpertRows& p,PortableRoute& r,std::string& e) { return l.step(token,x,p,r,e); }
inline bool step(PortableQsaLayer& l,int32_t,uint64_t pos,const std::vector<float>& x,
                 const PortableExpertRows& p,PortableRoute& r,std::string& e) { return l.step(pos,x,p,r,e); }
inline void near(const std::vector<float>& a,const std::vector<float>& b) {
    check(a.size()==b.size(),"chunk result size");
    for(size_t i=0;i<a.size();++i)check(std::isfinite(a[i])&&std::abs(a[i]-b[i])<=0.003f*(1+std::abs(b[i])),"chunk numerical gate failed");
}
// Existing borrowed CPU backend fault seam, not a runtime injection option.
struct PostFault {
    static inline decltype(ggml_backend_i::graph_compute) original=nullptr;
    static inline size_t calls=0,fail_at=0;
    static ggml_status compute(ggml_backend_t backend,ggml_cgraph* graph) {
        ++calls;
        if(calls==fail_at)return GGML_STATUS_FAILED;
        return original(backend,graph);
    }
};
inline uint64_t projection_calls(const PortableGdnLayer& l) { return l.input_projection_calls(); }
inline uint64_t projection_calls(const PortableQsaLayer&) { return 0; }
inline uint64_t projection_columns(const PortableGdnLayer& l) { return l.input_projection_columns(); }
inline uint64_t projection_columns(const PortableQsaLayer&) { return 0; }
inline std::array<uint64_t,3> causal_work(const PortableGdnLayer& l) {
    return {l.causal_chunk_calls(),l.causal_residual_handoffs(),l.causal_state_commits()};
}
inline std::array<uint64_t,3> causal_work(const PortableQsaLayer&) { return {}; }
template<class Layer,class Fixture> void exercise(Layer& chunk,Layer& serial,const Fixture& fixture,
    const PortableExpertRows& provider,ggml_backend_t backend,size_t graph_cap,size_t state_cap,const char* label,bool gpu=false,bool batched_projections=false,bool causal=false) {
    std::string error;double maximum=0,square=0,norm=0;size_t elements=0,completed_chunks=0,attention_tokens=0;
    const std::vector<int32_t> ids{5,248044,6,248056,0,8,9,7,11};
    const auto weight_bytes=chunk.weight_bytes();
    for(size_t width=1;width<=8;++width) {
        Snapshot first;
        for(int replay=0;replay<3;++replay) {
            const auto before_reset=chunk.chunk_work();
            check(chunk.reset(error)&&serial.reset(error),error);
            check(chunk.chunk_work().tokens==before_reset.tokens&&chunk.chunk_work().chunks==before_reset.chunks,"reset clears cumulative chunk receipts");
            for(size_t pos=0;pos<ids.size();) {
                const size_t T=std::min(width,ids.size()-pos);
                std::vector<float> input,expected;
                std::vector<PortableRoute> reference;
                for(size_t t=0;t<T;++t) {
                    auto row=fixture.input(int(pos+t)+(replay==1?11:0));
                    input.insert(input.end(),row.begin(),row.end());
                    PortableRoute route;
                    check(step(serial,ids[pos+t],pos+t,row,provider,route,error),error);
                    expected.insert(expected.end(),serial.last_residual().begin(),serial.last_residual().end());
                    reference.push_back(std::move(route));
                }
                const auto before=snapshot(chunk);const auto work=chunk.chunk_work();size_t calls=0;
                const PortableExpertRowsBatch dispatch=[&](int64_t layer,const auto& routes,auto& rows,auto& e) {
                    ++calls;same(snapshot(chunk),before);check(routes.size()==T,"layer batch provider T");
                    for(const auto& route:routes) {std::vector<float> token;check(provider(layer,route,token,e),e);
                        rows.insert(rows.end(),token.begin(),token.end());}
                    return true;
                };
                std::vector<PortableRoute> routes;
                const auto causal_before=causal_work(chunk);
                const auto projected_before=projection_calls(chunk), columns_before=projection_columns(chunk);
                check(chunk.run_chunk({ids.begin()+pos,ids.begin()+pos+T},pos,input,dispatch,routes,error),error);
                const bool projected=batched_projections&&T>1;
                check(projection_calls(chunk)==projected_before+size_t(projected) &&
                      projection_columns(chunk)==columns_before+(projected?T:0),"actual projection submissions and columns");
                const auto causal_after=causal_work(chunk);
                for(size_t i=0;i<3;++i) check(causal_after[i]==causal_before[i]+size_t(causal&&T>1),
                    "one actual causal submission, residual handoff and state commit per T>1 chunk");
                size_t projection_matrices=0, recurrence_nodes=0;
                for(const auto& p:chunk.placements()) if(p.phase=="gdn-causal-chunk") {
                    if(gpu&&!p.metadata_only) check(p.backend==ggml_backend_name(backend)&&p.fallback_reason.empty(),
                        "causal chunk operation not on selected Vulkan backend");
                    if(p.operation=="GATED_DELTA_NET") ++recurrence_nodes;
                }
                check(recurrence_nodes==(causal&&T>1?T:0),"one ordered recurrence node per causal token");
                for(const auto& p:chunk.placements()) if((p.phase=="gdn-input-projections"||p.phase=="gdn-causal-chunk")&&p.operation=="MUL_MAT" &&
                    (p.tensor=="attn_qkv.weight"||p.tensor=="attn_gate.weight"||p.tensor=="ssm_alpha.weight"||p.tensor=="ssm_beta.weight")) {
                    check(projected&&p.shape.size()>1&&p.shape[1]==int64_t(T),"projection matrix lacks actual T columns");
                    if(gpu)check(p.backend==ggml_backend_name(backend)&&p.fallback_reason.empty(),"input projection matrix not on Vulkan");
                    if(p.tensor=="attn_qkv.weight"||p.tensor=="attn_gate.weight"||p.tensor=="ssm_alpha.weight"||p.tensor=="ssm_beta.weight")
                        ++projection_matrices;
                }
                check(projection_matrices==(projected?4u:0u),"four actual GDN input projection matrices");
                check(calls==1&&routes.size()==T,"one provider per completed layer chunk");
                ++completed_chunks;attention_tokens+=T;
                const auto after=chunk.chunk_work();
                check(after.chunks==work.chunks+1&&after.tokens==work.tokens+T&&after.attention_calls==work.attention_calls+T&&
                      after.ffn_pre_calls==work.ffn_pre_calls+1&&after.provider_calls==work.provider_calls+1&&
                      after.ffn_post_calls==work.ffn_post_calls+1,"logical tokens versus graph/provider executions");
                same(snapshot(chunk,true),snapshot(serial,true),false);
                near(input,expected);
                for(size_t t=0;t<T;++t) {check(routes[t].ids==reference[t].ids,"chunk route IDs");
                    near(routes[t].activation,reference[t].activation);near(routes[t].weights,reference[t].weights);}
                for(size_t i=0;i<input.size();++i){const double delta=double(input[i])-expected[i];
                    maximum=std::max(maximum,std::abs(delta));square+=delta*delta;norm+=double(expected[i])*expected[i];++elements;}
                size_t matrices=0;
                for(const auto& p:chunk.placements()) if(p.phase=="pre-experts"&&p.operation=="MUL_MAT") {
                    check(p.shape.size()>1&&p.shape[1]==int64_t(T),"layer FFN graph lacks actual T columns");
                    if(gpu)check(p.backend==ggml_backend_name(backend)&&p.fallback_reason.empty(),"chunk FFN matrix not on Vulkan");
                    ++matrices;
                }
                check(matrices==8,"layer chunk FFN matrix coverage");
                check(chunk.weight_bytes()==weight_bytes&&chunk.memory_usage().host_weights+chunk.memory_usage().device_weights==weight_bytes,
                      "layer shared FFN weight ledger");
                check(chunk.graph_bytes()<=graph_cap&&chunk.state_bytes()<=state_cap,"aggregate layer chunk caps");
                pos+=T;
            }
            if(replay==0)first=snapshot(chunk,true);
            if(replay==2)same(snapshot(chunk,true),first);
            PortableRoute a,b;const auto continuation=fixture.input(22);
            check(step(chunk,2,9,continuation,provider,a,error)&&step(serial,2,9,continuation,provider,b,error),error);
            same(snapshot(chunk,true),snapshot(serial,true));check(a.ids==b.ids,"serial continuation after chunk");
        }
    }
    // Invalid middle input, provider rejection/throw, and an actual post graph
    // failure must poison both policies without completed publication.
    for(int failure=0;failure<(gpu?0:4);++failure) {
        check(chunk.reset(error),error);
        std::vector<float> input;
        for(int t=0;t<3;++t){const auto row=fixture.input(t);input.insert(input.end(),row.begin(),row.end());}
        if(failure==0)input[10240+7]=NAN;
        const auto saved=input;const auto before=snapshot(chunk);const auto work=chunk.chunk_work();
        const auto original=backend->iface.graph_compute;
        bool provider_called=false;
        PostFault::original=original;PostFault::calls=0;PostFault::fail_at=1;
        const PortableExpertRowsBatch dispatch=[&](int64_t layer,const auto& routes,auto& rows,auto& e) {
            provider_called=true;same(snapshot(chunk),before);
            if(failure==1){e="controlled chunk provider rejection";return false;}
            if(failure==2)throw std::runtime_error("controlled chunk provider throw");
            for(const auto& route:routes){std::vector<float> token;check(provider(layer,route,token,e),e);
                rows.insert(rows.end(),token.begin(),token.end());}
            backend->iface.graph_compute=PostFault::compute;return true;
        };
        std::vector<PortableRoute> routes;
        const bool ok=chunk.run_chunk({5,248044,6},0,input,dispatch,routes,error);
        backend->iface.graph_compute=original;
        check(!ok&&chunk.poisoned()&&routes.empty()&&exact(input,saved),"failed chunk exposes output or fails to poison");
        same(snapshot(chunk),before);
        check(chunk.chunk_work().tokens==work.tokens&&chunk.chunk_work().chunks==work.chunks&&
              chunk.chunk_work().attention_calls==work.attention_calls&&chunk.chunk_work().provider_calls==work.provider_calls,
              "failed chunk published completion counters");
        check(provider_called==(failure!=0),"middle attention failure order");
        if(failure==3)check(PostFault::calls==1,"actual FFN post graph failure reached");
        PortableRoute refused;
        check(!step(chunk,1,0,fixture.input(0),provider,refused,error),"poisoned layer accepted T1 retry");
        check(chunk.reset(error)&&!chunk.poisoned(),error);
        const auto row=fixture.input(0);auto next=row;
        check(chunk.run_chunk({5},0,next,[&](int64_t l,const auto& rs,auto& out,auto& e){return provider(l,rs[0],out,e);},routes,error),error);
    }
    if(batched_projections) check(projection_calls(chunk)==(gpu?48u:51u)&&projection_columns(chunk)==(gpu?180u:189u),
                                  "bounded projection submission totals including failed chunks");
    if(causal) {
        const auto work=causal_work(chunk);
        check(work[0]==(gpu?48u:51u)&&work[1]==work[0]&&work[2]==48u,"causal successful work versus deferred commits");
        std::printf("GDN causal chunk receipts: submissions=%llu residual_handoffs=%llu state_commits=%llu\n",
            (unsigned long long)work[0],(unsigned long long)work[1],(unsigned long long)work[2]);
    }
    if(batched_projections) std::printf("GDN input projection receipts: enabled=%d calls=%llu columns=%llu\n",int(batched_projections),
                (unsigned long long)projection_calls(chunk),(unsigned long long)projection_columns(chunk));
    std::printf("%s layer chunks T1-8/tails: max_abs=%.9g relative_l2=%.9g elements=%zu completed_chunks=%zu attention_tokens=%zu; causal state, routes, T-column execution, A/B/A passed; injected_failure_checks=%d\n",
                label,maximum,std::sqrt(square/(norm+1e-30)),elements,completed_chunks,attention_tokens,gpu?0:4);
}
} // namespace layer_chunk_fixture
