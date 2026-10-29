#pragma once
#include "strata/core/portable_gdn.hpp"
#include "strata/core/portable_qsa.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <type_traits>

namespace layer_chunk_fixture {
using namespace strata::core;
inline void check(bool ok,const std::string& error) { if(!ok)throw std::runtime_error(error); }
template<class T> void append(std::vector<uint8_t>& out,const std::vector<T>& values) {
    if(!values.empty()) { const auto* p=reinterpret_cast<const uint8_t*>(values.data()); out.insert(out.end(),p,p+values.size()*sizeof(T)); }
}
struct SnapshotField { const char* name; size_t offset, elements; bool floating; };
struct Snapshot {
    uint64_t tokens; std::vector<float> last; std::vector<uint8_t> history;
    std::vector<SnapshotField> fields;
};
template<class T> void append_field(Snapshot& s,const char* name,const std::vector<T>& values) {
    static_assert(sizeof(T)==4,"snapshot fields are F32 or 32-bit integers");
    s.fields.push_back({name,s.history.size(),values.size(),std::is_floating_point_v<T>});
    append(s.history,values); // Keep the existing byte-comparison payload/order.
}
inline Snapshot snapshot(const PortableGdnLayer& layer,bool = false) {
    Snapshot s{layer.tokens(),layer.last_residual(),{}};
    const auto& previous=layer.token_history();
    append_field(s,"token_history",std::vector<int32_t>(previous.begin(),previous.end()));
    if(layer.policy()==PortableGdnPolicy::eager_diagnostic) {
        append_field(s,"recurrent_state",layer.recurrent_state());append_field(s,"convolution_history",layer.convolution_history());
        append_field(s,"ple_convolution_history",layer.ple_convolution_history());
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
// Failure-only diagnostics. Metrics describe a mismatch; they never select a
// tolerance or change the original byte-exact publication/state/replay gate.
inline void field_difference(std::ostream& out,const char* name,const uint8_t* a,const uint8_t* b,
                             size_t elements,bool floating) {
    size_t different=0,first=elements,nonfinite_a=0,nonfinite_b=0;
    uint32_t first_a=0,first_b=0;double maximum=0,square=0,norm=0;
    for(size_t i=0;i<elements;++i) {
        uint32_t x,y;std::memcpy(&x,a+4*i,4);std::memcpy(&y,b+4*i,4);
        if(x!=y) {++different;if(first==elements){first=i;first_a=x;first_b=y;}}
        if(floating) {
            float f,g;std::memcpy(&f,&x,4);std::memcpy(&g,&y,4);
            nonfinite_a+=!std::isfinite(f);nonfinite_b+=!std::isfinite(g);
            if(std::isfinite(f)&&std::isfinite(g)) {
                const double delta=double(f)-g;
                maximum=std::max(maximum,std::abs(delta));square+=delta*delta;norm+=double(g)*g;
            }
        }
    }
    if(!different)return;
    out<<" field="<<name<<" elements="<<elements<<" different="<<different<<" first_index="<<first
       <<" actual_bits=0x"<<std::hex<<first_a<<" reference_bits=0x"<<first_b<<std::dec;
    if(floating)out<<std::setprecision(9)<<" finite_pair_max_abs="<<maximum
        <<" finite_pair_relative_l2="<<std::sqrt(square/(norm+1e-30))
        <<" actual_nonfinite="<<nonfinite_a<<" reference_nonfinite="<<nonfinite_b;
    else {int32_t x,y;std::memcpy(&x,&first_a,4);std::memcpy(&y,&first_b,4);
        out<<" actual_value="<<x<<" reference_value="<<y;}
    out<<'\n';
}
inline std::string snapshot_difference(const Snapshot& a,const Snapshot& b,bool last,const std::string& context) {
    std::ostringstream out;
    out<<"[chunk-snapshot] "<<context<<" tokens_actual="<<a.tokens<<" tokens_reference="<<b.tokens
       <<" history_bytes_actual="<<a.history.size()<<" history_bytes_reference="<<b.history.size()
       <<" history_bytes_equal="<<(a.history==b.history)<<" named_fields_actual="<<a.fields.size()<<" named_fields_reference="<<b.fields.size()
       <<" last_compared="<<last<<" last_bytes_equal="<<exact(a.last,b.last)<<'\n';
    for(size_t i=0;i<a.fields.size();++i) {
        const auto& f=a.fields[i];
        if(i>=b.fields.size()||std::string(f.name)!=b.fields[i].name||f.elements!=b.fields[i].elements||
           f.offset!=b.fields[i].offset||f.floating!=b.fields[i].floating) {
            out<<" field="<<f.name<<" layout_mismatch actual_elements="<<f.elements<<'\n';continue;
        }
        if(f.offset>a.history.size()||f.elements>(a.history.size()-f.offset)/4||
           f.offset>b.history.size()||f.elements>(b.history.size()-f.offset)/4) {
            out<<" field="<<f.name<<" invalid_snapshot_range\n";continue;
        }
        if(f.elements)field_difference(out,f.name,a.history.data()+f.offset,b.history.data()+f.offset,f.elements,f.floating);
    }
    if(last) {
        if(a.last.size()!=b.last.size())out<<" field=last_output actual_elements="<<a.last.size()<<" reference_elements="<<b.last.size()<<'\n';
        else if(!a.last.empty())field_difference(out,"last_output",reinterpret_cast<const uint8_t*>(a.last.data()),
            reinterpret_cast<const uint8_t*>(b.last.data()),a.last.size(),true);
    }
    return out.str();
}
inline void same(const Snapshot& a,const Snapshot& b,bool last=true,const std::string& context="unspecified comparison") {
    if(!(a.tokens==b.tokens&&a.history==b.history&&(!last||exact(a.last,b.last)))) {
        const auto detail=snapshot_difference(a,b,last,context);
        std::fputs(detail.c_str(),stderr);std::fflush(stderr);
    }
    check(a.tokens==b.tokens&&a.history==b.history&&(!last||exact(a.last,b.last)),"chunk committed snapshot differs");
}
inline std::string comparison_context(const char* label,bool batched,bool causal,size_t width,int replay,size_t pos,size_t T,const char* phase) {
    std::ostringstream out;out<<"mode="<<label<<" batched_projections="<<batched<<" causal="<<causal
        <<" width="<<width<<" replay="<<replay<<" position="<<pos<<" T="<<T<<" phase="<<phase;
    return out.str();
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
    std::fprintf(stderr,"[chunk-snapshot] begin mode=%s batched_projections=%d causal=%d gpu=%d\n",label,int(batched_projections),int(causal),int(gpu));
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
                    ++calls;same(snapshot(chunk),before,true,comparison_context(label,batched_projections,causal,width,replay,pos,T,"provider-publication"));check(routes.size()==T,"layer batch provider T");
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
                same(snapshot(chunk,true),snapshot(serial,true),false,comparison_context(label,batched_projections,causal,width,replay,pos,T,"after-chunk-vs-serial"));
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
            if(replay==2)same(snapshot(chunk,true),first,true,comparison_context(label,batched_projections,causal,width,replay,9,0,"reset-A-replay"));
            PortableRoute a,b;const auto continuation=fixture.input(22);
            check(step(chunk,2,9,continuation,provider,a,error)&&step(serial,2,9,continuation,provider,b,error),error);
            same(snapshot(chunk,true),snapshot(serial,true),true,comparison_context(label,batched_projections,causal,width,replay,9,1,"serial-continuation"));check(a.ids==b.ids,"serial continuation after chunk");
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
            provider_called=true;same(snapshot(chunk),before,true,comparison_context(label,batched_projections,causal,3,failure,0,3,"failure-provider-publication"));
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
        same(snapshot(chunk),before,true,comparison_context(label,batched_projections,causal,3,failure,0,3,"failed-chunk-publication"));
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
