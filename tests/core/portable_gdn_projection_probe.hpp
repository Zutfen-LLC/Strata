#pragma once
// Synthetic diagnostic only: no state, FFN, model, tolerance or backend policy changes.
#include "portable_gdn_fixture.hpp"
#include "../../src/core/portable_graph.hpp"
#include "portable_layer_chunk_fixture.hpp"
#include <map>
namespace gdn_projection_probe {
using namespace strata::core;
using portable_detail::require;
using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
inline Context context(){return Context(ggml_init({4u<<20,nullptr,true}),ggml_free);}
inline bool run(ggml_backend_t selected,bool force_f32,bool strided_f32=false) {
    constexpr int64_t H=2560,HC=4;
    const char* control=strided_f32?"strided-F32":(force_f32?"F32-src-acc":"DEFAULT");
    const char* names[]={"norm","hc_down","hc_low","hc_up","gated","mixed","inject","qkv","gate","alpha","beta"};
    gdn_fixture::Fixture fixture;
    portable_detail::Backend cpu(ggml_backend_cpu_init(),ggml_backend_free);
    require(bool(cpu),"projection probe CPU backend");
    auto weights_ctx=context();require(bool(weights_ctx),"projection probe metadata");
    std::map<std::string,ggml_tensor*> weights;
    for(const auto& view:fixture.views) {
        const auto name=view.name.substr(6);
        if(name.rfind("hc_attn_",0)!=0&&name!="attn_qkv.weight"&&name!="attn_gate.weight"&&
           name!="ssm_alpha.weight"&&name!="ssm_beta.weight")continue;
        auto* t=ggml_new_tensor(weights_ctx.get(),ggml_type(view.type),int(view.shape.size()),view.shape.data());
        weights[name]=t;
    }
    const auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(weights_ctx.get(),ggml_backend_get_default_buffer_type(selected));
    require(bound<=128u<<20,"projection probe weight bound");
    Buffer buffer(ggml_backend_alloc_ctx_tensors(weights_ctx.get(),selected),ggml_backend_buffer_free);
    require(bool(buffer)&&ggml_backend_buffer_get_size(buffer.get())<=bound,"projection probe weights/bound");
    ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for(const auto& view:fixture.views)if(weights.count(view.name.substr(6)))
        ggml_backend_tensor_set(weights.at(view.name.substr(6)),view.data,0,view.bytes);
    struct Graph {
        Context ctx{nullptr,ggml_free};
        portable_detail::Scheduler scheduler{nullptr,ggml_backend_sched_free};
        ggml_cgraph* graph=nullptr;ggml_tensor* input=nullptr;
        std::array<ggml_tensor*,11> stages{};
    };
    size_t graph_bytes=0;
    auto build=[&](int64_t T) {
        Graph g;g.ctx=context();require(bool(g.ctx),"projection probe graph metadata");auto* c=g.ctx.get();
        g.input=ggml_new_tensor_3d(c,GGML_TYPE_F32,H,HC,T);ggml_set_input(g.input);
        auto matrix=[&](const char* name,ggml_tensor* x) {
            if(strided_f32&&T>1){
                // Public GGML layout operations retain the original F32 values.
                // At the pin this excludes contiguous-input MMVQ for this op;
                // the backend packs the strided F32 input before its F32 kernel.
                const int64_t K=x->ne[0];
                auto* padded=ggml_pad(c,x,1,0,0,0);
                x=ggml_view_2d(c,padded,K,T,size_t(K+1)*sizeof(float),0);
            }
            auto* y=ggml_mul_mat(c,weights.at(name),x);
            if(force_f32){
                require(ggml_prec_set_acc(y,GGML_PREC_F32),"projection probe accumulator control rejected");
                require(ggml_prec_set_src(y,GGML_PREC_F32,1),"projection probe source control rejected");
            }
            return y;
        };
        auto& s=g.stages;
        s[0]=ggml_mul(c,ggml_reshape_2d(c,ggml_rms_norm(c,g.input,1e-6f),H*HC,T),weights.at("hc_attn_norm.weight"));
        s[1]=matrix("hc_attn_down.weight",s[0]);
        s[2]=ggml_silu(c,ggml_scale(c,s[1],0.25f));
        s[3]=matrix("hc_attn_up.weight",s[2]);
        s[4]=ggml_reshape_3d(c,ggml_mul(c,s[0],ggml_sigmoid(c,s[3])),H,HC,T);
        auto* mixed=ggml_cont(c,ggml_view_2d(c,s[4],H,T,H*HC*sizeof(float),0));
        for(int64_t i=1;i<HC;++i)mixed=ggml_add(c,mixed,ggml_view_2d(c,s[4],H,T,H*HC*sizeof(float),i*H*sizeof(float)));
        s[5]=ggml_scale(c,mixed,0.25f);
        s[6]=matrix("hc_attn_inject.weight",s[0]);
        s[7]=matrix("attn_qkv.weight",s[5]);s[8]=matrix("attn_gate.weight",s[5]);
        s[9]=matrix("ssm_alpha.weight",s[5]);s[10]=matrix("ssm_beta.weight",s[5]);
        g.graph=ggml_new_graph(c);
        for(size_t i=0;i<s.size();++i){ggml_set_name(s[i],names[i]);ggml_set_output(s[i]);ggml_build_forward_expand(g.graph,s[i]);}
        PortableFfnOptions options;options.accelerator=selected;options.request_accelerator=true;options.allow_cpu_fallback=false;
        std::vector<PortablePlacement> placement;
        g.scheduler=portable_detail::schedule(g.graph,"projection-probe",options,cpu,placement,graph_bytes);
        for(const auto& p:placement)if(!p.metadata_only)
            require(p.backend==ggml_backend_name(selected)&&p.fallback_reason.empty(),"projection probe backend placement");
        return g;
    };
    auto serial=build(1),batch=build(2);
    std::vector<float> input=fixture.input(0);const auto second=fixture.input(1);input.insert(input.end(),second.begin(),second.end());
    auto execute=[&](Graph& g,const float* input) {
        ggml_backend_tensor_set(g.input,input,0,ggml_nbytes(g.input));
        require(ggml_backend_sched_graph_compute(g.scheduler.get(),g.graph)==GGML_STATUS_SUCCESS,"projection probe graph failed");
        std::array<std::vector<float>,11> result;
        for(size_t i=0;i<result.size();++i){result[i].resize(size_t(ggml_nelements(g.stages[i])));
            ggml_backend_tensor_get(g.stages[i],result[i].data(),0,result[i].size()*sizeof(float));
            for(float f:result[i])require(std::isfinite(f),"projection probe nonfinite result");}
        return result;
    };
    const auto got=execute(batch,input.data());
    std::array<std::vector<float>,11> reference;
    for(size_t token=0;token<2;++token){const auto row=execute(serial,input.data()+token*H*HC);
        for(size_t i=0;i<row.size();++i)reference[i].insert(reference[i].end(),row[i].begin(),row[i].end());}
    bool equal=true;const char* first="none";
    for(size_t i=0;i<got.size();++i){
        const bool same=layer_chunk_fixture::exact(got[i],reference[i]);
        if(!same&&equal){equal=false;first=names[i];}
        std::fprintf(stderr,"[gdn-projection-probe] precision=%s stage=%s exact=%d\n",control,names[i],int(same));
        std::ostringstream detail;
        layer_chunk_fixture::field_difference(detail,names[i],reinterpret_cast<const uint8_t*>(got[i].data()),
            reinterpret_cast<const uint8_t*>(reference[i].data()),got[i].size(),true);
        std::fputs(detail.str().c_str(),stderr);
    }
    std::fprintf(stderr,"[gdn-projection-probe] precision=%s first_divergent_stage=%s qualification_evaluated=0\n",control,first);
    return equal;
}
}
