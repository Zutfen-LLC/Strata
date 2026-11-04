#include "strata/core/portable_gdn.hpp"
#include "strata/core/layer0_observer.hpp"
#include "portable_expert_timer.hpp"
#include "portable_gdn_finite.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "portable_graph.hpp"
#include "portable_gdn_projection.hpp"
#include "native_hc_boundary.hpp"
#include "strata/artifact/gguf_reader.hpp"
#ifdef STRATA_PORTABLE_NATIVE_HC
#include "portable_native_hc.hpp"
#include "portable_native_gdn.hpp"
#include "portable_native_gdn_preparation.hpp"
#include "portable_native_gdn_input.hpp"
#include "portable_native_ple_key.hpp"
#include "portable_native_ple_gate.hpp"
#include "portable_native_ple_value.hpp"
#include "portable_native_ple_conv.hpp"
#include "portable_native_qsa_wo.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <atomic>

namespace strata::core {
namespace {
using portable_detail::require;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr int64_t H = 2560, HC = 4, R = 320, D = 128, HK = 16, HV = 48, INNER = D * HV, CHANNELS = 2 * D * HK + INNER;
constexpr size_t STATES = D * D * HV, HISTORY = 3 * CHANNELS, RESIDUAL = H * HC;
// Two committed/candidate copies plus graph-read residuals, reset scratch and
// optional trace vectors. Graph input/output buffers are priced separately.
constexpr size_t PLE_HISTORY = 9 * RESIDUAL;
constexpr size_t PLE_STATE_BYTES = (2 * PLE_HISTORY + 12 * RESIDUAL) * sizeof(float);
constexpr size_t HOST_STATE_BYTES =
    (2 * STATES + 2 * HISTORY + 8 * RESIDUAL + 8 * CHANNELS + 4 * INNER) * sizeof(float);
const std::map<std::string, std::vector<int64_t>> &shapes() {
    static const std::map<std::string, std::vector<int64_t>> s{{"hc_attn_norm.weight", {H * HC}},
                                                               {"hc_attn_down.weight", {H * HC, R}},
                                                               {"hc_attn_up.weight", {R, H * HC}},
                                                               {"hc_attn_inject.weight", {H * HC, HC}},
                                                               {"attn_qkv.weight", {H, CHANNELS}},
                                                               {"attn_gate.weight", {H, INNER}},
                                                               {"ssm_alpha.weight", {H, HV}},
                                                               {"ssm_beta.weight", {H, HV}},
                                                               {"ssm_a", {HV}},
                                                               {"ssm_dt.bias", {HV}},
                                                               {"ssm_conv1d.weight", {4, CHANNELS}},
                                                               {"ssm_norm.weight", {D}},
                                                               {"ssm_out.weight", {INNER, H}}};
    return s;
}
const std::map<std::string, std::vector<int64_t>> &ple_shapes() {
    static const std::map<std::string, std::vector<int64_t>> s{
        {"ple_key.weight", {H, H * HC}},    {"ple_value.weight", {H, H}},      {"ple_norm_key.weight", {H, HC}},
        {"ple_norm_query.weight", {H, HC}}, {"ple_norm_conv.weight", {H, HC}}, {"ple_conv1d.weight", {4, H * HC}}};
    return s;
}
Context context() {
    Context c(ggml_init({4u << 20, nullptr, true}), ggml_free);
    require(bool(c), "portable GDN: metadata allocation failed");
    return c;
}
void finite(const std::vector<float> &x) {
    if (portable_detail::gdn_all_finite(x))
        return;
    for (float f : x)
        if (!std::isfinite(f))
            require(false, "portable GDN: nonfinite input/result");
}
void read_into(ggml_tensor *t, std::vector<float> &out, PortableGdnWork *work = nullptr) {
    if (size_t(ggml_nelements(t)) != out.size())
        require(false, "portable GDN: candidate readback size mismatch");
    if (!work) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
        finite(out);
        return;
    }
    {
        portable_detail::ScopedExpertTimer<> timer(true, work->candidate_readback_ms, work->candidate_readback_calls);
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    }
    {
        portable_detail::ScopedExpertTimer<> timer(true, work->candidate_finite_ms, work->candidate_finite_calls);
        finite(out);
    }
}
std::vector<float> read(ggml_tensor *t, PortableGdnWork *work = nullptr) {
    std::vector<float> out(size_t(ggml_nelements(t)));
    read_into(t, out, work);
    return out;
}
void write(ggml_tensor *t, const std::vector<float> &x) {
    require(size_t(ggml_nelements(t)) == x.size(), "portable GDN: input size mismatch");
    ggml_backend_tensor_set(t, x.data(), 0, x.size() * sizeof(float));
}
} // namespace

struct PortableGdnScratch::Impl {
    static constexpr size_t metadata_bound = 3 * (size_t(GGML_DEFAULT_GRAPH_SIZE) * 30 * 2 * sizeof(ggml_tensor) + (20u << 20));
    PortableFfnOptions options;
    portable_detail::Backend cpu{ggml_backend_cpu_init(), ggml_backend_free};
    std::array<portable_detail::Scheduler, 3> schedulers{{
        {nullptr, ggml_backend_sched_free}, {nullptr, ggml_backend_sched_free}, {nullptr, ggml_backend_sched_free}}};
    std::array<size_t, 3> workspace{};
    std::array<ggml_backend_sched_t, 3> active{};
    std::atomic_flag borrowed = ATOMIC_FLAG_INIT;
    explicit Impl(PortableFfnOptions o) : options(o) {
        require(bool(cpu), "portable GDN: shared CPU unavailable");
        require(o.graph_budget > 0 && o.graph_budget <= SIZE_MAX - metadata_bound,
                "portable GDN: invalid/overflowing shared graph budget");
        ggml_backend_cpu_set_n_threads(cpu.get(), o.threads);
    }
};
PortableGdnScratch::PortableGdnScratch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableGdnScratch::~PortableGdnScratch() = default;
std::shared_ptr<PortableGdnScratch> PortableGdnScratch::create(PortableFfnOptions options, std::string &error) {
    error.clear();
    try { return std::shared_ptr<PortableGdnScratch>(new PortableGdnScratch(std::make_unique<Impl>(options))); }
    catch (const std::exception &e) { error=e.what(); return nullptr; }
}
size_t PortableGdnScratch::host_metadata_bound() { return Impl::metadata_bound; }
PortableMemoryUsage PortableGdnScratch::memory_usage() const {
    const auto& s=*impl_;
    PortableMemoryUsage m;
    size_t workspace=0;
    for(size_t i=0;i<3;++i) {
        workspace+=s.workspace[i];
        auto* sched=s.active[i]?s.active[i]:s.schedulers[i].get();
        if(!sched) continue;
        m.host_graph+=ggml_backend_sched_get_buffer_size(sched,s.cpu.get());
        if(s.options.request_accelerator && s.options.accelerator)
            m.device_graph+=ggml_backend_sched_get_buffer_size(sched,s.options.accelerator);
    }
    require(m.host_graph <= s.options.graph_budget && m.device_graph <= s.options.graph_budget-m.host_graph &&
            workspace <= s.options.graph_budget-m.host_graph-m.device_graph,
            "portable GDN: shared retained scratch exceeds cap");
    m.host_extra_bound=s.options.graph_budget-m.host_graph+Impl::metadata_bound;
    if(s.options.request_accelerator) m.device_extra_bound=s.options.graph_budget-m.device_graph;
    return m;
}
size_t PortableGdnScratch::graph_bytes() const {
    const auto m=memory_usage();
    size_t bytes=m.host_graph+m.device_graph;
    for(auto workspace:impl_->workspace) bytes+=workspace;
    return bytes;
}

struct PortableGdnLayer::Impl {
    PortableGdnOptions options;
    portable_detail::Backend cpu{nullptr, ggml_backend_free};
    Context weights_ctx{nullptr, ggml_free}, ctx{nullptr, ggml_free};
    Context state_ctx{nullptr, ggml_free};
    Buffer state_buffer{nullptr, ggml_backend_buffer_free};
    size_t device_state_size = 0;
    bool state_healthy = true;
    Buffer weights_buffer{nullptr, ggml_backend_buffer_free};
#ifdef STRATA_PORTABLE_NATIVE_HC
    std::unique_ptr<portable_detail::PortableNativeHc> native_hc;
#endif
#ifdef STRATA_PORTABLE_NATIVE_HC
    std::unique_ptr<portable_detail::PortableNativeGdn> native_gdn;
    std::unique_ptr<portable_detail::PortableNativeGdnPreparation> native_preparation;
    std::unique_ptr<portable_detail::PortableNativePleValue> native_ple_value;
    std::unique_ptr<portable_detail::PortableNativeGdnInput> native_input_projection;
    std::unique_ptr<portable_detail::PortableNativePleKey> native_ple_key;
    std::unique_ptr<portable_detail::PortableNativePleConv> native_ple_conv;
    std::unique_ptr<portable_detail::PortableNativePleGate> native_ple_gate;
    std::unique_ptr<portable_detail::PortableNativeQsaWo> native_output_projection;
    std::unique_ptr<portable_detail::PortableNativeQsaWo::Candidate> native_output_candidate;
    std::vector<float> native_output_input;
#endif
    ggml_cgraph* native_prefix_graph=nullptr;
    ggml_cgraph* ple_prefix_graph=nullptr;
    std::array<ggml_tensor*,4> ple_conv_copies{},ple_conv_sources{};
    ggml_cgraph* ple_gate_prefix_graph=nullptr;
    ggml_tensor* ple_gate_broadcast=nullptr;
    ggml_tensor *ple_gate_key_copy=nullptr,*ple_gate_query_copy=nullptr;
    ggml_tensor* preparation_mixed=nullptr;
    size_t preparation_reserved=0;
    size_t preparation_credit()const {
#ifdef STRATA_PORTABLE_NATIVE_HC
        return preparation_reserved-(native_preparation?native_preparation->bytes():0);
#else
        return 0;
#endif
    }
    size_t native_gdn_bytes()const {
#ifdef STRATA_PORTABLE_NATIVE_HC
        return (native_ple_conv?native_ple_conv->bytes():0)+(native_ple_value?native_ple_value->bytes():0)+(native_gdn?native_gdn->bytes():0)+(native_preparation?native_preparation->bytes():0)+(native_input_projection?native_input_projection->bytes():0)+(native_output_projection?native_output_projection->bytes():0)+(native_ple_key?native_ple_key->bytes():0)+(native_ple_gate?native_ple_gate->bytes():0);
#else
        return 0;
#endif
    }
    size_t native_hc_bytes()const {
#ifdef STRATA_PORTABLE_NATIVE_HC
        return native_hc?native_hc->bytes():0;
#else
        return 0;
#endif
    }
    portable_detail::Scheduler scheduler{nullptr, ggml_backend_sched_free};
    std::unique_ptr<PortableFfn> ffn, chunk_ffn;
    PortableHcPendingWrite pending;
    bool chunk_active = false;
    size_t chunk_host_bound = 0;
    PortableLayerChunkWork chunk_work;
    // Graphs borrow the original immutable weight tensors; no second weight bank.
    std::map<std::string, ggml_tensor*> weights;
    Context projection_ctx{nullptr, ggml_free};
    portable_detail::Scheduler projection_scheduler{nullptr, ggml_backend_sched_free};
    portable_detail::Scheduler tail_scheduler{nullptr, ggml_backend_sched_free};
    ggml_cgraph *projection_graph = nullptr, *tail_graph = nullptr;
    ggml_tensor *projection_input = nullptr;
    std::array<ggml_tensor*,5> projection_outputs{}, projection_inputs{};
    ggml_tensor *produced_block=nullptr,*produced_inject=nullptr; // graph-owned until completed tuple readback
    using AttentionOutputs = std::array<ggml_tensor*,11>;
    AttentionOutputs tail_outputs{};
    size_t projection_tokens = 0, projection_bytes = 0;
    uint64_t projection_calls = 0, projection_columns = 0;
    uint64_t causal_calls = 0, residual_handoffs = 0, causal_commits = 0;
    ggml_tensor* causal_residuals=nullptr;
    AttentionOutputs causal_outputs{};
    bool causal() const { return options.causal_chunk && !options.ple_table; }
    bool batch_projections() const { return options.batch_input_projections && !options.ple_table; }
    AttentionOutputs outputs() const {
        return {attn_residual,state_output,history_output,qkv,conv,q,k,v,g,beta,attention};
    }
    void outputs(const AttentionOutputs& a) {
        attn_residual=a[0];state_output=a[1];history_output=a[2];qkv=a[3];conv=a[4];
        q=a[5];k=a[6];v=a[7];g=a[8];beta=a[9];attention=a[10];
    }
    AttentionOutputs build_tail(ggml_context* c, ggml_tensor* hidden, ggml_tensor* inject,
        ggml_tensor* qkv_input, ggml_tensor* z, ggml_tensor* alpha, ggml_tensor* beta_input,
        ggml_tensor* state, ggml_tensor* history) {
        const auto& w=weights;
        ggml_tensor* gated_output=nullptr;
#ifdef STRATA_PORTABLE_NATIVE_HC
        if(native_preparation) {
            qkv=qkv_input;conv=native_preparation->silu();
            q=ggml_view_4d(c,native_gdn->t[1],D,HK,1,1,D*sizeof(float),CHANNELS*sizeof(float),CHANNELS*sizeof(float),0);
            k=ggml_view_4d(c,native_gdn->t[1],D,HK,1,1,D*sizeof(float),CHANNELS*sizeof(float),CHANNELS*sizeof(float),D*HK*sizeof(float));
            v=ggml_view_4d(c,native_gdn->t[1],D,HV,1,1,D*sizeof(float),CHANNELS*sizeof(float),CHANNELS*sizeof(float),2*D*HK*sizeof(float));
            g=ggml_reshape_4d(c,native_gdn->t[2],1,HV,1,1);beta=ggml_reshape_4d(c,native_gdn->t[3],1,HV,1,1);
            native_prefix_graph=ggml_new_graph(c);
            // Actual borrowed inputs remain outputs through prefix completion.
            for(auto*t:{qkv,preparation_mixed}){ggml_set_output(t);ggml_build_forward_expand(native_prefix_graph,t);}
            if(!options.native_input_projection_t1)ggml_build_forward_expand(native_prefix_graph,ggml_cpy(c,z,native_gdn->t[4]));
            history_output=native_gdn->history;state_output=native_gdn->t[6];
            attention=ggml_reshape_2d(c,native_gdn->t[9],D,HV);
            gated_output=ggml_reshape_1d(c,native_gdn->t[7],INNER);
        } else
#endif
        {
        qkv=qkv_input;
        beta=ggml_reshape_4d(c,ggml_sigmoid(c,beta_input),1,HV,1,1);
        g=ggml_reshape_4d(c,ggml_mul(c,ggml_softplus(c,ggml_add(c,alpha,w.at("ssm_dt.bias"))),w.at("ssm_a")),1,HV,1,1);
        auto *window = ggml_concat(c, history, ggml_reshape_3d(c, qkv, 1, CHANNELS, 1), 0);
        history_output = ggml_cont(c, ggml_view_2d(c, window, 3, CHANNELS, 4 * sizeof(float), sizeof(float)));
        conv = ggml_silu(c, ggml_ssm_conv(c, window, w.at("ssm_conv1d.weight")));
        auto *qraw = ggml_view_4d(c, conv, D, HK, 1, 1, D * sizeof(float), CHANNELS * sizeof(float),
                                  CHANNELS * sizeof(float), 0);
        auto *kraw = ggml_view_4d(c, conv, D, HK, 1, 1, D * sizeof(float), CHANNELS * sizeof(float),
                                  CHANNELS * sizeof(float), D * HK * sizeof(float));
        v = ggml_view_4d(c, conv, D, HV, 1, 1, D * sizeof(float), CHANNELS * sizeof(float), CHANNELS * sizeof(float),
                         2 * D * HK * sizeof(float));
        q = ggml_scale(c, ggml_rms_norm(c, qraw, 1e-6f / float(D)), 1.0f / std::sqrt(float(D)));
        k = ggml_scale(c, ggml_rms_norm(c, kraw, 1e-6f / float(D)), 1.0f / std::sqrt(float(D)));
#ifdef STRATA_PORTABLE_NATIVE_HC
        if(native_gdn) {
            native_prefix_graph=ggml_new_graph(c);
            auto* packed=ggml_concat(c,ggml_concat(c,ggml_reshape_1d(c,q,D*HK),ggml_reshape_1d(c,k,D*HK),0),ggml_reshape_1d(c,v,INNER),0);
            for(auto pair:std::array<std::pair<ggml_tensor*,ggml_tensor*>,5>{{{packed,native_gdn->t[1]},{ggml_reshape_1d(c,g,HV),native_gdn->t[2]},{ggml_reshape_1d(c,beta,HV),native_gdn->t[3]},{z,native_gdn->t[4]},{history_output,native_gdn->history}}})
                ggml_build_forward_expand(native_prefix_graph,ggml_cpy(c,pair.first,pair.second));
            // Keep actual prefix trace operands alive through completion; never
            // let the suffix scheduler allocate/recompute this graph's outputs.
            for(auto*t:{qkv,conv,q,k,v,g,beta}){ggml_set_output(t);ggml_build_forward_expand(native_prefix_graph,t);}
            history_output=native_gdn->history;state_output=native_gdn->t[6];
            attention=ggml_reshape_2d(c,native_gdn->t[9],D,HV);
            gated_output=ggml_reshape_1d(c,native_gdn->t[7],INNER);
        } else
#endif
        {
            auto *result = ggml_gated_delta_net(c, q, k, v, g, beta, state, 1);
            attention = ggml_view_2d(c, result, D, HV, D * sizeof(float), 0);
            state_output = ggml_view_1d(c, result, STATES, INNER * sizeof(float));
            auto *normalized = ggml_mul(c, ggml_rms_norm(c, attention, 1e-6f), w.at("ssm_norm.weight"));
            auto *outgate = ggml_reshape_2d(c, ggml_sigmoid(c, z), D, HV);
            gated_output=ggml_reshape_1d(c, ggml_mul(c, normalized, outgate), INNER);
        }
        }
        ggml_tensor* projected=nullptr;
#ifdef STRATA_PORTABLE_NATIVE_HC
        if(native_output_projection)projected=native_output_projection->output();else
#endif
        projected=ggml_mul_mat(c,w.at("ssm_out.weight"),gated_output);
        if(options.native_hc_ffn_read) {
            // No eager multiply/add: actual half-1 read performs the native FMA.
            produced_block=projected;produced_inject=inject;
            attn_residual=hidden;
            return outputs();
        }
        auto *scatter =
            ggml_reshape_3d(c, ggml_scale(c, ggml_sigmoid(c, ggml_scale(c, inject, 0.25f)), 2.0f), 1, HC, 1);
        auto *repeat = ggml_repeat_4d(c, ggml_reshape_3d(c, projected, H, 1, 1), H, HC, 1, 1);
        attn_residual = ggml_add(c, hidden, ggml_mul(c, repeat, scatter));
        return outputs();
    }
    size_t chunk_graphs() const { return (chunk_ffn ? chunk_ffn->graph_bytes() : 0) + projection_bytes; }
    void prepare_projections(size_t T) {
        if (!batch_projections() || projection_tokens == T) return;
        // Drop the old shape before allocating its replacement. The tail and T1
        // schedulers keep their own buffers and are included in graphs_size.
        if(projection_scheduler) ggml_backend_sched_reset(projection_scheduler.get());
        if(!shared()) projection_scheduler.reset();
        projection_ctx.reset(); projection_bytes=0; projection_tokens=0;
        projection_graph=nullptr; projection_input=nullptr; projection_outputs={};
        causal_residuals=nullptr; causal_outputs={};
        placement.erase(std::remove_if(placement.begin(),placement.end(),[](const auto& p) {
            return p.phase=="gdn-input-projections" || p.phase=="gdn-causal-chunk";
        }),placement.end());
        if(T==1) return;
        projection_ctx=context(); auto* c=projection_ctx.get();
        projection_input=ggml_new_tensor_3d(c,GGML_TYPE_F32,H,HC,T); ggml_set_input(projection_input);
        auto* norm=ggml_mul(c,ggml_reshape_2d(c,ggml_rms_norm(c,projection_input,1e-6f),H*HC,T),weights.at("hc_attn_norm.weight"));
        // Share three padded F32 banks across all seven quantized input
        // projections. Views exclude the pinned backend's MMVQ conversion;
        // padding and backend packing do not change the input values.
        auto matrix=[&](const char* name,ggml_tensor* columns){
            auto* t=ggml_mul_mat(c,weights.at(name),columns);ggml_set_name(t,name);return t;
        };
        auto* norm_columns=portable_detail::gdn_f32_projection_columns(c,norm);
        auto* low=ggml_silu(c,ggml_scale(c,matrix("hc_attn_down.weight",norm_columns),0.25f));
        auto* low_columns=portable_detail::gdn_f32_projection_columns(c,low);
        auto* gates=ggml_sigmoid(c,matrix("hc_attn_up.weight",low_columns));
        auto* gated=ggml_reshape_3d(c,ggml_mul(c,norm,gates),H,HC,T);
        auto* mixed=ggml_cont(c,ggml_view_2d(c,gated,H,T,H*HC*sizeof(float),0));
        for(int64_t stream=1;stream<HC;++stream)
            mixed=ggml_add(c,mixed,ggml_view_2d(c,gated,H,T,H*HC*sizeof(float),stream*H*sizeof(float)));
        mixed=ggml_scale(c,mixed,0.25f);
        auto* mixed_columns=portable_detail::gdn_f32_projection_columns(c,mixed);
        projection_outputs[0]=matrix("hc_attn_inject.weight",norm_columns);
        const char* names[]={"attn_qkv.weight","attn_gate.weight","ssm_alpha.weight","ssm_beta.weight"};
        projection_graph=ggml_new_graph(c);
        for(size_t i=0;i<4;++i) {
            projection_outputs[i+1]=matrix(names[i],mixed_columns);
        }
        for(auto* t:projection_outputs) {ggml_set_output(t);ggml_build_forward_expand(projection_graph,t);}
        if(causal()) {
            // Each next token consumes its predecessor's candidate tensors.
            // No committed state is overwritten while this graph or FFN runs.
            const auto saved=outputs();
            auto* state=state_input; auto* history=history_input;
            causal_residuals=nullptr;
            for(size_t t=0;t<T;++t) {
                std::array<ggml_tensor*,5> column{};
                for(size_t i=0;i<5;++i) {
                    const int64_t n=projection_outputs[i]->ne[0];
                    column[i]=ggml_view_1d(c,projection_outputs[i],n,t*n*sizeof(float));
                }
                auto* hidden=ggml_view_3d(c,projection_input,H,HC,1,H*sizeof(float),RESIDUAL*sizeof(float),t*RESIDUAL*sizeof(float));
                causal_outputs=build_tail(c,hidden,column[0],column[1],column[2],column[3],column[4],state,history);
                state=ggml_reshape_4d(c,causal_outputs[1],D,D,HV,1);
                history=ggml_reshape_3d(c,causal_outputs[2],3,CHANNELS,1);
                causal_residuals=causal_residuals?ggml_concat(c,causal_residuals,causal_outputs[0],2):causal_outputs[0];
            }
            causal_outputs[1]=state; causal_outputs[2]=history;
            for(auto* t:{causal_residuals,state,history}) {ggml_set_output(t);ggml_build_forward_expand(projection_graph,t);}
            outputs(saved);
        }
        auto opts=options.execution;
        require(graphs_size+chunk_graphs()<opts.graph_budget,"portable GDN: no projection graph budget");
        // schedule_graph consumes the combined layer budget, including fixed graphs.
        projection_bytes=graphs_size+(chunk_ffn?chunk_ffn->graph_bytes():0);
        schedule_graph(projection_graph,causal()?"gdn-causal-chunk":"gdn-input-projections",1,projection_bytes,projection_scheduler,causal());
        projection_bytes-=graphs_size+(chunk_ffn?chunk_ffn->graph_bytes():0);
        projection_tokens=T;
    }

    void prepare(size_t T,std::string& error) {
        auto& s=*this;
        require(!options.native_hc_norm_inject&&!options.native_recurrence_t1,"native serial components: chunk API unsupported");
        require(T>=1&&T<=8,"portable layer: chunk size outside [1,8]");
        const size_t extra = (s.production() ? 0 : (STATES+HISTORY+(s.options.ple_table?PLE_HISTORY:0))*sizeof(float)) +
                             (3*T*RESIDUAL+RESIDUAL)*sizeof(float);
        require(s.host_state_bytes() <= s.options.state_budget && extra <= s.options.state_budget-s.host_state_bytes(),
                "portable GDN: chunk working state budget exceeded");
        s.chunk_host_bound = extra + (s.batch_projections() && T>1 ? T*(HC+CHANNELS+INNER+2*HV)*sizeof(float) : 0);
        require(s.chunk_host_bound <= s.options.state_budget-s.host_state_bytes(),
                "portable GDN: projection host scratch budget exceeded");
        if (T == 1 || (s.chunk_ffn && s.chunk_ffn->tokens() != int64_t(T))) s.chunk_ffn.reset();
        if (T > 1 && !s.chunk_ffn) {
            // A previous borrower can retain a larger projection shape than
            // this layer's fresh metadata. Admit that backing before allocating
            // another private FFN shape, rather than discovering it afterwards.
            const size_t fixed=shared()?s.ffn->graph_bytes()+options.scratch->graph_bytes():s.graphs_size+s.projection_bytes;
            require(fixed < s.options.execution.graph_budget,"portable GDN: no chunk graph budget");
            s.chunk_ffn = s.ffn->create_chunk_execution(int64_t(T),error,s.options.execution.graph_budget-fixed);
            require(bool(s.chunk_ffn),error);
        }
        s.prepare_projections(T);
    }
    struct CursorView {
        const std::vector<float>& state;
        const std::vector<float>& history;
        const std::vector<float>& ple_history;
        const std::array<int32_t,2>& previous;
    };
    struct AttentionResult {
        std::vector<float> residual;
        std::vector<float> block, inject;
        std::array<int32_t,2> previous;
        PortableGdnTrace trace;
    };
    AttentionResult attend(int32_t token, const std::vector<float>& residual,
                           CursorView cursor, bool want_trace, PortableGdnWork* work, bool projected = false,
                           const std::vector<float>* previous_output=nullptr,const std::vector<float>* previous_injection=nullptr) {
        auto& s = *this;
        const auto saved_outputs=projected?s.outputs():AttentionOutputs{};
        struct Restore { Impl& s; AttentionOutputs saved; bool enabled; ~Restore(){if(enabled)s.outputs(saved);} } restore{s,saved_outputs,projected};
        if(projected) s.outputs(s.tail_outputs);
        std::string error;
        finite(residual);
        const bool trace = want_trace;
        PortableGdnTrace next_trace;
        auto next_previous = cursor.previous;
        PortableGdnWork disabled;
        auto &timing = work ? *work : disabled;
        std::vector<float> next_residual,next_block,next_inject;
        std::array<uint32_t,16> ple_rows{};
        {
            portable_detail::ScopedExpertTimer<> attention_timer(work != nullptr,
                timing.attention_exchange_ms, timing.attention_exchange_calls);
            if (s.options.ple_table) {
                std::vector<float> embedding;
                require(s.options.ple_table->gather(token, cursor.previous, ple_rows, embedding, error), error);
#ifdef STRATA_PORTABLE_NATIVE_HC
                if (s.native_ple_key) s.native_ple_key->run(embedding);
                if (s.native_ple_value) s.native_ple_value->run(embedding);
                else
#endif
                write(s.ple_emb, embedding);
                if (!s.state_buffer)
                    write(s.ple_hist, cursor.ple_history);
                next_previous = {{cursor.previous[1], token}};
                if (trace) {
                    next_trace.ple_rows = ple_rows;
                    next_trace.ple_embedding = std::move(embedding);
                }
            }
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(s.native_hc && s.native_hc->pending_input) {
                require(!projected && previous_output && previous_injection,"native HC pending: explicit T1 tuple required");
                write(s.native_hc->pending_input,residual);write(s.native_hc->previous_output,*previous_output);write(s.native_hc->previous_inject,*previous_injection);
                s.native_hc->diagnostic_position(s.count);s.native_hc->run_pending(); // candidate R-prime is BOTH graph hidden and HC input
            } else
#endif
            {
                write(s.input, residual);
#ifdef STRATA_PORTABLE_NATIVE_HC
                if(s.native_hc) {require(!projected,"native HC serial: projected execution unsupported");s.native_hc->diagnostic_position(s.count);s.native_hc->run();}
#endif
            }
            if (!s.state_buffer) {
                write(s.state_input, cursor.state);
                write(s.history_input, cursor.history);
            }
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(s.native_ple_gate) {
                s.native_ple_gate->begin();
                require(!projected && s.ple_gate_prefix_graph && s.projection_scheduler,"native PLE gate: prefix unavailable");
                require(ggml_backend_sched_graph_compute(s.projection_scheduler.get(),s.ple_gate_prefix_graph)==GGML_STATUS_SUCCESS,"native PLE gate: normalization prefix failed");
                s.native_ple_gate->run(s.ple_gate_key_copy,s.ple_gate_query_copy,s.ple_raw_key,s.input);
            }
#endif
            ggml_status status;
            {
                portable_detail::ScopedExpertTimer<> compute_timer(work != nullptr,
                    timing.attention_compute_ms, timing.attention_compute_calls);
#ifdef STRATA_PORTABLE_NATIVE_HC
                if(s.native_ple_conv) {
                    auto* prefix=s.native_ple_gate?s.tail_scheduler.get():s.projection_scheduler.get();
                    require(ggml_backend_sched_graph_compute(prefix,s.ple_prefix_graph)==GGML_STATUS_SUCCESS,"native PLE conv: prefix graph failed");
                    s.native_ple_conv->run(s.ple_conv_copies.data(),s.ple_conv_sources.data());
                }
                if(s.native_gdn) {
                    portable_detail::PortableNativeGdnInput::Candidate candidate;
                    if(s.native_input_projection)candidate=s.native_input_projection->run(s.native_hc->mixed);
                    require(ggml_backend_sched_graph_compute(s.projection_scheduler.get(),s.native_prefix_graph)==GGML_STATUS_SUCCESS,"native GDN: prefix graph failed");
                    if(s.native_preparation)s.native_preparation->run(s.history_input,s.qkv,s.preparation_mixed,s.weights.at("ssm_conv1d.weight"),s.count);
                    s.native_gdn->run(s.state_input,s.weights.at("ssm_norm.weight"),s.count);
                    if(s.native_input_projection)s.native_input_projection->consumed(s.count,s.native_hc->mixed,{s.qkv,s.native_gdn->t[4]},candidate);
                    if(s.native_output_projection){s.native_output_input=read(s.native_gdn->t[7]);*s.native_output_candidate=s.native_output_projection->run(s.native_gdn->t[7],s.native_output_input);}
                }
#endif
                status = ggml_backend_sched_graph_compute(projected?s.tail_scheduler.get():s.scheduler.get(), projected?s.tail_graph:s.graph);
            }
            require(status == GGML_STATUS_SUCCESS, "portable GDN: attention graph failed");
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(s.native_ple_conv)s.native_ple_conv->verify_and_record(s.ple_result,s.ple_hist_out,s.count,token,cursor.previous);
            if(s.native_ple_gate)s.native_ple_gate->verify_and_record(s.ple_gate,s.ple_value,s.ple_gate_broadcast,s.count,token,cursor.previous,ple_rows);
            if(s.native_ple_key)s.native_ple_key->verify_and_record(s.ple_raw_key,s.count,token,cursor.previous,ple_rows);
            if(s.native_ple_value)s.native_ple_value->verify_and_record(s.ple_value,s.count,token,cursor.previous,ple_rows);
            if(s.native_hc)s.native_hc->record_first();
            if(trace && s.native_hc) {
                next_trace.native_hc_norm_inject=true;
                if(s.native_hc->pending_input) {
                    next_trace.native_hc_pending=true;next_trace.hc_pending_input=read(s.native_hc->pending_input);
                    next_trace.hc_previous_output=read(s.native_hc->previous_output);next_trace.hc_previous_inject=read(s.native_hc->previous_inject);
                    next_trace.hc_pending_gate=read(s.native_hc->pending_gate);next_trace.hc_pending_exp=read(s.native_hc->pending_exp);
                }
                next_trace.hc_residual=read(s.native_hc->residual);
                next_trace.hc_normalized=read(s.native_hc->normalized);
                next_trace.hc_rs=read(s.native_hc->rs);
                next_trace.hc_inject=read(s.native_hc->inject);
                if(s.native_hc->mixed) {
                    next_trace.native_hc_mix=true;next_trace.hc_low_pre=read(s.native_hc->low_pre);next_trace.hc_low=read(s.native_hc->low);
                    next_trace.hc_low_exp=read(s.native_hc->low_exp);next_trace.hc_up_pre=read(s.native_hc->up_pre);
                    next_trace.hc_gates=read(s.native_hc->gates);next_trace.hc_up_exp=read(s.native_hc->up_exp);next_trace.hc_mixed=read(s.native_hc->mixed);
                }
            }
#endif
            if (!s.production()) {
                read_into(s.state_output, s.candidate_state, work);
                read_into(s.history_output, s.candidate_history, work);
            }
            next_residual = read(s.attn_residual, work);
            if(s.options.native_hc_ffn_read) {
                // The native output owner survives the FFN call; the legacy
                // scheduler result must be copied before its arena is reused.
                if(!s.options.native_output_projection_t1)next_block=read(s.produced_block,work);
                next_inject=read(s.produced_inject,work);
            }
            auto* observer=s.options.execution.layer0_observer;
            if(observer && observer->active()) {
                const auto mixed=read(s.observed_mixed),inject=read(s.observed_inject);
                observer->put(Layer0Observer::attention_mixed,mixed.data(),mixed.size());
                observer->put(Layer0Observer::attention_inject,inject.data(),inject.size());
            }
            if (s.options.ple_table && !s.production())
                read_into(s.ple_hist_out, s.candidate_ple_history, work);
            if (trace) {
                next_trace.qkv = read(s.qkv);
                next_trace.convolution = read(s.conv);
                next_trace.query = read(s.q);
                next_trace.key = read(s.k);
                next_trace.value = read(s.v);
                next_trace.decay = read(s.g);
                next_trace.beta = read(s.beta);
                next_trace.attention = read(s.attention);
                if (s.options.ple_table) {
                    next_trace.ple_key = read(s.ple_key);
                    next_trace.ple_value = read(s.ple_value);
                    next_trace.ple_gate = read(s.ple_gate);
                    next_trace.ple_normalized = read(s.ple_norm);
                    next_trace.ple_convolution = read(s.ple_conv);
                    next_trace.ple_residual = read(s.ple_result);
                }
            }
        }
        return {std::move(next_residual),std::move(next_block),std::move(next_inject),next_previous,std::move(next_trace)};
    }
    void commit_device(PortableGdnWork* work, bool projected = false, bool causal = false) {
        auto& s = *this;
        PortableGdnWork disabled;
        auto& timing = work ? *work : disabled;
        if (s.state_buffer) {
            // Retain separate synchronous copies on this backend API. Diagnostics
            // validate full candidates; production only reads the FFN residual.
            // T1 and the causal graph invoke this after FFN. The legacy chunk
            // advances its working cursor after each serial attention step.
            // A recoverable copy exception poisons device state until reset;
            // public mirrors still describe the last successful token.
            portable_detail::ScopedExpertTimer<> commit_timer(work != nullptr,
                timing.state_commit_ms, timing.state_commit_calls);
            s.state_healthy = false;
            ggml_backend_tensor_copy(causal?s.causal_outputs[1]:(projected?s.tail_outputs[1]:s.state_output), s.state_input);
            ggml_backend_tensor_copy(causal?s.causal_outputs[2]:(projected?s.tail_outputs[2]:s.history_output), s.history_input);
            if (s.options.ple_table)
                ggml_backend_tensor_copy(s.ple_hist_out, s.ple_hist);
            ggml_backend_synchronize(s.options.execution.accelerator);
            s.state_healthy = true;
            if(causal) ++s.causal_commits;
        }
    }

    ggml_cgraph *graph = nullptr;
    ggml_tensor *observed_mixed = nullptr, *observed_inject = nullptr;
    ggml_tensor *input = nullptr, *state_input = nullptr, *history_input = nullptr, *attn_residual = nullptr;
    ggml_tensor *state_output = nullptr, *history_output = nullptr, *qkv = nullptr, *conv = nullptr, *q = nullptr,
                *k = nullptr, *v = nullptr, *g = nullptr, *beta = nullptr, *attention = nullptr;
    ggml_tensor *ple_emb = nullptr, *ple_hist = nullptr, *ple_hist_out = nullptr, *ple_key = nullptr, *ple_raw_key = nullptr,
                *ple_value = nullptr, *ple_gate = nullptr, *ple_norm = nullptr, *ple_conv = nullptr,
                *ple_result = nullptr;
    std::array<int32_t, 2> previous{{-1, -1}};
    std::vector<float> ple_history;
    std::vector<float> state, history, last;
    // Separate host banks, already priced by HOST_STATE_BYTES/PLE_STATE_BYTES.
    // Only a complete successful step swaps these into the public mirrors.
    std::vector<float> candidate_state, candidate_history, candidate_ple_history;
    std::vector<PortablePlacement> placement;
    size_t weights_size = 0, graphs_size = 0, attention_device_weights = 0;
    uint64_t count = 0;

    bool production() const { return options.policy == PortableGdnPolicy::device_production; }
    size_t host_state_bytes() const {
        return (options.native_ple_gate_t1?PortableGdnOptions::native_ple_gate_host_bound:0)+(options.native_ple_conv_t1?PortableGdnOptions::native_ple_conv_host_bound:0)+(options.native_ple_value_t1?PortableGdnOptions::native_ple_value_host_bound:0)+(options.native_ple_key_t1?PortableGdnOptions::native_ple_key_host_bound:0)+(options.native_output_projection_t1?PortableGdnOptions::native_output_projection_host_bound:0)+(options.native_input_projection_t1?PortableGdnOptions::native_input_projection_host_bound:0)+(options.native_preparation_weights?PortableGdnOptions::native_preparation_host_bound:0)+(options.native_recurrence_t1?PortableGdnOptions::native_recurrence_host_bound:0)+(options.native_hc_norm_inject?(options.native_hc_pending?PortableGdnOptions::native_hc_pending_host_bound:(options.native_hc_mix?PortableGdnOptions::native_hc_read_host_bound:PortableGdnOptions::native_hc_host_bound)):0)+HOST_STATE_BYTES - (production() ? (2 * STATES + 2 * HISTORY) * sizeof(float) : 0) +
            (options.ple_table ? PLE_STATE_BYTES - (production() ? 2 * PLE_HISTORY * sizeof(float) : 0) : 0);
    }

    PortableGdnScratch::Impl* shared() const { return options.scratch ? options.scratch->impl_.get() : nullptr; }
    void schedule_graph(ggml_cgraph* g, const char* phase, size_t index, size_t& bytes,
                        portable_detail::Scheduler& sched, bool credit=false) {
        if(!shared()) { auto execution=options.execution;require(preparation_credit()<=execution.graph_budget,"native preparation graph reserve");execution.graph_budget-=preparation_credit();sched=portable_detail::schedule(g,phase,execution,cpu,placement,bytes,credit); return; }
        auto& o=*shared();
        const size_t workspace=std::max(o.workspace[index],ggml_graph_plan(g,options.execution.threads,nullptr).work_size);
        size_t used=preparation_credit()+ffn->graph_bytes()+native_hc_bytes()+native_gdn_bytes()+(chunk_ffn?chunk_ffn->graph_bytes():0);
        const std::array<ggml_backend_sched_t,3> all{{scheduler.get(),projection_scheduler.get(),tail_scheduler.get()}};
        for(size_t i=0;i<3;++i) {
            const size_t work=i==index?workspace:o.workspace[i];
            require(used<=options.execution.graph_budget && work<=options.execution.graph_budget-used,
                    "portable GDN: shared workspace budget exceeded");
            used+=work;
            if(i==index || !all[i]) continue;
            const size_t retained=ggml_backend_sched_get_buffer_size(all[i],o.cpu.get())+
                (options.execution.request_accelerator && options.execution.accelerator ?
                 ggml_backend_sched_get_buffer_size(all[i],options.execution.accelerator):0);
            require(retained<=options.execution.graph_budget-used,"portable GDN: shared retained budget exceeded");
            used+=retained;
        }
        const size_t before=used;
        portable_detail::schedule_reusing(g,phase,options.execution,o.cpu,placement,used,sched,credit);
        o.workspace[index]=workspace;
        o.active[index]=sched.get();
        bytes+=used-before+workspace;
    }
    void build_graphs() {
        // The pinned allocator reuses same-shaped plans without comparing
        // OUTPUT flags. An ordinary layer's plan does not retain HC snapshots.
        // Replan only this opt-in serial observation, before assigning backends;
        // reserve_size would discard those assignments at this GGML pin.
        if(options.execution.layer0_observer && !options.execution.layer0_observer->finished())
            scheduler.reset();
        ctx=context();
        const auto& w=weights;
        auto *c = ctx.get();
        input = ggml_new_tensor_3d(c, GGML_TYPE_F32, H, HC, 1);
#ifdef STRATA_PORTABLE_NATIVE_HC
        if(native_hc)input=native_hc->residual;
#endif
        const bool keep_state = options.resident_state && options.execution.request_accelerator &&
                                options.execution.accelerator;
        if (keep_state && !state_ctx)
            state_ctx = context();
        auto *sc = keep_state ? state_ctx.get() : c;
        if (!keep_state || !state_input) {
            state_input = ggml_new_tensor_4d(sc, GGML_TYPE_F32, D, options.native_recurrence_t1?HV:D, options.native_recurrence_t1?D:HV, 1);
            history_input = ggml_new_tensor_3d(sc, GGML_TYPE_F32, 3, CHANNELS, 1);
        }
        for (auto *t : {input, state_input, history_input})
            ggml_set_input(t);
        auto *hidden = input;
        if (options.ple_table) {
#ifdef STRATA_PORTABLE_NATIVE_HC
            ple_emb = native_ple_value ? native_ple_value->input() : ggml_new_tensor_2d(c, GGML_TYPE_F32, H, 1);
#else
            ple_emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, H, 1);
#endif
            if (!keep_state || !ple_hist) ple_hist = ggml_new_tensor_2d(sc, GGML_TYPE_F32, 9, RESIDUAL);
            ggml_set_input(ple_emb);
            ggml_set_input(ple_hist);
            auto grouped = [&](ggml_tensor *x, const char *name) {
                return ggml_mul(c, ggml_rms_norm(c, ggml_reshape_3d(c, x, H, HC, 1), 1e-6f), w.at(name));
            };
            ggml_tensor* raw_key = nullptr;
#ifdef STRATA_PORTABLE_NATIVE_HC
            if (native_ple_key) raw_key = native_ple_key->output_bank(); else
#endif
                raw_key = ggml_mul_mat(c, w.at("ple_key.weight"), ple_emb);
            ple_raw_key = raw_key;
            auto *key_normalized = grouped(raw_key, "ple_norm_key.weight");
            ple_key = key_normalized;
#ifdef STRATA_PORTABLE_NATIVE_HC
            ple_value = native_ple_value ? native_ple_value->output() : ggml_mul_mat(c, w.at("ple_value.weight"), ple_emb);
#else
            ple_value = ggml_mul_mat(c, w.at("ple_value.weight"), ple_emb);
#endif
            auto *query = grouped(input, "ple_norm_query.weight");
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_ple_gate) {
                ple_gate_prefix_graph=ggml_new_graph(c);
                ple_gate_key_copy=ggml_cpy(c,key_normalized,native_ple_gate->key());
                ple_gate_query_copy=ggml_cpy(c,query,native_ple_gate->query());
                ggml_build_forward_expand(ple_gate_prefix_graph,ple_gate_key_copy);
                ggml_build_forward_expand(ple_gate_prefix_graph,ple_gate_query_copy);
                native_ple_gate->bind(ple_gate_key_copy,ple_gate_query_copy,raw_key,input);
                ple_key=native_ple_gate->key();ple_gate=native_ple_gate->gate();
            } else
#endif
            {
                auto *score = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, ple_key, query)), 1.0f / std::sqrt(float(H)));
                auto *mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, score), 1e-6f, INFINITY));
                ple_gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, score), mag));
            }
            auto *value = ggml_repeat_4d(c, ggml_reshape_3d(c, ple_value, H, 1, 1), H, HC, 1, 1);
            auto *gated_value = ggml_mul(c, value, ple_gate);
            ple_gate_broadcast = gated_value;
            ple_norm = grouped(gated_value, "ple_norm_conv.weight");
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_ple_conv) {
                ple_prefix_graph=ggml_new_graph(c);
                ple_conv_sources={ple_hist,ple_norm,input,gated_value};
                for(size_t i=0;i<4;++i){
                    ggml_set_output(ple_conv_sources[i]);
                    ple_conv_copies[i]=ggml_cpy(c,ple_conv_sources[i],native_ple_conv->input(i));
                    ggml_set_output(ple_conv_copies[i]);
                    ggml_build_forward_expand(ple_prefix_graph,ple_conv_copies[i]);
                }
                for(auto* t:{ple_key,ple_value,ple_gate,ple_norm}) {ggml_set_output(t);ggml_build_forward_expand(ple_prefix_graph,t);}
                native_ple_conv->bind(ple_conv_copies.data(),ple_conv_sources.data());
                ple_hist_out=ggml_reshape_2d(c,native_ple_conv->history(),9,RESIDUAL);
                ple_conv=ggml_reshape_3d(c,native_ple_conv->convolution(),H,HC,1);
                ple_result=ggml_reshape_3d(c,native_ple_conv->result(),H,HC,1);
            } else
#endif
            {
            auto *padded = ggml_concat(c, ple_hist, ggml_reshape_2d(c, ple_norm, 1, RESIDUAL), 0);
            ple_hist_out = ggml_cont(c, ggml_view_2d(c, padded, 9, RESIDUAL, 10 * sizeof(float), sizeof(float)));
            ggml_tensor *sum = nullptr;
            for (int tap = 0; tap < 4; ++tap) {
                auto *shifted = ggml_reshape_1d(c,
                                                ggml_cont(c, ggml_view_2d(c, padded, 1, RESIDUAL, 10 * sizeof(float),
                                                                          size_t(tap * 3) * sizeof(float))),
                                                RESIDUAL);
                auto *weight =
                    ggml_reshape_1d(c,
                                    ggml_cont(c, ggml_view_2d(c, w.at("ple_conv1d.weight"), 1, RESIDUAL,
                                                              w.at("ple_conv1d.weight")->nb[1],
                                                              size_t(tap) * w.at("ple_conv1d.weight")->nb[0])),
                                    RESIDUAL);
                if (weight->type != GGML_TYPE_F32)
                    weight = ggml_cast(c, weight, GGML_TYPE_F32);
                auto *term = ggml_mul(c, shifted, weight);
                sum = sum ? ggml_add(c, sum, term) : term;
            }
            ple_conv = ggml_reshape_3d(c, ggml_silu(c, sum), H, HC, 1);
            ple_result = ggml_add(c, input, ggml_add(c, gated_value, ple_conv));
            }
            ggml_set_name(ple_result, "ple_residual");
            hidden = ple_result;
        }
        auto build = [&](bool external) {
            auto *norm =
                ggml_mul(c, ggml_reshape_2d(c, ggml_rms_norm(c, hidden, 1e-6f), H * HC, 1), w.at("hc_attn_norm.weight"));
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_hc)norm=native_hc->normalized;
#endif
            ggml_tensor* mixed=nullptr;
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_hc && native_hc->mixed)mixed=native_hc->mixed;else
#endif
            {
            auto *low = ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, w.at("hc_attn_down.weight"), norm), 0.25f));
            auto *gates = ggml_sigmoid(c, ggml_mul_mat(c, w.at("hc_attn_up.weight"), low));
            auto *gated = ggml_reshape_2d(c, ggml_mul(c, norm, gates), H, HC);
            mixed = ggml_cont(c, ggml_view_1d(c, gated, H, 0));
            for (int64_t s = 1; s < HC; ++s)
                mixed = ggml_add(c, mixed, ggml_view_1d(c, gated, H, size_t(s * H) * sizeof(float)));
            mixed = ggml_scale(c, mixed, 0.25f);
            }
            ggml_tensor* inject=nullptr;
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_hc)inject=native_hc->inject;else
#endif
            inject=ggml_mul_mat(c,w.at("hc_attn_inject.weight"),norm);
            // Keep only the opt-in serial observer's actual HC outputs alive.
            // T1 attention uses this graph; chunk tail projections are separate.
            if(!external && options.execution.layer0_observer) {
                observed_mixed=mixed;observed_inject=inject;
                ggml_set_output(mixed);ggml_set_output(inject);
            }
            auto leaf=[&](int64_t n) {auto* t=ggml_new_tensor_1d(c,GGML_TYPE_F32,n);ggml_set_input(t);return t;};
            if(external) for(size_t i=0;i<5;++i) projection_inputs[i]=leaf(std::array<int64_t,5>{HC,CHANNELS,INNER,HV,HV}[i]);
            if(external) inject=projection_inputs[0];
            ggml_tensor *z=nullptr;
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_input_projection){qkv=native_input_projection->projection(0);z=native_input_projection->projection(1);}
            else
#endif
            {qkv = external ? projection_inputs[1] : ggml_mul_mat(c, w.at("attn_qkv.weight"), mixed);
             z = external ? projection_inputs[2] : ggml_mul_mat(c, w.at("attn_gate.weight"), mixed);}
            auto* beta_input=options.native_preparation_t1 ? nullptr : external ? projection_inputs[4] : ggml_mul_mat(c,w.at("ssm_beta.weight"),mixed);
            auto* alpha=options.native_preparation_t1 ? nullptr : external ? projection_inputs[3] : ggml_mul_mat(c,w.at("ssm_alpha.weight"),mixed);
            preparation_mixed=mixed;
            outputs(build_tail(c,hidden,inject,qkv,z,alpha,beta_input,state_input,history_input));
            graph = ggml_new_graph(c);
            for (auto *t : {attn_residual, state_output, history_output, qkv, conv, q, k, v, g, beta, attention}) {
                if(options.native_recurrence_t1 && (t==qkv||t==conv||t==q||t==k||t==v||t==g||t==beta))continue;
                ggml_set_output(t);
                ggml_build_forward_expand(graph, t);
            }
            if(options.native_hc_ffn_read)for(auto* t:{produced_block,produced_inject}) {
                ggml_set_output(t);ggml_build_forward_expand(graph,t);
            }
            if (options.ple_table) {
                for (auto *t : {ple_hist_out, ple_key, ple_value, ple_gate, ple_norm, ple_conv, ple_result}) {
#ifdef STRATA_PORTABLE_NATIVE_HC
                    if(native_ple_conv && (t==ple_key||t==ple_value||t==ple_gate||t==ple_norm))continue;
#endif
                    ggml_set_output(t);
                    ggml_build_forward_expand(graph, t);
                }
                if (!production() && ple_history.empty()) {
                    ple_history.assign(PLE_HISTORY, 0);
                    candidate_ple_history.resize(PLE_HISTORY);
                }
            }
            if (keep_state && !external) {
              if (!state_buffer) {
                // Native session ownership: one committed state allocation per layer.
                // Candidate graph outputs remain separate until attention AND FFN pass.
                auto *backend = options.execution.accelerator;
                const size_t bound = ggml_backend_alloc_ctx_tensors_from_buft_size(
                    state_ctx.get(), ggml_backend_get_default_buffer_type(backend));
                require(bound <= options.resident_state_budget, "portable GDN: resident state budget exceeded");
                state_buffer.reset(ggml_backend_alloc_ctx_tensors(state_ctx.get(), backend));
                require(bool(state_buffer), "portable GDN: resident state allocation failed");
                device_state_size = ggml_backend_buffer_get_size(state_buffer.get());
                require(device_state_size <= options.resident_state_budget,
                        "portable GDN: actual resident state budget exceeded");
                ggml_backend_buffer_clear(state_buffer.get(), 0);
                ggml_backend_synchronize(backend);
              }
                // The copy API requires identical layouts. These are metadata-only
                // views of the unchanged candidate bytes, allocated before scheduling.
                state_output = ggml_reshape_4d(c, state_output, D, options.native_recurrence_t1?HV:D, options.native_recurrence_t1?D:HV, 1);
                history_output = ggml_reshape_3d(c, history_output, 3, CHANNELS, 1);
                for (auto *t : {state_output, history_output}) {
                    ggml_set_output(t);
                    ggml_build_forward_expand(graph, t);
                }
            }
            if(external && keep_state) {
                state_output=ggml_reshape_4d(c,state_output,D,D,HV,1);
                history_output=ggml_reshape_3d(c,history_output,3,CHANNELS,1);
                for(auto* t:{state_output,history_output}) {ggml_set_output(t);ggml_build_forward_expand(graph,t);}
            }
            if(options.native_ple_gate_t1) {
                require(!external && ple_gate_prefix_graph,"native PLE gate: serial graph required");
                schedule_graph(ple_gate_prefix_graph,"native-ple-gate-prefix",1,graphs_size,projection_scheduler);
            }
            if(options.native_recurrence_t1) {
                require(!external,"native GDN: external/chunk graph unsupported");
                schedule_graph(native_prefix_graph,"native-gdn-prefix",1,graphs_size,projection_scheduler);
            }
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_ple_conv) {
                if(native_ple_gate)schedule_graph(ple_prefix_graph,"native-ple-conv-prefix",2,graphs_size,tail_scheduler);
                else schedule_graph(ple_prefix_graph,"native-ple-conv-prefix",1,graphs_size,projection_scheduler);
            }
#endif
            auto& sched=external?tail_scheduler:scheduler;
            schedule_graph(graph,external?"gdn-chunk-recurrence":"gdn-attention",external?2:0,graphs_size,sched);
            if(external) { tail_graph=graph; tail_outputs=outputs(); }
        };
        build(false);
        if(batch_projections() && !causal()) {
            const auto saved=outputs(); auto* saved_graph=graph;
            build(true); outputs(saved); graph=saved_graph;
        }
    }
    // A lease spans attention, host readback, the provider, and state commit.
    // Synchronize and forget all old tensor identities before freeing metadata.
    struct Borrow {
        Impl& s;
        bool held=false;
        explicit Borrow(Impl& layer, bool rebuild=true) : s(layer) {
            auto* o=s.shared();
            if(!o) return;
            require(!o->borrowed.test_and_set(std::memory_order_acquire), "portable GDN: shared scratch already borrowed");
            held=true;
            if(rebuild) s.placement.erase(std::remove_if(s.placement.begin(),s.placement.end(),[](const auto& p){ return p.phase.rfind("gdn-",0)==0; }),s.placement.end());
            s.scheduler=std::move(o->schedulers[0]);
            s.projection_scheduler=std::move(o->schedulers[1]);
            s.tail_scheduler=std::move(o->schedulers[2]);
            o->active={s.scheduler.get(),s.projection_scheduler.get(),s.tail_scheduler.get()};
            try { if(rebuild) s.build_graphs(); } catch(...) { release(); throw; }
        }
        void release() {
            if(!held) return;
            auto& o=*s.shared();
            for(auto* sched:{s.scheduler.get(),s.projection_scheduler.get(),s.tail_scheduler.get()}) if(sched) {
                if(s.ctx || s.projection_ctx) ggml_backend_sched_synchronize(sched);
                ggml_backend_sched_reset(sched);
            }
            // reset() takes a lease without graph metadata. Its state clear
            // performs its own synchronization and can report a failed reset.
            if(s.ctx || s.projection_ctx) {
                if(o.options.accelerator) ggml_backend_synchronize(o.options.accelerator);
                ggml_backend_synchronize(o.cpu.get());
            }
            o.schedulers[0]=std::move(s.scheduler);
            o.schedulers[1]=std::move(s.projection_scheduler);
            o.schedulers[2]=std::move(s.tail_scheduler);
            o.active={};
            s.ctx.reset(); s.projection_ctx.reset();
            s.graph=nullptr; s.tail_graph=nullptr; s.projection_graph=nullptr; s.native_prefix_graph=nullptr; s.ple_prefix_graph=nullptr; s.ple_gate_prefix_graph=nullptr; s.ple_gate_broadcast=nullptr; s.ple_gate_key_copy=nullptr; s.ple_gate_query_copy=nullptr;
            s.projection_tokens=0; s.projection_bytes=0;
            s.graphs_size=s.ffn->graph_bytes()+s.native_hc_bytes()+s.native_gdn_bytes();
            if(!s.state_buffer) { s.state_input=nullptr; s.history_input=nullptr; s.ple_hist=nullptr; }
            held=false; o.borrowed.clear(std::memory_order_release);
        }
        ~Borrow() { release(); }
    };

    Impl(int64_t layer, const std::vector<PortableTensorView> &tensors, PortableGdnOptions opts) : options(opts) {
        require(layer >= 0 && layer < 48 && layer % 4 != 3,
                "portable GDN: expected a non-QSA layer in [0,47]");
        require(!options.execution.defer_hc_write,"portable GDN: configure deferred write through layer option");
        require(!options.defer_ffn_hc_write || (portable_detail::native_hc_boundary(int(layer)).defer_ffn && !options.native_hc_ffn_write && !production() &&
            !options.batch_input_projections && !options.causal_chunk),"deferred HC write: eager ordinary native boundary only");
        require(!options.execution.native_hc_final_write,"portable GDN: configure FFN write through layer option");
        require(!options.native_hc_ffn_write||options.native_hc_ffn_read,"native FFN final write requires pending FFN read");
        require(!options.native_ple_value_t1 || (layer == 1 && options.ple_table && !production() && !options.batch_input_projections && !options.causal_chunk && options.execution.request_accelerator && options.execution.accelerator && !options.execution.allow_cpu_fallback && (options.resident_attention_weights || options.execution.resident_dense_weights) && !options.native_ple_value_receipt.empty()),
                "native PLE value: layer 1 serial eager strict accelerator, resident roles and receipt required");
#ifndef STRATA_PORTABLE_NATIVE_HC
        require(!options.native_ple_value_t1, "native PLE value: native stage build required");
#endif
        require(!options.native_preparation_weights || (layer!=1 && options.native_recurrence_t1),"native GDN preparation weights: ordinary native recurrence required; PLE unsupported");
        require(!options.native_preparation_t1 || (options.native_preparation_weights&&!options.native_preparation_receipt.empty()),"native GDN preparation: explicit matched BF16 residency and receipt required");
        require(!options.native_input_projection_t1 || (layer!=1 && options.native_hc_mix && options.native_preparation_t1 && options.native_recurrence_t1 && !options.native_input_projection_receipt.empty()),"native GDN input projection: ordinary native HC, preparation, recurrence and receipt required");
        require(!options.native_output_projection_t1 || (layer!=1 && options.native_hc_ffn_read && options.native_preparation_t1 && options.native_recurrence_t1 && !options.native_output_projection_receipt.empty()),"native GDN output projection: ordinary native recurrence, FFN read and receipt required");
        require(!options.native_ple_gate_t1 || (layer==1 && options.ple_table && !production() && !options.batch_input_projections && !options.causal_chunk && !options.native_recurrence_t1 && !options.native_hc_norm_inject && options.execution.request_accelerator && options.execution.accelerator && !options.execution.allow_cpu_fallback && (options.resident_attention_weights || options.execution.resident_dense_weights) && !options.native_ple_gate_receipt.empty()), "native PLE gate: layer 1 serial eager strict accelerator, resident weights and receipt required");
        require(!options.native_ple_key_t1 || (layer == 1 && options.ple_table && !production() && !options.batch_input_projections && !options.causal_chunk && options.execution.request_accelerator && options.execution.accelerator && !options.execution.allow_cpu_fallback && (options.resident_attention_weights || options.execution.resident_dense_weights) && !options.native_ple_key_receipt.empty()), "native PLE key: layer 1 serial eager strict accelerator, resident role and receipt required");
        require(!options.native_ple_conv_t1 || (layer == 1 && options.ple_table && !production() && !options.batch_input_projections && !options.causal_chunk && !options.native_hc_norm_inject && options.execution.request_accelerator && options.execution.accelerator && !options.execution.allow_cpu_fallback && (options.resident_attention_weights || options.execution.resident_dense_weights) && !options.native_ple_conv_receipt.empty()), "native PLE conv: layer 1 serial eager strict accelerator, resident role and receipt required");
#ifndef STRATA_PORTABLE_NATIVE_HC
        require(!options.native_ple_key_t1, "native PLE key: native stage build required");
        require(!options.native_ple_conv_t1, "native PLE conv: native stage build required");
        require(!options.native_ple_gate_t1, "native PLE gate: native stage build required");
#endif
        if(options.native_output_projection_t1){const char* fused=std::getenv("STRATA_QFUSE");require(!fused||std::atoi(fused)==0,"native GDN output projection: fused quantization unsupported");}
        if(options.native_recurrence_t1) {
            require(!options.ple_table&&!options.batch_input_projections&&!options.causal_chunk&&!production(),"native GDN T1: eager ordinary graph only");
            require(options.resident_state && options.execution.request_accelerator && options.execution.accelerator && !options.execution.allow_cpu_fallback &&
                    (options.resident_attention_weights||options.execution.resident_dense_weights)&&!options.native_recurrence_receipt.empty(),"native GDN T1: strict accelerator, resident state/attention and receipt required");
#ifndef STRATA_PORTABLE_NATIVE_HC
            throw std::runtime_error("native GDN T1: strict-FMA native stage build required");
#endif
        }
        require(!options.execution.native_hc_pending_read,"portable GDN: configure FFN read through layer option, not raw execution option");
        require(!options.native_hc_ffn_read || (options.native_hc_mix &&
                (!options.native_hc_pending || portable_detail::native_hc_boundary(int(layer)).pending_attention) &&
                (options.execution.resident_dense_weights || options.resident_ffn_weights) && !options.execution.layer0_observer),
                "native GDN-to-FFN: composed attention/resident dense required; legacy observer unsupported");
        require(!options.native_hc_pending||options.native_hc_mix,"native HC pending: composed read required");
        require(!options.native_hc_pending||!options.execution.layer0_observer,"native HC pending: legacy observer lacks pending tuple schema; use bounded layer trace");
        require(!options.native_hc_mix||options.native_hc_norm_inject,"native HC read: norm/injection required");
        if(options.native_hc_norm_inject) {
            require((layer==0 || !options.native_hc_pending || portable_detail::native_hc_boundary(int(layer)).pending_attention) && !options.ple_table && !production() && !options.batch_input_projections && !options.causal_chunk,
                    "native HC serial: eager native boundary only; PLE unsupported");
            require(options.execution.request_accelerator && options.execution.accelerator && !options.execution.allow_cpu_fallback &&
                    (options.resident_attention_weights || options.execution.resident_dense_weights) && !options.native_hc_receipt_prefix.empty(),
                    "native HC serial: strict accelerator, resident attention weights and receipt prefix required");
#ifndef STRATA_PORTABLE_NATIVE_HC
            throw std::runtime_error("native HC serial: strict-FMA native-stage build required");
#else
            portable_detail::native_hc_environment();
#endif
        }

        require(!options.ple_table || layer == 1,
                "portable GDN: this model supports PLE only at layer 1");
        require(!production() || (options.resident_state && options.execution.request_accelerator &&
                    options.execution.accelerator && !options.execution.allow_cpu_fallback),
                "portable GDN: production requires resident state and strict selected backend");
        require(!options.causal_chunk || (options.batch_input_projections && production()),
                "portable GDN: causal chunk requires production and batched input projections");
        if(auto* o=shared()) {
            const auto& a=o->options; const auto& b=options.execution;
            require(a.accelerator==b.accelerator && a.request_accelerator==b.request_accelerator &&
                    a.allow_cpu_fallback==b.allow_cpu_fallback && a.threads==b.threads && a.graph_budget==b.graph_budget,
                    "portable GDN: incompatible shared scratch backend/options");
        }
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        auto expected_shapes = shapes();
        if (options.ple_table)
            expected_shapes.insert(ple_shapes().begin(), ple_shapes().end());
        require(options.state_budget >= host_state_bytes(),
                "portable GDN: host state budget exceeded");
        std::vector<PortableTensorView> fw, aw;
        for (const auto &t : tensors) {
            require(t.name.rfind(prefix, 0) == 0, "portable GDN: tensor belongs to another layer");
            (expected_shapes.count(t.name.substr(prefix.size())) ? aw : fw).push_back(t);
        }
        require(aw.size() == expected_shapes.size() && fw.size() == 9,
                "portable GDN: missing/extra attention, PLE or FFN tensors");
        std::string error;
        auto ffn_options=options.execution;
        ffn_options.resident_dense_weights=options.execution.resident_dense_weights || options.resident_ffn_weights;
        ffn_options.native_hc_pending_read=options.native_hc_ffn_read;
        ffn_options.native_hc_final_write=options.native_hc_ffn_write;
        ffn_options.defer_hc_write=options.defer_ffn_hc_write;
        ffn_options.native_hc_receipt_prefix=options.native_hc_receipt_prefix+".ffn";
        if(options.scratch) {
            const size_t retained=options.scratch->graph_bytes();
            require(retained<ffn_options.graph_budget,"portable GDN: no private FFN graph budget");
            ffn_options.graph_budget-=retained;
        }
        ffn = PortableFfn::create({H, HC, R, 640, 512, 10, 1e-6f}, layer, fw, ffn_options, error);
        require(bool(ffn), error);
        weights_size = ffn->weight_bytes();
        graphs_size = ffn->graph_bytes();
        placement = ffn->placements();
        cpu.reset(ggml_backend_cpu_init());
        require(bool(cpu), "portable GDN: CPU unavailable");
        ggml_backend_cpu_set_n_threads(cpu.get(), options.execution.threads);
        weights_ctx = context();
        std::map<std::string, ggml_tensor *> w;
        auto native_elements=[&](const std::string& key)->size_t {
            if(options.native_ple_value_t1 && key=="ple_value.weight")return H*H;
            if(options.native_hc_norm_inject && key=="hc_attn_inject.weight")return 40960;
            if(options.native_hc_mix && (key=="hc_attn_down.weight"||key=="hc_attn_up.weight"))return 3276800;
            if(options.native_preparation_weights && (key=="ssm_alpha.weight"||key=="ssm_beta.weight"))return H*HV;
            return 0;
        };
        for (const auto &view : aw) {
            const auto key = view.name.substr(prefix.size());
            const auto &expected = expected_shapes.at(key);
            const bool flat_norm = key.rfind("ple_norm_", 0) == 0 && view.shape == std::vector<int64_t>{H * HC};
            require(!w.count(key) && (view.shape == expected || flat_norm),
                    "portable GDN: bad/duplicate tensor shape: " + view.name);
            const bool quantized = expected.size() == 2 && key != "ssm_conv1d.weight" &&
                                   (key.rfind("ple_", 0) != 0 || key == "ple_key.weight" || key == "ple_value.weight");
            const bool half_conv = key == "ple_conv1d.weight" && view.type == 1;
            require((view.type == (quantized ? 8u : 0u) || half_conv) && view.data,
                    "portable GDN: expected fleet Q8_0/F32 encoding: " + view.name);
            const size_t native_count=native_elements(key);
            const bool ab_bf16=options.native_preparation_weights&&(key=="ssm_alpha.weight"||key=="ssm_beta.weight");
            auto *t = ab_bf16?ggml_new_tensor(weights_ctx.get(),GGML_TYPE_BF16,int(expected.size()),expected.data()):native_count?ggml_new_tensor_1d(weights_ctx.get(),GGML_TYPE_I32,int64_t(native_count/2)):
                ggml_new_tensor(weights_ctx.get(),ggml_type(view.type),int(expected.size()),expected.data());
            ggml_set_name(t, view.name.c_str());
            const size_t stored=ggml_nbytes(t);
            require((native_count?view.bytes==native_count/32*34:stored==view.bytes) && stored <= options.execution.weight_budget - weights_size,
                    "portable GDN: weight size/budget mismatch");
            weights_size += stored;
            w[key] = t;
        }
        const bool resident = (options.resident_attention_weights || options.execution.resident_dense_weights) &&
                              options.execution.request_accelerator &&
                              options.execution.accelerator;
        auto *weight_backend = resident ? options.execution.accelerator : cpu.get();
        const size_t allocation = ggml_backend_alloc_ctx_tensors_from_buft_size(
            weights_ctx.get(), ggml_backend_get_default_buffer_type(weight_backend));
        require(allocation <= options.execution.weight_budget - ffn->weight_bytes(),
                "portable GDN: aligned weight budget exceeded");
        weights_buffer.reset(ggml_backend_alloc_ctx_tensors(weights_ctx.get(), weight_backend));
        require(bool(weights_buffer), "portable GDN: weight allocation failed");
        ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto &view : aw) {
            const auto key=view.name.substr(prefix.size());
            const size_t native_count=native_elements(key);
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(native_count) {
                // Stream one original compat-BF16 conversion at a time: no
                // duplicate resident Q8 bank or retained host down/up copies.
                const auto converted=portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(view.data),view.bytes,native_count);
                ggml_backend_tensor_set(w.at(key),converted.data(),0,converted.size()*2);
            } else
#endif
            ggml_backend_tensor_set(w.at(key),view.data,0,view.bytes);
        }
        const size_t actual_weights = ggml_backend_buffer_get_size(weights_buffer.get());
        require(actual_weights <= options.execution.weight_budget - ffn->weight_bytes(),
                "portable GDN: actual aligned weight budget exceeded");
        attention_device_weights = resident ? actual_weights : 0;
        weights_size = ffn->weight_bytes() + actual_weights;
        weights=w;
#ifdef STRATA_PORTABLE_NATIVE_HC
        if(options.native_ple_conv_t1){
            auto role=std::find_if(aw.begin(),aw.end(),[&](const auto& v){return v.name==prefix+"ple_conv1d.weight";});
            require(role!=aw.end() && role->type==1 && role->bytes==81920,"native PLE conv: F16 role required");
            require(graphs_size<options.execution.graph_budget,"native PLE conv: no graph room");
            require(std::filesystem::create_directory(options.native_ple_conv_receipt),"native PLE conv: fresh receipt required");
            auto path=(std::filesystem::path(options.native_ple_conv_receipt)/"effective.spv").string();
            native_ple_conv=std::make_unique<portable_detail::PortableNativePleConv>(options.execution.accelerator,role->data,role->bytes,path.c_str(),options.execution.graph_budget-graphs_size);
            graphs_size+=native_ple_conv->bytes();
            placement.push_back({"native-ple","NATIVE_CONV_T1_DIAGNOSTIC","normalized input; candidate history","F32/F16",ggml_backend_name(options.execution.accelerator),"",{4,RESIDUAL}});
        }
        if(options.native_ple_gate_t1) {
            require(graphs_size<options.execution.graph_budget,"native PLE gate: no graph room");
            native_ple_gate=std::make_unique<portable_detail::PortableNativePleGate>(options.execution.accelerator,options.native_ple_gate_receipt,options.execution.graph_budget-graphs_size);
            graphs_size+=native_ple_gate->bytes();
            placement.push_back({"native-ple","NATIVE_PLE_GATE_T1_DIAGNOSTIC","device normalized PLE key/query; actual value broadcast","F32",ggml_backend_name(options.execution.accelerator),"",{H,HC}});
        }
        if (options.native_ple_key_t1) {
            require(graphs_size < options.execution.graph_budget, "native PLE key: no graph room");
            auto role = std::find_if(aw.begin(), aw.end(), [&](const auto& v) { return v.name == prefix + "ple_key.weight"; });
            require(role != aw.end() && role->type == 8, "native PLE key: Q8_0 role required");
            native_ple_key = std::make_unique<portable_detail::PortableNativePleKey>(options.execution.accelerator,
                w.at("ple_key.weight"), role->data, role->bytes, options.native_ple_key_receipt, options.execution.graph_budget - graphs_size);
            graphs_size += native_ple_key->bytes();
            placement.push_back({"native-ple", "NATIVE_Q8_KEY_T1_DIAGNOSTIC", "gathered embedding; existing PLE history", "Q8_0/Q8_1", ggml_backend_name(options.execution.accelerator), "", {H, H * HC}});
        }
        if(options.native_ple_value_t1) {
            auto role=std::find_if(aw.begin(),aw.end(),[&](const auto& v){return v.name==prefix+"ple_value.weight";});
            require(role!=aw.end() && role->type==8,"native PLE value: Q8_0 role required");
            auto converted=portable_detail::native_hc_q8_bf16(static_cast<const uint8_t*>(role->data),role->bytes,H*H);
            require(graphs_size<options.execution.graph_budget,"native PLE value: no graph room");
            native_ple_value=std::make_unique<portable_detail::PortableNativePleValue>(options.execution.accelerator,w.at("ple_value.weight"),std::move(converted),options.native_ple_value_receipt,options.execution.graph_budget-graphs_size);
            graphs_size+=native_ple_value->bytes();
            placement.push_back({"native-ple", "NATIVE_BF16_VALUE_T1_DIAGNOSTIC", "gathered embedding; existing PLE history", "F32/BF16", ggml_backend_name(options.execution.accelerator), "", {H,H}});
        }
        if(options.native_recurrence_t1) {
            require(graphs_size<options.execution.graph_budget,"native GDN: no graph room");
            native_gdn=std::make_unique<portable_detail::PortableNativeGdn>(options.execution.accelerator,options.execution.graph_budget-graphs_size,options.native_recurrence_receipt,options.native_recurrence_record_first);
            graphs_size+=native_gdn->bytes();
        }
        if(options.native_hc_norm_inject) {
            require(graphs_size<options.execution.graph_budget,"native HC serial: no graph budget");
            native_hc=std::make_unique<portable_detail::PortableNativeHc>(options.execution.accelerator,w.at("hc_attn_norm.weight"),
                w.at("hc_attn_inject.weight"),options.execution.graph_budget-graphs_size,options.native_hc_receipt_prefix,
                options.native_hc_mix?w.at("hc_attn_down.weight"):nullptr,options.native_hc_mix?w.at("hc_attn_up.weight"):nullptr,options.native_hc_pending);
            graphs_size+=native_hc->bytes();
            placement.push_back({"native-hc",options.native_hc_pending?"NATIVE_BF16_PENDING_READ_T1_DIAGNOSTIC":(options.native_hc_mix?"NATIVE_BF16_READ_SERIAL_T1":"NATIVE_BF16_NORM_INJECT_SERIAL_T1"),options.native_hc_pending?"explicit value-owned prior FFN tuple":(options.native_hc_mix?"serial attention read; FFN write mode explicit":"serial norm/injection; GGML mixed remains"), "F32/BF16",ggml_backend_name(options.execution.accelerator),"",{H,HC,1}});
        }
#endif
#ifdef STRATA_PORTABLE_NATIVE_HC
        if(options.native_preparation_weights){preparation_reserved=portable_detail::PortableNativeGdnPreparation::estimate(options.execution.accelerator);require(graphs_size<=options.execution.graph_budget&&preparation_reserved<=options.execution.graph_budget-graphs_size,"native GDN preparation reserved graph admission");}
        if(options.native_preparation_t1){native_preparation=std::make_unique<portable_detail::PortableNativeGdnPreparation>(options.execution.accelerator,*native_gdn,w.at("ssm_alpha.weight"),w.at("ssm_beta.weight"),w.at("ssm_dt.bias"),w.at("ssm_a"),options.native_preparation_receipt,options.execution.graph_budget-graphs_size,options.native_preparation_record_first);graphs_size+=native_preparation->bytes();}
        if(options.native_input_projection_t1){
            require(graphs_size<options.execution.graph_budget,"native GDN input projection: no graph room");
            std::array<ggml_tensor*,2> weights{{w.at("attn_qkv.weight"),w.at("attn_gate.weight")}};
            std::array<const void*,2> raw{};std::array<size_t,2> sizes{};const std::array<const char*,2> names{{"attn_qkv.weight","attn_gate.weight"}};
            for(size_t i=0;i<2;++i){auto role=std::find_if(aw.begin(),aw.end(),[&](const auto&v){return v.name==prefix+names[i];});require(role!=aw.end()&&role->type==8,"native GDN input projection: Q8_0 role required");raw[i]=role->data;sizes[i]=role->bytes;}
            native_input_projection=std::make_unique<portable_detail::PortableNativeGdnInput>(options.execution.accelerator,weights,raw,sizes,options.native_input_projection_receipt,options.execution.graph_budget-graphs_size,options.native_input_projection_record_first);
            graphs_size+=native_input_projection->bytes();native_gdn->t[4]=native_input_projection->projection(1);
        }
        if(options.native_output_projection_t1){
            require(graphs_size<options.execution.graph_budget,"native GDN output projection: no graph room");
            auto role=std::find_if(aw.begin(),aw.end(),[&](const auto&v){return v.name==prefix+"ssm_out.weight";});
            require(role!=aw.end()&&role->type==8&&role->shape==std::vector<int64_t>({6144,2560}),"native GDN output projection: Q8_0 role required");
            native_output_projection=std::make_unique<portable_detail::PortableNativeQsaWo>(options.execution.accelerator,w.at("ssm_out.weight"),role->data,role->bytes,options.native_output_projection_receipt,options.execution.graph_budget-graphs_size,options.native_output_projection_record_first,portable_detail::PortableNativeQsaWo::InputDomain::gdn_output);
            native_output_candidate=std::make_unique<portable_detail::PortableNativeQsaWo::Candidate>();
            graphs_size+=native_output_projection->bytes();
        }
#endif
        if (options.scratch) { Borrow borrow(*this); }
        else build_graphs();
        if (!production()) {
            state.assign(STATES, 0);
            history.assign(HISTORY, 0);
            candidate_state.resize(STATES);
            candidate_history.resize(HISTORY);
        }
    }
};

PortableGdnLayer::PortableGdnLayer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableGdnLayer::~PortableGdnLayer() = default;
bool PortableGdnLayer::native_hc_compiled() {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return true;
#else
    return false;
#endif
}
std::unique_ptr<PortableGdnLayer> PortableGdnLayer::create(int64_t layer,
                                                           const std::vector<PortableTensorView> &tensors,
                                                           PortableGdnOptions options, std::string &error) {
    error.clear();
    try {
        return std::unique_ptr<PortableGdnLayer>(new PortableGdnLayer(std::make_unique<Impl>(layer, tensors, options)));
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
std::unique_ptr<PortableGdnLayer> PortableGdnLayer::from_gguf(const GgufModel &model, int64_t layer,
                                                              PortableGdnOptions options, std::string &error) {
    error.clear();
    try {
        require(layer >= 0 && layer < 48 && layer % 4 != 3,
                "portable GDN: expected a non-QSA layer in [0,47]");
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto *arch = model.meta().get("general.architecture");
        require(arch && arch->s == "qwen4exp", "portable GDN: requires general.architecture=qwen4exp");
        const auto geometry_error = check_architecture(model.meta());
        require(geometry_error.empty(), "portable GDN: " + geometry_error);
        const auto *ple = model.meta().get("qwen4exp.ple.layers");
        bool configured_ple = false;
        if (ple) {
            require(ple->type == MetaType::ARRAY && ple->items.size() <= 1,
                    "portable GDN: malformed/unsupported PLE layer list");
            for (const auto &v : ple->items) {
                require(v.is_num() && v.u == 1, "portable GDN: supports PLE only at layer 1");
                configured_ple = layer == 1;
            }
        }
        require(!options.ple_table || configured_ple,
                "portable GDN: PLE table supplied for a layer without configured PLE");
        if (configured_ple) {
            options.ple_table = PortablePleTable::from_gguf(model, error);
            require(bool(options.ple_table), error);
        }
        for (auto pair : {std::pair<const char *, uint64_t>{"embedding_length", H},
                          {"block_count", 48},
                          {"ssm.state_size", D},
                          {"ssm.group_count", HK},
                          {"ssm.time_step_rank", HV},
                          {"ssm.inner_size", INNER},
                          {"ssm.conv_kernel", 4},
                          {"hyper_connection.count", HC},
                          {"hyper_connection.low_rank", R},
                          {"expert_feed_forward_length", 640}}) {
            const auto *value = model.meta().get("qwen4exp." + std::string(pair.first));
            require(value && value->is_num() && value->u == pair.second,
                    "portable GDN: missing/mismatched " + std::string(pair.first));
        }
        const auto *experts = model.meta().get("qwen4exp.expert_count");
        const auto *used = model.meta().get("qwen4exp.expert_used_count");
        require(experts && experts->u == 512 && used && used->u == 10, "portable GDN: expected 512 experts/top10");
        std::vector<PortableTensorView> views;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto &t : model.shard(s).tensors()) {
                if (t.name.rfind(prefix, 0) != 0)
                    continue;
                const auto suffix = t.name.substr(prefix.size());
                if (!shapes().count(suffix) && suffix.rfind("ple_", 0) != 0 &&
                    suffix.rfind("hc_ffn_", 0) != 0 &&
                    (suffix.rfind("ffn_", 0) != 0 || suffix.find("_exps") != std::string::npos))
                    continue;
                require(model.in_bounds(t, s), "portable GDN: tensor outside shard");
                PortableTensorView v{
                    t.name, {}, t.type, model.shard(s).tensor_data(t), size_t(tensor_payload_bytes(t))};
                for (auto d : t.shape) {
                    require(d <= INT64_MAX, "portable GDN: dimension overflow");
                    v.shape.push_back(int64_t(d));
                }
                views.push_back(std::move(v));
            }
        return create(layer, views, options, error);
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
bool PortableGdnLayer::reset(std::string &error) {
    error.clear();
    try {
        Impl::Borrow borrow(*impl_,false);
        impl_->state_healthy = false;
        require(!impl_->chunk_active, "portable GDN: chunk execution active");
        std::vector<PortablePlacement> next_placement;
        if(impl_->chunk_host_bound) {
            next_placement=impl_->placement;
            next_placement.erase(std::remove_if(next_placement.begin(),next_placement.end(),[](const auto& p){
                return p.phase != "gdn-attention" && p.phase != "gdn-chunk-recurrence" && p.phase != "gdn-input-projections" && p.phase != "gdn-causal-chunk";
            }),next_placement.end());
            next_placement.insert(next_placement.begin(),impl_->ffn->placements().begin(),impl_->ffn->placements().end());
        }
        impl_->chunk_ffn.reset(); impl_->chunk_host_bound = 0;
        std::vector<float> zero(RESIDUAL, 0);
        require(impl_->ffn->reset(zero, error), error);
        if (impl_->state_buffer) {
            impl_->state_healthy = false;
            ggml_backend_buffer_clear(impl_->state_buffer.get(), 0);
            ggml_backend_synchronize(impl_->options.execution.accelerator);
            impl_->state_healthy = true;
        }
        // Private candidates may contain a rejected step. Keep their capacity:
        // every successful step fully reads and validates them before swapping.
        std::fill(impl_->state.begin(), impl_->state.end(), 0);
        std::fill(impl_->history.begin(), impl_->history.end(), 0);
        std::fill(impl_->ple_history.begin(), impl_->ple_history.end(), 0);
        impl_->previous = {{-1, -1}};
        impl_->last.clear();impl_->pending={};
        impl_->count = 0;
        if(!next_placement.empty()) impl_->placement.swap(next_placement);
        impl_->state_healthy = true;
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}
bool PortableGdnLayer::step(const std::vector<float> &residual, const PortableExpertRows &dispatch,
                            PortableRoute &route, std::string &error, PortableGdnTrace *trace, PortableGdnWork *work) {
    return step(-1, residual, dispatch, route, error, trace, work);
}
bool PortableGdnLayer::step(int32_t token, const std::vector<float> &residual, const PortableExpertRows &dispatch,
                            PortableRoute &route, std::string &error, PortableGdnTrace *trace, PortableGdnWork *work) {
    return step_impl(token,residual,nullptr,nullptr,dispatch,route,error,trace,work);
}
bool PortableGdnLayer::step_pending(const std::vector<float>& residual,const std::vector<float>& previous_output,
    const std::vector<float>& previous_injection,const PortableExpertRows& dispatch,PortableRoute& route,
    std::string& error,PortableGdnTrace* trace,PortableGdnWork* work) {
    return step_impl(-1,residual,&previous_output,&previous_injection,dispatch,route,error,trace,work);
}
bool PortableGdnLayer::step_impl(int32_t token,const std::vector<float>& residual,
    const std::vector<float>* previous_output,const std::vector<float>* previous_injection,
    const PortableExpertRows& dispatch,PortableRoute& route,std::string& error,PortableGdnTrace* trace,PortableGdnWork* work) {
    error.clear();
    route = {};
    if (trace)
        *trace = {};
    try {
        auto &s = *impl_;
        Impl::Borrow borrow(s);
        require(!s.chunk_active, "portable GDN: chunk execution active");
        require(s.state_healthy, "portable GDN: resident state poisoned; reset required");
        require(!s.production() || !trace, "portable GDN: trace requires eager diagnostic policy");
        require(residual.size() == RESIDUAL && bool(dispatch), "portable GDN: residual/callback required");
        require(s.options.native_hc_pending==bool(previous_output&&previous_injection),"native HC pending: explicit matching tuple mode required");
        if(previous_output) {require(previous_output->size()==H&&previous_injection->size()==HC,"native HC pending: tuple shape");finite(*previous_output);finite(*previous_injection);}
        require(s.count < UINT64_MAX, "portable GDN: token counter exhausted");
        auto attended = s.attend(token, residual, {s.state,s.history,s.ple_history,s.previous}, trace != nullptr, work,false,previous_output,previous_injection);
        auto next_residual = std::move(attended.residual);
        auto next_previous = attended.previous;
        auto next_trace = std::move(attended.trace);
        PortableGdnWork disabled;
        auto &timing = work ? *work : disabled;
        PortableRoute candidate;
        {
            portable_detail::ScopedExpertTimer<> ffn_timer(work != nullptr, timing.ffn_ms, timing.ffn_calls);
            require(s.ffn->reset(next_residual, error), error);
            bool ok;
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(s.native_output_projection)ok=s.ffn->step_pending_device(s.native_output_projection->output(),attended.inject,dispatch,candidate,error,trace?&next_trace.ffn_hc:nullptr,s.count);
            else
#endif
            ok=s.options.native_hc_ffn_read ? s.ffn->step_pending(attended.block,attended.inject,dispatch,candidate,error,trace?&next_trace.ffn_hc:nullptr,s.count):s.ffn->step(dispatch,candidate,error);
            require(ok, error);
#ifdef STRATA_PORTABLE_NATIVE_HC
            if(s.native_output_projection)s.native_output_projection->consumed(s.count,s.native_output_input,s.produced_block,*s.native_output_candidate);
#endif
            next_residual = s.ffn->residual(); // allocate before the no-throw public commit
        }
        auto next_pending=s.ffn->pending_write(); // allocate complete producer candidate before state commit
        std::vector<PortablePlacement> next_placement;
        if(s.chunk_host_bound) {
            next_placement=s.placement;
            next_placement.erase(std::remove_if(next_placement.begin(),next_placement.end(),[](const auto& p){
                return p.phase != "gdn-attention" && p.phase != "gdn-chunk-recurrence" && p.phase != "gdn-input-projections" && p.phase != "gdn-causal-chunk";
            }),next_placement.end());
            next_placement.insert(next_placement.begin(),s.ffn->placements().begin(),s.ffn->placements().end());
        }
        s.commit_device(work);
#ifdef STRATA_PORTABLE_NATIVE_HC
        if(s.native_gdn)s.native_gdn->record_commit(s.state_input,s.count,s.native_preparation?s.history_input:nullptr);
#endif
        if (!s.production()) {
            s.ple_history.swap(s.candidate_ple_history);
            s.state.swap(s.candidate_state);
            s.history.swap(s.candidate_history);
        }
        s.previous = next_previous;
        s.last.swap(next_residual);s.pending.swap(next_pending);
        ++s.count;
        if(!next_placement.empty()) s.placement.swap(next_placement);
        route = std::move(candidate);
        if (trace)
            *trace = std::move(next_trace);
        return true;
    } catch (const std::exception &e) {
        if (impl_->production())
            impl_->state_healthy = false;
        error = e.what();
        return false;
    } catch (...) {
        if (impl_->production())
            impl_->state_healthy = false;
        error = "portable GDN: unknown execution failure; reset required";
        return false;
    }
}
bool PortableGdnLayer::prepare_chunk(size_t tokens,std::string& error) {
    error.clear();
    try { require(!impl_->options.native_hc_norm_inject && !impl_->options.native_recurrence_t1 && !impl_->options.defer_ffn_hc_write,"native HC serial: chunk API unsupported"); Impl::Borrow borrow(*impl_); require(!impl_->chunk_active,"portable layer: active chunk"); impl_->prepare(tokens,error); return true; }
    catch(const std::exception& e) { error=e.what(); return false; }
}
bool PortableGdnLayer::run_chunk(const std::vector<int32_t>& tokens, uint64_t position0,
    std::vector<float>& residual_rows, const PortableExpertRowsBatch& dispatch,
    std::vector<PortableRoute>& routes, std::string& error) {
    routes.clear(); error.clear();
    auto& s = *impl_;
    try {
        require(!s.options.native_hc_norm_inject && !s.options.defer_ffn_hc_write,"native HC serial: chunk API unsupported");
        Impl::Borrow borrow(s);
        require(s.state_healthy && !s.chunk_active, "portable GDN: poisoned or active chunk; reset required");
        const size_t T = tokens.size();
        require(T >= 1 && T <= 8 && residual_rows.size() == T*RESIDUAL && bool(dispatch),
                "portable GDN: invalid chunk shape/callback");
        require(position0 == s.count && T <= UINT64_MAX-s.count && T <= UINT64_MAX-s.chunk_work.tokens,
                "portable GDN: noncontiguous/exhausted chunk");
        struct Active { bool& flag; explicit Active(bool& f):flag(f){flag=true;} ~Active(){flag=false;} } active(s.chunk_active);
        s.prepare(T,error);
        auto state = s.state, history = s.history, ple_history = s.ple_history;
        auto previous = s.previous;
        std::vector<float> attended_rows; attended_rows.reserve(T*RESIDUAL);
        const bool projected=s.batch_projections() && T>1;
        const bool causal=s.causal() && T>1;
        require(!causal || (s.causal_calls<UINT64_MAX && s.residual_handoffs<UINT64_MAX && s.causal_commits<UINT64_MAX),
                "portable GDN: exhausted causal chunk counters");
        std::array<std::vector<float>,5> projections;
        if(projected) {
            require(s.projection_calls<UINT64_MAX && T<=UINT64_MAX-s.projection_columns,
                    "portable GDN: exhausted input projection counters");
            finite(residual_rows); write(s.projection_input,residual_rows);
            require(ggml_backend_sched_graph_compute(s.projection_scheduler.get(),s.projection_graph)==GGML_STATUS_SUCCESS,
                    "portable GDN: input projection graph failed");
            ++s.projection_calls; s.projection_columns+=T;
            if(causal) ++s.causal_calls;
            for(size_t i=0;i<5;++i) projections[i]=read(s.projection_outputs[i]);
        }
        if(causal) {
            // Keep the five bulk reads for the existing finite checks. Their
            // device tensors feed recurrence directly; never upload host slices.
            attended_rows=read(s.causal_residuals);
            ++s.residual_handoffs;
        } else for(size_t t=0;t<T;++t) {
            if(projected) for(size_t i=0;i<5;++i) {
                const size_t n=size_t(ggml_nelements(s.projection_inputs[i]));
                ggml_backend_tensor_set(s.projection_inputs[i],projections[i].data()+t*n,0,n*sizeof(float));
            }
            std::vector<float> input(residual_rows.begin()+t*RESIDUAL,residual_rows.begin()+(t+1)*RESIDUAL);
            auto next = s.attend(tokens[t],input,{state,history,ple_history,previous},false,nullptr,projected);
            // The recurrence/conv/PLE cursor advances now, before the next
            // attention. Public diagnostic banks remain unchanged until FFN.
            s.commit_device(nullptr,projected);
            if(!s.production()) {
                state.swap(s.candidate_state); history.swap(s.candidate_history);
                ple_history.swap(s.candidate_ple_history);
            }
            previous = next.previous;
            attended_rows.insert(attended_rows.end(),next.residual.begin(),next.residual.end());
        }
        auto& ffn = T == 1 ? *s.ffn : *s.chunk_ffn;
        const auto before = ffn.work();
        require(ffn.reset(attended_rows,error),error);
        std::vector<PortableRoute> candidate;
        if(T == 1) {
            PortableRoute route;
            require(ffn.step([&](int64_t l,const PortableRoute& r,std::vector<float>& rows,std::string& e) {
                return dispatch(l,{r},rows,e);
            },route,error),error);
            candidate.push_back(std::move(route));
        } else require(ffn.step_batch(dispatch,candidate,error),error);
        auto completed = ffn.residual();
        std::vector<float> last(completed.end()-RESIDUAL,completed.end());
        auto placement = s.placement;
        placement.erase(std::remove_if(placement.begin(),placement.end(),[](const auto& p){
            return p.phase != "gdn-attention" && p.phase != "gdn-chunk-recurrence" && p.phase != "gdn-input-projections" && p.phase != "gdn-causal-chunk";
        }),placement.end());
        placement.insert(placement.begin(),ffn.placements().begin(),ffn.placements().end());
        const auto after = ffn.work();
        auto work = s.chunk_work;
        ++work.chunks; work.tokens += T; work.attention_calls += T;
        work.ffn_pre_calls += after.pre_graph_calls-before.pre_graph_calls;
        work.provider_calls += after.provider_calls-before.provider_calls;
        work.ffn_post_calls += after.post_graph_calls-before.post_graph_calls;
        require(s.state_healthy,"portable GDN: chunk invalidated; reset required");
        if(causal) s.commit_device(nullptr,false,true);
        if(!s.production()) { s.state.swap(state); s.history.swap(history); s.ple_history.swap(ple_history); }
        s.previous = previous; s.last.swap(last); s.count += T;
        s.placement.swap(placement); s.chunk_work = work;
        residual_rows.swap(completed); routes.swap(candidate);
        return true;
    } catch(const std::exception& e) { s.state_healthy=false; error=e.what(); return false; }
      catch(...) { s.state_healthy=false; error="portable GDN: unknown chunk failure; reset required"; return false; }
}
uint64_t PortableGdnLayer::causal_chunk_calls() const { return impl_->causal_calls; }
uint64_t PortableGdnLayer::causal_residual_handoffs() const { return impl_->residual_handoffs; }
uint64_t PortableGdnLayer::causal_state_commits() const { return impl_->causal_commits; }
uint64_t PortableGdnLayer::input_projection_calls() const { return impl_->projection_calls; }
uint64_t PortableGdnLayer::input_projection_columns() const { return impl_->projection_columns; }
PortableLayerChunkWork PortableGdnLayer::chunk_work() const { return impl_->chunk_work; }
bool PortableGdnLayer::poisoned() const { return !impl_->state_healthy; }
PortableGdnPolicy PortableGdnLayer::policy() const { return impl_->options.policy; }
const std::vector<float> &PortableGdnLayer::ple_convolution_history() const {
    require(!impl_->production(), "portable GDN: state inspection requires eager diagnostic policy");
    return impl_->ple_history;
}
const std::array<int32_t, 2> &PortableGdnLayer::token_history() const { return impl_->previous; }
const std::vector<float> &PortableGdnLayer::recurrent_state() const {
    require(!impl_->production(), "portable GDN: state inspection requires eager diagnostic policy");
    return impl_->state;
}
const std::vector<float> &PortableGdnLayer::convolution_history() const {
    require(!impl_->production(), "portable GDN: state inspection requires eager diagnostic policy");
    return impl_->history;
}
const std::vector<float> &PortableGdnLayer::last_residual() const { return impl_->last; }
uint64_t PortableGdnLayer::tokens() const { return impl_->count; }
uint64_t PortableGdnLayer::native_ffn_hc_write_calls() const {return impl_->ffn->native_hc_write_calls();}
uint64_t PortableGdnLayer::native_ffn_hc_pending_calls() const {return impl_->ffn->native_hc_pending_calls();}
const PortableHcPendingWrite& PortableGdnLayer::pending_write() const {return impl_->pending;}
uint64_t PortableGdnLayer::deferred_hc_outputs() const {return impl_->ffn->work().deferred_hc_outputs;}
uint64_t PortableGdnLayer::native_hc_pending_calls() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_hc?impl_->native_hc->pending_calls():0;
#else
    return 0;
#endif
}
uint64_t PortableGdnLayer::native_hc_mix_calls() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_hc?impl_->native_hc->mix_calls():0;
#else
    return 0;
#endif
}
uint64_t PortableGdnLayer::native_hc_calls() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_hc?impl_->native_hc->calls():0;
#else
    return 0;
#endif
}
const std::vector<PortablePlacement> &PortableGdnLayer::placements() const { return impl_->placement; }
size_t PortableGdnLayer::weight_bytes() const { return impl_->weights_size; }
size_t PortableGdnLayer::graph_bytes() const {
    return impl_->shared() ? impl_->native_hc_bytes()+impl_->native_gdn_bytes()+impl_->ffn->graph_bytes()+(impl_->chunk_ffn?impl_->chunk_ffn->graph_bytes():0)
                           : impl_->graphs_size+impl_->chunk_graphs();
}
const PortableGdnScratch* PortableGdnLayer::scratch_owner() const { return impl_->options.scratch.get(); }
size_t PortableGdnLayer::state_bytes() const { return impl_->host_state_bytes()+impl_->chunk_host_bound; }
size_t PortableGdnLayer::native_input_projection_bytes() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_input_projection ? impl_->native_input_projection->bytes() : 0;
#else
    return 0;
#endif
}
uint64_t PortableGdnLayer::native_input_projection_calls(unsigned counter) const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_input_projection ? impl_->native_input_projection->calls(counter) : 0;
#else
    (void)counter;
    return 0;
#endif
}
size_t PortableGdnLayer::native_output_projection_bytes() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_output_projection ? impl_->native_output_projection->bytes() : 0;
#else
    return 0;
#endif
}
uint64_t PortableGdnLayer::native_ple_gate_calls() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_ple_gate ? impl_->native_ple_gate->calls() : 0;
#else
    return 0;
#endif
}
uint64_t PortableGdnLayer::native_ple_key_calls() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_ple_key ? impl_->native_ple_key->calls() : 0;
#else
    return 0;
#endif
}
uint64_t PortableGdnLayer::native_output_projection_calls(unsigned counter) const {
#ifdef STRATA_PORTABLE_NATIVE_HC
    return impl_->native_output_projection ? impl_->native_output_projection->calls(counter) : 0;
#else
    (void)counter;
    return 0;
#endif
}
PortableMemoryUsage PortableGdnLayer::memory_usage() const {
    const auto &s = *impl_;
    auto m = s.ffn->memory_usage();
    if(s.chunk_ffn) { std::string error; require(s.ffn->combined_memory_usage(*s.chunk_ffn,m,error),error); }
    m.device_state += s.device_state_size;
    // Extra metadata context plus the copy API's bounded host staging fallback.
    if (s.state_buffer)
        m.host_extra_bound += (4u << 20) + STATES * sizeof(float);
    m.device_graph += s.native_hc_bytes()+s.native_gdn_bytes();
    m.device_weights += s.attention_device_weights;
    m.host_weights = s.weights_size - m.device_weights;
    if(!s.shared()) {
        m.host_graph += ggml_backend_sched_get_buffer_size(s.scheduler.get(), s.cpu.get());
        if (s.options.execution.accelerator && s.options.execution.request_accelerator)
            m.device_graph += ggml_backend_sched_get_buffer_size(s.scheduler.get(), s.options.execution.accelerator);
        for(auto* sched:{s.tail_scheduler.get(),s.projection_scheduler.get()}) if(sched) {
            m.host_graph+=ggml_backend_sched_get_buffer_size(sched,s.cpu.get());
            if(s.options.execution.accelerator && s.options.execution.request_accelerator)
                m.device_graph+=ggml_backend_sched_get_buffer_size(sched,s.options.execution.accelerator);
        }
        if(s.tail_graph) m.host_extra_bound+=ggml_graph_plan(s.tail_graph,s.options.execution.threads,nullptr).work_size;
        if(s.projection_ctx) m.host_extra_bound+=(4u<<20)+ggml_graph_plan(s.projection_graph,s.options.execution.threads,nullptr).work_size;
    }
    if(s.native_prefix_graph)m.host_extra_bound+=ggml_graph_plan(s.native_prefix_graph,s.options.execution.threads,nullptr).work_size;
    const size_t buffers = m.host_graph + m.device_graph;
    const size_t allowance = graph_bytes() > buffers ? graph_bytes() - buffers : 0;
    m.host_extra_bound += state_bytes() + (8u << 20) + allowance +
                          (s.shared()?0:ggml_graph_plan(s.graph, s.options.execution.threads, nullptr).work_size);
    if (s.options.execution.request_accelerator) {
        m.device_extra_bound += s.preparation_credit()+allowance + (8u << 20);
        portable_detail::account_matrix_conversion(m,s.options.execution.accelerator,size_t(H*CHANNELS)*sizeof(float));
        // Strided F32 inputs use the backend's distinct persistent prealloc_y
        // packing bank. Price the largest permitted shape even after replacing
        // it with a smaller graph; backend scratch retains its high-water size.
        // Padded graph tensors are already included in scheduler graph bytes.
        if(s.batch_projections()) m.device_extra_bound+=8*RESIDUAL*sizeof(float);
    }
    return m;
}
} // namespace strata::core

namespace strata::core {
uint64_t PortableGdnLayer::native_recurrence_calls() const {
#ifdef STRATA_PORTABLE_NATIVE_HC
 return impl_->native_gdn?impl_->native_gdn->calls:0;
#else
 return 0;
#endif
}
}

namespace strata::core {
uint64_t PortableGdnLayer::native_convolution_calls()const {
#ifdef STRATA_PORTABLE_NATIVE_HC
 return impl_->native_preparation?impl_->native_preparation->conv_calls():0;
#else
 return 0;
#endif
}
uint64_t PortableGdnLayer::native_ab_calls()const {
#ifdef STRATA_PORTABLE_NATIVE_HC
 return impl_->native_preparation?impl_->native_preparation->ab_calls():0;
#else
 return 0;
#endif
}
}
