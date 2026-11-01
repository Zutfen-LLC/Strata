#pragma once
#include "strata/core/portable_model.hpp"
#include "native_hc_layout.hpp"
#include <stdexcept>
#include <cstdlib>
namespace strata::core::portable_detail {
// Shared model-to-layer policy seam. Explicit layer-0 residency never changes
// attention-only GDN residency, another layer, the head, or dense topology.
inline bool model_resident_dense(const PortableModelOptions& opts,size_t index,bool output) {
    return opts.dense_resident_weights || (!output && index==0 && opts.layer0_resident_weights);
}
inline size_t model_hc_host_bound(const PortableModelOptions& opts,size_t layer) {
    if(opts.native_hc_pending_layer1_2 && layer==1)return PortableFfnOptions::deferred_hc_host_bound;
    if(opts.native_hc_pending_layer1_2 && layer==2)return PortableGdnOptions::native_hc_pending_host_bound;
    if(layer || !opts.native_hc_norm_inject_layer0)return 0;
    return (opts.native_hc_mix_layer0 ? PortableGdnOptions::native_hc_read_host_bound : PortableGdnOptions::native_hc_host_bound)+
        (opts.native_hc_ffn_read_layer0 ? PortableFfnOptions::native_hc_pending_host_bound : 0)+
        (opts.native_hc_ffn_write_layer0 ? PortableFfnOptions::native_hc_write_host_bound : 0);
}
inline void model_gdn_hc_options(const PortableModelOptions& opts,size_t layer,PortableGdnOptions& o) {
    o.native_hc_norm_inject=opts.native_hc_norm_inject_layer0 && layer==0;
    o.native_hc_mix=opts.native_hc_mix_layer0 && layer==0;
    o.native_hc_ffn_read=opts.native_hc_ffn_read_layer0 && layer==0;
    o.native_hc_ffn_write=opts.native_hc_ffn_write_layer0 && layer==0;
    if(o.native_hc_norm_inject)o.native_hc_receipt_prefix=opts.native_hc_receipt_prefix;
    if(opts.native_hc_pending_layer1_2) {
        o.defer_ffn_hc_write=layer==1;
        if(layer==2) {
            o.native_hc_pending=true;o.native_hc_mix=true;o.native_hc_norm_inject=true;
            o.native_hc_receipt_prefix=opts.native_hc_pending_receipt_prefix;
        }
    }
}
// Argument/mode validation only. No backend query, model access or allocation;
// never a physical capability, metadata or memory-admission success receipt.
inline void validate_model_modes(const PortableModelOptions& opts,size_t selected_accelerators) {
    const auto require=[](bool ok,const char* error){if(!ok)throw std::runtime_error(error);};
        require(!opts.expert_native_profile.empty() || (!opts.expert_native_primary_slots && !opts.expert_native_helper_slots &&
            opts.expert_native_plan_prefix.empty()),"native capacity options require a profile");
        require(opts.expert_native_q8_observation.empty() || (!opts.expert_devices.empty() && !opts.allow_cpu_fallback),
                "native Q8 observation requires explicit expert plans and strict accelerator execution");
        require(!opts.native_hc_pending_layer1_2 || (!opts.prompt_chunk_size && !opts.gdn_production &&
            !opts.gdn_batch_input_projections && !opts.gdn_causal_chunk && selected_accelerators && !opts.allow_cpu_fallback &&
            (opts.gdn_resident_weights || opts.dense_resident_weights) && !opts.native_hc_pending_receipt_prefix.empty()),
            "native HC layer1-2 requires serial eager strict accelerator, resident attention and receipt prefix");
        const char* boundary_observer=std::getenv("STRATA_LAYER0_OBSERVER");
        require(!opts.native_hc_pending_layer1_2 || !boundary_observer || !*boundary_observer,"native HC layer1-2: legacy observer unsupported");
        require(!opts.native_hc_ffn_write_layer0||opts.native_hc_ffn_read_layer0,"native FFN final write requires pending FFN read");
        require(!opts.native_hc_ffn_read_layer0 || (opts.native_hc_mix_layer0 &&
            (opts.layer0_resident_weights || opts.dense_resident_weights)),
            "native FFN HC layer0 requires composed attention and explicit layer0/dense resident weights");
        const char* observer=std::getenv("STRATA_LAYER0_OBSERVER");
        require(!opts.native_hc_ffn_read_layer0 || !observer || !*observer,
            "native FFN HC layer0: legacy observer lacks pending tuple; unsupported");
        require(!opts.native_hc_mix_layer0 || opts.native_hc_norm_inject_layer0,"native HC read layer0 requires native norm/injection");
        require(!opts.native_hc_norm_inject_layer0 || (!opts.prompt_chunk_size && !opts.gdn_production && !opts.gdn_batch_input_projections &&
            selected_accelerators != 0 && !opts.allow_cpu_fallback && (opts.gdn_resident_weights || opts.dense_resident_weights || opts.layer0_resident_weights) &&
            !opts.native_hc_receipt_prefix.empty()),"native HC layer0 requires serial eager strict accelerator, resident weights and receipt prefix");
        require(!opts.native_shared || (selected_accelerators != 0 && !opts.allow_cpu_fallback && !opts.native_shared_receipt_prefix.empty()),
                "native shared requires explicit accelerators, strict execution and private receipt prefix");
        require(!opts.native_router || (selected_accelerators != 0 && !opts.allow_cpu_fallback && !opts.native_router_receipt_prefix.empty()),
                "native router requires explicit accelerators, strict execution and private receipt prefix");
        require(!opts.native_moe_combine || (!opts.expert_native_q8_observation.empty() && selected_accelerators != 0),
                "native MoE combine requires native Q8 observation and explicit accelerators");
        require(opts.prompt_chunk_size==0 || (opts.prompt_chunk_size>=2&&opts.prompt_chunk_size<=8),
                "portable model: prompt chunk size must be zero or [2,8]");
        require(!opts.gdn_batch_input_projections || opts.prompt_chunk_size,
                "portable model: batched GDN input projections require prompt chunks");
        require(!opts.gdn_causal_chunk || (opts.prompt_chunk_size && opts.gdn_batch_input_projections && opts.gdn_production),
                "portable model: causal GDN chunks require prompt chunks, batched projections and production");

    require(opts.context>=1 && opts.context<=256 && opts.threads>=1 && opts.threads<=256,
            "portable model: context [1,256]/thread limits");
    require(opts.expert_workers>=1 && opts.expert_workers<=256,"portable model: expert workers must be in [1,256]");
    require(!opts.qsa_production || (opts.qsa_reuse_attention && selected_accelerators && !opts.allow_cpu_fallback),
            "portable model: QSA production requires reuse and strict accelerators");
    require(!opts.gdn_production || (opts.gdn_resident_state && selected_accelerators && !opts.allow_cpu_fallback),
            "portable model: GDN production requires resident state and strict accelerators");
    if(opts.native_hc_norm_inject_layer0 || opts.native_hc_pending_layer1_2) {
        require(PortableGdnLayer::native_hc_compiled(),"native HC serial: strict-FMA native-stage build required");
        native_hc_environment();
    }
}
}
