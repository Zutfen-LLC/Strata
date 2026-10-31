#pragma once
#include "strata/core/portable_model.hpp"
#include "native_hc_layout.hpp"
#include <stdexcept>
namespace strata::core::portable_detail {
// Argument/mode validation only. No backend query, model access or allocation;
// never a physical capability, metadata or memory-admission success receipt.
inline void validate_model_modes(const PortableModelOptions& opts,size_t selected_accelerators) {
    const auto require=[](bool ok,const char* error){if(!ok)throw std::runtime_error(error);};
        require(!opts.expert_native_profile.empty() || (!opts.expert_native_primary_slots && !opts.expert_native_helper_slots &&
            opts.expert_native_plan_prefix.empty()),"native capacity options require a profile");
        require(opts.expert_native_q8_observation.empty() || (!opts.expert_devices.empty() && !opts.allow_cpu_fallback),
                "native Q8 observation requires explicit expert plans and strict accelerator execution");
        require(!opts.native_hc_norm_inject_layer0 || (!opts.prompt_chunk_size && !opts.gdn_production && !opts.gdn_batch_input_projections &&
            selected_accelerators != 0 && !opts.allow_cpu_fallback && (opts.gdn_resident_weights || opts.dense_resident_weights) &&
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
    if(opts.native_hc_norm_inject_layer0) {
        require(PortableGdnLayer::native_hc_compiled(),"native HC serial: strict-FMA native-stage build required");
        native_hc_environment();
    }
}
}
