#include "strata/core/portable_model.hpp"
#include "portable_graph.hpp"
#include "portable_router_work.hpp"
#include "portable_shared_work.hpp"
#include "portable_memory.hpp"
#include "portable_native_capacity.hpp"
#include "strata/core/layer0_observer.hpp"
#include "strata/core/expert_profile.hpp"
#include "portable_topology.hpp"
#include "portable_cpu_experts.hpp"
#include "portable_expert_timer.hpp"
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
#include "portable_local_experts.hpp"
#include "strata/worker/vulkan_executor.hpp"
#endif
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/worker/local_source.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace strata::core {
using portable_detail::require;
namespace {
constexpr size_t H = 2560, HC = 4, FF = 640, NE = 512, K = 10, L = 48;
void finite(const std::vector<float> &x, const char *what) {
    for (float v : x)
        require(std::isfinite(v), std::string("portable model: nonfinite ") + what);
}
PortableTensorView view(const GgufModel &m, const std::string &name) {
    size_t shard = 0;
    const auto *t = m.find(name, &shard);
    require(t && m.in_bounds(*t, shard), "portable model: missing/out-of-bounds " + name);
    PortableTensorView v{name, {}, t->type, m.shard(shard).tensor_data(*t), size_t(tensor_payload_bytes(*t))};
    for (uint64_t d : t->shape) {
        require(d <= INT64_MAX, "portable model: dimension overflow");
        v.shape.push_back(int64_t(d));
    }
    return v;
}
struct Layer {
    std::unique_ptr<PortableGdnLayer> gdn;
    std::unique_ptr<PortableQsaLayer> qsa;
    bool reset(std::string &error) { return gdn ? gdn->reset(error) : qsa->reset(error); }
    bool step(int32_t token, uint64_t pos, std::vector<float> &residual, const PortableExpertRows &experts,
              std::string &error, PortableGdnWork *work) {
        PortableRoute route;
        const bool ok =
            gdn ? gdn->step(token, residual, experts, route, error, nullptr, work) : qsa->step(pos, residual, experts, route, error);
        if (ok)
            residual = gdn ? gdn->last_residual() : qsa->last_residual();
        return ok;
    }
    bool prepare(size_t T,std::string& error) { return gdn?gdn->prepare_chunk(T,error):qsa->prepare_chunk(T,error); }
    bool chunk(const std::vector<int32_t>& tokens,uint64_t pos,std::vector<float>& rows,
               const PortableExpertRowsBatch& experts,std::string& error) {
        std::vector<PortableRoute> routes;
        return gdn?gdn->run_chunk(tokens,pos,rows,experts,routes,error):qsa->run_chunk(tokens,pos,rows,experts,routes,error);
    }
    PortableMemoryUsage memory() const { return gdn ? gdn->memory_usage() : qsa->memory_usage(); }
    const std::vector<PortablePlacement> &placements() const { return gdn ? gdn->placements() : qsa->placements(); }
};
void inventory(const GgufModel &m) {
    const std::set<std::string> globals{"token_embd.weight",     "output.weight",       "output_hc_norm.weight",
                                        "output_hc_down.weight", "output_hc_up.weight", "per_layer_token_embd.weight"};
    const std::set<std::string> dense{
        "hc_attn_norm.weight",   "hc_attn_down.weight",       "hc_attn_up.weight",     "hc_attn_inject.weight",
        "hc_ffn_norm.weight",    "hc_ffn_down.weight",        "hc_ffn_up.weight",      "hc_ffn_inject.weight",
        "ffn_gate_inp.weight",   "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
        "ffn_down_shexp.weight", "ffn_gate_exps.weight",      "ffn_up_exps.weight",    "ffn_down_exps.weight"};
    const std::set<std::string> gdn{"attn_qkv.weight",   "attn_gate.weight", "ssm_a",
                                    "ssm_dt.bias",       "ssm_alpha.weight", "ssm_beta.weight",
                                    "ssm_conv1d.weight", "ssm_norm.weight",  "ssm_out.weight"};
    const std::set<std::string> qsa{"attn_q.weight",         "attn_k.weight",         "attn_v.weight",
                                    "attn_output.weight",    "attn_q_norm.weight",    "attn_k_norm.weight",
                                    "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight",
                                    "indexer.k_norm.weight"};
    const std::set<std::string> ple{"ple_key.weight",        "ple_value.weight",     "ple_norm_key.weight",
                                    "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"};
    for (size_t s = 0; s < m.size(); ++s) {
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (const auto &t : m.shard(s).tensors()) {
            require(m.in_bounds(t, s), "portable model: unknown encoding/truncated tensor " + t.name);
            ranges.emplace_back(t.offset, t.offset + tensor_payload_bytes(t));
            if (globals.count(t.name))
                continue;
            // Unused MTP has no session state or execution. Validate its byte
            // extent/encoding, but do not read or load its payload.
            if (t.name.rfind("nextn.", 0) == 0 || t.name.rfind("mtp.", 0) == 0 || t.name.rfind("blk.48.", 0) == 0)
                continue;
            require(t.name.rfind("blk.", 0) == 0, "portable model: unsupported global tensor " + t.name);
            const size_t end = t.name.find('.', 4);
            require(end != std::string::npos, "portable model: malformed block tensor");
            const auto number = t.name.substr(4, end - 4);
            const int l = std::stoi(number);
            require(l >= 0 && l < int(L) && std::to_string(l) == number, "portable model: invalid block index");
            const auto key = t.name.substr(end + 1);
            require(dense.count(key) || (l % 4 == 3 ? qsa.count(key) : gdn.count(key)) || (l == 1 && ple.count(key)),
                    "portable model: unsupported block tensor " + t.name);
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i)
            require(ranges[i - 1].second <= ranges[i].first, "portable model: overlapping tensor payloads");
    }
}
} // namespace

struct PortableModelSession::Impl {
    PortableModelOptions opts;
    std::unique_ptr<Layer0Observer> observer;
    GgufModel model; // before all consumers; destroyed last
    std::unique_ptr<portable_detail::CpuExperts> experts;
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
    std::unique_ptr<portable_detail::LocalExperts> local_experts;
#endif
    PortableTensorView embedding;
    std::map<ggml_backend_t, std::shared_ptr<PortableQsaAttention>> qsa_attention;
    std::map<ggml_backend_t, std::shared_ptr<PortableGdnScratch>> gdn_scratch;
    std::vector<Layer> layers;
    std::unique_ptr<PortableHead> head;
    PortableModelReport report;
    std::vector<PortablePlacement> placement;
    uint64_t pos = 0;
    int32_t eos_id = -1;
    bool poison = false, eos_seen = false;
    double provider_ms = 0;
    uint64_t provider_calls = 0;
    PortableModelWork model_work;
    size_t vocabulary = 0;
    Impl(const std::vector<std::string> &shards, const std::string &manifest, PortableModelOptions options)
        : opts(std::move(options)), model(shards) {
        if(Layer0Observer::requested()) {
            require(!opts.prompt_chunk_size && !opts.gdn_production,"layer0 observer requires existing serial eager diagnostic policy");
            require(opts.host_cap>=Layer0Observer::host_bound,"layer0 observer host cap exceeded");
            observer=std::make_unique<Layer0Observer>();
            std::clog<<"layer0_observer native_snapshot_bytes=0 host_bound_bytes="<<Layer0Observer::host_bound<<'\n';
        }
        require(!opts.expert_native_profile.empty() || (!opts.expert_native_primary_slots && !opts.expert_native_helper_slots &&
            opts.expert_native_plan_prefix.empty()),"native capacity options require a profile");
        require(opts.expert_native_q8_observation.empty() || (!opts.expert_devices.empty() && !opts.allow_cpu_fallback),
                "native Q8 observation requires explicit expert plans and strict accelerator execution");
        require(!opts.native_shared || (!opts.accelerators.empty() && !opts.allow_cpu_fallback && !opts.native_shared_receipt_prefix.empty()),
                "native shared requires explicit accelerators, strict execution and private receipt prefix");
        require(!opts.native_router || (!opts.accelerators.empty() && !opts.allow_cpu_fallback && !opts.native_router_receipt_prefix.empty()),
                "native router requires explicit accelerators, strict execution and private receipt prefix");
        require(!opts.native_moe_combine || (!opts.expert_native_q8_observation.empty() && !opts.accelerators.empty()),
                "native MoE combine requires native Q8 observation and explicit accelerators");
        require(opts.prompt_chunk_size==0 || (opts.prompt_chunk_size>=2&&opts.prompt_chunk_size<=8),
                "portable model: prompt chunk size must be zero or [2,8]");
        require(!opts.gdn_batch_input_projections || opts.prompt_chunk_size,
                "portable model: batched GDN input projections require prompt chunks");
        require(!opts.gdn_causal_chunk || (opts.prompt_chunk_size && opts.gdn_batch_input_projections && opts.gdn_production),
                "portable model: causal GDN chunks require prompt chunks, batched projections and production");
        if(opts.prompt_chunk_size) opts.prompt_head_elision=true;
        report.chunk_memory_is_shape_envelope=opts.prompt_chunk_size!=0;
        require(shards.size() == 6 && model.size() == 6, "portable model: this fleet path requires all six shards");
        require(opts.context >= 1 && opts.context <= 256 && opts.threads >= 1 && opts.threads <= 256,
                "portable model: context [1,256]/thread limits");
        require(opts.accelerators.size() <= 2 && opts.device_caps.size() == opts.accelerators.size(),
                "portable model: supply a separate cap for each of at most two accelerators");
        require((opts.device_policy_caps.empty() || opts.device_policy_caps.size()==opts.accelerators.size()) &&
                (opts.device_free_reserves.empty() || opts.device_free_reserves.size()==opts.accelerators.size()),
                "portable model: diagnostic budget evidence must follow accelerator slots");
        require(!opts.qsa_production || (opts.qsa_reuse_attention && !opts.accelerators.empty() && !opts.allow_cpu_fallback),
                "portable model: QSA production requires reuse and strict accelerators");
        require(!opts.gdn_production || (opts.gdn_resident_state && !opts.accelerators.empty() &&
                    !opts.allow_cpu_fallback),
                "portable model: GDN production requires resident state and strict accelerators");
        portable_detail::validate_dense_topology(opts);
        std::set<ggml_backend_dev_t> devices;
        for (auto *a : opts.accelerators) {
            require(a != nullptr, "portable model: missing requested accelerator");
            const auto device = ggml_backend_get_device(a);
            require(device != nullptr, "portable model: requested accelerator has no device identity");
            require(devices.insert(device).second, "portable model: duplicate physical accelerator device");
        }
        require(check_architecture(model.meta()).empty(), "portable model: unsupported architecture");
        require(opts.expert_workers >= 1 && opts.expert_workers <= 256,
                "portable model: expert workers must be in [1,256]");
        const size_t expert_host_bound =
            10 * (3 * (H * FF / 32 * 34)) + (32u << 20) + size_t(opts.expert_workers + 1) * (9u << 20);
        require(expert_host_bound <= opts.host_cap, "portable model: host cap cannot admit expert staging/pool");
        inventory(model);
        embedding = view(model, "token_embd.weight");
        require(embedding.type == 8 && embedding.shape.size() == 2 && embedding.shape[0] == H &&
                    embedding.shape[1] > 0 && embedding.shape[1] <= 1048576,
                "portable model: expected Q8 token embedding [2560,vocabulary]");
        vocabulary = size_t(embedding.shape[1]);
        auto token_id = [&](const char *key) {
            const auto *value = model.meta().get(key);
            require(value && value->is_num() && value->type != MetaType::F32 && value->type != MetaType::F64 &&
                        value->type != MetaType::BOOL && value->u < vocabulary,
                    "portable model: " + std::string(key) + " requires an in-vocabulary integer token ID");
            return int32_t(value->u);
        };
        eos_id = token_id("tokenizer.ggml.eos_token_id");
        const auto *ple_layers = model.meta().get("qwen4exp.ple.layers");
        if (ple_layers && !ple_layers->items.empty()) {
            // PLE's predecessor-history boundary is independent of the
            // tokenizer's generated end-of-turn token. The fleet has 248044
            // and 248046 respectively; PortablePleTable validates its own config.
            (void)token_id("qwen4exp.ple.eos_token_id");
        }
        for (size_t s = 0; s < model.size(); ++s)
            report.mapped_bytes += model.shard(s).file_size();
        experts = std::make_unique<portable_detail::CpuExperts>(model, manifest, opts);
        report.expert_staging = experts->bytes();
        report.retained_manifest_identity_a = experts->identity().identity.a;
        report.retained_manifest_identity_b = experts->identity().identity.b;
        report.identity_evidence =
            std::string("SWLOCAL1 expert ranges/file sizes matched; selected expert digests verified ") +
            (opts.expert_dispatch_oracle ? "on every dispatch (staging oracle); " :
             "on first use after source metadata changes (immutable sources required); ") +
            "six GGUF headers/inventory/bounds checked; no new full-file hashes";
        // Pool-owned split scratch is capped at 96 experts, MAXT=8. Include
        // its allocation, per-worker scratch/stacks and bounded manifest/index.
        report.host_state_bound = (32u << 20) + size_t(opts.expert_workers + 1) * (9u << 20);
        if(observer)report.host_state_bound+=Layer0Observer::host_bound;
        if(opts.prompt_chunk_size) report.host_state_bound+=(4u<<20)+2*8*H*HC*sizeof(float);
        report.device_admission.assign(opts.accelerators.size(), 0);
        report.device_scheduler.assign(opts.accelerators.size(), 0);
        report.device_state.assign(opts.accelerators.size(), 0);
        report.device_weights.assign(opts.accelerators.size(), 0);
        report.device_workspace_bound.assign(opts.accelerators.size(), 0);
        report.device_token_temporary_bound.assign(opts.accelerators.size(), 0);
        report.device_shared_conversion_bound.assign(opts.accelerators.size(), 0);
        auto execution = [&](size_t index, bool output) {
            PortableFfnOptions o;
            o.threads = opts.threads;
            if(index==0 && !output)o.layer0_observer=observer.get();
            o.native_shared=!output && opts.native_shared;
            if(o.native_shared)o.native_shared_receipt_prefix=opts.native_shared_receipt_prefix;
            o.native_router = !output && opts.native_router;
            if(o.native_router)o.native_router_receipt_prefix=opts.native_router_receipt_prefix;
            o.native_moe_combine = !output && opts.native_moe_combine;
            if(o.native_moe_combine) o.native_moe_combine_receipt_prefix = opts.expert_native_q8_observation + ".combine";
            o.allow_cpu_fallback = opts.allow_cpu_fallback;
            o.resident_dense_weights = opts.dense_resident_weights;
            o.weight_budget = output ? opts.head_weight_cap : opts.layer_weight_cap;
            o.graph_budget = output ? opts.head_graph_cap : opts.layer_graph_cap;
            if (!opts.accelerators.empty()) {
                o.accelerator = opts.accelerators[device(index)];
                o.request_accelerator = true;
            }
            return o;
        };
        std::string error;
        // Native SessionState::qsa_bufs: one serial scratch owner per physical
        // dense backend. Reserve its whole shape-growth cap before any layer.
        for (size_t l = 3; l < L; l += 4) {
            const auto execution_options = execution(l, false);
            auto &owner = qsa_attention[execution_options.accelerator];
            if (!owner) {
                preflight(device(l), l, "qsa-shared-attention", 0, opts.layer_graph_cap, PortableQsaAttention::host_metadata_bound());
                owner = PortableQsaAttention::create(execution_options, error);
                require(bool(owner), error);
                account(owner->memory_usage(), device(l));
            }
        }
        for(size_t l=0;l<L;++l) if(l%4!=3) {
            const auto execution_options=execution(l,false);
            auto& owner=gdn_scratch[execution_options.accelerator];
            if(!owner) {
                preflight(device(l),l,"gdn-shared-scratch",0,opts.layer_graph_cap,PortableGdnScratch::host_metadata_bound());
                owner=PortableGdnScratch::create(execution_options,error);
                require(bool(owner),error);
                account(owner->memory_usage(),device(l));
            }
        }
        layers.reserve(L);
        for (size_t l = 0; l < L; ++l) {
            // Fixed schedulers obey layer_graph_cap. Native backend-private
            // workspaces are admitted separately from those scheduler buffers.
            const size_t chunk_private = opts.prompt_chunk_size ? (8u<<20)+size_t(H*HC*320)*sizeof(float) : 0;
            const size_t private_workspace = (l % 4 == 3 ? (24u<<20) : (136u<<20))+chunk_private;
            require(!opts.prompt_chunk_size || opts.layer_graph_cap<=SIZE_MAX/2,"portable model: chunk graph bound overflow");
            const size_t graph_preflight=opts.layer_graph_cap*(opts.prompt_chunk_size?2:1);
            require(graph_preflight <= SIZE_MAX - private_workspace, "portable model: graph cap overflow");
            const bool resident = !opts.accelerators.empty() &&
                                  (opts.dense_resident_weights || (opts.gdn_resident_weights && l % 4 != 3));
            const size_t state_bound = portable_detail::layer_state_preflight_bytes(
                opts, l, opts.accelerators.empty() ? nullptr : opts.accelerators[device(l)]);
            require(state_bound <= SIZE_MAX - graph_preflight - private_workspace,
                    "portable model: resident state preflight overflow");
            const size_t resident_bound = resident ? opts.layer_weight_cap : 0;
            require(resident_bound <= SIZE_MAX - graph_preflight - private_workspace - state_bound,
                    "portable model: resident weight preflight overflow");
            // Pre-admit the complete shape envelope BEFORE any layer/chunk
            // allocation. A shape keeps one active chunk pair. Fieldwise
            // maxima may combine graph and unused-workspace high waters, so
            // reserve two graph caps plus the explicit FFN private allowance.
            preflight(device(l),l,l%4==3?"qsa":"gdn",opts.layer_weight_cap,
                      graph_preflight+private_workspace+resident_bound+state_bound,128u<<20);
            Layer layer;
            if (l % 4 == 3) {
                PortableQsaOptions o;
                o.execution = execution(l, false);
                o.capacity = opts.context;
                o.reuse_attention = true; // session ownership; standalone legacy API is unchanged
                o.production = opts.qsa_production;
                o.attention = qsa_attention.at(o.execution.accelerator);
                layer.qsa = PortableQsaLayer::from_gguf(model, int64_t(l), o, error);
                require(bool(layer.qsa), "portable model: layer " + std::to_string(l) + ": " + error);
            } else {
                PortableGdnOptions o;
                o.causal_chunk = opts.gdn_causal_chunk;
                o.batch_input_projections = opts.gdn_batch_input_projections;
                o.resident_state = opts.gdn_resident_state;
                o.policy = opts.gdn_production ? PortableGdnPolicy::device_production : PortableGdnPolicy::eager_diagnostic;
                o.resident_attention_weights = opts.gdn_resident_weights;
                o.execution = execution(l, false);
                o.scratch = gdn_scratch.at(o.execution.accelerator);
                layer.gdn = PortableGdnLayer::from_gguf(model, int64_t(l), o, error);
                require(bool(layer.gdn), "portable model: layer " + std::to_string(l) + ": " + error);
            }
            auto memory=layer.memory();
            if(opts.prompt_chunk_size) {
                auto include=[&](const PortableMemoryUsage& candidate) {
                    for(auto member:{&PortableMemoryUsage::host_weights,&PortableMemoryUsage::device_weights,
                        &PortableMemoryUsage::host_graph,&PortableMemoryUsage::device_graph,
                        &PortableMemoryUsage::host_extra_bound,&PortableMemoryUsage::device_extra_bound,
                        &PortableMemoryUsage::device_token_temporary_bound,&PortableMemoryUsage::device_shared_conversion_bound,
                        &PortableMemoryUsage::device_state}) memory.*member=std::max(memory.*member,candidate.*member);
                };
                for(size_t T=1;T<=opts.prompt_chunk_size;++T) {
                    // Recheck the reserved module envelope before every shape
                    // allocation; no account-after-allocation admission gate.
                    preflight(device(l),l,l%4==3?"qsa-chunk-shape":"gdn-chunk-shape",opts.layer_weight_cap,
                              graph_preflight+private_workspace+resident_bound+state_bound,128u<<20);
                    const bool prepared=layer.prepare(T,error);
                    require(prepared,"portable model: chunk preflight layer "+std::to_string(l)+": "+error);
                    include(layer.memory());
                }
                const size_t device_total=memory.device_graph+memory.device_extra_bound+
                    memory.device_token_temporary_bound+memory.device_shared_conversion_bound+memory.device_state+memory.device_weights;
                require(device_total<=graph_preflight+private_workspace+resident_bound+state_bound,
                        "portable model: chunk memory exceeds reserved module envelope");
                require(memory.host_weights+memory.host_graph+memory.host_extra_bound<=
                        opts.layer_weight_cap+graph_preflight+private_workspace+resident_bound+state_bound+(128u<<20),
                        "portable model: chunk host memory exceeds reserved module envelope");
            }
            account(memory, device(l));
            append_placements(layer.placements(), int64_t(l));
            layers.push_back(std::move(layer));
        }
        // Initial admission charged the full shared cap with zero buffers.
        // Reclassify actual primed allocations without changing total admission.
        report.qsa_attention_resources = qsa_attention.size();
        for (const auto &[backend, owner] : qsa_attention) {
            const auto memory = owner->memory_usage();
            host_graph += memory.host_graph;
            report.host_scheduler = host_graph;
            report.host_state_bound -= memory.host_graph;
            report.graph_admission_bound += memory.host_graph;
            report.qsa_attention_host_scheduler += memory.host_graph;
            if (backend) {
                const size_t die = size_t(std::find(opts.accelerators.begin(), opts.accelerators.end(), backend) - opts.accelerators.begin());
                report.device_scheduler[die] += memory.device_graph;
                report.device_workspace_bound[die] -= memory.device_graph;
                report.qsa_attention_device_scheduler += memory.device_graph;
            }
        }
        report.gdn_scratch_resources=gdn_scratch.size();
        for(const auto& [backend,owner]:gdn_scratch) {
            const auto memory=owner->memory_usage();
            host_graph+=memory.host_graph;
            report.host_scheduler=host_graph;
            report.host_state_bound-=memory.host_graph;
            report.graph_admission_bound+=memory.host_graph;
            report.gdn_scratch_host_scheduler+=memory.host_graph;
            if(backend) {
                const size_t die=size_t(std::find(opts.accelerators.begin(),opts.accelerators.end(),backend)-opts.accelerators.begin());
                report.device_scheduler[die]+=memory.device_graph;
                report.device_workspace_bound[die]-=memory.device_graph;
                report.gdn_scratch_device_scheduler+=memory.device_graph;
            }
        }
        std::vector<PortableTensorView> hw;
        for (const char *name :
             {"output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", "output.weight"})
            hw.push_back(view(model, name));
        require(hw[1].type == 8 && hw[2].type == 8 && hw[3].type == 8,
                "portable model: fleet output HC/head must be Q8_0");
        const size_t head_resident_bound = opts.dense_resident_weights && !opts.accelerators.empty()
                                               ? opts.head_weight_cap : 0;
        require(head_resident_bound <= SIZE_MAX - opts.head_graph_cap,
                "portable model: resident head preflight overflow");
        const PortableHeadGeometry head_geometry{H, HC, 320, int64_t(vocabulary), 1e-6f};
        const size_t aligned_head_weights = head_resident_bound ? PortableHead::weight_allocation_size(
            opts.accelerators[device(L-1)], head_geometry, hw) : 0;
        require(aligned_head_weights <= head_resident_bound, "portable model: aligned head weights exceed configured cap");
        if (head_resident_bound)
            std::clog << "head_weight_preflight slot=" << device(L-1)
                      << " backend=" << ggml_backend_name(opts.accelerators[device(L-1)])
                      << " weight_cap_bytes=" << head_resident_bound << " aligned_weight_bytes=" << aligned_head_weights
                      << " graph_cap_bytes=" << opts.head_graph_cap
                      << " next_estimated_bytes=" << opts.head_graph_cap + aligned_head_weights << '\n';
        // Keep the full graph/workspace cap and the previous host envelope.
        // Only replace the device's policy weight ceiling with allocator sizing.
        preflight(device(L - 1), L, "output-head", opts.head_weight_cap,
                  opts.head_graph_cap + head_resident_bound, 64u << 20, opts.head_graph_cap + aligned_head_weights);
        head = PortableHead::create(head_geometry, hw, execution(L - 1, true), error, opts.gdn_production || opts.qsa_production || opts.prompt_head_elision);
        require(bool(head), error);
        account(head->memory_usage(), device(L - 1));
        append_placements(head->placements(), int64_t(L));
        // Explicit native mode resolves capacity and ownership only now, then
        // freezes canonical plans. Fixed-plan mode never enters this branch.
        if(!opts.expert_native_profile.empty()) {
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
            require(opts.dense_topology==PortableDenseTopology::primary && opts.accelerators.size()==2 &&
                opts.expert_devices.size()==2 && opts.expert_devices[0].accelerator==0 && opts.expert_devices[1].accelerator==1 &&
                !opts.expert_native_q8_observation.empty() && !opts.expert_native_plan_prefix.empty() &&
                opts.expert_native_primary_slots && opts.expert_native_helper_slots,"native capacity: requires primary/two native Q8 devices and explicit requests/output");
            for(const auto& d:opts.expert_devices)require(d.plan_path.empty() && d.native_frozen_plan.empty(),"native capacity excludes fixed plans");
            std::vector<std::pair<int32_t,int32_t>> ranked;int64_t profile_slots=0;
            const auto profile_time=std::filesystem::last_write_time(opts.expert_native_profile);
            const auto profile_size=std::filesystem::file_size(opts.expert_native_profile);
            require(profile_size<=4u<<20,"native capacity: profile file exceeds bound");
            require(read_expert_profile_bounded(opts.expert_native_profile,L,NE,ranked,profile_slots,error,L*NE),error);
            require(profile_time==std::filesystem::last_write_time(opts.expert_native_profile) && profile_size==std::filesystem::file_size(opts.expert_native_profile),"native capacity: profile changed during read");
            std::array<size_t,2> room{},available{},total{};
            for(size_t die=0;die<2;++die) {
                const auto& d=opts.expert_devices[die];
                ggml_backend_dev_memory(ggml_backend_get_device(opts.accelerators[die]),&available[die],&total[die]);
                room[die]=portable_detail::NativeCapacityRoom{report.device_admission[die],opts.device_caps[die],
                    available[die],total[die],d.free_reserve,d.tensor_cap,d.workspace_cap}.tensor_room();
            }
            auto estimate=[&](size_t die,size_t slots) {
                worker::NativeQ8AllocationEstimate e;
                require(worker::estimate_vulkan_native_q8(opts.accelerators[die],slots,e,error),error);return e;
            };
            const auto q=portable_detail::native_primary_quota(opts.expert_native_primary_slots,NE,
                std::vector<size_t>(L,5222400),room[0],[&](size_t n){return estimate(0,n).total_bytes;});
            auto selected=portable_detail::native_ranked_selection(ranked,L,NE,q);
            const auto helper=portable_detail::native_fit_count(std::min(opts.expert_native_helper_slots,selected.remaining.size()),
                room[1],[&](size_t n){return estimate(1,n).total_bytes;});
            require(helper>0,"native capacity: no unclaimed helper expert fits");selected.remaining.resize(helper);
            std::array<worker::GpuPlan,2> plans;
            for(size_t die=0;die<2;++die) {
                auto& d=opts.expert_devices[die];const auto ceiling=d.tensor_cap;
                auto pairs=die==0?selected.primary:selected.remaining;std::sort(pairs.begin(),pairs.end());
                std::vector<std::pair<uint16_t,uint16_t>> canonical(pairs.begin(),pairs.end());
                require(worker::make_gpu_plan(experts->identity(),canonical,plans[die],error),error);
                const auto e=estimate(die,pairs.size());require(e.total_bytes<=room[die],"native capacity: projected plan exceeds room");
                d.tensor_cap=e.total_bytes;
                d.plan_path=opts.expert_native_plan_prefix+".slot-"+std::to_string(die)+".plan";
                require(worker::encode_gpu_plan(plans[die],d.native_frozen_plan,error),error);
                require(!std::filesystem::exists(d.plan_path),"native capacity: plan output already exists");
                std::clog<<"native_capacity slot="<<die<<" requested_slots="<<(die?opts.expert_native_helper_slots:opts.expert_native_primary_slots)
                    <<" primary_quota="<<q<<" selected="<<pairs.size()<<" dense_admitted_bytes="<<report.device_admission[die]
                    <<" effective_limit_bytes="<<opts.device_caps[die]<<" available_after_dense_bytes="<<available[die]
                    <<" reserve_bytes="<<d.free_reserve<<" tensor_ceiling_bytes="<<ceiling<<" tensor_room_bytes="<<room[die]
                    <<" admitted_tensor_bytes="<<d.tensor_cap<<" weight_bytes="<<e.weight_bytes<<" scratch_bytes="<<e.scratch_bytes
                    <<" blocks="<<e.blocks<<" block_limit="<<e.block_limit<<" workspace_bytes="<<d.workspace_cap<<'\n';
            }
            // Existing no-replace publisher. Partial receipts survive failure;
            // no resident allocation has occurred and neither old plan is changed.
            for(size_t die=0;die<2;++die)require(worker::write_gpu_plan(opts.expert_devices[die].plan_path,plans[die],error),error);
#else
            require(false,"native capacity planning requires local Vulkan experts");
#endif
        }
        // Dense/head ledgers are final before expert allocation on the same physical dies.
        if (!opts.expert_devices.empty()) {
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
            require(!opts.expert_dispatch_oracle, "resident experts exclude staging oracle");
            std::set<size_t> assigned;
            size_t host_extra = 0;
            for (const auto& d:opts.expert_devices) {
                require(d.accelerator<opts.accelerators.size() && assigned.insert(d.accelerator).second,
                        "resident expert device missing or duplicated");
                require(d.tensor_cap>0 && d.workspace_cap>0 && d.tensor_cap<=SIZE_MAX-d.workspace_cap,
                        "invalid resident expert budgets");
                const size_t budget=d.tensor_cap+d.workspace_cap;
                // Observe the same post-dense free memory used by native cache
                // sizing. Diagnostic only: admission still uses the unchanged ledger guard.
                size_t available_after_dense=0,total_after_dense=0;
                ggml_backend_dev_memory(ggml_backend_get_device(opts.accelerators[d.accelerator]),
                                        &available_after_dense,&total_after_dense);
                const portable_detail::DeviceAdmissionContext context{d.accelerator,L,"resident-experts",
                    opts.device_policy_caps.empty()?0:opts.device_policy_caps[d.accelerator],
                    opts.device_free_reserves.empty()?0:opts.device_free_reserves[d.accelerator]};
                portable_detail::preflight_resident_expert_memory(report,d.accelerator,opts.device_caps[d.accelerator],
                    d.tensor_cap,d.workspace_cap,available_after_dense,total_after_dense,&context);
                worker::GpuPlan plan;
                require(d.native_frozen_plan.empty()?worker::read_gpu_plan(d.plan_path,plan,error):
                    worker::decode_gpu_plan(d.native_frozen_plan.data(),d.native_frozen_plan.size(),plan,error),error);
                if(observer && d.accelerator<2) {
                    // Retained until begin; not part of numerical execution.
                    for(size_t expert=0;expert<NE;++expert)(d.accelerator?observer->helper:observer->primary)[expert]=plan.owns(0,int(expert))?1:0;
                }
                const size_t per_expert=(2u<<20)+(32u<<10); // ggml metadata + persistent host boundaries
                require(plan.assets.size()<=(SIZE_MAX-(16u<<20))/per_expert,"resident host metadata overflow");
                const size_t h=plan.assets.size()*per_expert+(16u<<20); // load blob, plan/rows/request slack
                require(host_extra<=SIZE_MAX-h,"resident host budget overflow");
                host_extra+=h;
                report.device_admission[d.accelerator]+=budget;
            }
            require(report.host_weights+report.host_state_bound+host_graph+report.expert_staging<=opts.host_cap &&
                    host_extra<=opts.host_cap-report.host_weights-report.host_state_bound-host_graph-report.expert_staging,
                    "resident expert host metadata/staging exceeds cap");
            report.host_state_bound+=host_extra;
            auto memory=[](ggml_backend_t backend) {
                size_t free=0,total=0;
                ggml_backend_dev_memory(ggml_backend_get_device(backend),&free,&total);
                return std::make_pair(free,total);
            };
            size_t native_slot=0;
            auto factory=[&](ggml_backend_t backend,size_t budget) {
                if(opts.expert_native_q8_observation.empty())return worker::create_vulkan_executor_on_backend(backend,budget);
                return worker::create_vulkan_native_q8_observation(backend,budget,
                    opts.expert_native_q8_observation+".slot-"+std::to_string(native_slot++));
            };
            local_experts=std::make_unique<portable_detail::LocalExperts>(model,*experts,opts,
                factory,memory);
            report.expert_devices=local_experts->report();
#else
            throw std::runtime_error("local resident experts unavailable in this build");
#endif
        }
        require(reset(error), error);
    }
    void append_placements(const std::vector<PortablePlacement> &source, int64_t layer) {
        for (const auto &record : source) {
            auto p = record;
            p.layer = layer;
            placement.push_back(std::move(p));
        }
    }
    size_t device(size_t layer) const { return portable_detail::dense_device(opts, layer); }
    void preflight(size_t die, size_t layer, const char* module, size_t weight, size_t graph, size_t state,
                   size_t device_next = 0) const {
        // Bound one new module before creating its scheduler/device copies.
        // Reconcile to actual per-backend buffers plus allowances afterwards.
        // A nonzero device_next replaces only the device estimate, never the host envelope.
        size_t used = report.host_weights + report.host_state_bound + host_graph + report.expert_staging;
        for (size_t bytes : {weight, graph, state}) {
            require(used <= opts.host_cap && bytes <= opts.host_cap - used,
                    "portable model: next module exceeds host admission cap");
            used += bytes;
        }
        if (!opts.accelerators.empty()) {
            const portable_detail::DeviceAdmissionContext context{
                die, layer, module,
                opts.device_policy_caps.empty() ? 0 : opts.device_policy_caps[die],
                opts.device_free_reserves.empty() ? 0 : opts.device_free_reserves[die]};
            portable_detail::preflight_device_memory(report.device_admission[die], opts.device_caps[die],
                                                     device_next ? device_next : graph, &context);
        }
    }
    void account(const PortableMemoryUsage &m, size_t die) {
        report.host_weights += m.host_weights;
        report.host_state_bound += m.host_extra_bound;
        report.graph_admission_bound += m.host_graph + m.device_graph + m.device_extra_bound;
        host_graph += m.host_graph;
        report.host_scheduler = host_graph;
        require(report.host_weights <= opts.host_cap &&
                    report.host_state_bound <= opts.host_cap - report.host_weights &&
                    host_graph <= opts.host_cap - report.host_weights - report.host_state_bound &&
                    report.expert_staging <= opts.host_cap - report.host_weights - report.host_state_bound - host_graph,
                "portable model: host anonymous admission cap exceeded");
        if (!opts.accelerators.empty()) {
            const size_t previous_conversion=report.device_shared_conversion_bound[die];
            const size_t previous_temporary=report.device_token_temporary_bound[die];
            portable_detail::account_device_memory(report,die,opts.device_caps[die],m);
            report.graph_admission_bound+=report.device_token_temporary_bound[die]-previous_temporary+
                report.device_shared_conversion_bound[die]-previous_conversion;
        }
    }
    size_t host_graph = 0;
    bool reset(std::string &error) {
        poison = true; // reset failure cannot make partly reset state usable
        for (auto &l : layers)
            if (!l.reset(error))
                return false;
        pos = 0;
        eos_seen = false;
        poison = false;
        return true;
    }
    bool advance_chunk(const std::vector<int32_t>& tokens,std::string& error) {
        error.clear();
        if(poison) { error="portable model: session poisoned; reset/replay required"; return false; }
        portable_detail::ScopedExpertTimer<> step_timer(opts.expert_timings,model_work.step_ms,model_work.step_calls);
        try {
            const size_t T=tokens.size();
            require(opts.prompt_chunk_size&&T>=1&&T<=opts.prompt_chunk_size&&T<=opts.context-pos&&!eos_seen,
                    "portable model: invalid prompt chunk policy/context");
            std::vector<float> residual(T*H*HC);
            for(size_t t=0;t<T;++t) {
                require(tokens[t]>=0&&size_t(tokens[t])<vocabulary,"portable model: token ID out of range");
                const auto* row=static_cast<const uint8_t*>(embedding.data)+size_t(tokens[t])*(H/32*34);
                for(size_t b=0;b<H/32;++b)dequantize_q8_0(row+b*34,residual.data()+t*H*HC+b*32);
                for(size_t stream=1;stream<HC;++stream)
                    std::copy_n(residual.begin()+t*H*HC,H,residual.begin()+t*H*HC+stream*H);
            }
            finite(residual,"chunk embedding");
            std::ostringstream trace;
            portable_detail::RouterParticipation router_participation(opts.native_router);
            portable_detail::SharedParticipation shared_participation(opts.native_shared);
            const PortableExpertRowsBatch dispatch=[&](int64_t layer,const auto& routes,auto& rows,auto& e) {
                portable_detail::ScopedExpertTimer<> timer(opts.expert_timings,provider_ms,provider_calls);
                router_participation.observe(routes);shared_participation.observe(routes);
                if(opts.expert_record_routes) for(size_t t=0;t<T;++t) {
                    trace<<"expert_route position="<<pos+t<<" layer="<<layer<<" ids=";
                    for(size_t k=0;k<routes[t].ids.size();++k)trace<<(k?",":"")<<routes[t].ids[k];
                    trace<<'\n';
                }
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
                if(local_experts)return local_experts->run_batch(layer,routes,rows,e);
#endif
                return experts->run_batch(layer,routes,rows,e);
            };
            for(size_t l=0;l<layers.size();++l) {
                const bool gdn=bool(layers[l].gdn);const double before=provider_ms;
                {
                    portable_detail::ScopedExpertTimer<> timer(opts.expert_timings,gdn?model_work.gdn_ms:model_work.qsa_ms,
                                                               gdn?model_work.gdn_calls:model_work.qsa_calls);
                    const bool completed=layers[l].chunk(tokens,pos,residual,dispatch,error);
                    require(completed,"portable model: chunk position "+std::to_string(pos)+" layer "+std::to_string(l)+": "+error);
                }
                if(opts.expert_timings)(gdn?model_work.gdn_provider_ms:model_work.qsa_provider_ms)+=provider_ms-before;
            }
            std::vector<PortablePlacement> next;
            for(size_t l=0;l<layers.size();++l) for(const auto& original:layers[l].placements()) {
                auto record=original;record.layer=int64_t(l);next.push_back(std::move(record));
            }
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
            if(local_experts)report.expert_devices=local_experts->report();
#endif
            if(opts.expert_record_routes)std::cerr<<trace.str();
            placement.swap(next);pos+=T;
            router_participation.commit(model_work);shared_participation.commit(model_work);
            ++model_work.completed_forward_calls;model_work.completed_layer_calls+=L;
            model_work.completed_ffn_pre_calls+=L;model_work.completed_provider_calls+=L;model_work.completed_ffn_post_calls+=L;
            model_work.logical_tokens+=T;model_work.heads_skipped+=T;
            ++model_work.prompt_chunks;model_work.prompt_chunk_tokens+=T;
            model_work.chunk_layer_calls+=L;model_work.chunk_ffn_pre_calls+=L;
            model_work.chunk_provider_calls+=L;model_work.chunk_ffn_post_calls+=L;
            return true;
        } catch(const std::exception& e) { poison=true;error=e.what();return false; }
          catch(...) { poison=true;error="portable model: unknown chunk failure; reset/replay required";return false; }
    }
    bool step(int32_t token, std::vector<float> &logits, std::string &error, PortableTokenKind kind, bool project = true) {
        portable_detail::ScopedExpertTimer<> step_timer(opts.expert_timings, model_work.step_ms, model_work.step_calls);
        error.clear();
        logits.clear();
        if (poison) {
            error = "portable model: session poisoned; explicit reset and replay required";
            return false;
        }
        try {
            require(!eos_seen, "portable model: EOS already consumed; reset required");
            require(pos < opts.context, "portable model: context exhausted; reset required");
            require(token >= 0 && size_t(token) < vocabulary, "portable model: token ID out of range");
            std::vector<float> residual(H * HC);
            const auto *row = static_cast<const uint8_t *>(embedding.data) + size_t(token) * (H / 32 * 34);
            for (size_t b = 0; b < H / 32; ++b)
                dequantize_q8_0(row + b * 34, residual.data() + b * 32);
            finite(residual, "embedding");
            for (size_t stream = 1; stream < HC; ++stream)
                std::copy_n(residual.begin(), H, residual.begin() + stream * H);
            if(observer && pos==0 && !observer->finished()) {
                observer->begin(token);
                observer->put(Layer0Observer::primary_ownership,observer->primary.data(),observer->primary.size());
                observer->put(Layer0Observer::helper_ownership,observer->helper.data(),observer->helper.size());
                observer->put(Layer0Observer::embedding,residual.data(),residual.size());
                const auto& state=layers[0].gdn->recurrent_state();
                const auto& history=layers[0].gdn->convolution_history();
                observer->put(Layer0Observer::state_before,state.data(),state.size());
                observer->put(Layer0Observer::history_before,history.data(),history.size());
            }
            portable_detail::RouterParticipation router_participation(opts.native_router);
            portable_detail::SharedParticipation shared_participation(opts.native_shared);
            const PortableExpertRows dispatch = [&](int64_t l, const PortableRoute &r, std::vector<float> &rows,
                                                    std::string &e) {
                portable_detail::ScopedExpertTimer<> timer(opts.expert_timings, provider_ms, provider_calls);
                router_participation.observe(r);shared_participation.observe(r);
                if (opts.expert_record_routes) {
                    std::cerr << "expert_route position=" << pos << " layer=" << l << " ids=";
                    for(size_t i=0;i<r.ids.size();++i) std::cerr << (i ? "," : "") << r.ids[i];
                    std::cerr << '\n';
                }
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
                if (local_experts) return local_experts->run(l,r,rows,e);
#endif
                return experts->run(l, r, rows, e);
            };
            for (size_t l = 0; l < layers.size(); ++l) {
                const bool gdn = bool(layers[l].gdn);
                const double provider_before = provider_ms;
                {
                    portable_detail::ScopedExpertTimer<> layer_timer(opts.expert_timings,
                        gdn ? model_work.gdn_ms : model_work.qsa_ms,
                        gdn ? model_work.gdn_calls : model_work.qsa_calls);
                    const bool ok = layers[l].step(token, pos, residual, dispatch, error,
                        opts.expert_timings ? &model_work.gdn_phases : nullptr);
                    require(ok, "portable model: token position " + std::to_string(pos) + " layer " + std::to_string(l) +
                                    ": " + error);
                }
                if(l==0 && observer && observer->active()) {
                    const auto& state=layers[0].gdn->recurrent_state();
                    const auto& history=layers[0].gdn->convolution_history();
                    observer->put(Layer0Observer::state_after,state.data(),state.size());
                    observer->put(Layer0Observer::history_after,history.data(),history.size());
                    observer->put(Layer0Observer::materialized_post_residual,residual.data(),residual.size());
                    observer->finish();
                }
                if (opts.expert_timings)
                    (gdn ? model_work.gdn_provider_ms : model_work.qsa_provider_ms) += provider_ms-provider_before;
            }
            if (project) {
                ++model_work.heads_executed;
                portable_detail::ScopedExpertTimer<> head_timer(opts.expert_timings, model_work.head_ms, model_work.head_calls);
                const bool head_ok = head->run(residual, logits, error);
                require(head_ok, "portable model: output head: " + error);
            } else {
                ++model_work.heads_skipped;
            }
            placement.clear();
            for (size_t l = 0; l < layers.size(); ++l)
                append_placements(layers[l].placements(), int64_t(l));
            if (project) append_placements(head->placements(), int64_t(L));
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
            if (local_experts) report.expert_devices=local_experts->report();
#endif
            ++pos; ++model_work.logical_tokens;
            router_participation.commit(model_work);shared_participation.commit(model_work);
            ++model_work.completed_forward_calls;model_work.completed_layer_calls+=L;
            model_work.completed_ffn_pre_calls+=L;model_work.completed_provider_calls+=L;model_work.completed_ffn_post_calls+=L;
            eos_seen = kind == PortableTokenKind::generated && token == eos_id;
            return true;
        } catch (const std::exception &e) {
            poison = true;
            logits.clear();
            error = e.what();
            return false;
        }
    }
};
PortableModelSession::PortableModelSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PortableModelSession::~PortableModelSession() = default;
std::unique_ptr<PortableModelSession> PortableModelSession::open(const std::vector<std::string> &shards,
                                                                 const std::string &manifest, PortableModelOptions opts,
                                                                 std::string &error) {
    error.clear();
    try {
        return std::unique_ptr<PortableModelSession>(
            new PortableModelSession(std::make_unique<Impl>(shards, manifest, std::move(opts))));
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
bool PortableModelSession::step(int32_t token, std::vector<float> &logits, std::string &error, PortableTokenKind kind) {
    return impl_->step(token, logits, error, kind);
}
bool PortableModelSession::advance(int32_t token, std::string &error) {
    std::vector<float> omitted;
    return impl_->step(token, omitted, error, PortableTokenKind::input, false);
}
bool PortableModelSession::advance_chunk(const std::vector<int32_t>& tokens,std::string& error) {
    return impl_->advance_chunk(tokens,error);
}
bool PortableModelSession::reset(std::string &error) {
    error.clear();
    try {
        return impl_->reset(error);
    } catch (const std::exception &e) {
        impl_->poison = true;
        error = e.what();
        return false;
    }
}
uint64_t PortableModelSession::position() const { return impl_->pos; }
bool PortableModelSession::poisoned() const { return impl_->poison; }
bool PortableModelSession::ended() const { return impl_->eos_seen; }
int32_t PortableModelSession::eos() const { return impl_->eos_id; }
size_t PortableModelSession::vocabulary() const { return impl_->vocabulary; }
PortableExpertWork PortableModelSession::expert_work() const {
    auto work=impl_->experts->work();
    work.provider_ms=impl_->provider_ms;
    work.provider_calls=impl_->provider_calls;
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
    if (impl_->local_experts) {
        work.gpu_entries=impl_->local_experts->gpu_entries();
        work.cpu_entries=impl_->local_experts->cpu_entries();
    }
#endif
    return work;
}
PortableModelWork PortableModelSession::model_work() const {
    auto work=impl_->model_work;
    for(const auto& layer:impl_->layers) if(layer.gdn) {
        work.gdn_causal_submissions+=layer.gdn->causal_chunk_calls();
        work.gdn_causal_residual_handoffs+=layer.gdn->causal_residual_handoffs();
        work.gdn_causal_state_commits+=layer.gdn->causal_state_commits();
    }
    return work;
}
const PortableModelReport &PortableModelSession::report() const { return impl_->report; }
const std::vector<PortablePlacement> &PortableModelSession::placements() const { return impl_->placement; }
bool PortableModelSession::generate(const std::vector<int32_t> &prompt, size_t max_new, PortableGeneration &result,
                                    std::string &error) {
    result = {};
    error.clear();
    if (impl_->poison) {
        error = "portable model: session poisoned; reset/replay required";
        return false;
    }
    // Generation preflight is side-effect free. step errors always poison.
    if (prompt.empty() || prompt.size() > impl_->opts.context - impl_->pos || max_new > impl_->opts.context) {
        error = "portable model: empty prompt or generation/context bound";
        return false;
    }
    std::vector<float> logits;
    size_t first=0;
    if(impl_->opts.prompt_chunk_size) while(first+1<prompt.size()) {
        const size_t count=std::min(impl_->opts.prompt_chunk_size,prompt.size()-1-first);
        if(!advance_chunk({prompt.begin()+first,prompt.begin()+first+count},error))return false;
        first+=count;
    }
    for (size_t i = first; i < prompt.size(); ++i) {
        if (!(impl_->opts.prompt_head_elision && i + 1 < prompt.size()
                  ? advance(prompt[i], error) : step(prompt[i], logits, error)))
            return false;
    }
    for (size_t n = 0; n < max_new; ++n) {
        if (position() == impl_->opts.context) {
            result.stop = PortableStop::context;
            return true;
        }
        // max_element returns the first (lowest ID) in an exact tie.
        const auto token = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (!step(token, logits, error, PortableTokenKind::generated))
            return false;
        result.tokens.push_back(token);
        if (ended()) {
            result.stop = PortableStop::eos;
            return true;
        }
    }
    result.stop = PortableStop::max_new;
    return true;
}
} // namespace strata::core
