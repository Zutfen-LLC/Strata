#include "strata/core/portable_model.hpp"
#include "portable_graph.hpp"
#include "portable_memory.hpp"
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
#include <set>

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
    GgufModel model; // before all consumers; destroyed last
    std::unique_ptr<portable_detail::CpuExperts> experts;
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
    std::unique_ptr<portable_detail::LocalExperts> local_experts;
#endif
    PortableTensorView embedding;
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
        require(shards.size() == 6 && model.size() == 6, "portable model: this fleet path requires all six shards");
        require(opts.context >= 1 && opts.context <= 256 && opts.threads >= 1 && opts.threads <= 256,
                "portable model: context [1,256]/thread limits");
        require(opts.accelerators.size() <= 2 && opts.device_caps.size() == opts.accelerators.size(),
                "portable model: supply a separate cap for each of at most two accelerators");
        require(!opts.qsa_production || (opts.qsa_reuse_attention && !opts.accelerators.empty() && !opts.allow_cpu_fallback),
                "portable model: QSA production requires reuse and strict accelerators");
        require(!opts.gdn_production || (opts.gdn_resident_state && !opts.accelerators.empty() &&
                    !opts.allow_cpu_fallback),
                "portable model: GDN production requires resident state and strict accelerators");
        require(opts.accelerators.size() < 2 || (opts.split >= 1 && opts.split < L),
                "portable model: invalid contiguous split");
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
        layers.reserve(L);
        for (size_t l = 0; l < L; ++l) {
            // Fixed schedulers obey layer_graph_cap. Native backend-private
            // workspaces are admitted separately from those scheduler buffers.
            const size_t private_workspace = l % 4 == 3 ? (24u << 20) : (136u << 20);
            require(opts.layer_graph_cap <= SIZE_MAX - private_workspace, "portable model: graph cap overflow");
            const bool resident = !opts.accelerators.empty() &&
                                  (opts.dense_resident_weights || (opts.gdn_resident_weights && l % 4 != 3));
            const size_t state_bound = opts.gdn_resident_state && l % 4 != 3 && !opts.accelerators.empty()
                                           ? PortableGdnOptions{}.resident_state_budget : 0;
            require(state_bound <= SIZE_MAX - opts.layer_graph_cap - private_workspace,
                    "portable model: resident state preflight overflow");
            const size_t resident_bound = resident ? opts.layer_weight_cap : 0;
            require(resident_bound <= SIZE_MAX - opts.layer_graph_cap - private_workspace - state_bound,
                    "portable model: resident weight preflight overflow");
            preflight(device(l), opts.layer_weight_cap,
                      opts.layer_graph_cap + private_workspace + resident_bound + state_bound, 128u << 20);
            Layer layer;
            if (l % 4 == 3) {
                PortableQsaOptions o;
                o.execution = execution(l, false);
                o.capacity = opts.context;
                o.reuse_attention = opts.qsa_reuse_attention;
                o.production = opts.qsa_production;
                layer.qsa = PortableQsaLayer::from_gguf(model, int64_t(l), o, error);
                require(bool(layer.qsa), "portable model: layer " + std::to_string(l) + ": " + error);
            } else {
                PortableGdnOptions o;
                o.resident_state = opts.gdn_resident_state;
                o.policy = opts.gdn_production ? PortableGdnPolicy::device_production : PortableGdnPolicy::eager_diagnostic;
                o.resident_attention_weights = opts.gdn_resident_weights;
                o.execution = execution(l, false);
                layer.gdn = PortableGdnLayer::from_gguf(model, int64_t(l), o, error);
                require(bool(layer.gdn), "portable model: layer " + std::to_string(l) + ": " + error);
            }
            account(layer.memory(), device(l));
            append_placements(layer.placements(), int64_t(l));
            layers.push_back(std::move(layer));
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
        preflight(device(L - 1), opts.head_weight_cap, opts.head_graph_cap + head_resident_bound, 64u << 20);
        head = PortableHead::create({H, HC, 320, int64_t(vocabulary), 1e-6f}, hw, execution(L - 1, true), error, opts.gdn_production || opts.qsa_production);
        require(bool(head), error);
        account(head->memory_usage(), device(L - 1));
        append_placements(head->placements(), int64_t(L));
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
                require(report.device_admission[d.accelerator]<=opts.device_caps[d.accelerator] &&
                        budget<=opts.device_caps[d.accelerator]-report.device_admission[d.accelerator],
                        "dense plus resident expert tensors/runtime exceed per-device cap");
                worker::GpuPlan plan;
                require(worker::read_gpu_plan(d.plan_path,plan,error),error);
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
            local_experts=std::make_unique<portable_detail::LocalExperts>(model,*experts,opts,
                worker::create_vulkan_executor_on_backend,memory);
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
    size_t device(size_t layer) const { return opts.accelerators.size() == 2 && layer >= opts.split ? 1 : 0; }
    void preflight(size_t die, size_t weight, size_t graph, size_t state) const {
        // Bound one new module before creating its scheduler/device copies.
        // Reconcile to actual per-backend buffers plus allowances afterwards.
        size_t used = report.host_weights + report.host_state_bound + host_graph + report.expert_staging;
        for (size_t bytes : {weight, graph, state}) {
            require(used <= opts.host_cap && bytes <= opts.host_cap - used,
                    "portable model: next module exceeds host admission cap");
            used += bytes;
        }
        if (!opts.accelerators.empty())
            require(report.device_admission[die] <= opts.device_caps[die] &&
                        graph <= opts.device_caps[die] - report.device_admission[die],
                    "portable model: next module exceeds separate die admission cap");
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
    bool step(int32_t token, std::vector<float> &logits, std::string &error, PortableTokenKind kind) {
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
            const PortableExpertRows dispatch = [&](int64_t l, const PortableRoute &r, std::vector<float> &rows,
                                                    std::string &e) {
                portable_detail::ScopedExpertTimer<> timer(opts.expert_timings, provider_ms, provider_calls);
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
                if (opts.expert_timings)
                    (gdn ? model_work.gdn_provider_ms : model_work.qsa_provider_ms) += provider_ms-provider_before;
            }
            {
                portable_detail::ScopedExpertTimer<> head_timer(opts.expert_timings, model_work.head_ms, model_work.head_calls);
                const bool head_ok = head->run(residual, logits, error);
                require(head_ok, "portable model: output head: " + error);
            }
            placement.clear();
            for (size_t l = 0; l < layers.size(); ++l)
                append_placements(layers[l].placements(), int64_t(l));
            append_placements(head->placements(), int64_t(L));
#ifdef STRATA_PORTABLE_LOCAL_EXPERTS
            if (local_experts) report.expert_devices=local_experts->report();
#endif
            ++pos;
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
PortableModelWork PortableModelSession::model_work() const { return impl_->model_work; }
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
    for (int32_t token : prompt) {
        if (!step(token, logits, error))
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
