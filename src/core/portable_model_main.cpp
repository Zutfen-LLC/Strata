#include "ggml-backend.h"
#include "portable_stdio.hpp"
#include "portable_device_budget.hpp"
#include <csignal>
#ifndef _WIN32
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#endif
#include "strata/artifact/gguf_split.hpp"
#include "strata/core/portable_model.hpp"
#ifdef STRATA_PORTABLE_MODEL_VULKAN
#include "ggml-vulkan.h"
#endif
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int signal) { interrupted = signal; }
// Polling only stdin keeps idle SIGTERM/SIGINT cleanup bounded. No listener.
bool request_line(std::string& line) {
    line.clear();
#ifndef _WIN32
    for (;;) {
        if(interrupted) throw std::runtime_error("resident interrupted");
        pollfd fd{STDIN_FILENO,POLLIN,0};
        const int ready=::poll(&fd,1,100);
        if(ready<0) { if(errno==EINTR) continue; throw std::runtime_error("stdin poll failed"); }
        if(!ready) continue;
        char c;
        const auto n=::read(STDIN_FILENO,&c,1);
        if(n<0) { if(errno==EINTR) continue; throw std::runtime_error("stdin read failed"); }
        if(!n) {
            if(!line.empty()) throw std::runtime_error("incomplete request at EOF");
            return false;
        }
        if(c=='\n') return true;
        if(line.size()==4096) throw std::runtime_error("request line exceeds 4096 bytes");
        line.push_back(c);
    }
#else
    throw std::runtime_error("--stdio requires POSIX stdin polling");
#endif
}

std::vector<int64_t> numbers(const std::string &text) {
    std::vector<int64_t> out;
    std::istringstream in(text);
    std::string part;
    while (std::getline(in, part, ',')) {
        size_t used = 0;
        const auto value = std::stoll(part, &used);
        if (used != part.size())
            throw std::runtime_error("invalid integer list");
        out.push_back(value);
    }
    if (out.empty() || text.back() == ',')
        throw std::runtime_error("empty integer list item");
    return out;
}
} // namespace
int main(int argc, char **argv) {
    try {
        strata::core::PortableModelOptions opts;
        std::string model, manifest, token_text, dump, executed_dump, placement_path, timings_path;
        using Clock = std::chrono::steady_clock;
        Clock::time_point setup_start, prompt_start, decode_start, decode_end, last_prompt_start, last_selection;
        std::vector<int64_t> devices, caps, free_bytes, expert_tensors, expert_workspaces, expert_reserves;
        size_t max_new = 0, max_requests = 4;
        bool resident = false;
        auto integer = [](const std::string &s) {
            const auto v = numbers(s);
            if (v.size() != 1 || v[0] < 0)
                throw std::runtime_error("expected one nonnegative integer");
            return size_t(v[0]);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--expert-timings") { opts.expert_timings = true; continue; }
            if (key == "--stdio") { resident = true; continue; }
            if (key == "--list-vulkan") {
                if (argc != 2) throw std::runtime_error("--list-vulkan must be used alone");
#ifdef STRATA_PORTABLE_MODEL_VULKAN
                auto reg = ggml_backend_vk_reg();
                for (size_t index = 0; index < ggml_backend_reg_dev_count(reg); ++index) {
                    ggml_backend_dev_props props{};
                    ggml_backend_dev_get_props(ggml_backend_reg_dev_get(reg, index), &props);
                    // Same logical enumeration and PCI extension as the model backend.
                    std::cout << index << '\t' << (props.device_id ? props.device_id : "-")
                              << '\t' << props.memory_free << '\t' << props.memory_total
                              << '\t' << props.description << '\n';
                }
                return 0;
#else
                throw std::runtime_error("this build has no Vulkan backend");
#endif
            }
            if (key == "--help") {
                std::cout << "portable_model --list-vulkan (read-only logical index/PCI/free/total/description TSV)\n"
                             "portable_model --model ANY_SHARD --manifest SWLOCAL1 --tokens ID,ID,...\n"
                             " [--context 16] [--max-new 0] [--threads 1] [--expert-workers 1]\n"
                             " [--vulkan 0,1 --device-cap-mib 7680,7680 (or scalar) --split 32] [--host-cap-mib 32768]\n"
                             " [--dense-topology split|primary (default split; primary keeps dense/state/head on slot 0)]\n"
                             " [--device-free-bytes N,N (optional external free evidence; only tightens admission)]\n"
                             " [--gdn-production (requires --gdn-resident-state and strict Vulkan)]\n"
                             " [--qsa-production (requires --qsa-reuse-attention and strict Vulkan)]\n"
                             " [--qsa-reuse-attention (trial, default off)]\n"
                             " [--gdn-resident-state (trial, default off)]\n"
                             " [--gdn-resident-weights | --dense-resident-weights (trials, default off)]\n"
                             " [--expert-dispatch-oracle] [--allow-cpu-fallback] [--expert-timings (stdio only)]\n"
                             " [--expert-plan SLOT:FILE] [--expert-tensor-mib N,N] [--expert-workspace-mib N,N]\n"
                             " [--expert-free-reserve-mib N,N] [--expert-check-rows] [--expert-record-routes]\n"
                             " [--prompt-head-elision (serial head elision; incompatible with --dump-logits)]\n"
                             " [--gdn-causal-chunk (production chunks; default off)] [--gdn-batch-input-projections (requires prompt chunks; default off)] [--prompt-chunk-size 2..8 (implies head elision; default off)]\n"
                             " [--dump-logits DIRECTORY | --dump-executed-logits EMPTY_DIRECTORY] [--placements FILE] [--timings FILE]\n"
                             " Or --stdio [--max-requests 4]: GEN <max_new> <id,id,...> / QUIT; JSONL replies.\n"
                             "Default T=1 serial text; optional bounded prompt chunks; all 48 layers; CPU experts/host PLE.\n"
                             "Every consumed prompt/generated token can dump its full vocabulary as pos-N.f32.\n";
                return 0;
            }
            if (key == "--expert-check-rows" || key == "--expert-record-routes") {
                if(key == "--expert-check-rows") opts.expert_check_rows=true;
                else opts.expert_record_routes=true;
                continue;
            }
            if (key == "--expert-dispatch-oracle") {
                opts.expert_dispatch_oracle = true;
                continue;
            }
            if (key == "--prompt-head-elision") {
                opts.prompt_head_elision = true;
                continue;
            }
            if (key == "--qsa-production") {
                opts.qsa_production = true;
                continue;
            }
            if (key == "--qsa-reuse-attention") {
                opts.qsa_reuse_attention = true;
                continue;
            }
            if (key == "--gdn-production") {
                opts.gdn_production = true;
                continue;
            }
            if (key == "--gdn-resident-state") {
                opts.gdn_resident_state = true;
                continue;
            }
            if (key == "--dense-resident-weights") {
                opts.dense_resident_weights = true;
                continue;
            }
            if (key == "--gdn-causal-chunk") {
                opts.gdn_causal_chunk = true;
                continue;
            }
            if (key == "--gdn-batch-input-projections") {
                opts.gdn_batch_input_projections = true;
                continue;
            }
            if (key == "--gdn-resident-weights") {
                opts.gdn_resident_weights = true;
                continue;
            }
            if (key == "--allow-cpu-fallback") {
                opts.allow_cpu_fallback = true;
                continue;
            }
            if (i + 1 >= argc)
                throw std::runtime_error("missing value: " + key);
            const std::string value = argv[++i];
            if (key == "--model")
                model = value;
            else if (key == "--manifest")
                manifest = value;
            else if (key == "--tokens")
                token_text = value;
            else if (key == "--context")
                opts.context = integer(value);
            else if (key == "--max-requests")
                max_requests = integer(value);
            else if (key == "--prompt-chunk-size") {
                opts.prompt_chunk_size=integer(value);
                if(opts.prompt_chunk_size<2||opts.prompt_chunk_size>8)throw std::runtime_error("prompt chunk size must be in [2,8]");
                opts.prompt_head_elision=true;
            }
            else if (key == "--max-new")
                max_new = integer(value);
            else if (key == "--threads" || key == "--expert-workers") {
                const auto n = integer(value);
                if (n < 1 || n > 256)
                    throw std::runtime_error("thread/worker count must be in [1,256]");
                if (key == "--threads")
                    opts.threads = int(n);
                else
                    opts.expert_workers = int(n);
            } else if (key == "--split")
                opts.split = integer(value);
            else if (key == "--dense-topology") {
                if (value == "split") opts.dense_topology = strata::core::PortableDenseTopology::split;
                else if (value == "primary") opts.dense_topology = strata::core::PortableDenseTopology::primary;
                else throw std::runtime_error("dense topology must be split or primary");
            }
            else if (key == "--host-cap-mib") {
                const auto n = integer(value);
                if (n > (SIZE_MAX >> 20))
                    throw std::runtime_error("host cap overflow");
                opts.host_cap = n << 20;
            } else if (key == "--vulkan")
                devices = numbers(value);
            else if (key == "--device-cap-mib")
                caps = numbers(value);
            else if (key == "--device-free-bytes")
                free_bytes = numbers(value);
            else if (key == "--dump-logits")
                dump = value;
            else if (key == "--dump-executed-logits")
                executed_dump = value;
            else if (key == "--expert-plan") {
                const auto colon=value.find(':');
                if(colon==std::string::npos || colon+1==value.size()) throw std::runtime_error("expert plan requires SLOT:FILE");
                strata::core::PortableExpertDevice d;
                d.accelerator=integer(value.substr(0,colon)); d.plan_path=value.substr(colon+1);
                opts.expert_devices.push_back(std::move(d));
            } else if (key == "--expert-tensor-mib") expert_tensors=numbers(value);
            else if (key == "--expert-workspace-mib") expert_workspaces=numbers(value);
            else if (key == "--expert-free-reserve-mib") expert_reserves=numbers(value);
            else if (key == "--timings")
                timings_path = value;
            else if (key == "--placements")
                placement_path = value;
            else
                throw std::runtime_error("unknown option: " + key);
        }
        if(opts.gdn_causal_chunk && (!opts.prompt_chunk_size || !opts.gdn_batch_input_projections || !opts.gdn_production))
            throw std::runtime_error("--gdn-causal-chunk requires --prompt-chunk-size, --gdn-batch-input-projections and --gdn-production");
        if(opts.gdn_batch_input_projections && !opts.prompt_chunk_size)
            throw std::runtime_error("--gdn-batch-input-projections requires --prompt-chunk-size");
        if (!executed_dump.empty() && (!dump.empty() || !opts.prompt_head_elision || resident))
            throw std::runtime_error("--dump-executed-logits requires fresh head elision and excludes --dump-logits/--stdio");
        if (opts.prompt_head_elision && !dump.empty())
            throw std::runtime_error("--prompt-head-elision cannot dump skipped logits; omit it for exhaustive --dump-logits");
        if (opts.expert_timings && !resident) throw std::runtime_error("--expert-timings requires --stdio");
        if (resident && (!token_text.empty() || max_new || !dump.empty() || !timings_path.empty()))
            throw std::runtime_error("--stdio takes requests on stdin; --tokens/--max-new/--dump-logits/--timings incompatible");
        if (max_requests < 1 || max_requests > 64) throw std::runtime_error("--max-requests must be in [1,64]");
#ifdef _WIN32
        if(resident) throw std::runtime_error("--stdio requires POSIX stdin polling");
#endif
        if (model.empty() || manifest.empty() || (!resident && token_text.empty()))
            throw std::runtime_error("--model/--manifest/--tokens required; see --help");
        const auto raw_tokens = resident ? std::vector<int64_t>{} : numbers(token_text);
        std::vector<int32_t> tokens;
        for (int64_t t : raw_tokens) {
            if (t < 0 || t > INT32_MAX)
                throw std::runtime_error("invalid token ID");
            tokens.push_back(int32_t(t));
        }
        if (tokens.size() > opts.context || max_new > opts.context)
            throw std::runtime_error("prompt/max-new exceeds context bound");
        if (caps.size() == 1 && devices.size() == 2) caps.push_back(caps[0]);
        if (!free_bytes.empty() && free_bytes.size() != devices.size())
            throw std::runtime_error("separate free-memory observations required for each device");
        if (devices.size() != caps.size() || devices.size() > 2)
            throw std::runtime_error("separate cap required for each Vulkan device");
        if (devices.size() == 2 && devices[0] == devices[1])
            throw std::runtime_error("duplicate Vulkan device index");
        if(resident) {
            std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt); std::signal(SIGPIPE,SIG_IGN);
        }
        if (resident || !timings_path.empty())
            setup_start = Clock::now();
        const auto budget=[&](const std::vector<int64_t>& values,size_t slot,size_t fallback) {
            if(values.empty()) return fallback;
            if(values.size()!=devices.size() || values[slot]<=0 || uint64_t(values[slot])>(SIZE_MAX>>20))
                throw std::runtime_error("separate valid expert budgets required for all device slots");
            return size_t(values[slot])<<20;
        };
        if(opts.expert_devices.empty() && (!expert_tensors.empty() || !expert_workspaces.empty() || !expert_reserves.empty() || opts.expert_check_rows))
            throw std::runtime_error("expert budgets/check require an expert plan");
        for(auto& d:opts.expert_devices) {
            if(d.accelerator>=devices.size()) throw std::runtime_error("expert plan slot outside Vulkan devices");
            d.tensor_cap=budget(expert_tensors,d.accelerator,d.tensor_cap);
            d.workspace_cap=budget(expert_workspaces,d.accelerator,d.workspace_cap);
            d.free_reserve=budget(expert_reserves,d.accelerator,d.free_reserve);
        }
        std::vector<std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>> backends;
        std::vector<size_t> device_totals, device_reserves;
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i] < 0 || devices[i] > INT32_MAX)
                throw std::runtime_error("invalid Vulkan index");
            const size_t policy_cap = strata::core::portable_detail::device_mib(caps[i]);
#ifdef STRATA_PORTABLE_MODEL_VULKAN
            if (devices[i] >= ggml_backend_vk_get_device_count())
                throw std::runtime_error("Vulkan device index out of range");
            auto *backend = ggml_backend_vk_init(size_t(devices[i]));
            if (!backend)
                throw std::runtime_error("requested Vulkan backend unavailable");
            backends.emplace_back(backend, ggml_backend_free);
            opts.accelerators.push_back(backend);
            size_t available = 0, total = 0;
            ggml_backend_vk_get_device_memory(int(devices[i]), &available, &total);
            if (!total || available > total)
                throw std::runtime_error("missing/contradictory Vulkan memory evidence");
            size_t reserve = 512u << 20;
            for (const auto& expert : opts.expert_devices)
                if (expert.accelerator == i) reserve = std::max(reserve, expert.free_reserve);
            if (!free_bytes.empty()) {
                if (free_bytes[i] <= 0 || uint64_t(free_bytes[i]) > total || uint64_t(free_bytes[i]) > SIZE_MAX)
                    throw std::runtime_error("invalid external free-memory observation");
                available = std::min(available, size_t(free_bytes[i]));
            }
            opts.device_caps.push_back(strata::core::portable_detail::device_admission_limit(
                policy_cap, available, total, reserve));
            opts.device_policy_caps.push_back(policy_cap);
            opts.device_free_reserves.push_back(reserve);
            device_totals.push_back(total);
            device_reserves.push_back(reserve);
            std::cerr << "device=" << devices[i] << " backend=" << ggml_backend_name(backend)
                      << " policy_cap_bytes=" << policy_cap << " available_bytes=" << available
                      << " reserve_bytes=" << reserve << " effective_limit_bytes=" << opts.device_caps.back()
                      << " cap_bytes=" << opts.device_caps.back() << '\n';
#else
            throw std::runtime_error("this build has no Vulkan backend");
#endif
        }
        auto device_memory = [&](const char *stage, uint64_t position) {
#ifdef STRATA_PORTABLE_MODEL_VULKAN
            for (size_t i = 0; i < devices.size(); ++i) {
                size_t available = 0, total = 0;
                char description[256]{};
                ggml_backend_vk_get_device_memory(int(devices[i]), &available, &total);
                if (total != device_totals[i] || available > total || available < device_reserves[i])
                    throw std::runtime_error("Vulkan capacity changed or runtime free reserve breached");
                ggml_backend_vk_get_device_description(int(devices[i]), description, sizeof(description));
                std::cerr << "device_memory stage=" << stage << " position=" << position << " device=" << devices[i]
                          << " total_bytes=" << total << " available_bytes=" << available;
                if (available <= total)
                    std::cerr << " heap_used_bytes=" << total - available;
                std::cerr << " description=" << description << '\n';
            }
#else
            (void)stage;
            (void)position;
#endif
        };
        device_memory("before-model", 0);
        if (!dump.empty() && !std::filesystem::is_directory(dump))
            throw std::runtime_error("--dump-logits directory must already exist");
        if (!executed_dump.empty() && (!std::filesystem::is_directory(executed_dump) ||
                                      !std::filesystem::is_empty(executed_dump)))
            throw std::runtime_error("--dump-executed-logits directory must exist and be empty");
        std::string error;
        auto paths = strata::gguf_split_paths(model);
        auto source_paths = paths;
        source_paths.push_back(manifest);
        for(const auto& d:opts.expert_devices) source_paths.push_back(d.plan_path);
        // Capture before loading; observed change requires a new process.
        std::unique_ptr<strata::core::stdio_detail::Sources> sources;
        if(resident) sources=std::make_unique<strata::core::stdio_detail::Sources>(source_paths);
        auto session = strata::core::PortableModelSession::open(paths, manifest, opts, error);
        if (!session)
            throw std::runtime_error(error);
        device_memory("model-loaded", 0);
        const auto &report = session->report();
        if(report.chunk_memory_is_shape_envelope)
            std::cerr << "prompt_chunk_admission shape_envelope=1 prompt_chunk_size=" << opts.prompt_chunk_size << '\n';
        std::cerr << "mapped_bytes=" << report.mapped_bytes << " host_weight_bytes=" << report.host_weights
                  << " host_scheduler=" << report.host_scheduler << " host_state_bound=" << report.host_state_bound
                  << " expert_staging=" << report.expert_staging << " graph_admission=" << report.graph_admission_bound
                  << '\n';
        for (size_t i = 0; i < report.device_admission.size(); ++i)
            std::cerr << "die=" << i << " actual_weight_bytes=" << report.device_weights[i]
                      << " actual_state_bytes=" << report.device_state[i]
                      << (report.chunk_memory_is_shape_envelope?" scheduler_shape_envelope_bytes=":" actual_scheduler_bytes=") << report.device_scheduler[i]
                      << " workspace_and_transient_bound=" << report.device_workspace_bound[i]+report.device_token_temporary_bound[i]+report.device_shared_conversion_bound[i]
                      << " backend_private_bound=" << report.device_workspace_bound[i]
                      << " token_temporary_bound=" << report.device_token_temporary_bound[i]
                      << " shared_conversion_bound=" << report.device_shared_conversion_bound[i]
                      << " admitted_bytes=" << report.device_admission[i]
                      << " policy_cap_bytes=" << strata::core::portable_detail::device_mib(caps[i])
                      << " effective_limit_bytes=" << opts.device_caps[i] << " reserve_bytes=" << device_reserves[i]
                      << " cap_bytes=" << opts.device_caps[i]
                      << " headroom_bytes=" << opts.device_caps[i] - report.device_admission[i] << '\n';
        std::cerr << report.identity_evidence << '\n';
        std::cerr << (opts.prompt_chunk_size ? "Bounded prompt chunks; T=1 final prompt/decode; " : "T=1; ");
        if(opts.expert_devices.empty())
            std::cerr << "CPU experts; host PLE; recurrent state and inter-layer residuals use host/device transfers. No performance claim.\n";
        else
            std::cerr << "resident Vulkan/CPU expert rows; host PLE; original-order host merge. Vulkan hybrid Q8 is not CPU bit parity.\n";
        const auto expert_report=[&](const char* stage,uint64_t position) {
            if(opts.expert_devices.empty()) return;
            std::cerr << std::setprecision(17);
            for(const auto& d:session->report().expert_devices)
                std::cerr << "expert_device stage=" << stage << " position=" << position << " slot=" << d.accelerator
                    << " backend=" << ggml_backend_name(opts.accelerators[d.accelerator])
                    << " numerics=VULKAN_HYBRID_Q8 assigned=" << d.assigned << " weight_bytes=" << d.weight_bytes << " tensor_bytes=" << d.tensor_bytes
                    << " tensor_scratch_bytes=" << d.tensor_scratch_bytes << " tensor_cap=" << d.tensor_cap << " workspace_cap=" << d.workspace_cap
                    << " metadata_bound=" << d.metadata_bound << " staging_peak=" << d.staging_peak
                    << " free_before=" << d.free_before << " free_after=" << d.free_after << " peak_growth=" << d.peak_growth
                    << " requests=" << d.requests << " entries=" << d.entries << " row_checks=" << d.row_checks
                    << " row_max_abs=" << d.row_max_abs << " row_relative_l2=" << d.row_relative_l2 << '\n';
        };
        expert_report("loaded",0);
        size_t request_id=0;
        strata::core::PortableExpertWork before;
        strata::core::PortableModelWork model_before;
        Clock::time_point request_start;
        std::vector<uint64_t> device_entries_before, device_calls_before;
        std::vector<double> device_ms_before;
        if(resident) {
            sources->check();
            if(interrupted) throw std::runtime_error("resident interrupted during load");
            std::cout << std::setprecision(17) << "{\"ready\":true,\"schema\":1,\"load_ms\":"
                << std::chrono::duration<double,std::milli>(Clock::now()-setup_start).count()
                << ",\"context\":" << opts.context << ",\"max_requests\":" << max_requests << "}\n" << std::flush;
            if(!std::cout) throw std::runtime_error("ready write failed");
        }
        do {
        if(resident) {
            if(request_id==max_requests) break;
            std::string line;
            if(!request_line(line) || line=="QUIT") break;
            // Parse completely before reset or other mutation.
            auto request=strata::core::stdio_detail::parse(line,opts.context,session->vocabulary());
            request_start=Clock::now();
            sources->check();
            if(session->poisoned()) throw std::runtime_error("resident session poisoned");
            before=session->expert_work();
            model_before=session->model_work();
            device_entries_before.clear();
            device_calls_before.clear(); device_ms_before.clear();
            for(const auto& d:session->report().expert_devices) {
                device_calls_before.push_back(d.infer_rows_calls); device_ms_before.push_back(d.infer_rows_ms);
            }
            for(const auto& d:session->report().expert_devices) device_entries_before.push_back(d.entries);
            if(!session->reset(error)) throw std::runtime_error(error);
            tokens=std::move(request.tokens); max_new=request.max_new; ++request_id;
        }
        std::vector<float> logits;
        std::vector<int32_t> executed_tokens;
        const auto heads_before = session->model_work();
        const auto emit_row = [&](uint64_t pos,int32_t token,bool generated,bool project) {
            std::cout << "{\"position\":" << pos << ",\"token\":" << token
                      << ",\"generated\":" << (generated ? "true" : "false");
            if (opts.prompt_head_elision)
                std::cout << ",\"schema\":2,\"head_policy\":\"prompt_head_elision\",\"logits_available\":"
                          << (project ? "true" : "false");
            if(opts.prompt_chunk_size)
                std::cout << ",\"chunk_policy\":\"layer_major_v1\",\"prompt_chunk_size\":" << opts.prompt_chunk_size;
            std::cout << ",\"greedy_next\":";
            if (project) std::cout << std::max_element(logits.begin(), logits.end()) - logits.begin();
            else std::cout << "null";
            std::cout << "}\n";
        };
        const auto chunk_fields = [&](std::ostream& out) {
            if(!opts.prompt_chunk_size)return;
            const auto after=session->model_work();
            out << ",\"chunk_policy\":\"layer_major_v1\",\"prompt_chunk_size\":" << opts.prompt_chunk_size
                << ",\"gdn_subphases_available\":false"
                << ",\"model_forward_calls_delta\":" << after.completed_forward_calls-heads_before.completed_forward_calls
                << ",\"model_layer_calls_delta\":" << after.completed_layer_calls-heads_before.completed_layer_calls
                << ",\"ffn_pre_calls_delta\":" << after.completed_ffn_pre_calls-heads_before.completed_ffn_pre_calls
                << ",\"provider_calls_delta\":" << after.completed_provider_calls-heads_before.completed_provider_calls
                << ",\"ffn_post_calls_delta\":" << after.completed_ffn_post_calls-heads_before.completed_ffn_post_calls
                << ",\"logical_tokens_delta\":" << after.logical_tokens-heads_before.logical_tokens
                << ",\"prompt_chunks_delta\":" << after.prompt_chunks-heads_before.prompt_chunks
                << ",\"prompt_chunk_tokens_delta\":" << after.prompt_chunk_tokens-heads_before.prompt_chunk_tokens
                << ",\"chunk_layer_calls_delta\":" << after.chunk_layer_calls-heads_before.chunk_layer_calls
                << ",\"chunk_ffn_pre_calls_delta\":" << after.chunk_ffn_pre_calls-heads_before.chunk_ffn_pre_calls
                << ",\"chunk_provider_calls_delta\":" << after.chunk_provider_calls-heads_before.chunk_provider_calls
                << ",\"chunk_ffn_post_calls_delta\":" << after.chunk_ffn_post_calls-heads_before.chunk_ffn_post_calls;
            if(opts.gdn_causal_chunk) out << ",\"gdn_causal_chunk\":true"
                << ",\"gdn_causal_submissions_delta\":" << after.gdn_causal_submissions-heads_before.gdn_causal_submissions
                << ",\"gdn_causal_residual_handoffs_delta\":" << after.gdn_causal_residual_handoffs-heads_before.gdn_causal_residual_handoffs
                << ",\"gdn_causal_state_commits_delta\":" << after.gdn_causal_state_commits-heads_before.gdn_causal_state_commits;
            if(opts.gdn_batch_input_projections) out << ",\"gdn_batch_input_projections\":true";
        };
        auto consume = [&](int32_t token, bool generated, bool project = true) {
            if(resident) {
                if(interrupted) throw std::runtime_error("resident interrupted");
                sources->check();
            }
            const uint64_t pos = session->position();
            if (!project) logits.clear();
            if (!(project ? session->step(token, logits, error,
                               generated ? strata::core::PortableTokenKind::generated
                                         : strata::core::PortableTokenKind::input)
                          : session->advance(token, error)))
                throw std::runtime_error(error);
            if(resident && interrupted) throw std::runtime_error("resident interrupted during token");
            device_memory("token-complete", pos);
            const auto work = session->expert_work();
            std::cerr << "expert_work position=" << pos << " hits=" << work.hits
                      << " verifications=" << work.verifications << " copied_bytes=" << work.copied_bytes
                      << " digest_bytes=" << work.digest_bytes << " native_batches=" << work.native_batches
                      << " source_invalidations=" << work.source_invalidations << '\n';
            if (!opts.expert_devices.empty()) {
                std::cerr << "expert_dispatch position=" << pos << " gpu_entries=" << work.gpu_entries
                          << " cpu_entries=" << work.cpu_entries << '\n';
                expert_report("token-complete",pos);
            }
            if (project && (!dump.empty() || !executed_dump.empty())) {
                // compare_logits.py consumes little-endian raw F32. This CLI is
                // for the existing little-endian x86 CPU/Vulkan execution path.
                const uint16_t one = 1;
                if (*reinterpret_cast<const uint8_t *>(&one) != 1)
                    throw std::runtime_error("logit dump requires little-endian host");
                std::ofstream file((executed_dump.empty()?dump:executed_dump) + "/pos-" + std::to_string(pos) + ".f32", std::ios::binary);
                file.write(reinterpret_cast<const char *>(logits.data()),
                           std::streamsize(logits.size() * sizeof(float)));
                if (!file)
                    throw std::runtime_error("logit dump write failed");
                if(!executed_dump.empty())executed_tokens.push_back(token);
            }
            emit_row(pos,token,generated,project);
        };
        if (resident || !timings_path.empty())
            prompt_start = Clock::now();
        // Native DONE places the last prompt forward step in decode. Keep
        // that selection boundary as well as this CLI's full-consumption phases.
        size_t first=0;
        if(opts.prompt_chunk_size) while(first+1<tokens.size()) {
            const size_t count=std::min(opts.prompt_chunk_size,tokens.size()-1-first);
            if(resident) {
                if(interrupted)throw std::runtime_error("resident interrupted");
                sources->check();
            }
            const uint64_t pos=session->position();
            if(!session->advance_chunk({tokens.begin()+first,tokens.begin()+first+count},error))throw std::runtime_error(error);
            if(resident&&interrupted)throw std::runtime_error("resident interrupted during chunk");
            device_memory("chunk-complete",pos);
            // No row is published until all 48 layers completed this chunk.
            for(size_t t=0;t<count;++t)emit_row(pos+t,tokens[first+t],false,false);
            first+=count;
        }
        for (size_t i = first; i < tokens.size(); ++i) {
            if ((resident || !timings_path.empty()) && i + 1 == tokens.size())
                last_prompt_start = Clock::now();
            consume(tokens[i], false, !opts.prompt_head_elision || i + 1 == tokens.size());
        }
        if (resident || !timings_path.empty())
            decode_start = Clock::now();
        size_t generated = 0;
        while (!session->ended() && generated < max_new && session->position() < opts.context) {
            const int32_t next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (resident || !timings_path.empty())
                last_selection = Clock::now();
            consume(next, true);
            ++generated;
        }
        if (resident || !timings_path.empty())
            decode_end = Clock::now();
        if (opts.prompt_head_elision) {
            const auto heads = session->model_work();
            std::cerr << "prompt_head_work executed=" << heads.heads_executed-heads_before.heads_executed
                      << " skipped=" << heads.heads_skipped-heads_before.heads_skipped << '\n';
        }
        const char *stop = session->ended() ? "eos" : session->position() == opts.context ? "context" : "max_new";
        if(!executed_dump.empty()) {
            // A manifest is published only after the complete forward sequence.
            // Missing prompt positions are absent, never zero-filled logit rows.
            std::ofstream file(executed_dump+"/manifest.json");
            file << "{\"schema\":1,\"format\":\"executed_heads_f32le_v1\",\"head_policy\":\"prompt_head_elision\","
                 << "\"chunk_policy\":\"" << (opts.prompt_chunk_size?"layer_major_v1":"serial_t1")
                 << "\",\"prompt_chunk_size\":" << opts.prompt_chunk_size
                 << ",\"vocabulary\":" << session->vocabulary() << ",\"prompt_tokens\":[";
            for(size_t i=0;i<tokens.size();++i)file << (i?",":"") << tokens[i];
            file << "],\"output_count\":" << generated << ",\"stop\":\"" << stop << "\",\"frames\":[";
            if(executed_tokens.size()!=generated+1)throw std::runtime_error("executed logit frame count mismatch");
            for(size_t i=0;i<executed_tokens.size();++i) {
                const auto position=tokens.size()-1+i;
                file << (i?",":"") << "{\"position\":" << position << ",\"token\":" << executed_tokens[i]
                     << ",\"role\":\"" << (!i?"final_prompt":i+1==executed_tokens.size()?"terminal":"generated")
                     << "\",\"file\":\"pos-" << position << ".f32\"}";
            }
            file << "]}\n";
            if(!file)throw std::runtime_error("executed logit manifest write failed");
        }
        std::cout << "{\"stop\":\"" << stop << "\",\"position\":" << session->position() << "}\n";
        if(resident) {
            sources->check();
            const auto after=session->expert_work();
            const auto ms=[](Clock::time_point a,Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
            std::cout << std::setprecision(17) << "{\"done\":" << request_id
                << ",\"setup_reset_ms\":" << ms(request_start,prompt_start)
                << ",\"prompt_ms\":" << ms(prompt_start,decode_start)
                << ",\"decode_ms\":" << ms(decode_start,decode_end)
                << ",\"selection_prompt_ms\":" << ms(prompt_start,last_prompt_start)
                << ",\"selection_decode_ms\":" << (generated?ms(last_prompt_start,last_selection):0.)
                << ",\"terminal_feed_ms\":" << (generated?ms(last_selection,decode_end):0.)
                << ",\"prompt_count\":" << tokens.size() << ",\"output_count\":" << generated
                << ",\"selection_prompt_steps\":" << tokens.size()-1 << ",\"selection_decode_steps\":" << generated
                << ",\"cpu_hits_delta\":" << after.hits-before.hits
                << ",\"cpu_verifications_delta\":" << after.verifications-before.verifications
                << ",\"cpu_copied_bytes_delta\":" << after.copied_bytes-before.copied_bytes
                << ",\"cpu_digest_bytes_delta\":" << after.digest_bytes-before.digest_bytes
                << ",\"source_invalidations_delta\":" << after.source_invalidations-before.source_invalidations
                << ",\"cpu_pool_gu_ms_delta\":" << after.pool_gu_ms-before.pool_gu_ms
                << ",\"cpu_pool_quant_ms_delta\":" << after.pool_quant_ms-before.pool_quant_ms
                << ",\"cpu_pool_down_ms_delta\":" << after.pool_down_ms-before.pool_down_ms
                << ",\"cpu_pool_elapsed_ms_delta\":" << after.pool_elapsed_ms-before.pool_elapsed_ms
                << ",\"gpu_entries_delta\":" << after.gpu_entries-before.gpu_entries
                << ",\"cpu_entries_delta\":" << after.cpu_entries-before.cpu_entries
                << ",\"gpu_entries_per_device_delta\":[";
            for(size_t i=0;i<device_entries_before.size();++i)
                std::cout << (i ? "," : "") << session->report().expert_devices[i].entries-device_entries_before[i];
            std::cout << "]";
            if (opts.prompt_head_elision) {
                const auto heads = session->model_work();
                std::cout << ",\"head_policy\":\"prompt_head_elision\",\"receipt_schema\":2"
                          << ",\"heads_executed_delta\":" << heads.heads_executed-heads_before.heads_executed
                          << ",\"heads_skipped_delta\":" << heads.heads_skipped-heads_before.heads_skipped;
            }
            chunk_fields(std::cout);
            if(opts.expert_timings) {
                std::cout << ",\"expert_provider_ms_delta\":" << after.provider_ms-before.provider_ms
                    << ",\"expert_provider_calls_delta\":" << after.provider_calls-before.provider_calls
                    << ",\"gpu_infer_rows_ms_delta\":[";
                for(size_t i=0;i<device_ms_before.size();++i)
                    std::cout << (i ? "," : "") << session->report().expert_devices[i].infer_rows_ms-device_ms_before[i];
                std::cout << "],\"gpu_infer_rows_calls_delta\":[";
                for(size_t i=0;i<device_calls_before.size();++i)
                    std::cout << (i ? "," : "") << session->report().expert_devices[i].infer_rows_calls-device_calls_before[i];
                std::cout << "]";
                const auto model_after=session->model_work();
                const double step=model_after.step_ms-model_before.step_ms;
                const double gdn=model_after.gdn_ms-model_before.gdn_ms;
                const double qsa=model_after.qsa_ms-model_before.qsa_ms;
                const double head=model_after.head_ms-model_before.head_ms;
                const auto &ga=model_after.gdn_phases, &gb=model_before.gdn_phases;
                const double attention=ga.attention_exchange_ms-gb.attention_exchange_ms;
                const double compute=ga.attention_compute_ms-gb.attention_compute_ms;
                const double ffn=ga.ffn_ms-gb.ffn_ms;
                const double gp=model_after.gdn_provider_ms-model_before.gdn_provider_ms;
                if(!opts.prompt_chunk_size) std::cout << ",\"gdn_attention_exchange_ms_delta\":" << attention
                    << ",\"gdn_attention_compute_ms_delta\":" << compute
                    << ",\"gdn_candidate_readback_ms_delta\":" << ga.candidate_readback_ms-gb.candidate_readback_ms
                    << ",\"gdn_candidate_finite_ms_delta\":" << ga.candidate_finite_ms-gb.candidate_finite_ms
                    << ",\"gdn_state_commit_ms_delta\":" << ga.state_commit_ms-gb.state_commit_ms
                    << ",\"gdn_candidate_readback_calls_delta\":" << ga.candidate_readback_calls-gb.candidate_readback_calls
                    << ",\"gdn_candidate_finite_calls_delta\":" << ga.candidate_finite_calls-gb.candidate_finite_calls
                    << ",\"gdn_state_commit_calls_delta\":" << ga.state_commit_calls-gb.state_commit_calls
                    << ",\"gdn_ffn_ms_delta\":" << ffn
                    << ",\"gdn_attention_surrounding_ms_delta\":" << attention-compute
                    << ",\"gdn_ffn_non_provider_ms_delta\":" << ffn-gp
                    << ",\"gdn_outer_residual_ms_delta\":" << gdn-attention-ffn
                    << ",\"gdn_attention_exchange_calls_delta\":" << ga.attention_exchange_calls-gb.attention_exchange_calls
                    << ",\"gdn_attention_compute_calls_delta\":" << ga.attention_compute_calls-gb.attention_compute_calls
                    << ",\"gdn_ffn_calls_delta\":" << ga.ffn_calls-gb.ffn_calls;
                std::cout << ",\"model_step_ms_delta\":" << step
                    << ",\"gdn_layer_ms_delta\":" << gdn
                    << ",\"qsa_layer_ms_delta\":" << qsa
                    << ",\"output_head_ms_delta\":" << head
                    << ",\"model_residual_ms_delta\":" << step-gdn-qsa-head
                    << ",\"gdn_provider_ms_delta\":" << model_after.gdn_provider_ms-model_before.gdn_provider_ms
                    << ",\"qsa_provider_ms_delta\":" << model_after.qsa_provider_ms-model_before.qsa_provider_ms
                    << ",\"model_step_calls_delta\":" << model_after.step_calls-model_before.step_calls
                    << ",\"gdn_layer_calls_delta\":" << model_after.gdn_calls-model_before.gdn_calls
                    << ",\"qsa_layer_calls_delta\":" << model_after.qsa_calls-model_before.qsa_calls
                    << ",\"output_head_calls_delta\":" << model_after.head_calls-model_before.head_calls;
            }
            std::cout << "}\n" << std::flush;
            if(!std::cout) throw std::runtime_error("request result write failed");
        }
        if (!timings_path.empty()) {
            const auto ms = [](Clock::time_point a, Clock::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::ofstream file(timings_path);
            file << std::setprecision(17)
                 << "{\"schema\":1,\"setup_ms\":" << ms(setup_start, prompt_start)
                 << ",\"prompt_ms\":" << ms(prompt_start, decode_start)
                 << ",\"decode_ms\":" << ms(decode_start, decode_end)
                 << ",\"prompt_count\":" << tokens.size() << ",\"output_count\":" << generated
                 << ",\"selection_prompt_ms\":" << ms(prompt_start, last_prompt_start)
                 << ",\"selection_decode_ms\":" << (generated ? ms(last_prompt_start, last_selection) : 0.0)
                 << ",\"terminal_feed_ms\":" << (generated ? ms(last_selection, decode_end) : 0.0)
                 << ",\"selection_prompt_steps\":" << tokens.size() - 1
                 << ",\"selection_decode_steps\":" << generated
                 << ",\"prompt_steps\":" << tokens.size() << ",\"decode_steps\":" << generated;
            chunk_fields(file);
            file << "}\n";
            if (!file)
                throw std::runtime_error("timing report write failed");
        }
        } while(resident);
        if (!placement_path.empty() && (!resident || request_id)) {
            std::ofstream file(placement_path);
            file << "layer\tphase\top\ttensor\ttype\tbackend\tmetadata\tfallback\n";
            for (const auto &p : session->placements())
                file << p.layer << '\t' << p.phase << '\t' << p.operation << '\t' << p.tensor << '\t' << p.type << '\t'
                     << p.backend << '\t' << p.metadata_only << '\t' << p.fallback_reason << '\n';
            if (!file)
                throw std::runtime_error("placement report write failed");
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
