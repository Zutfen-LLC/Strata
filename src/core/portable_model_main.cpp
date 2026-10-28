#include "ggml-backend.h"
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
        std::string model, manifest, token_text, dump, placement_path, timings_path;
        using Clock = std::chrono::steady_clock;
        Clock::time_point setup_start, prompt_start, decode_start, decode_end, last_prompt_start, last_selection;
        std::vector<int64_t> devices, caps;
        size_t max_new = 0;
        auto integer = [](const std::string &s) {
            const auto v = numbers(s);
            if (v.size() != 1 || v[0] < 0)
                throw std::runtime_error("expected one nonnegative integer");
            return size_t(v[0]);
        };
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "portable_model --model ANY_SHARD --manifest SWLOCAL1 --tokens ID,ID,...\n"
                             " [--context 16] [--max-new 0] [--threads 1] [--expert-workers 1]\n"
                             " [--vulkan 0,1 --device-cap-mib 7680,7680 --split 32] [--host-cap-mib 32768]\n"
                             " [--expert-dispatch-oracle] [--allow-cpu-fallback]\n"
                             " [--dump-logits DIRECTORY] [--placements FILE] [--timings FILE]\n"
                             "T=1 serial text; all 48 layers; fresh embedding per token; CPU experts/host PLE.\n"
                             "Every consumed prompt/generated token can dump its full vocabulary as pos-N.f32.\n";
                return 0;
            }
            if (key == "--expert-dispatch-oracle") {
                opts.expert_dispatch_oracle = true;
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
            else if (key == "--host-cap-mib") {
                const auto n = integer(value);
                if (n > (SIZE_MAX >> 20))
                    throw std::runtime_error("host cap overflow");
                opts.host_cap = n << 20;
            } else if (key == "--vulkan")
                devices = numbers(value);
            else if (key == "--device-cap-mib")
                caps = numbers(value);
            else if (key == "--dump-logits")
                dump = value;
            else if (key == "--timings")
                timings_path = value;
            else if (key == "--placements")
                placement_path = value;
            else
                throw std::runtime_error("unknown option: " + key);
        }
        if (model.empty() || manifest.empty() || token_text.empty())
            throw std::runtime_error("--model/--manifest/--tokens required; see --help");
        const auto raw_tokens = numbers(token_text);
        std::vector<int32_t> tokens;
        for (int64_t t : raw_tokens) {
            if (t < 0 || t > INT32_MAX)
                throw std::runtime_error("invalid token ID");
            tokens.push_back(int32_t(t));
        }
        if (tokens.size() > opts.context || max_new > opts.context)
            throw std::runtime_error("prompt/max-new exceeds context bound");
        if (devices.size() != caps.size() || devices.size() > 2)
            throw std::runtime_error("separate cap required for each Vulkan device");
        if (devices.size() == 2 && devices[0] == devices[1])
            throw std::runtime_error("duplicate Vulkan device index");
        if (!timings_path.empty())
            setup_start = Clock::now();
        std::vector<std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>> backends;
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i] < 0 || caps[i] <= 0 || caps[i] > 8192)
                throw std::runtime_error("invalid Vulkan index or per-die cap (<=8192 MiB)");
#ifdef STRATA_PORTABLE_MODEL_VULKAN
            if (devices[i] >= ggml_backend_vk_get_device_count())
                throw std::runtime_error("Vulkan device index out of range");
            auto *backend = ggml_backend_vk_init(size_t(devices[i]));
            if (!backend)
                throw std::runtime_error("requested Vulkan backend unavailable");
            backends.emplace_back(backend, ggml_backend_free);
            opts.accelerators.push_back(backend);
            opts.device_caps.push_back(size_t(caps[i]) << 20);
            std::cerr << "device=" << devices[i] << " backend=" << ggml_backend_name(backend)
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
        std::string error;
        auto session = strata::core::PortableModelSession::open(strata::gguf_split_paths(model), manifest, opts, error);
        if (!session)
            throw std::runtime_error(error);
        device_memory("model-loaded", 0);
        const auto &report = session->report();
        std::cerr << "mapped_bytes=" << report.mapped_bytes << " host_weight_bytes=" << report.host_weights
                  << " host_scheduler=" << report.host_scheduler << " host_state_bound=" << report.host_state_bound
                  << " expert_staging=" << report.expert_staging << " graph_admission=" << report.graph_admission_bound
                  << '\n';
        for (size_t i = 0; i < report.device_admission.size(); ++i)
            std::cerr << "die=" << i << " actual_scheduler_bytes=" << report.device_scheduler[i]
                      << " workspace_and_transient_bound=" << report.device_workspace_bound[i]
                      << " admitted_bytes=" << report.device_admission[i] << " cap_bytes=" << opts.device_caps[i]
                      << " headroom_bytes=" << opts.device_caps[i] - report.device_admission[i] << '\n';
        std::cerr << report.identity_evidence << '\n'
                  << "T=1; CPU experts; host PLE; recurrent state and inter-layer residuals use host/device transfers. "
                     "No performance claim.\n";
        std::vector<float> logits;
        auto consume = [&](int32_t token, bool generated) {
            const uint64_t pos = session->position();
            if (!session->step(token, logits, error,
                               generated ? strata::core::PortableTokenKind::generated
                                         : strata::core::PortableTokenKind::input))
                throw std::runtime_error(error);
            device_memory("token-complete", pos);
            const auto work = session->expert_work();
            std::cerr << "expert_work position=" << pos << " hits=" << work.hits
                      << " verifications=" << work.verifications << " copied_bytes=" << work.copied_bytes
                      << " digest_bytes=" << work.digest_bytes << " native_batches=" << work.native_batches
                      << " source_invalidations=" << work.source_invalidations << '\n';
            if (!dump.empty()) {
                // compare_logits.py consumes little-endian raw F32. This CLI is
                // for the existing little-endian x86 CPU/Vulkan execution path.
                const uint16_t one = 1;
                if (*reinterpret_cast<const uint8_t *>(&one) != 1)
                    throw std::runtime_error("logit dump requires little-endian host");
                std::ofstream file(dump + "/pos-" + std::to_string(pos) + ".f32", std::ios::binary);
                file.write(reinterpret_cast<const char *>(logits.data()),
                           std::streamsize(logits.size() * sizeof(float)));
                if (!file)
                    throw std::runtime_error("logit dump write failed");
            }
            std::cout << "{\"position\":" << pos << ",\"token\":" << token
                      << ",\"generated\":" << (generated ? "true" : "false")
                      << ",\"greedy_next\":" << std::max_element(logits.begin(), logits.end()) - logits.begin()
                      << "}\n";
        };
        if (!timings_path.empty())
            prompt_start = Clock::now();
        // Native DONE places the last prompt forward step in decode. Keep
        // that selection boundary as well as this CLI's full-consumption phases.
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (!timings_path.empty() && i + 1 == tokens.size())
                last_prompt_start = Clock::now();
            consume(tokens[i], false);
        }
        if (!timings_path.empty())
            decode_start = Clock::now();
        size_t generated = 0;
        while (!session->ended() && generated < max_new && session->position() < opts.context) {
            const int32_t next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (!timings_path.empty())
                last_selection = Clock::now();
            consume(next, true);
            ++generated;
        }
        if (!timings_path.empty())
            decode_end = Clock::now();
        const char *stop = session->ended() ? "eos" : session->position() == opts.context ? "context" : "max_new";
        std::cout << "{\"stop\":\"" << stop << "\",\"position\":" << session->position() << "}\n";
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
                 << ",\"prompt_steps\":" << tokens.size() << ",\"decode_steps\":" << generated << "}\n";
            if (!file)
                throw std::runtime_error("timing report write failed");
        }
        if (!placement_path.empty()) {
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
