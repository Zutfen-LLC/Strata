#include "portable_native_hc_test_backend.hpp"
#include "portable_ple_fixture.hpp"
#include "portable_model_preflight.hpp"
#include <iostream>

int main() {
    try {
        ple_fixture::Fixture f;
        std::string error;
        {
            PortableModelOptions model;
            model.native_ple_value_t1 = true;
            model.native_ple_value_receipt_prefix = "ple";
            model.gdn_resident_weights = true;
            model.allow_cpu_fallback = false;
            validate_model_modes(model, 1);
            auto off = model; off.native_ple_value_t1 = false;
            for (size_t l = 0; l < 48; ++l) {
                PortableGdnOptions per_layer;
                model_gdn_hc_options(model, l, per_layer);
                check(per_layer.native_ple_value_t1 == (l == 1), "PLE value leaked to another layer");
                check(model_hc_host_bound(model,l) - model_hc_host_bound(off,l) ==
                      (l == 1 ? PortableGdnOptions::native_ple_value_host_bound : 0), "model host envelope mismatch");
            }
            auto bad = model; bad.allow_cpu_fallback = true;
            bool rejected = false; try { validate_model_modes(bad, 1); } catch (const std::exception&) { rejected = true; }
            check(rejected, "model accepted native PLE with fallback");
        }
        auto table = ple_fixture::small_table(error);
        check(bool(table), error);
        auto root = std::filesystem::temp_directory_path() /
            ("ple-native-value-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fresh test root");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        original_compute = cpu->iface.graph_compute;
        cpu->iface.graph_compute = check_compute;
        PortableGdnOptions options;
        options.execution.threads = 1;
        options.execution.accelerator = cpu.get();
        options.execution.request_accelerator = true;
        options.execution.allow_cpu_fallback = false;
        options.resident_attention_weights = true;
        options.ple_table = table;
        auto baseline = PortableGdnLayer::create(1, f.views, options, error);
        check(bool(baseline), error);
        options.native_ple_value_t1 = true;
        options.native_ple_value_receipt = (root / "candidate").string();
        options.state_budget += PortableGdnOptions::native_ple_value_host_bound;
        { auto bad = options; bad.execution.allow_cpu_fallback = true;
          check(!PortableGdnLayer::create(1, f.views, bad, error), "CPU fallback accepted"); }
        { auto bad = options; bad.state_budget = baseline->state_bytes() + PortableGdnOptions::native_ple_value_host_bound - 1;
          check(!PortableGdnLayer::create(1, f.views, bad, error), "short host budget accepted"); }
        auto layer = PortableGdnLayer::create(1, f.views, options, error);
        check(bool(layer), error);
        check(layer->weight_bytes() == baseline->weight_bytes() + (2560 * 2560 * 2 - 2560 * 2560 / 32 * 34),
              "BF16 replacement weight accounting");
        check(layer->reset(error), error);
        PortableRoute route;
        auto before = layer->ple_convolution_history();
        fail_ple_value = true;
        check(!layer->step(19, f.input(0), ple_fixture::experts(), route, error), "component failure accepted");
        check(layer->tokens() == 0 && layer->ple_convolution_history() == before && layer->token_history() == std::array<int32_t, 2>{{-1,-1}},
              "component failure committed PLE state");
        check(!std::filesystem::exists(root / "candidate" / "first-candidate"), "failed projection produced candidate receipt");
        fail_ple_value = false;
        observe_ple_value = true;
        PortableGdnTrace trace;
        check(layer->step(19, f.input(0), ple_fixture::experts(), route, error, &trace), error);
        check(ple_value_links > 0, "PLE graph did not consume actual value bank");
        auto candidate_dir = root / "candidate" / "first-candidate";
        auto bytes = [](const std::filesystem::path& p){std::ifstream f(p,std::ios::binary);return std::vector<char>(std::istreambuf_iterator<char>(f),{});};
        const auto captured_input = bytes(candidate_dir / "input.f32le");
        const auto captured_value = bytes(candidate_dir / "value.f32le");
        const auto captured_meta = bytes(candidate_dir / "metadata.json");
        const std::string metadata(captured_meta.begin(),captured_meta.end());
        check(captured_input.size()==trace.ple_embedding.size()*4 && !std::memcmp(captured_input.data(),trace.ple_embedding.data(),captured_input.size()), "captured input differs from gathered embedding");
        check(captured_value.size()==trace.ple_value.size()*4 && !std::memcmp(captured_value.data(),trace.ple_value.data(),captured_value.size()), "captured value differs from consumed bank");
        check(metadata.find("\"position\":0,\"token\":19,\"previous\":[-1,-1]")!=std::string::npos && metadata.find("\"committed\":false")!=std::string::npos, "candidate identity or commit label differs");
        ple_fixture::reference(f, f.input(0), before, trace, *layer);
        const auto first = layer->last_residual();
        const auto history = layer->ple_convolution_history();
        const auto token_history = layer->token_history();
        PortableExpertRows refuse = [](int64_t, const PortableRoute&, std::vector<float>&, std::string& e) { e = "intentional refusal"; return false; };
        check(!layer->step(20, f.input(1), refuse, route, error), "provider refusal accepted");
        check(layer->tokens() == 1 && layer->ple_convolution_history() == history && layer->token_history() == token_history,
              "failed FFN committed PLE history");
        PortableGdnTrace second;
        check(layer->step(20, f.input(1), ple_fixture::experts(), route, error, &second), error);
        check(layer->token_history() == std::array<int32_t, 2>{{19, 20}}, "retry advanced wrong token history");
        ple_fixture::reference(f, f.input(1), history, second, *layer);
        check(layer->reset(error), error);
        check(layer->step(19, f.input(0), ple_fixture::experts(), route, error), error);
        check(layer->last_residual() == first && layer->ple_convolution_history() == history, "reset/retry differs");
        check(bytes(candidate_dir / "input.f32le")==captured_input && bytes(candidate_dir / "metadata.json")==captured_meta, "reset overwrote lifetime receipt");
        observe_ple_value = false;
        layer.reset();
        options.native_ple_value_receipt = (root / "failed-candidate").string();
        auto failed = PortableGdnLayer::create(1, f.views, options, error);
        check(bool(failed) && failed->reset(error), error);
        auto failed_history = failed->ple_convolution_history();
        check(!failed->step(19, f.input(0), refuse, route, error), "first FFN refusal accepted");
        check(failed->tokens()==0 && failed->ple_convolution_history()==failed_history, "first candidate failure committed state");
        const auto failure_meta = bytes(root / "failed-candidate" / "first-candidate" / "metadata.json");
        check(std::string(failure_meta.begin(),failure_meta.end()).find("\"committed\":false")!=std::string::npos, "failed candidate claimed commit");
        failed.reset(); baseline.reset();
        std::filesystem::remove_all(root);
        std::cout << "PLE native value ownership, history and rollback PASS (host provider)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
