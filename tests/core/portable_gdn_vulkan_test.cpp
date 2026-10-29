#include "portable_layer_chunk_fixture.hpp"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"
#include "portable_gdn_fixture.hpp"
#include "portable_gdn_production.hpp"
#include <cstdlib>
#include <limits>
using namespace strata::core;
using namespace gdn_fixture;
int main(int argc, char **argv) {
    try {
        int index = 0;
        std::string expected;
        for (int i = 1; i < argc; i += 2) {
            check(i + 1 < argc, "usage: --device INDEX --expect-device SUBSTRING");
            if (std::string(argv[i]) == "--device") {
                char *end = nullptr;
                long n = std::strtol(argv[i + 1], &end, 10);
                check(end != argv[i + 1] && *end == 0 && n >= 0 && n < GGML_VK_MAX_DEVICES, "bad index");
                index = int(n);
            } else if (std::string(argv[i]) == "--expect-device")
                expected = argv[i + 1];
            else
                check(false, "unknown option");
        }
        check(ggml_backend_vk_get_device_count() > index, "Vulkan device required; no skipped pass");
        char description[256] = {};
        ggml_backend_vk_get_device_description(index, description, sizeof(description));
        check(expected.empty() || std::string(description).find(expected) != std::string::npos, "unexpected device");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> device(ggml_backend_vk_init(size_t(index)),
                                                                           ggml_backend_free);
        check(bool(device) && !ggml_backend_supports_buft(device.get(), ggml_backend_cpu_buffer_type()),
              "distinct Vulkan buffers required");
        // Same-device serial control isolates multirow FFN kernel differences.
        for(bool batched:{false,true}) for(bool production:{false,true}) for(bool causal:{false,true}) {
            if(causal && (!batched || !production)) continue;
            gdn_fixture::Fixture fixture;std::string error;
            PortableGdnOptions options;options.resident_state=true;
            options.execution.accelerator=device.get();options.execution.request_accelerator=true;
            options.execution.allow_cpu_fallback=false;options.execution.resident_dense_weights=true;
            options.policy=production?PortableGdnPolicy::device_production:PortableGdnPolicy::eager_diagnostic;
            options.batch_input_projections=batched;
            options.causal_chunk=causal;
            auto chunk=PortableGdnLayer::create(0,fixture.views,options,error);
            options.batch_input_projections=false;
            options.causal_chunk=false;
            auto serial=PortableGdnLayer::create(0,fixture.views,options,error);
            check(bool(chunk)&&bool(serial),error);
            layer_chunk_fixture::exercise(*chunk,*serial,fixture,gdn_fixture::experts(),device.get(),
                options.execution.graph_budget,options.state_budget,production?"GDN production Vulkan":"GDN diagnostic Vulkan",true,batched,causal);
        }
        for (int trial : {0, 1, 2, 3, 4, 5}) {
            const int mode = trial % 3;
            const bool state_resident = trial >= 3;
            std::printf("state residency=%d\n", int(state_resident));
            const bool resident = mode != 0;
            std::printf("dense weight fixture: mode=%d (default=0 GDN-only=1 dense=2)\n", mode);
            Fixture f;
            PortableGdnOptions options;
            options.resident_state = state_resident;
            options.resident_attention_weights = mode == 1;
            options.execution.resident_dense_weights = mode == 2;
            options.execution.accelerator = device.get();
            options.execution.request_accelerator = true;
            std::string error;
            if (state_resident) production_policy(f, options);
            auto layer = PortableGdnLayer::create(0, f.views, options, error);
            check(bool(layer), error);
            check((layer->memory_usage().device_state > 0) == state_resident, "resident state ledger");
            check((layer->memory_usage().device_weights > 0) == resident, "resident weight ledger");
            check(layer->memory_usage().host_weights + layer->memory_usage().device_weights == layer->weight_bytes(),
                  "host/device weight allocation reconciliation");
            auto cpu = PortableGdnLayer::create(0, f.views, {}, error);
            check(bool(cpu), error);
            size_t gdn = 0, conv = 0, mat = 0;
            for (const auto &p : layer->placements())
                if (p.phase == "gdn-attention" && !p.metadata_only) {
                    check(p.backend == ggml_backend_name(device.get()) && p.fallback_reason.empty(),
                          "strict attention placement");
                    if (p.operation == "GATED_DELTA_NET")
                        ++gdn;
                    if (p.operation == "SSM_CONV")
                        ++conv;
                    if (p.operation == "MUL_MAT")
                        ++mat;
                }
            check(gdn == 1 && conv == 1 && mat == 8, "actual attention operation coverage");
            PortableRoute route;
            PortableGdnTrace trace;
            const auto *state0=layer->recurrent_state().data(), *history0=layer->convolution_history().data();
            const float *state1=nullptr, *history1=nullptr;
            PortableGdnWork work;
            for (int t = 0; t < 2; ++t) {
                auto state = layer->recurrent_state(), history = layer->convolution_history();
                PortableRoute cpu_route;
                check(cpu->step(f.input(t), experts(), cpu_route, error), error);
                check(layer->step(f.input(t), experts(), route, error, &trace, &work), error);
                candidate_work(work, t + 1, state_resident ? t + 1 : 0, false);
                if(t==0) {
                    state1=layer->recurrent_state().data();history1=layer->convolution_history().data();
                    check(state1!=state0 && history1!=history0,"Vulkan candidate banks do not alias committed mirrors");
                }
                check(layer->recurrent_state().data()==(t%2?state0:state1) &&
                      layer->convolution_history().data()==(t%2?history0:history1),"Vulkan reuses two host banks");
                reference(f, state, history, trace, *layer);
                close(layer->last_residual(), cpu->last_residual(), 0.003f, "full layer CPU/Vulkan output");
                close(route.weights, cpu_route.weights, 0.003f, "full layer route weights");
                check(route.ids == cpu_route.ids, "full layer route IDs");
            }
            auto state = layer->recurrent_state(), history = layer->convolution_history(), last = layer->last_residual();
            check(!layer->step(
                      f.input(2), [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) { return false; },
                      route, error),
                  "Vulkan callback rejection");
            close(layer->recurrent_state(), state, 0, "Vulkan state rollback");
            close(layer->convolution_history(), history, 0, "Vulkan history rollback");
            close(layer->last_residual(), last, 0, "Vulkan residual rollback");
            for (int failure = 0; failure < 2; ++failure) {
                check(!layer->step(f.input(2),
                    [failure](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &) -> bool {
                        if (failure == 0) throw std::runtime_error("callback threw");
                        rows.assign(25600, std::numeric_limits<float>::quiet_NaN());
                        return true;
                    }, route, error, &trace), "Vulkan callback failure accepted");
                check(layer->recurrent_state() == state && layer->convolution_history() == history &&
                      layer->last_residual() == last && layer->tokens() == 2 && trace.attention.empty(),
                      "Vulkan failed candidate committed");
            }
            // A valid retry must consume the last committed device state.
            PortableRoute cpu_route;
            check(cpu->step(f.input(2), experts(), cpu_route, error), error);
            check(layer->step(f.input(2), experts(), route, error, &trace), error);
            reference(f, state, history, trace, *layer);
            close(layer->last_residual(), cpu->last_residual(), 0.003f, "Vulkan retry after rollback");
            check(layer->tokens() == 3 && layer->reset(error), "Vulkan reset");
            state = layer->recurrent_state();
            history = layer->convolution_history();
            check(layer->step(f.input(0), experts(), route, error, &trace), error);
            reference(f, state, history, trace, *layer);
            std::printf("portable_gdn_vulkan_test: device=%s actual H2560/S128/Hk16/Hv48; backend=%s GATED_DELTA_NET=%zu "
                        "SSM_CONV=%zu attention_MUL_MAT=%zu strict execution, independent state, full layer CPU parity, "
                        "reset and rollback passed\n",
                        description, ggml_backend_name(device.get()), gdn, conv, mat);
        } // state OFF/ON crossed with all three weight placements
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
