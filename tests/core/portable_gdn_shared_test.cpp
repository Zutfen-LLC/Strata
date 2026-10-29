#include "portable_gdn_shared_fixture.hpp"
#include "portable_gdn_budget_fixture.hpp"
int main() {
    try {
        gdn_budget_fixture::retained_projection_transition();
        gdn_shared_fixture::run({});
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
        for(bool production:{false,true}) for(bool causal:{false,true}) {
            if(causal && !production)continue;
            strata::core::PortableGdnOptions options;
            options.resident_state=true;
            options.execution.accelerator=backend.get(); options.execution.request_accelerator=true;
            options.execution.resident_dense_weights=true;
            options.policy=production?strata::core::PortableGdnPolicy::device_production:strata::core::PortableGdnPolicy::eager_diagnostic;
            options.batch_input_projections=true;options.causal_chunk=causal;
            gdn_shared_fixture::run(options);
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
