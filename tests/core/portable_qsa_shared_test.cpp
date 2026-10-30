#include "portable_qsa_shared_fixture.hpp"
#include "ggml-cpu.h"
int main() {
    try {
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
        strata::core::PortableQsaOptions options;
        options.execution.accelerator=backend.get(); options.execution.request_accelerator=true;
        qsa_fixture::shared_attention_tests(options);
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
