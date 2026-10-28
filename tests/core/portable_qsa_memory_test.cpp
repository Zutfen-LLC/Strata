#include "portable_qsa_fixture.hpp"
#include "ggml-cpu.h"
#include <cstdio>
using namespace qsa_fixture;
int main(){
    try{
        Fixture f;
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
        PortableQsaOptions opts;opts.capacity=1;
        opts.execution.accelerator=backend.get();opts.execution.request_accelerator=true;
        std::string error;
        auto layer=PortableQsaLayer::create(3,f.views,opts,error);check(bool(layer),error);
        const size_t fixed=layer->graph_bytes();const auto before=layer->memory_usage();
        check(fixed<opts.execution.graph_budget,"fixture has temporary capacity");
        check(before.device_token_temporary_bound==opts.execution.graph_budget-fixed,
              "temporary-only budget excludes all fixed_graphs private workspace");
        check(before.device_shared_conversion_bound==0,"unverified CPU backend keeps conversion bound additive");
        const size_t ffn_private=size_t(2560*4*320)*sizeof(float)+(8u<<20);
        check(before.device_extra_bound+before.device_token_temporary_bound==
              ffn_private+opts.execution.graph_budget-before.host_graph-before.device_graph,
              "one-layer total bound conserved after lifetime split");
        PortableRoute route;bool called=false;
        check(layer->step(0,f.input(0),[&](int64_t,const PortableRoute&,std::vector<float>& rows,std::string&){
            called=true;
            check(layer->memory_usage().device_token_temporary_bound==before.device_token_temporary_bound,
                  "temporary allowance remains live through expert callback");
            rows.assign(25600,0);return true;
        },route,error),error);
        check(called && layer->graph_bytes()>fixed,"token temporary executed");
        check(layer->memory_usage().device_token_temporary_bound==before.device_token_temporary_bound,
              "last-token graph diagnostics do not become retained scheduler bytes");
        std::puts("portable_qsa_memory_test passed: fixed-private versus token-temporary conservation and callback overlap");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
