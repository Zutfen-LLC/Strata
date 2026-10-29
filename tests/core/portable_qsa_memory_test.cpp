#include "portable_qsa_production_fixture.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <new>
namespace { bool fail_next_allocation = false; }
void *operator new(std::size_t n) {
    if (fail_next_allocation) { fail_next_allocation = false; throw std::bad_alloc(); }
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void *p) noexcept { std::free(p); }
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
using namespace qsa_fixture;
int main(){
    try{
        Fixture f;
        for (bool reuse : {false, true}) {
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
        PortableQsaOptions opts;opts.capacity=9;opts.reuse_attention=reuse;
        opts.execution.accelerator=backend.get();opts.execution.request_accelerator=true;
        std::string error;
        auto layer=PortableQsaLayer::create(3,f.views,opts,error);check(bool(layer),error);
        const size_t fixed=layer->graph_bytes();const auto before=layer->memory_usage();
        if (reuse) {
            auto tight=opts;tight.execution.graph_budget=fixed-1;
            check(!PortableQsaLayer::create(3,f.views,tight,error),"capacity priming rejects inadequate graph cap");
        }
        check(fixed<opts.execution.graph_budget,"fixture has temporary capacity");
        check(before.device_token_temporary_bound==(reuse ? 0 : opts.execution.graph_budget-fixed),
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
        check(called && (reuse ? layer->graph_bytes()==fixed : layer->graph_bytes()>fixed),"attention scratch execution ledger");
        check(layer->memory_usage().device_token_temporary_bound==before.device_token_temporary_bound,
              "last-token graph diagnostics do not become retained scheduler bytes");
        check(layer->reset(error),error);
        const auto after=layer->memory_usage();
        check(after.host_graph==before.host_graph && after.device_graph==before.device_graph &&
              after.device_extra_bound==before.device_extra_bound &&
              after.device_token_temporary_bound==before.device_token_temporary_bound,"reset retained memory ledger");
        }
        {
            std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
            PortableQsaOptions opts; opts.execution.accelerator=backend.get(); opts.execution.request_accelerator=true;
            for (bool combine : {false, true}) {
                opts.device_combine=combine;
                production_tests(opts, f.views, [&](int t){return f.input(t);});
            }
            opts.device_combine=false;
            opts.capacity=256; opts.reuse_attention=true; opts.production=true;
            std::string error;
            auto layer=PortableQsaLayer::create(3,f.views,opts,error); check(bool(layer),error);
            auto merged_options=opts; merged_options.device_combine=true;
            auto merged=PortableQsaLayer::create(3,f.views,merged_options,error); check(bool(merged),error);
            const auto baseline_memory=layer->memory_usage(), merged_memory=merged->memory_usage();
            const auto device_bound=[](const auto& m) {return m.device_graph+m.device_extra_bound+m.device_token_temporary_bound+m.device_shared_conversion_bound+m.device_state+m.device_weights;};
            check(device_bound(baseline_memory)==device_bound(merged_memory),"unchanged whole-graph device admission envelope");
            check(baseline_memory.host_extra_bound+baseline_memory.host_graph+baseline_memory.device_graph==
                  merged_memory.host_extra_bound+merged_memory.host_graph+merged_memory.device_graph,
                  "host allowance reconciles the unchanged graph cap");
            std::printf("QSA C256 selected-CPU admission: old_graph=%zu merged_graph=%zu old_device_bound=%zu merged_device_bound=%zu\n",
                layer->graph_bytes(),merged->graph_bytes(),device_bound(baseline_memory),device_bound(merged_memory));
            std::printf("QSA C256 admission host: old=%zu merged=%zu actual_scheduler_device: old=%zu merged=%zu\n",
                baseline_memory.host_extra_bound+baseline_memory.host_graph,merged_memory.host_extra_bound+merged_memory.host_graph,
                baseline_memory.device_graph,merged_memory.device_graph);
            auto tight=merged_options; tight.execution.graph_budget=merged->graph_bytes()-1;
            check(!PortableQsaLayer::create(3,f.views,tight,error),"merged capacity priming enforces unchanged graph cap");
            PortableQsaState state; check(layer->inspect_state(state,error),error);
            const size_t raw=(state.keys.size()+state.values.size())*2+
                (state.pooled.size()+state.spare.size()+state.tail.size()+state.bias.size()+state.mapping.size())*4;
            check(raw==562176,"C256 raw persistent payload derived from allocated tensor shapes");
            check(layer->memory_usage().device_state>=raw,"aligned actual state ledger");
            check(PortableQsaLayer::device_state_allocation_size(backend.get(),opts.capacity)==
                  layer->memory_usage().device_state,"preallocation estimate equals actual aligned buffer");
            for (size_t invalid : {size_t(0), size_t(262145), SIZE_MAX}) {
                bool refused=false;
                try { PortableQsaLayer::device_state_allocation_size(backend.get(),invalid); }
                catch(const std::exception&) { refused=true; }
                check(refused,"invalid/overflowing estimate capacity rejected before tensor construction");
            }
            auto limited=opts; limited.state_budget=layer->state_bytes()-1;
            check(!PortableQsaLayer::create(3,f.views,limited,error),"actual aligned device state budget enforced");
            fail_next_allocation=true;
            const bool reset_ok=layer->reset(error);
            check(!reset_ok && layer->poisoned(),"reset initialization failure retains poison");
            PortableRoute route;
            check(!layer->step(0,f.input(0),experts(),route,error),"failed reset cannot resume");
            check(layer->reset(error) && !layer->poisoned(),"successful initialization clears poison");
        }
        std::puts("portable_qsa_memory_test passed: fixed-private versus token-temporary conservation and callback overlap");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
