#pragma once
#include "portable_gdn_fixture.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
namespace gdn_budget_fixture {
using namespace strata::core;
using gdn_fixture::check;
// A CPU buffer type can report extra input storage. Calibrate an independent
// FFN plan to the exact remaining cap, then reuse that requirement across a
// GDN borrower transition. Count actual backend allocations after planning
// sees the T2 input; weight and metadata creation are outside this counter.
struct AllocationProbe {
    static inline decltype(ggml_backend_buffer_type_i::get_alloc_size) original_size=nullptr;
    static inline decltype(ggml_backend_buffer_type_i::alloc_buffer) original_alloc=nullptr;
    static inline size_t input_bytes=0,allocations=0;
    static inline bool seen=false;
    static size_t size(ggml_backend_buffer_type_t buft,const ggml_tensor* tensor) {
        if(tensor->op==GGML_OP_NONE && std::string(ggml_get_name(tensor))=="residual" && tensor->ne[2]==2) {
            seen=true;return std::max(input_bytes,ggml_nbytes(tensor));
        }
        return original_size?original_size(buft,tensor):ggml_nbytes(tensor);
    }
    static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t buft,size_t bytes) {
        if(seen)++allocations;
        return original_alloc(buft,bytes);
    }
    struct Scope {
        ggml_backend_buffer_type_t buft;
        explicit Scope(size_t bytes):buft(ggml_backend_cpu_buffer_type()) {
            original_size=buft->iface.get_alloc_size;original_alloc=buft->iface.alloc_buffer;
            input_bytes=bytes;allocations=0;seen=false;
            buft->iface.get_alloc_size=size;buft->iface.alloc_buffer=allocate;
        }
        ~Scope(){buft->iface.get_alloc_size=original_size;buft->iface.alloc_buffer=original_alloc;}
    };
};
inline void retained_projection_transition() {
    gdn_fixture::Fixture fixture;std::string error;
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    PortableGdnOptions options;options.execution.accelerator=backend.get();options.execution.request_accelerator=true;
    options.batch_input_projections=true;options.resident_state=true;
    options.scratch=PortableGdnScratch::create(options.execution,error);check(bool(options.scratch),error);
    auto a=PortableGdnLayer::create(0,fixture.views,options,error);
    auto views=fixture.views;for(auto& view:views)view.name.replace(0,6,"blk.2.");
    auto b=PortableGdnLayer::create(2,views,options,error);check(bool(a)&&bool(b),error);
    const size_t before=options.scratch->graph_bytes();
    check(a->prepare_chunk(8,error),error);
    const size_t retained=options.scratch->graph_bytes();
    check(retained>before,"T8 adds retained projection backing/workspace");
    const size_t private_t1=b->graph_bytes();
    check(retained+private_t1<options.execution.graph_budget,"remaining FFN cap exists");
    const size_t remaining=options.execution.graph_budget-retained-private_t1;
    std::vector<PortableTensorView> weights;
    for(const auto& view:fixture.views)if(view.name.rfind("blk.0.hc_ffn_",0)==0 || view.name.rfind("blk.0.ffn_",0)==0)weights.push_back(view);
    auto plan=[&](size_t padded) {
        AllocationProbe::Scope probe(padded);
        auto execution=options.execution;execution.graph_budget=remaining;
        auto control=PortableFfn::create({2560,4,320,640,512,10,1e-6f},0,weights,execution,error,2);
        check(AllocationProbe::seen,"independent FFN planner sees padded T2 input");
        return AllocationProbe::allocations>0;
    };
    // Find the first backend input requirement that cannot reach allocation
    // under the independently computed cap. This does not consult GDN's plan.
    size_t low=2*10240*sizeof(float),high=options.execution.graph_budget;
    check(plan(low) && !plan(high),"independent tight-cap controls bracket admission");
    while(low+1<high){const size_t middle=low+(high-low)/2;if(plan(middle))low=middle;else high=middle;}
    size_t allocations=0;bool ok=false;
    {
        AllocationProbe::Scope probe(high);
        ok=b->prepare_chunk(2,error);
        allocations=AllocationProbe::allocations;
        check(AllocationProbe::seen,"replacement T2 FFN planning reached");
    }
    check(!ok && allocations==0 && error.find("budget")!=std::string::npos,
          "retained T8 projection must reject other borrower's T2 FFN before backend allocation");
    check(options.scratch->graph_bytes()==retained && a->tokens()==0 && b->tokens()==0,
          "rejected planning preserves retained backing and sequence state");
    check(b->reset(error) && b->prepare_chunk(2,error),error);
    std::printf("[gdn-shared-budget] T8_to_other_T2=1 retained_projection_family=%zu private_t1=%zu remaining_cap=%zu last_admitted_input=%zu first_rejected_input=%zu rejected_backend_allocations=%zu\n",
        retained,private_t1,remaining,low,high,allocations);
}
}
