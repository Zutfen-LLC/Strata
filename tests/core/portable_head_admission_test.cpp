#include "../../src/core/portable_head_admission.hpp"
#include "../../src/core/portable_memory.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <cstring>
#ifdef STRATA_PORTABLE_MODEL_VULKAN
#include "ggml-vulkan.h"
#endif
using namespace strata::core;
namespace D = strata::core::portable_detail;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
}
int main(int argc,char** argv) {
    try {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2u<<20,nullptr,true}),ggml_free);
        check(bool(ctx), "metadata context");
        auto* x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q8_0,2560,248320);
        auto* y=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2560,1);
        auto* output=ggml_mul_mat(ctx.get(),x,y);
        const std::string pci="0000:07:00.0";
        auto limit=D::match_storage_limit(pci,{{"0000:0b:00.0",4294967295ull},{pci,4294967295ull}});
        check(ggml_nbytes(x)==675430400,"exact Q8 output head size");
        check(D::direct_head_q8_matvec(output,limit),"verified contiguous Q8/F32 T1 head fits storage range");
        check(D::direct_head_q8_matvec(output,{pci,675430400}),"range equality does not trigger huge split");
        for (auto missing : {D::PhysicalStorageLimit{pci,675430399}, {pci,0}, {"",4294967295ull},
                             D::match_storage_limit(pci,{}), D::match_storage_limit("",{{pci,4294967295ull}}),
                             D::match_storage_limit(pci,{{"0000:03:00.0",4294967295ull}}),
                             D::match_storage_limit(pci,{{pci,4294967295ull},{pci,4294967295ull}})})
            check(!D::direct_head_q8_matvec(output,missing),"missing/ambiguous/mismatched PCI or range must retain allowance");
        check(!D::direct_head_q8_matvec(nullptr,limit),"missing graph");
        auto mutate=[&](auto change){
            const ggml_tensor saved_x=*x,saved_y=*y,saved_out=*output;
            change();
            check(!D::direct_head_q8_matvec(output,limit),"unsupported head contract must retain allowance");
            *x=saved_x;*y=saved_y;*output=saved_out;
        };
        mutate([&]{x->type=GGML_TYPE_F32;});
        mutate([&]{y->type=GGML_TYPE_F16;});
        mutate([&]{output->type=GGML_TYPE_F16;});
        mutate([&]{x->ne[0]=1280;});
        mutate([&]{x->ne[1]=248319;});
        mutate([&]{x->ne[2]=2;});
        mutate([&]{x->ne[3]=2;});
        mutate([&]{y->ne[0]=1280;});
        mutate([&]{y->ne[1]=output->ne[1]=2;});
        mutate([&]{y->ne[2]=2;});
        mutate([&]{output->ne[0]=248319;});
        mutate([&]{output->ne[3]=2;});
        mutate([&]{x->nb[1]+=34;});
        mutate([&]{y->nb[0]+=4;});
        mutate([&]{output->nb[0]+=4;});
        mutate([&]{x->view_src=y;});
        mutate([&]{output->op=GGML_OP_ADD;});
        mutate([&]{output->op_params[1]=1;});
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
        check(D::head_storage_limit(cpu.get()).max_storage_buffer_range==0,"CPU backend has no Vulkan proof");
        check(D::head_storage_limit(nullptr).max_storage_buffer_range==0,"missing backend has no proof");

        constexpr size_t M=1u<<20;
        // Full vocabulary metadata only: no tensor payloads or head graph allocated.
        // Match the loader's Q8 schema, with one dummy byte that sizing must not read.
        const uint8_t dummy=0;
        std::vector<PortableTensorView> weights{
            {"output_hc_norm.weight",{10240},0,&dummy,40960},
            {"output_hc_down.weight",{10240,320},8,&dummy,3481600},
            {"output_hc_up.weight",{320,10240},8,&dummy,3481600},
            {"output.weight",{2560,248320},8,&dummy,675430400}};
        const PortableHeadGeometry geometry{2560,4,320,248320,1e-6f};
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> vk(nullptr,ggml_backend_free);
        if(argc!=1) {
            check(argc==3 && std::string(argv[1])=="--vulkan","usage: portable_head_admission_test [--vulkan INDEX]");
#ifdef STRATA_PORTABLE_MODEL_VULKAN
            size_t used=0;const auto index=std::stoul(argv[2],&used);
            check(used==std::strlen(argv[2]) && index<size_t(ggml_backend_vk_get_device_count()),"Vulkan index");
            vk.reset(ggml_backend_vk_init(index));check(bool(vk),"Vulkan backend");
#else
            throw std::runtime_error("Vulkan not compiled");
#endif
        }
        PortableFfnOptions estimate_options;estimate_options.weight_budget=1ull<<30;
        const auto host_estimate=PortableHead::admission(geometry,weights,estimate_options,true);
        check(host_estimate.scheduler_bound>=675430400 && host_estimate.scheduler_bound<800*M,
              "full Q8 head source copy remains charged without payload allocation");
        std::printf("head_shape_host scheduler_bound=%zu cpu_workspace=%zu total=%zu verified=0\n",
                    host_estimate.scheduler_bound,host_estimate.cpu_workspace,host_estimate.graph_bound);
        check(!host_estimate.verified_vulkan && !host_estimate.direct_q8,"CPU metadata cannot earn Vulkan credit");
        check(host_estimate.graph_bound==host_estimate.scheduler_bound+host_estimate.cpu_workspace,
              "host metadata has no device workspace/status allocation");
        estimate_options.request_accelerator=true;estimate_options.accelerator=cpu.get();
        const auto unknown_estimate=PortableHead::admission(geometry,weights,estimate_options,true);
        check(!unknown_estimate.verified_vulkan && !unknown_estimate.direct_q8,
              "unverified backend retains fallback, even with supported shapes");
        check(unknown_estimate.device_workspace==2542796800ull+16056320,"unknown path retains full conversion");
        check(unknown_estimate.guard_bound>=(1u<<20)+64,"integer status and resource allowance retained");
        if(vk) {
            estimate_options.accelerator=vk.get();estimate_options.allow_cpu_fallback=false;
            const auto physical=PortableHead::admission(geometry,weights,estimate_options,true);
            check(physical.verified_vulkan && physical.direct_q8,"physical direct-Q8 contract required; no fallback");
            check(physical.device_workspace==13107200+16056320,"physical proven HC/other workspace unchanged");
            check(physical.graph_bound<1024*M,"metadata head bound below 1GiB without removing Q8 copy");
            D::preflight_device_memory(6000*M,7500*M,physical.graph_bound);
            D::preflight_device_memory(6000*M,6000*M+physical.graph_bound,physical.graph_bound);
            bool short_room=false;
            try {D::preflight_device_memory(6000*M,6000*M+physical.graph_bound-1,physical.graph_bound);}
            catch(const std::exception&){short_room=true;}
            check(short_room,"exact head metadata bound refuses one byte short");
            estimate_options.allow_cpu_fallback=true;
            check(!PortableHead::admission(geometry,weights,estimate_options,true).verified_vulkan,
                  "fallback execution cannot earn tight Vulkan admission");
            std::printf("head_shape_physical scheduler_bound=%zu cpu_workspace=%zu device_workspace=%zu guard=%zu total=%zu direct=1 verified=1\n",
                physical.scheduler_bound,physical.cpu_workspace,physical.device_workspace,physical.guard_bound,physical.graph_bound);
        }
        const size_t aligned=PortableHead::weight_allocation_size(cpu.get(),geometry,weights);
        check(aligned>=682434560 && aligned<1024*M,"complete aligned head weights below policy ceiling");
        bool legacy_refused=false;
        try { D::preflight_device_memory(2500*M,7500*M,4096*M+1024*M); }
        catch(const std::exception&) { legacy_refused=true; }
        check(legacy_refused,"reproduce fixed head ceiling refusal before construction");
        // Only weight sizing changes; retain the entire graph/workspace ceiling.
        D::preflight_device_memory(2500*M,7500*M,4096*M+aligned);
        D::preflight_device_memory(2500*M,2500*M+4096*M+aligned,4096*M+aligned);
        bool underestimated=false;
        try { D::preflight_device_memory(2500*M,2500*M+4096*M+aligned-1,4096*M+aligned); }
        catch(const std::exception&) { underestimated=true; }
        check(underestimated,"one byte below complete aligned head bound still refuses");
        auto invalid_weights=weights;invalid_weights[3].bytes--;
        bool invalid=false;
        try { PortableHead::weight_allocation_size(cpu.get(),geometry,invalid_weights); }
        catch(const std::exception&) { invalid=true; }
        check(invalid,"head estimate validates payload metadata");
        invalid=false;
        try { PortableHead::weight_allocation_size(nullptr,geometry,weights); }
        catch(const std::exception&) { invalid=true; }
        check(invalid,"head estimate requires selected backend");
        // Small F32 tensors expose padding: a payload sum is not a backend bound.
        std::vector<PortableTensorView> small{
            {"output_hc_norm.weight",{4},0,&dummy,16},
            {"output_hc_down.weight",{4,1},0,&dummy,16},
            {"output_hc_up.weight",{1,4},0,&dummy,16},
            {"output.weight",{1,3},0,&dummy,12}};
        check(PortableHead::weight_allocation_size(cpu.get(),{1,4,1,3,1e-6f},small)>60,
              "head estimate includes backend tensor padding");
        const size_t full=2560ull*248320*4, hc=2560ull*4*320*4;
        const size_t other=(248320ull*16+2560*4*4)*4;
        check(full==2542796800ull && hc==13107200 && other==16056320,"head component arithmetic");
        check(D::head_conversion_bound(full,hc,false)==full,"conservative path unchanged");
        check(D::head_conversion_bound(full,hc,true)==hc,"HC conversion retained");
        auto fresh=[] {
            PortableModelReport r;r.device_admission={0};r.device_scheduler={0};r.device_state={0};r.device_weights={0};r.device_workspace_bound={0};
            r.device_shared_conversion_bound={0};r.device_token_temporary_bound={0};return r;
        };
        auto initial=fresh();PortableMemoryUsage layer;
        layer.device_graph=64*M;layer.device_extra_bound=8*M;
        layer.device_shared_conversion_bound=100*M;layer.device_token_temporary_bound=128*M;
        D::account_device_memory(initial,0,8ull<<30,layer);
        auto corrected=initial,conservative=initial;
        PortableMemoryUsage head;head.device_graph=32*M;head.device_extra_bound=other;
        head.device_shared_conversion_bound=D::head_conversion_bound(full,hc,true);
        D::account_device_memory(corrected,0,8ull<<30,head);
        head.device_shared_conversion_bound=D::head_conversion_bound(full,hc,false);
        D::account_device_memory(conservative,0,8ull<<30,head);
        check(conservative.device_admission[0]-corrected.device_admission[0]==full-100*M,
              "only proven output conversion growth removed from shared ledger");
        check(corrected.device_scheduler[0]==96*M && corrected.device_workspace_bound[0]==8*M+other &&
              corrected.device_shared_conversion_bound[0]==100*M && corrected.device_token_temporary_bound[0]==128*M,
              "scheduler, other private workspace, existing conversion and QSA peak unchanged");
        PortableMemoryUsage expert;expert.device_graph=160*M;expert.device_extra_bound=32*M;
        const auto before=corrected.device_admission[0];
        D::account_device_memory(corrected,0,8ull<<30,expert);
        check(corrected.device_admission[0]==before+192*M,"expert tensors/workspace remain additive");
        auto capped=initial;bool refused=false;
        try { D::account_device_memory(capped,0,initial.device_admission[0],head); }
        catch(const std::exception&) { refused=true; }
        check(refused && capped.device_admission==initial.device_admission,"physical cap ledger remains fail closed");
        std::puts("portable_head_admission_test passed: exact shape/range/PCI proof, conservative fallbacks, component ledger");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr,"%s\n",error.what());return 1;
    }
}
