#include "../../src/core/portable_head_admission.hpp"
#include "../../src/core/portable_memory.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <memory>
#include <stdexcept>
using namespace strata::core;
namespace D = strata::core::portable_detail;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
}
int main() {
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
