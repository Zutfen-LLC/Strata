#include "portable_native_ple_key_batch.hpp"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
using namespace strata::core::portable_detail;
namespace {
void need(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
const ggml_tensor* resident = nullptr;
const ggml_tensor* last_packed = nullptr;
size_t quant_calls = 0, dot_calls = 0, creates = 0, quant_budget = 0, dot_budget = 0;
bool fail_quant = false, fail_dot = false;
}
namespace strata::core::portable_detail {
struct StrataVkNativeStages {};
struct StrataVkQ8Wo {};
StrataVkNativeStages* strata_vk_native_stages_create_quant_rcp_observation(ggml_backend_t, const char* p, size_t budget) {
    need(budget >= 512, "quant budget"); quant_budget = budget; ++creates; std::ofstream(p) << "CPU mock"; return new StrataVkNativeStages;
}
size_t strata_vk_native_stages_bytes(const StrataVkNativeStages*) { return 512; }
void strata_vk_native_stages_destroy(StrataVkNativeStages* p) noexcept { delete p; }
StrataVkQ8Wo* strata_vk_head_vocab_create(ggml_backend_t, const char* p, size_t budget) {
    need(budget >= 512, "dot budget"); dot_budget = budget; ++creates; std::ofstream(p) << "CPU mock"; return new StrataVkQ8Wo;
}
size_t strata_vk_q8_wo_bytes(const StrataVkQ8Wo*) { return 512; }
void strata_vk_q8_wo_destroy(StrataVkQ8Wo* p) noexcept { delete p; }
void strata_vk_ple_key_quantize_bounded(StrataVkNativeStages*, const ggml_tensor* input, ggml_tensor* packed, size_t tokens) {
    ++quant_calls; need(tokens == 1 || tokens == 2, "quant tokens");
    need(ggml_nbytes(input) == tokens * 2560 * 4 && ggml_nbytes(packed) == tokens * 2880, "quant bank spans");
    if (fail_quant) throw std::runtime_error("injected quant failure");
    std::vector<float> x(tokens * 2560); ggml_backend_tensor_get(input, x.data(), 0, x.size()*4);
    std::vector<uint8_t> q(tokens * 2880, 0);
    for (size_t t=0; t<tokens; ++t) {
        need(x[t*2560] == float(3 + 2*t), "input row offset");
        for (size_t b=0; b<80; ++b) q[t*2880 + b*36 + 4] = uint8_t(11 + t);
    }
    ggml_backend_tensor_set(packed, q.data(), 0, q.size()); last_packed = packed;
}
void strata_vk_ple_key_run_bounded(StrataVkQ8Wo*, const ggml_tensor* weight, const ggml_tensor* packed, ggml_tensor* output, size_t rows, size_t tokens) {
    ++dot_calls;
    need(rows == 10240 && (tokens == 1 || tokens == 2) && weight->view_src == resident && packed == last_packed, "dot operands");
    need(ggml_nbytes(weight) == ggml_nbytes(resident) && ggml_nbytes(packed) == tokens*2880 && ggml_nbytes(output) == tokens*rows*4, "dot bank spans");
    std::vector<uint8_t> q(tokens*2880); ggml_backend_tensor_get(packed, q.data(), 0, q.size());
    for (size_t t=0; t<tokens; ++t) for (size_t b=0; b<80; ++b)
        need(q[t*2880+b*36+4] == 11+t, "packed row or block offset");
    if (fail_dot) throw std::runtime_error("injected dot failure");
    std::vector<float> y(tokens*rows);
    for (size_t t=0; t<tokens; ++t) for (size_t r=0; r<rows; ++r) y[t*rows+r] = float(1 + t*rows + r);
    ggml_backend_tensor_set(output, y.data(), 0, y.size()*4);
}
}
int main(int argc, char** argv) { try {
    need(argc == 2, "scratch root");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({16384,nullptr,true}), ggml_free);
    need(bool(backend) && bool(ctx), "CPU setup");
    auto* weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 2560, 10240); resident = weight;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> wb(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    need(bool(wb), "weight allocation"); std::vector<uint8_t> role(ggml_nbytes(weight), 0);
    ggml_backend_tensor_set(weight, role.data(), 0, role.size());
    auto dir = std::filesystem::path(argv[1]) / ("ple-key-batch-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto budget = PortableNativePleKeyBatch::estimate(backend.get());
    bool rejected = false; try { PortableNativePleKeyBatch bad(backend.get(),weight,role.data(),role.size(),dir.string(),budget-1); } catch (...) { rejected = true; }
    need(rejected && creates == 0 && !std::filesystem::exists(dir), "short budget dispatched");
    PortableNativePleKeyBatch owner(backend.get(),weight,role.data(),role.size(),dir.string(),budget);
    need(owner.bytes() <= budget && creates == 2 && !owner.valid(), "admission");
    need(quant_budget == (3u << 20) && dot_budget == quant_budget - 512 + ggml_backend_buffer_get_size(wb.get()),
         "component budget must subtract the bank and quant allocations");
    auto* bank = owner.output_bank(2); need(ggml_nbytes(bank)==2*10240*4, "full output bank");
    std::vector<float> input(5120,0); input[0]=3; input[2560]=5;
    need(owner.run(input,2)==bank && owner.valid() && owner.calls()==1 && quant_calls==1 && dot_calls==1, "T2 operation counts");
    std::vector<float> out(20480); ggml_backend_tensor_get(bank,out.data(),0,out.size()*4);
    for (size_t i=0;i<out.size();++i) need(std::isfinite(out[i]) && out[i]==float(i+1), "T2 row canary");
    std::vector<float> tail(2560,0); tail[0]=3;
    need(owner.run(tail,1)==owner.output_bank(1) && owner.valid() && owner.calls()==2 && quant_calls==2 && dot_calls==2, "T1 tail");
    std::vector<float> saved(out.size()); ggml_backend_tensor_get(bank,saved.data(),0,saved.size()*4);
    need(saved==out, "T1 modified inactive output row or first row");
    fail_dot=true; rejected=false; try { owner.run(input,2); } catch (...) { rejected=true; }
    need(rejected && !owner.valid() && owner.calls()==2 && quant_calls==3 && dot_calls==3, "dot failure accepted");
    std::vector<float> after(out.size()); ggml_backend_tensor_get(bank,after.data(),0,after.size()*4); need(after==saved, "failed dot changed output");
    fail_dot=false; fail_quant=true; rejected=false; try { owner.run(input,2); } catch (...) { rejected=true; }
    need(rejected && !owner.valid() && owner.calls()==2 && quant_calls==4 && dot_calls==3, "quant failure accepted");
    ggml_backend_tensor_get(bank,after.data(),0,after.size()*4); need(after==saved, "failed quant changed output");
    fail_quant=false; input[0]=std::numeric_limits<float>::infinity(); rejected=false; try { owner.run(input,2); } catch (...) { rejected=true; }
    need(rejected && !owner.valid() && owner.calls()==2 && quant_calls==4 && dot_calls==3, "invalid input dispatched");
    std::puts("PLE key batch CPU owner mock passed"); return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
