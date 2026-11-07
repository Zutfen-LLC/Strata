#include "native_component_host_bound.hpp"
#include "native_q8_gu_spirv.hpp"
#include "strata_native_ple_gate_t1_spv.hpp"
#include <stdexcept>
#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (false)
#include <limits>
using namespace strata::core::portable_detail;
static void check(const uint32_t* code,size_t count) {
    auto strict=native_stage_float_controls(code,count);
    CHECK(strict.size()==count+18);
    auto result=native_component_float_controls(code,count,NativeComponentAddressing::descriptor);
    CHECK(result.size()==count+12);
    CHECK(native_component_rewrite_host_bound(count)==2*(count+18)*sizeof(uint32_t));
}
int main() {
    const uint32_t synthetic[]={0x07230203,0x10000,0,8,0,
        (2u<<16)|spv::OpCapability,spv::CapabilityShader,
        (3u<<16)|spv::OpMemoryModel,spv::AddressingModelLogical,spv::MemoryModelGLSL450,
        (4u<<16)|spv::OpEntryPoint,spv::ExecutionModelGLCompute,1,0};
    check(synthetic,sizeof(synthetic)/4);
    check(strata_native_ple_gate_t1_spv,sizeof(strata_native_ple_gate_t1_spv)/4);
    bool overflow=false; try { (void)native_component_rewrite_host_bound(std::numeric_limits<size_t>::max()); } catch(const std::overflow_error&) { overflow=true; } CHECK(overflow);
    for (auto kind : {NativeComponentHostKind::hc, NativeComponentHostKind::hc_pending,
                      NativeComponentHostKind::hc_write, NativeComponentHostKind::hc_mix,
                      NativeComponentHostKind::hc_head_mix, NativeComponentHostKind::ple_conv,
                      NativeComponentHostKind::ple_gate, NativeComponentHostKind::ple_norm,
                      NativeComponentHostKind::ple_key_quant, NativeComponentHostKind::ple_key_mmvq,
                      NativeComponentHostKind::ple_value}) {
        auto actual = native_component_factory_host_bound(kind, 20);
        CHECK(actual >= native_component_metadata_arena_bytes + native_component_rewrite_host_bound(0));
        CHECK(native_component_factory_host_bound(kind, 21) == actual + 1);
        int factory_calls = 0;
        bool short_rejected = false;
        try { native_component_admit(actual, actual - 1, [&]{ ++factory_calls; return 0; }); }
        catch (const std::length_error&) { short_rejected = true; }
        CHECK(short_rejected && factory_calls == 0);
        native_component_admit(actual, actual, [&]{ ++factory_calls; return 0; });
        CHECK(factory_calls == 1);
        bool path_overflow = false;
        try { (void)native_component_factory_host_bound(kind, std::numeric_limits<size_t>::max()); }
        catch (const std::overflow_error&) { path_overflow = true; }
        CHECK(path_overflow);
    }
    int calls=0; auto bound=native_component_rewrite_host_bound(100); bool rejected=false;
    try { native_component_admit(bound,bound-1,[&]{++calls;return 0;}); } catch(const std::length_error&) { rejected=true; }
    CHECK(rejected && !calls); native_component_admit(bound,bound,[&]{++calls;return 0;}); CHECK(calls==1);
}
