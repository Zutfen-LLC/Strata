#pragma once
#include "../../src/core/portable_logits_guard.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <bit>

inline void logits_guard_test(ggml_backend_t backend) {
    using namespace strata::core;
    auto check = [](bool ok, const char *message) { if (!ok) throw std::runtime_error(message); };
    portable_detail::PortableLogitsGuard guard(backend, 2u << 20);
    check(guard.host_bound() > 0, "guard host resource accounting");
    if (guard.device_bytes()) {
        check(guard.device_bytes() >= 64 && guard.device_resource_bound() == (1u << 20), "guard aligned device ledger");
        bool refused = false;
        try { portable_detail::PortableLogitsGuard too_small(backend, 0); }
        catch (const std::runtime_error &) { refused = true; }
        check(refused, "guard device admission must precede allocation");
    }
    for (int64_t n : {1, 15, 16, 17, 255, 256, 257, 4097, 248320})
    for (int offset : {0, 1}) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({4096, nullptr, true}), ggml_free);
        check(bool(ctx), "guard context");
        auto *storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n + offset);
        auto *input = offset ? ggml_view_1d(ctx.get(), storage, n, sizeof(float)) : storage;
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
        check(bool(buffer), "guard input allocation");
        const uint32_t valid[] = {0, 0x80000000u, 1, 0x80000001u, 0x00800000u, 0x80800000u, 0x7f7fffffu, 0xff7fffffu};
        std::vector<uint32_t> values(size_t(n), 0x7f800001u);
        auto run = [&] {
            // Upload raw words: no host floating arithmetic or classification of
            // the input. Vulkan check executes the same bridge as PortableHead.
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(uint32_t));
            return guard.check(input);
        };
        check(!run(), "first guard invocation must reject signaling NaNs");
        for (size_t i = 0; i < values.size(); ++i) values[i] = valid[i % std::size(valid)];
        check(run(), "guard rejects finite edges");
        for (size_t at : {size_t(0), size_t(n / 2), size_t(n - 1)}) {
            const uint32_t saved = values[at];
            for (uint32_t bad : {0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u}) {
                values[at] = bad;
                check(!run(), "guard missed infinity/quiet/signaling NaN");
                values[at] = saved;
                check(run(), "all status words must refresh after failure");
            }
        }
        check(run(), "guard repeated finite result");
    }
}
