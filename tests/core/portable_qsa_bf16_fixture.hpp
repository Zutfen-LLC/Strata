#pragma once
#include "portable_qsa_fixture.hpp"
#include "strata/artifact/dequant.hpp"

namespace qsa_fixture {
struct Bf16Fixture {
    Fixture base;
    std::array<std::vector<uint8_t>, 2> bf16;
    std::array<std::vector<float>, 2> widened;
    std::vector<PortableTensorView> views, f32_views;
    Bf16Fixture() : views(base.views), f32_views(base.views) {
        for (size_t role = 0; role < 2; ++role) {
            const size_t weight = 10 + role;
            const size_t elements = size_t(2560 * (role == 0 ? 512 : 128));
            widened[role].resize(elements);
            ggml_get_type_traits(GGML_TYPE_Q8_0)
                ->to_float(base.data[weight].data(), widened[role].data(), int64_t(elements));
            // A finite BF16 value outside FP16 range must not be downcast by
            // loading or by the resulting F32 matrix graph.
            widened[role][33] = 65536.0f;
            bf16[role].resize(elements * 2);
            for (size_t i = 0; i < elements; ++i) {
                uint32_t bits;
                std::memcpy(&bits, &widened[role][i], 4);
                check((bits & 0xffffu) == 0, "fixture values must be exactly BF16");
                const uint16_t high = uint16_t(bits >> 16);
                bf16[role][2 * i] = uint8_t(high);
                bf16[role][2 * i + 1] = uint8_t(high >> 8);
            }
            const std::string name = role == 0 ? "blk.3.indexer.q_proj.weight" : "blk.3.indexer.k_proj.weight";
            for (auto &v : views)
                if (v.name == name) {
                    v.type = 30;
                    v.data = bf16[role].data();
                    v.bytes = bf16[role].size();
                }
            for (auto &v : f32_views)
                if (v.name == name) {
                    v.type = 0;
                    v.data = widened[role].data();
                    v.bytes = widened[role].size() * sizeof(float);
                }
        }
    }
};
} // namespace qsa_fixture
