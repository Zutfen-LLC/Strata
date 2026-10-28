#include "gguf_fixture.hpp"
#include "portable_qsa_bf16_fixture.hpp"
#include <chrono>
using namespace qsa_fixture;
namespace {
std::vector<fixture::Kv> metadata() {
    std::vector<fixture::Kv> kv{fixture::str("general.architecture", "qwen4exp")};
    for (auto pair : {std::pair<const char *, uint64_t>{"embedding_length", 2560},
                      {"block_count", 48},
                      {"expert_count", 512},
                      {"expert_used_count", 10},
                      {"expert_feed_forward_length", 640},
                      {"expert_shared_feed_forward_length", 640},
                      {"hyper_connection.count", 4},
                      {"hyper_connection.low_rank", 320},
                      {"attention.head_count", 24},
                      {"attention.head_count_kv", 2},
                      {"attention.key_length", 256},
                      {"attention.value_length", 256},
                      {"attention.indexer.head_count", 4},
                      {"attention.indexer.key_length", 128},
                      {"attention.indexer.top_k", 2048},
                      {"rope.dimension_count", 64},
                      {"full_attention_interval", 4}})
        kv.push_back(fixture::u32("qwen4exp." + std::string(pair.first), pair.second));
    kv.push_back(fixture::f32("qwen4exp.rope.freq_base", 1e7f));
    kv.push_back(fixture::f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f));
    kv.push_back(fixture::u32s("qwen4exp.rope.dimension_sections", {11, 11, 10, 0}));
    std::vector<uint64_t> ratios(48, 0);
    for (size_t i = 3; i < 48; i += 4)
        ratios[i] = 4;
    kv.push_back(fixture::u32s("qwen4exp.attention.compress_ratios", ratios));
    return kv;
}
// Existing fixture writer supplies the GGUF metadata; stream actual fleet-type
// payloads separately so malformed metadata tests need no large payload copy.
void model_file(const std::filesystem::path &path, const std::vector<fixture::Kv> &kv,
                const std::vector<PortableTensorView> &views) {
    std::vector<uint8_t> b;
    auto put = [&](const void *p, size_t n) {
        auto *v = static_cast<const uint8_t *>(p);
        b.insert(b.end(), v, v + n);
    };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    auto str = [&](const std::string &v) {
        u64(v.size());
        put(v.data(), v.size());
    };
    u32(0x46554747);
    u32(3);
    u64(views.size());
    u64(kv.size());
    for (const auto &k : kv) {
        str(k.key);
        u32(k.type);
        if (k.type == 8)
            str(k.s);
        else if (k.type == 4)
            u32(uint32_t(k.u));
        else if (k.type == 6)
            put(&k.f, 4);
        else if (k.type == 9) {
            u32(4);
            u64(k.arr.size());
            for (auto v : k.arr)
                u32(uint32_t(v));
        } else
            check(false, "GGUF fixture type");
    }
    size_t end = 0;
    std::vector<size_t> offsets;
    for (const auto &v : views) {
        str(v.name);
        u32(uint32_t(v.shape.size()));
        for (auto d : v.shape)
            u64(uint64_t(d));
        u32(v.type);
        u64(end);
        offsets.push_back(end);
        end = (end + v.bytes + 31) / 32 * 32;
    }
    b.resize((b.size() + 31) / 32 * 32);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(b.data()), std::streamsize(b.size()));
    for (size_t i = 0; i < views.size(); ++i) {
        file.seekp(std::streamoff(b.size() + offsets[i]));
        file.write(static_cast<const char *>(views[i].data), std::streamsize(views[i].bytes));
    }
    check(bool(file), "GGUF fixture write");
}
} // namespace
int main() {
    const auto root =
        std::filesystem::temp_directory_path() /
        ("strata-qsa-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directory(root);
        const auto path = root / "layer.gguf";
        const auto kv = metadata();
        Fixture f;
        std::string error;
        PortableQsaOptions options;
        options.capacity = 2;
        model_file(path, kv, f.views);
        std::unique_ptr<PortableQsaLayer> bound;
        {
            auto model = strata::GgufModel::open(path.string());
            bound = PortableQsaLayer::from_gguf(model, 3, options, error);
            check(bool(bound), error);
        }
        auto control = PortableQsaLayer::create(3, f.views, options, error);
        check(bool(control), error);
        PortableRoute route, expected;
        check(bound->step(0, f.input(0), experts(), route, error), error);
        check(control->step(0, f.input(0), experts(), expected, error), error);
        close(bound->last_residual(), control->last_residual(), 0, "GGUF owned binding survives model close");
        check(bound->keys() == control->keys() && route.ids == expected.ids, "GGUF exact binding");
        auto late_views = f.views;
        for (auto &v : late_views)
            v.name = "blk.47." + v.name.substr(6);
        model_file(path, kv, late_views);
        {
            auto model = strata::GgufModel::open(path.string());
            auto late = PortableQsaLayer::from_gguf(model, 47, options, error);
            check(bool(late), error);
            check(late->step(0, f.input(0),
                [&](int64_t id, const PortableRoute &r, std::vector<float> &rows, std::string &e) {
                    check(id == 47, "GGUF late QSA dispatch ID");
                    return experts()(3, r, rows, e);
                }, route, error), error);
            close(late->last_residual(), control->last_residual(), 0, "GGUF late QSA binding");
        }
        {
            Bf16Fixture mixed;
            model_file(path, kv, mixed.views);
            std::unique_ptr<PortableQsaLayer> widened;
            {
                auto model = strata::GgufModel::open(path.string());
                widened = PortableQsaLayer::from_gguf(model, 3, options, error);
                check(bool(widened), error);
            }
            auto f32 = PortableQsaLayer::create(3, mixed.f32_views, options, error);
            check(bool(f32), error);
            for (auto &data : mixed.bf16) { data.clear(); data.shrink_to_fit(); }
            check(widened->step(0, mixed.base.input(0), experts(), route, error), error);
            check(f32->step(0, mixed.base.input(0), experts(), expected, error), error);
            close(widened->last_residual(), f32->last_residual(), 0, "BF16 GGUF binding survives closed mapping/source lifetime");
        }
        auto rejection = [&](const std::vector<fixture::Kv> &bad, const char *label) {
            model_file(path, bad, {});
            auto model = strata::GgufModel::open(path.string());
            check(!PortableQsaLayer::from_gguf(model, 3, options, error),
                  std::string("metadata rejection: ") + label);
            check(error.find("missing/extra") == std::string::npos,
                  std::string("metadata accepted before empty tensor rejection: ") + label);
        };
        for (size_t i = 0; i < kv.size(); ++i) {
            auto bad = kv;
            bad.erase(bad.begin() + std::ptrdiff_t(i));
            // Shared FF width is optional upstream and defaults to routed width.
            if (kv[i].key != "qwen4exp.expert_shared_feed_forward_length")
                rejection(bad, kv[i].key.c_str());
            bad = kv;
            if (bad[i].type == 8)
                bad[i].s = "wrong";
            else if (bad[i].type == 4)
                bad[i].u += 1;
            else if (bad[i].type == 6)
                bad[i].f *= 2;
            else
                bad[i].arr[0] += 1;
            rejection(bad, kv[i].key.c_str());
        }
        for (const char *key :
             {"rope.scaling.factor", "rope.scale_linear", "rope.scaling.alpha", "rope.scaling.attn_factor",
              "attention.scale", "rope.scaling.yarn_ext_factor", "rope.scaling.yarn_attn_factor",
              "rope.scaling.yarn_beta_fast", "rope.scaling.yarn_beta_slow", "rope.scaling.yarn_log_multiplier",
              "rope.freq_factors"}) {
            auto bad = kv;
            bad.push_back(fixture::f32("qwen4exp." + std::string(key), 2.0f));
            rejection(bad, key);
        }
        for (const char *key : {"rope.scaling.finetuned", "rope.scaling.original_context_length"}) {
            auto bad = kv;
            bad.push_back(fixture::u32("qwen4exp." + std::string(key), 1));
            rejection(bad, key);
        }
        auto scaled = kv;
        scaled.push_back(fixture::str("qwen4exp.rope.scaling.type", "yarn"));
        rejection(scaled, "scaling.type");
        auto ple = kv;
        ple.push_back(fixture::u32s("qwen4exp.ple.layers", {3}));
        rejection(ple, "PLE at QSA layer");
        auto placeholder = kv;
        placeholder.back().arr.assign(48, 0);
        rejection(placeholder, "placeholder ratios");
        // create() validates every actual attention type/shape/byte count before graph execution.
        for (size_t i = 9; i < f.views.size(); ++i) {
            auto views = f.views;
            views[i].type = views[i].type == 8 ? 0 : 8;
            check(!PortableQsaLayer::create(3, views, options, error), "incorrect attention GGUF type accepted");
            views = f.views;
            views[i].bytes -= 1;
            check(!PortableQsaLayer::create(3, views, options, error), "incorrect payload size accepted");
        }
        std::filesystem::remove_all(root);
        std::printf("PASS portable QSA GGUF actual Q8_0/F32 owned binding; required metadata and unsupported "
                    "scaling/type rejection\n");
    } catch (const std::exception &e) {
        std::filesystem::remove_all(root);
        std::fprintf(stderr, "FAIL GGUF QSA: %s\n", e.what());
        return 1;
    }
}
