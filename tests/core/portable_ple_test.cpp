#include "gguf_fixture.hpp"
#include "portable_ple_fixture.hpp"
#include <chrono>
using namespace ple_fixture;
namespace {
// Independent newest-first hash oracle, including current EOS retaining its
// predecessors and missing/EOS predecessors cutting all older context.
std::array<uint32_t, 16> oracle(int32_t token, std::array<int32_t, 2> prev, const PortablePleConfig &c) {
    uint64_t ctx[3] = {uint64_t(token), uint64_t(c.eos), uint64_t(c.eos)};
    if (prev[1] >= 0 && prev[1] != c.eos) {
        ctx[1] = uint64_t(prev[1]);
        if (prev[0] >= 0 && prev[0] != c.eos)
            ctx[2] = uint64_t(prev[0]);
    }
    std::array<uint32_t, 16> out{};
    for (size_t h = 0; h < 16; ++h) {
        uint64_t hash = ctx[0] * c.hash.mult[0] ^ ctx[1] * c.hash.mult[1];
        if (h >= 8)
            hash ^= ctx[2] * c.hash.mult[2];
        out[h] = uint32_t(c.hash.offset[h] + hash % c.hash.vocab[h]);
    }
    return out;
}
std::vector<fixture::Kv> metadata() {
    std::vector<fixture::Kv> kv{fixture::str("general.architecture", "qwen4exp"),
                                fixture::u32s("qwen4exp.ple.layers", {1})};
    for (auto pair : {std::pair<const char *, uint64_t>{"block_count", 48},
                      {"embedding_length", 2560},
                      {"expert_count", 512},
                      {"expert_used_count", 10},
                      {"expert_feed_forward_length", 640},
                      {"expert_shared_feed_forward_length", 640},
                      {"attention.head_count", 24},
                      {"attention.head_count_kv", 2},
                      {"ssm.state_size", 128},
                      {"ssm.group_count", 16},
                      {"ssm.time_step_rank", 48},
                      {"ssm.inner_size", 6144},
                      {"ssm.conv_kernel", 4},
                      {"hyper_connection.count", 4},
                      {"hyper_connection.low_rank", 320},
                      {"embedding_length_per_layer_input", 160},
                      {"ple.ngram_size", 3},
                      {"ple.heads_per_ngram", 8},
                      {"ple.conv_kernel", 4},
                      {"ple.eos_token_id", 248044},
                      {"ple.image_token_id", 248056}})
        kv.push_back(fixture::u32("qwen4exp." + std::string(pair.first), pair.second));
    const auto c = strata::kernels::ple_artifact_consts();
    auto arr = [&](const char *name, const uint64_t *v, size_t n) {
        auto a = fixture::u32s("qwen4exp.ple." + std::string(name), {v, v + n});
        a.elem = 10;
        kv.push_back(a);
    };
    arr("layer_multipliers", c.mult, 3);
    arr("head_offsets", c.offset, 16);
    arr("head_vocab_sizes", c.vocab, 16);
    return kv;
}
// Sparse actual-table geometry: writes metadata and selected rows only. No
// allocation, fill or scan of the 54 GB logical table. Also places it behind
// a real layer payload to test nonzero table offsets in a multi-tensor shard.
void sparse(const std::filesystem::path &path, const std::vector<fixture::Kv> &kv,
            const std::vector<PortableTensorView> &views, bool table = true) {
    std::vector<uint8_t> header;
    auto put = [&](const void *p, size_t n) {
        auto b = static_cast<const uint8_t *>(p);
        header.insert(header.end(), b, b + n);
    };
    auto u32 = [&](uint32_t n) { put(&n, 4); };
    auto u64 = [&](uint64_t n) { put(&n, 8); };
    auto str = [&](const std::string &s) {
        u64(s.size());
        put(s.data(), s.size());
    };
    u32(0x46554747);
    u32(3);
    u64(views.size() + (table ? 1 : 0));
    u64(kv.size());
    for (const auto &k : kv) {
        str(k.key);
        u32(k.type);
        if (k.type == 8)
            str(k.s);
        else if (k.type == 4)
            u32(uint32_t(k.u));
        else if (k.type == 9) {
            u32(k.elem);
            u64(k.arr.size());
            for (auto v : k.arr) {
                if (k.elem == 10)
                    u64(v);
                else
                    u32(uint32_t(v));
            }
        } else
            check(false, "sparse metadata type");
    }
    uint64_t at = 0;
    std::vector<uint64_t> offsets;
    for (const auto &v : views) {
        str(v.name);
        u32(uint32_t(v.shape.size()));
        for (auto d : v.shape)
            u64(uint64_t(d));
        u32(v.type);
        u64(at);
        offsets.push_back(at);
        at = (at + v.bytes + 31) / 32 * 32;
    }
    const uint64_t table_offset = at;
    if (table) {
        str("per_layer_token_embd.weight");
        u32(2);
        u64(160);
        u64(320001536);
        u32(8);
        u64(at);
        at += 320001536ull * 170;
    }
    header.resize((header.size() + 31) / 32 * 32);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(header.data()), std::streamsize(header.size()));
    file.seekp(std::streamoff(header.size() + at - 1));
    file.put(0);
    for (size_t i = 0; i < views.size(); ++i) {
        file.seekp(std::streamoff(header.size() + offsets[i]));
        file.write(static_cast<const char *>(views[i].data), std::streamsize(views[i].bytes));
    }
    if (table) {
        const auto rows = oracle(0, {{-1, -1}}, {});
        const auto bytes = table_bytes(16);
        for (size_t h = 0; h < 16; ++h) {
            file.seekp(std::streamoff(header.size() + table_offset + uint64_t(rows[h]) * 170));
            file.write(reinterpret_cast<const char *>(bytes.data() + h * 170), 170);
        }
    }
    check(bool(file), "sparse GGUF write");
}
} // namespace
int main() {
    try {
        std::string error;
        auto table = small_table(error);
        check(bool(table), error);
        const auto bytes = table_bytes(272);
        const auto config = small_config();
        for (auto token : {0, 7, 248044, 248056, INT32_MAX})
            for (auto prev : {std::array<int32_t, 2>{{-1, -1}},
                              {{5, 0}},
                              {{248044, 9}},
                              {{11, 248044}},
                              {{-1, 8}},
                              {{INT32_MAX, INT32_MAX}}}) {
                std::array<uint32_t, 16> rows{};
                std::vector<float> emb;
                check(table->gather(token, prev, rows, emb, error), error);
                check(rows == oracle(token, prev, config), "independent hash/EOS");
                std::vector<float> reference(2560);
                for (size_t h = 0; h < 16; ++h)
                    ggml_get_type_traits(GGML_TYPE_Q8_0)
                        ->to_float(bytes.data() + size_t(rows[h]) * 170, reference.data() + h * 160, 160);
                close(emb, reference, 0, "pinned ggml Q8 row decoder/head layout");
            }
        PortableTensorView tv{"per_layer_token_embd.weight", {160, 272}, 8, bytes.data(), bytes.size()};
        check(!PortablePleTable::copy(tv, config, 1, error), "copy budget");
        auto bad = config;
        bad.hash.vocab[15] = UINT64_MAX;
        check(!PortablePleTable::copy(tv, bad, bytes.size(), error), "hash overflow/range");
        tv.type = 0;
        check(!PortablePleTable::copy(tv, config, bytes.size(), error), "encoding rejection");
        std::array<uint32_t, 16> rejected_rows{};
        rejected_rows.fill(123);
        std::vector<float> rejected_emb{5};
        check(!table->gather(-1, {{-1, -1}}, rejected_rows, rejected_emb, error), "negative current token");
        check(rejected_rows[0] == 123 && rejected_emb == std::vector<float>{5}, "gather failure preserves outputs");
        auto nonfinite_bytes = bytes;
        const uint16_t infinity = 0x7c00;
        for (size_t r = 0; r < 272; ++r)
            std::memcpy(nonfinite_bytes.data() + r * 170, &infinity, 2);
        auto nonfinite = PortablePleTable::copy(
            {"per_layer_token_embd.weight", {160, 272}, 8, nonfinite_bytes.data(), nonfinite_bytes.size()}, config,
            nonfinite_bytes.size(), error);
        check(bool(nonfinite), error);
        check(!nonfinite->gather(0, {{-1, -1}}, rejected_rows, rejected_emb, error), "nonfinite table row rejected");
        Fixture f;
        PortableGdnOptions options;
        options.ple_table = table;
        auto layer = PortableGdnLayer::create(1, f.views, options, error);
        check(bool(layer), error);
        exercise(*layer, f);
        // Upstream permits flattened norm tensors and retains F16 conv weights.
        auto reshaped = f.views;
        std::vector<uint16_t> half_conv(40960);
        for (size_t i = 0; i < half_conv.size(); ++i)
            half_conv[i] = ggml_fp32_to_fp16(f.f32(5, i));
        for (auto &v : reshaped) {
            if (v.name.find("ple_norm_") != std::string::npos)
                v.shape = {10240};
            if (v.name == "blk.1.ple_conv1d.weight") {
                v.type = 1;
                v.bytes = half_conv.size() * 2;
                v.data = half_conv.data();
            }
        }
        auto half_layer = PortableGdnLayer::create(1, reshaped, options, error);
        check(bool(half_layer), error);
        PortableRoute half_route;
        check(half_layer->step(0, f.input(0), experts(), half_route, error), error);
        close(half_layer->last_residual(), layer->last_residual(), 0.003f, "F16 convolution/flat norm binding");
        check(!PortableGdnLayer::create(3, f.views, options, error), "QSA rejected");
        options.state_budget = 16ull << 20; // default remains sufficient for both layers
        check(layer->state_bytes() <= options.state_budget, "PLE host state accounted");
        options.state_budget = layer->state_bytes() - 1;
        check(!PortableGdnLayer::create(1, f.views, options, error), "PLE state budget");
        const auto root = std::filesystem::temp_directory_path() /
                          ("strata-ple-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fixture dir");
        struct Cleanup {
            std::filesystem::path p;
            ~Cleanup() {
                std::error_code e;
                std::filesystem::remove_all(p, e);
            }
        } cleanup{root};
        const auto path = root / "configured.gguf";
        const auto kv = metadata();
        sparse(path, kv, f.views);
        std::unique_ptr<PortableGdnLayer> bound;
        {
            auto model = strata::GgufModel::open(path.string());
            bound = PortableGdnLayer::from_gguf(model, 1, {}, error);
            check(bool(bound), error);
        }
        PortableRoute route;
        PortableGdnTrace trace;
        check(bound->step(0, f.input(0), experts(), route, error, &trace), error);
        check(trace.ple_rows == oracle(0, {{-1, -1}}, {}), "actual metadata hash");
        check(bound->weight_bytes() == layer->weight_bytes() && bound->graph_bytes() < (32ull << 20),
              "mapped table excluded from accelerator weights/graph");
        reference(f, f.input(0), std::vector<float>(92160), trace, *bound);
        // Corrupt each required metadata value independently; reject before
        // reading or allocating any tensor payload. Preserve integer widths.
        for (size_t i = 1; i < kv.size(); ++i) {
            if (kv[i].key.find("ple.") == std::string::npos && kv[i].key.find("per_layer_input") == std::string::npos)
                continue;
            auto corrupt = kv;
            if (corrupt[i].type == 9)
                corrupt[i].arr[0] += 1;
            else
                corrupt[i].u += 1;
            sparse(root / "bad.gguf", corrupt, {}, false);
            auto model = strata::GgufModel::open((root / "bad.gguf").string());
            check(!PortablePleTable::from_gguf(model, error), "corrupt metadata accepted: " + corrupt[i].key);
            corrupt = kv;
            corrupt.erase(corrupt.begin() + ptrdiff_t(i));
            sparse(root / "bad.gguf", corrupt, {}, false);
            auto missing = strata::GgufModel::open((root / "bad.gguf").string());
            check(!PortablePleTable::from_gguf(missing, error), "missing metadata accepted");
        }
        std::printf("portable_ple_test: small-table hash/EOS/zero/image/overflow vectors, "
                    "pinned Q8 decoder, actual metadata sparse mapped table binding, "
                    "independent PLE and GDN scalar oracles, 10 tokens, reset, failure "
                    "rollback, QSA rejection passed; weights=%zu graphs=%zu state=%zu\n",
                    layer->weight_bytes(), layer->graph_bytes(), layer->state_bytes());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
