#pragma once
#include "gguf_fixture.hpp"
#include "strata/core/portable_model.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/worker/local_source.hpp"
#include <chrono>
#include <map>

namespace model_fixture {
inline void check(bool b, const std::string &e) {
    if (!b)
        throw std::runtime_error(e);
}
struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t type;
    uint64_t offset = 0, bytes = 0;
};
struct SparseModel {
    std::filesystem::path root;
    std::vector<std::string> paths;
    strata::worker::LocalManifest manifest;
    struct Shard {
        std::vector<Tensor> tensors;
        uint64_t start = 0, size = 0;
    };
    std::array<Shard, 6> shards;
    strata::kernels::cpu::NativeFmt fmt;
    static constexpr size_t H = 2560, V = 248320;
    explicit SparseModel(bool headers_only = false) {
        root = std::filesystem::temp_directory_path() /
               ("strata-model-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fixture mkdir");
        for (size_t s = 0; s < 6; ++s)
            paths.push_back((root / ("model-0000" + std::to_string(s + 1) + "-of-00006.gguf")).string());
        auto add = [&](size_t s, std::string name, std::vector<uint64_t> shape, uint32_t type) {
            strata::TensorInfo t;
            t.shape = shape;
            t.type = type;
            shards[s].tensors.push_back({std::move(name), std::move(shape), type, 0, strata::tensor_payload_bytes(t)});
        };
        if (headers_only) {
            add(0, "token_embd.weight", {H, V}, 8);
            for (size_t i = 0; i < 6; ++i) {
                auto keys = i == 0 ? metadata() : std::vector<fixture::Kv>{};
                keys.push_back(fixture::u32("split.count", 6));
                keys.push_back(fixture::u32("split.no", i));
                keys.push_back(fixture::u32("split.tensors.count", 1));
                write(i, keys);
            }
            return; // no weight writes, native layout construction or payload reads
        }
        for (size_t l = 0; l < 48; ++l) {
            const size_t s = l / 16;
            const auto p = "blk." + std::to_string(l) + ".";
            for (const char *family : {"attn", "ffn"}) {
                add(s, p + "hc_" + family + "_norm.weight", {10240}, 0);
                add(s, p + "hc_" + family + "_down.weight", {10240, 320}, 8);
                add(s, p + "hc_" + family + "_up.weight", {320, 10240}, 8);
                add(s, p + "hc_" + family + "_inject.weight", {10240, 4}, 8);
            }
            add(s, p + "ffn_gate_inp.weight", {2560, 512}, 0);
            add(s, p + "ffn_gate_inp_shexp.weight", {2560}, 0);
            add(s, p + "ffn_gate_shexp.weight", {2560, 640}, 8);
            add(s, p + "ffn_up_shexp.weight", {2560, 640}, 8);
            add(s, p + "ffn_down_shexp.weight", {640, 2560}, 8);
            if (l % 4 == 3) {
                for (const auto &v :
                     std::vector<std::pair<std::string, std::vector<uint64_t>>>{{"attn_q.weight", {2560, 12288}},
                                                                                {"attn_k.weight", {2560, 512}},
                                                                                {"attn_v.weight", {2560, 512}},
                                                                                {"attn_output.weight", {6144, 2560}},
                                                                                {"indexer.q_proj.weight", {2560, 512}},
                                                                                {"indexer.k_proj.weight", {2560, 128}}})
                    add(s, p + v.first, v.second, v.first.rfind("indexer.", 0) == 0 ? 30 : 8);
                add(s, p + "attn_q_norm.weight", {256}, 0);
                add(s, p + "attn_k_norm.weight", {256}, 0);
                add(s, p + "indexer.q_norm.weight", {128}, 0);
                add(s, p + "indexer.k_norm.weight", {128}, 0);
            } else {
                add(s, p + "attn_qkv.weight", {2560, 10240}, 8);
                add(s, p + "attn_gate.weight", {2560, 6144}, 8);
                add(s, p + "ssm_alpha.weight", {2560, 48}, 8);
                add(s, p + "ssm_beta.weight", {2560, 48}, 8);
                add(s, p + "ssm_a", {48}, 0);
                add(s, p + "ssm_dt.bias", {48}, 0);
                add(s, p + "ssm_conv1d.weight", {4, 10240}, 0);
                add(s, p + "ssm_norm.weight", {128}, 0);
                add(s, p + "ssm_out.weight", {6144, 2560}, 8);
            }
            if (l == 1) {
                add(s, p + "ple_key.weight", {2560, 10240}, 8);
                add(s, p + "ple_value.weight", {2560, 2560}, 8);
                for (const char *key : {"ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight"})
                    add(s, p + key, {2560, 4}, 0);
                add(s, p + "ple_conv1d.weight", {4, 10240}, 0);
            }
            const size_t es = 3 + l / 16;
            add(es, p + "ffn_gate_exps.weight", {2560, 640, 512}, 8);
            add(es, p + "ffn_up_exps.weight", {2560, 640, 512}, 8);
            add(es, p + "ffn_down_exps.weight", {640, 2560, 512}, 8);
        }
        add(0, "token_embd.weight", {H, V}, 8);
        add(0, "output_hc_norm.weight", {10240}, 0);
        add(0, "output_hc_down.weight", {10240, 320}, 8);
        add(0, "output_hc_up.weight", {320, 10240}, 8);
        add(0, "output.weight", {H, V}, 8);
        add(0, "per_layer_token_embd.weight", {160, 320001536}, 8);
        auto kv = metadata();
        size_t count = 0;
        for (const auto &s : shards)
            count += s.tensors.size();
        for (size_t s = 0; s < 6; ++s) {
            auto keys = s == 0 ? kv : std::vector<fixture::Kv>{};
            keys.push_back(fixture::u32("split.count", 6));
            keys.push_back(fixture::u32("split.no", s));
            keys.push_back(fixture::u32("split.tensors.count", count));
            write(s, keys);
        }
        // Only norms and a few Q8 blocks are nonzero. The huge PLE and expert
        // tensors are sparse files, not huge allocations or weight scans.
        for (size_t s = 0; s < 6; ++s)
            for (const auto &t : shards[s].tensors)
                if (t.type == 0 && t.name.find("norm") != std::string::npos) {
                    std::vector<float> ones(size_t(t.bytes / 4), 1);
                    patch(s, t.name, 0, ones.data(), t.bytes);
                }
        for (int token : {5, 6, 7, 248044, 248046}) {
            std::vector<uint8_t> row(H / 32 * 34);
            for (size_t b = 0; b < H / 32; ++b) {
                const uint16_t d = 0x3c00;
                std::memcpy(row.data() + b * 34, &d, 2);
                std::memset(row.data() + b * 34 + 2, 1, 32);
            }
            const int8_t first = token == 6 ? -3 : 2;
            std::memcpy(row.data() + 2, &first, 1);
            if (token == 7)
                row[3] = 64; // makes the dedicated generation-EOS row win
            patch(0, "token_embd.weight", size_t(token) * row.size(), row.data(), row.size());
        }
        for (size_t r = 0; r < 2; ++r) {
            uint8_t block[34]{};
            const uint16_t d = 0x3c00;
            std::memcpy(block, &d, 2);
            const int8_t code = r ? -1 : 1;
            std::memcpy(block + 2, &code, 1);
            patch(0, "output.weight", r * (H / 32 * 34), block, sizeof(block));
        }
        {
            uint8_t block[34]{};
            const uint16_t scale = 0x3400; // exactly1/4, on the second component
            std::memcpy(block, &scale, 2);
            block[3] = 1;
            patch(0, "output.weight", 248046ull * (H / 32 * 34), block, sizeof(block));
        }
        std::string error;
        check(strata::kernels::cpu::native_fmt(8, 8, 2560, 640, fmt, error), error);
        manifest.n_layers = 48;
        manifest.n_expert = 512;
        manifest.n_embd = 2560;
        manifest.n_ff = 640;
        manifest.gu_type = manifest.d_type = 8;
        manifest.index_digest = {123, 456};
        for (size_t s = 3; s < 6; ++s)
            manifest.files.push_back({std::filesystem::path(paths[s]).filename().string(), shards[s].size});
        manifest.layers.resize(48);
        const char *roles[]{"gate", "up", "down"};
        for (size_t l = 0; l < 48; ++l)
            for (size_t r = 0; r < 3; ++r) {
                const size_t s = 3 + l / 16;
                const auto &t = find(s, "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight");
                manifest.layers[l][r] = {uint32_t(s - 3), shards[s].start + t.offset, t.bytes / 512};
            }
        // Nonzero expert 0 in layer 0 makes role layout and weighted reduction
        // observable. Other experts are zero, never a fake provider callback.
        std::vector<uint8_t> zero(fmt.bytes, 0), blob = zero;
        const size_t offsets[]{0, fmt.up_off, fmt.down_off};
        for (size_t r = 0; r < 3; ++r) {
            const uint16_t d = r == 1 ? 0x4000 : 0x3c00;
            std::memcpy(blob.data() + offsets[r], &d, 2);
            blob[offsets[r] + 2] = 1;
            patch(3, "blk.0.ffn_" + std::string(roles[r]) + "_exps.weight", 0, blob.data() + offsets[r], 34);
        }
        manifest.digests.assign(48 * 512, strata::worker::blob_digest(zero));
        manifest.digests[0] = strata::worker::blob_digest(blob);
        const auto zd = strata::worker::blob_digest(zero.data(), fmt.up_off);
        manifest.role_digests.assign(48 * 512, {zd, zd, zd});
        // gate/up/down have equal bytes at this geometry.
        for (size_t r = 0; r < 3; ++r)
            manifest.role_digests[0][r] = strata::worker::blob_digest(blob.data() + offsets[r], fmt.up_off);
        check(strata::worker::write_local_manifest((root / "manifest").string(), manifest, error), error);
    }
    ~SparseModel() {
        std::error_code e;
        std::filesystem::remove_all(root, e);
    }
    const Tensor &find(size_t s, const std::string &name) const {
        for (const auto &t : shards[s].tensors)
            if (t.name == name)
                return t;
        throw std::runtime_error("fixture missing tensor " + name);
    }
    void patch(size_t s, const std::string &name, uint64_t offset, const void *data, size_t bytes) {
        const auto &t = find(s, name);
        check(offset <= t.bytes && bytes <= t.bytes - offset, "fixture patch bounds");
        std::fstream file(paths[s], std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(std::streamoff(shards[s].start + t.offset + offset));
        file.write(static_cast<const char *>(data), std::streamsize(bytes));
        check(bool(file), "fixture patch");
    }
    static std::vector<fixture::Kv> metadata() {
        std::vector<fixture::Kv> kv{fixture::str("general.architecture", "qwen4exp")};
        for (const auto &v : std::vector<std::pair<std::string, uint64_t>>{{"embedding_length", 2560},
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
                                                                           {"full_attention_interval", 4},
                                                                           {"ssm.state_size", 128},
                                                                           {"ssm.group_count", 16},
                                                                           {"ssm.time_step_rank", 48},
                                                                           {"ssm.inner_size", 6144},
                                                                           {"ssm.conv_kernel", 4},
                                                                           {"embedding_length_per_layer_input", 160},
                                                                           {"ple.ngram_size", 3},
                                                                           {"ple.heads_per_ngram", 8},
                                                                           {"ple.conv_kernel", 4},
                                                                           {"ple.eos_token_id", 248044},
                                                                           {"ple.image_token_id", 248056}})
            kv.push_back(fixture::u32("qwen4exp." + v.first, v.second));
        kv.push_back(fixture::u32("tokenizer.ggml.eos_token_id", 248046));
        kv.push_back(fixture::u32("tokenizer.ggml.bos_token_id", 248044));
        kv.push_back(fixture::u32("tokenizer.ggml.padding_token_id", 248044));
        kv.push_back(fixture::f32("qwen4exp.rope.freq_base", 1e7f));
        kv.push_back(fixture::f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f));
        kv.push_back(fixture::u32s("qwen4exp.rope.dimension_sections", {11, 11, 10, 0}));
        std::vector<uint64_t> ratios(48, 0);
        for (size_t l = 3; l < 48; l += 4)
            ratios[l] = 4;
        kv.push_back(fixture::u32s("qwen4exp.attention.compress_ratios", ratios));
        kv.push_back(fixture::u32s("qwen4exp.ple.layers", {1}));
        const auto hash = strata::kernels::ple_artifact_consts();
        kv.push_back(fixture::u32s("qwen4exp.ple.layer_multipliers", std::vector<uint64_t>(hash.mult, hash.mult + 3)));
        // Multipliers exceed u32: the sparse writer below emits u64 arrays.
        kv.back().elem = 10;
        kv.push_back(fixture::u32s("qwen4exp.ple.head_offsets", std::vector<uint64_t>(hash.offset, hash.offset + 16)));
        kv.push_back(
            fixture::u32s("qwen4exp.ple.head_vocab_sizes", std::vector<uint64_t>(hash.vocab, hash.vocab + 16)));
        return kv;
    }
    void write(size_t s, const std::vector<fixture::Kv> &kv) {
        std::vector<uint8_t> b;
        auto put = [&](uint64_t v, size_t n) {
            for (size_t i = 0; i < n; ++i)
                b.push_back(uint8_t(v >> (8 * i)));
        };
        auto text = [&](const std::string &v) {
            put(v.size(), 8);
            b.insert(b.end(), v.begin(), v.end());
        };
        put(0x46554747, 4);
        put(3, 4);
        put(shards[s].tensors.size(), 8);
        put(kv.size(), 8);
        for (const auto &k : kv) {
            text(k.key);
            put(k.type, 4);
            if (k.type == 8)
                text(k.s);
            else if (k.type == 4)
                put(k.u, 4);
            else if (k.type == 6) {
                uint32_t f;
                std::memcpy(&f, &k.f, 4);
                put(f, 4);
            } else if (k.type == 9) {
                put(k.elem, 4);
                put(k.arr.size(), 8);
                for (auto v : k.arr)
                    put(v, k.elem == 10 ? 8 : 4);
            } else
                throw std::runtime_error("unsupported fixture metadata");
        }
        uint64_t offset = 0;
        for (auto &t : shards[s].tensors) {
            t.offset = offset;
            text(t.name);
            put(t.shape.size(), 4);
            for (auto d : t.shape)
                put(d, 8);
            put(t.type, 4);
            put(offset, 8);
            offset = (offset + t.bytes + 31) / 32 * 32;
        }
        b.resize((b.size() + 31) / 32 * 32);
        shards[s].start = b.size();
        shards[s].size = b.size() + offset;
        std::ofstream file(paths[s], std::ios::binary);
        file.write(reinterpret_cast<const char *>(b.data()), std::streamsize(b.size()));
        file.seekp(std::streamoff(shards[s].size - 1));
        file.put(0);
        check(bool(file), "sparse fixture write");
    }
};
} // namespace model_fixture
