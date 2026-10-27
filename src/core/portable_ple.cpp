#include "strata/core/portable_ple.hpp"
#include "portable_graph.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::core {
using portable_detail::require;
namespace {
bool integer(const MetaValue &v) {
    return v.is_num() && v.type != MetaType::F32 && v.type != MetaType::F64 && v.type != MetaType::BOOL;
}
} // namespace
struct PortablePleTable::Impl {
    PortablePleConfig config;
    uint64_t count = 0;
    const uint8_t *data = nullptr;
    std::vector<uint8_t> owned;
    std::unique_ptr<GgufFile> mapping;
    void validate(const PortableTensorView &view) {
        require(view.name == "per_layer_token_embd.weight" && view.type == 8 && view.shape.size() == 2 &&
                    view.shape[0] == 160 && view.shape[1] > 0 && view.data,
                "portable PLE: expected Q8_0 table [160, rows]");
        count = uint64_t(view.shape[1]);
        require(count <= UINT32_MAX && count <= SIZE_MAX / 170 && view.bytes == count * 170,
                "portable PLE: table size mismatch");
        require(config.eos == kernels::PLE_EOS_TOKEN_ID && config.image == 248056,
                "portable PLE: unsupported EOS/image token metadata");
        for (int h = 0; h < 16; ++h) {
            require(config.hash.vocab[h] > 0 && config.hash.offset[h] < count &&
                        config.hash.vocab[h] <= count - config.hash.offset[h],
                    "portable PLE: hash range outside table");
            require(config.hash.offset[h] == (h ? config.hash.offset[h - 1] + config.hash.vocab[h - 1] : 0),
                    "portable PLE: head offsets must be contiguous");
        }
        data = static_cast<const uint8_t *>(view.data);
    }
};
std::shared_ptr<PortablePleTable> PortablePleTable::copy(const PortableTensorView &view, PortablePleConfig config,
                                                         size_t budget, std::string &error) {
    error.clear();
    try {
        auto p = std::make_shared<Impl>();
        p->config = config;
        p->validate(view);
        require(view.bytes <= budget, "portable PLE: host table copy budget exceeded");
        p->owned.assign(p->data, p->data + view.bytes);
        p->data = p->owned.data();
        return std::shared_ptr<PortablePleTable>(new PortablePleTable(std::move(p)));
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
std::shared_ptr<PortablePleTable> PortablePleTable::from_gguf(const GgufModel &model, std::string &error) {
    error.clear();
    try {
        const auto &meta = model.meta();
        const auto *arch = meta.get("general.architecture");
        require(arch && arch->s == "qwen4exp", "portable PLE: requires qwen4exp");
        auto p = std::make_shared<Impl>();
        for (auto pair : {std::pair<const char *, uint64_t>{"embedding_length_per_layer_input", 160},
                          {"ple.ngram_size", 3},
                          {"ple.heads_per_ngram", 8},
                          {"ple.conv_kernel", 4},
                          {"ple.eos_token_id", 248044},
                          {"ple.image_token_id", 248056}}) {
            const auto *v = meta.get("qwen4exp." + std::string(pair.first));
            require(v && integer(*v) && v->u == pair.second,
                    "portable PLE: missing/mismatched " + std::string(pair.first));
        }
        const auto *layers = meta.get("qwen4exp.ple.layers");
        require(layers && layers->type == MetaType::ARRAY && layers->items.size() == 1 && integer(layers->items[0]) &&
                    layers->items[0].u == 1,
                "portable PLE: expected layers [1]");
        auto array = [&](const char *key, uint64_t *out, size_t n) {
            const auto *v = meta.get("qwen4exp.ple." + std::string(key));
            require(v && v->type == MetaType::ARRAY && v->items.size() == n,
                    "portable PLE: missing/mismatched " + std::string(key));
            for (size_t i = 0; i < n; ++i) {
                require(integer(v->items[i]), "portable PLE: noninteger hash metadata");
                out[i] = v->items[i].u;
            }
        };
        array("layer_multipliers", p->config.hash.mult, 3);
        array("head_offsets", p->config.hash.offset, 16);
        array("head_vocab_sizes", p->config.hash.vocab, 16);
        const auto fleet = kernels::ple_artifact_consts();
        require(std::equal(p->config.hash.mult, p->config.hash.mult + 3, fleet.mult) &&
                    std::equal(p->config.hash.offset, p->config.hash.offset + 16, fleet.offset) &&
                    std::equal(p->config.hash.vocab, p->config.hash.vocab + 16, fleet.vocab),
                "portable PLE: hash metadata differs from configured fleet model");
        size_t found = 0;
        for (size_t s = 0; s < model.size(); ++s)
            for (const auto &t : model.shard(s).tensors())
                if (t.name == "per_layer_token_embd.weight") {
                    require(++found == 1 && model.in_bounds(t, s), "portable PLE: duplicate/out-of-bounds table");
                    require(t.shape == std::vector<uint64_t>({160, 320001536}),
                            "portable PLE: fleet table shape mismatch");
                    p->mapping = std::make_unique<GgufFile>(model.shard(s).path());
                    const auto *mapped = p->mapping->find(t.name);
                    require(mapped && mapped->shape == t.shape && mapped->type == t.type && mapped->offset == t.offset,
                            "portable PLE: table changed during binding");
                    const auto payload = p->mapping->file_size() - p->mapping->data_start();
                    require(mapped->offset <= payload && tensor_payload_bytes(*mapped) <= payload - mapped->offset,
                            "portable PLE: mapped table outside shard");
                    p->validate({t.name,
                                 {160, 320001536},
                                 t.type,
                                 p->mapping->tensor_data(*mapped),
                                 size_t(tensor_payload_bytes(t))});
                }
        require(found == 1, "portable PLE: missing per_layer_token_embd.weight");
        return std::shared_ptr<PortablePleTable>(new PortablePleTable(std::move(p)));
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
bool PortablePleTable::gather(int32_t token, const std::array<int32_t, 2> &previous, std::array<uint32_t, 16> &rows,
                              std::vector<float> &embedding, std::string &error) const {
    error.clear();
    try {
        require(token >= 0, "portable PLE: token ID required (use configured image "
                            "ID for image positions)");
        std::array<uint32_t, 16> next{};
        kernels::ngram_rows(&token, previous.data(), 1, impl_->config.hash, next.data());
        std::vector<float> values(2560);
        for (size_t h = 0; h < 16; ++h) {
            require(next[h] < impl_->count, "portable PLE: row outside table");
            for (int b = 0; b < 5; ++b)
                strata::dequantize_q8_0(impl_->data + size_t(next[h]) * 170 + size_t(b) * 34,
                                        values.data() + h * 160 + size_t(b) * 32);
        }
        for (float v : values)
            require(std::isfinite(v), "portable PLE: nonfinite table row");
        rows = next;
        embedding.swap(values);
        return true;
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }
}
uint64_t PortablePleTable::rows() const { return impl_->count; }
const PortablePleConfig &PortablePleTable::config() const { return impl_->config; }
} // namespace strata::core
