#pragma once
#include "strata/core/portable_ffn.hpp"
#include "strata/kernels/ngram.hpp"
#include <array>

namespace strata::core {
// Fixed fleet geometry; hash vocabularies may be small in independent fixtures.
struct PortablePleConfig {
    kernels::PleConsts hash = kernels::ple_artifact_consts();
    int32_t eos = kernels::PLE_EOS_TOKEN_ID, image = 248056;
};

// Host-only Q8_0 table. The GGUF factory retains its own read-only mapping;
// copy() is a bounded small-table seam. No ggml tensor references this table.
class PortablePleTable {
  public:
    static std::shared_ptr<PortablePleTable> copy(const PortableTensorView &view, PortablePleConfig config,
                                                  size_t budget, std::string &error);
    static std::shared_ptr<PortablePleTable> from_gguf(const GgufModel &model, std::string &error);
    bool gather(int32_t token, const std::array<int32_t, 2> &previous, std::array<uint32_t, 16> &rows,
                std::vector<float> &embedding, std::string &error) const;
    uint64_t rows() const;
    const PortablePleConfig &config() const;

  private:
    struct Impl;
    explicit PortablePleTable(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<Impl> impl_;
};
} // namespace strata::core
