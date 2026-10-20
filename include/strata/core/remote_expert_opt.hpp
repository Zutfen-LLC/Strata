#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class ExpertSource;

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    void attach(RemoteExperts& remote); // before RemoteExperts::open
    bool init(std::string& err);        // after all helper caches are open
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

    // inferswarm #349: the remote CPU-expert worker's partial joins the SAME h_sum_/h_mask_ the helper
    // tiers use, with no helper GPU of its own.  `partials` is n_tok x n_embd ALREADY-weighted floats (the
    // worker applied the routing weights in entry order); `owned` is n_tok x k (1 = this partial includes
    // the entry), so copy_rows() zeroes those entries' rows in the mapped buffer and the GPU's own
    // weighted combine skips them - one contribution each, in the documented order (every helper tier's
    // partial first, the CPU worker's on top).  Call between begin() and end(), from the pool callback.
    void accumulate_host(const float* partials, const uint8_t* owned, int n_tok, int k);
    // The current window's routing weights and token span as begin() published them: the dispatch reads
    // the same weights the verifier holds (the pool callback's signature carries none).
    const float* cur_weights() const { return weights_; }
    int cur_token_begin() const { return token_begin_; }
    int cur_tokens() const { return tokens_; }

private:
    friend class RemoteExperts;
    static size_t metadata_bytes();
    bool active() const { return weights_ != nullptr; }
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    std::vector<Peer> peers_;
    float* h_sum_ = nullptr;
    float* m_sum_ = nullptr;
    int32_t* h_mask_ = nullptr;
    int32_t* m_mask_ = nullptr;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
};
} // namespace strata::core
