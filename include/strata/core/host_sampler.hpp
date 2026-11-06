// Pure host-logit implementation of the native ordinary sampling chain.
#pragma once
#include <strata/kernels/sampler.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {
// logit_bias in SamplerParams is device-only; pass host bias separately.
inline bool validate_host_sampler(const kernels::SamplerParams& p, std::string& error) {
    if (p.logit_bias || p.gumbel || !std::isfinite(p.temperature) ||
        (p.temperature > 0 && !std::isfinite(1.0f/p.temperature)) || !std::isfinite(p.top_p) ||
        p.top_p < 0 || p.top_p > 1 || !std::isfinite(p.min_p) || p.min_p < 0 || p.min_p > 1 ||
        p.min_keep < 1 || !std::isfinite(p.penalty_repeat) || p.penalty_repeat <= 0 ||
        !std::isfinite(p.penalty_freq) || !std::isfinite(p.penalty_present)) {
        error = "host sampler: unsupported parameters"; return false;
    }
    return true;
}
inline float host_philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t a=uint32_t(counter), b=uint32_t(counter>>32), c=uint32_t(seed), d=uint32_t(seed>>32);
    for (uint32_t i=0;i<10;++i) {
        const uint64_t x=uint64_t(0x9E3779B9u)*a, y=uint64_t(0xBB67AE85u)*c;
        a=uint32_t(y>>32)^b^i; b=uint32_t(y); c=uint32_t(x>>32)^d; d=uint32_t(x);
    }
    return float(a>>8)*(1.0f/16777216.0f);
}
inline bool sample_host_logits(const std::vector<float>& logits, const std::vector<int32_t>& history,
                               const kernels::SamplerParams& p, const std::vector<float>& bias,
                               int32_t& token, std::string& error) {
    if (!validate_host_sampler(p,error)) return false;
    if (logits.empty() || logits.size()>size_t(std::numeric_limits<int32_t>::max()) ||
        (!bias.empty() && bias.size()!=logits.size())) { error="host sampler: invalid vocabulary or bias"; return false; }
    std::vector<size_t> counts(logits.size());
    size_t window=p.penalty_last_n>0 ? std::min(history.size(),size_t(p.penalty_last_n)) : 0;
    for (size_t i=history.size()-window;i<history.size();++i)
        if (history[i]>=0 && size_t(history[i])<counts.size()) ++counts[size_t(history[i])];
    struct Entry { float value; int32_t id; };
    std::vector<Entry> entries; entries.reserve(logits.size());
    for (size_t i=0;i<logits.size();++i) {
        if (logits[i] == std::numeric_limits<float>::infinity() ||
            (!bias.empty() && (std::isnan(bias[i]) || bias[i] == std::numeric_limits<float>::infinity()))) {
            error="host sampler: unsupported logit or bias"; return false;
        }
        float v=logits[i]; if (!bias.empty()) v+=bias[i];
        if (counts[i]) { v=v<=0 ? v*p.penalty_repeat : v/p.penalty_repeat;
            v-=float(counts[i])*p.penalty_freq+p.penalty_present; }
        if (v == std::numeric_limits<float>::infinity()) { error="host sampler: positive infinity"; return false; }
        if (std::isfinite(v)) entries.push_back({v,int32_t(i)});
    }
    if (entries.empty()) { error="host sampler: no finite logits"; return false; }
    auto order=[](const Entry& a,const Entry& b) { return a.value>b.value || (a.value==b.value && a.id<b.id); };
    if (p.greedy || p.temperature<=0) { token=std::min_element(entries.begin(),entries.end(),order)->id; return true; }
    size_t k=std::min(entries.size(),size_t(p.top_k>=1 && p.top_k<64 ? p.top_k : 64));
    std::partial_sort(entries.begin(),entries.begin()+k,entries.end(),order); entries.resize(k);
    if (p.top_p<1) {
        double sum=0; for (auto e:entries) sum+=std::exp(double(e.value)-double(entries[0].value));
        double cum=0; size_t cut=k;
        for (size_t i=0;i<k;++i) { cum+=std::exp(double(entries[i].value)-double(entries[0].value))/sum; if (cum>=double(p.top_p)) { cut=i+1; break; } }
        entries.resize(std::min(k,std::max(cut,size_t(p.min_keep))));
    }
    if (p.min_p>0) { float threshold=entries[0].value+std::log(p.min_p); size_t n=1;
        while (n<entries.size() && entries[n].value>=threshold) ++n;
        entries.resize(n); }
    const float inv=1.0f/p.temperature, maximum=entries[0].value*inv;
    if (!std::isfinite(maximum)) { error="host sampler: scaled logit overflow"; return false; }
    std::vector<double> weights; double sum=0;
    for (auto e:entries) { float scaled=e.value*inv; if (!std::isfinite(scaled)) { error="host sampler: scaled logit overflow"; return false; } double w=std::exp(double(scaled)-double(maximum)); weights.push_back(w); sum+=w; }
    const float u=host_philox_uniform(p.seed,p.counter); double cum=0;
    for (size_t i=0;i<entries.size();++i) { cum+=weights[i]/sum; if (double(u)<cum) { token=entries[i].id; return true; } }
    token=entries.back().id; return true;
}
} // namespace strata::core
