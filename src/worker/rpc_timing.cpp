// src/worker/rpc_timing.cpp - fixed-memory latency histograms for the CPU-worker diagnostic.
#include "strata/worker/rpc_timing.hpp"

#include <algorithm>
#include <cstdio>

namespace strata::worker {
namespace {

const char* phase_name(RpcTimingPhase phase) {
    switch (phase) {
    case RpcTimingPhase::ClientPrepare: return "prepare";
    case RpcTimingPhase::ClientQueueWake: return "queue_wake";
    case RpcTimingPhase::ClientEncodeSend: return "encode_send";
    case RpcTimingPhase::ClientReceiveDecode: return "receive_decode";
    case RpcTimingPhase::ClientFinishWait: return "finish_wait";
    case RpcTimingPhase::ClientResultMerge: return "result_merge";
    case RpcTimingPhase::WorkerDecodeValidate: return "decode_validate";
    case RpcTimingPhase::WorkerQuantPrepare: return "quant_prepare";
    case RpcTimingPhase::WorkerPool: return "pool";
    case RpcTimingPhase::WorkerWeightedReduce: return "weighted_reduce";
    case RpcTimingPhase::WorkerEncodeSend: return "encode_send";
    case RpcTimingPhase::WorkerRequest: return "request";
    case RpcTimingPhase::Count: break;
    }
    return "unknown";
}

} // namespace

RpcRequestShape rpc_request_shape(const std::vector<uint8_t>& mask, const std::vector<uint16_t>& ids) {
    RpcRequestShape shape;
    const size_t n = std::min(mask.size(), ids.size());
    for (size_t i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        ++shape.entries;
        bool first = true;
        for (size_t j = 0; j < i; ++j) first = first && (!mask[j] || ids[j] != ids[i]);
        shape.distinct_experts += first;
    }
    return shape;
}

void RpcLatencyDistribution::add(uint64_t ns) {
    ++count;
    total_ns += ns;
    max_ns = std::max(max_ns, ns);
    const uint64_t us = (ns + 999) / 1000;
    size_t bucket = 0;
    uint64_t upper = 1;
    while (bucket + 1 < kBuckets && us > upper) {
        ++bucket;
        upper <<= 1;
    }
    ++buckets[bucket];
}

void RpcLatencyDistribution::reset() {
    count = total_ns = max_ns = 0;
    buckets.fill(0);
}

uint64_t RpcLatencyDistribution::percentile_us(unsigned percentile) const {
    if (!count) return 0;
    percentile = std::max(1u, std::min(100u, percentile));
    const uint64_t want = (count * percentile + 99) / 100;
    uint64_t seen = 0;
    for (size_t i = 0; i < buckets.size(); ++i) {
        seen += buckets[i];
        if (seen >= want)
            return i + 1 == buckets.size() ? UINT64_MAX : (uint64_t) 1 << i;
    }
    return UINT64_MAX;
}

void RpcTimingCounters::reset() {
    calls_ = successes_ = failures_ = 0;
    request_bytes_ = reply_bytes_ = tokens_ = entries_ = experts_ = 0;
    token_shapes_.fill(0);
    for (auto& phase : phases_) phase.reset();
}

void RpcTimingCounters::add_phase(RpcTimingPhase phase, uint64_t ns) {
    if (!enabled_ || phase == RpcTimingPhase::Count) return;
    phases_[(size_t) phase].add(ns);
}

void RpcTimingCounters::finish_call(uint8_t tokens, uint32_t entries, uint32_t experts, uint64_t request_bytes,
                                    uint64_t reply_bytes, bool success) {
    if (!enabled_) return;
    ++calls_;
    success ? ++successes_ : ++failures_;
    request_bytes_ += request_bytes;
    reply_bytes_ += reply_bytes;
    tokens_ += tokens;
    entries_ += entries;
    experts_ += experts;
    ++token_shapes_[std::min<size_t>(tokens, token_shapes_.size() - 1)];
}

std::string RpcTimingCounters::report(const char* side) const {
    if (!enabled_) return {};
    char head[768];
    std::snprintf(head, sizeof head,
                  "rpc_timing host=%s calls=%llu ok=%llu failed=%llu request_bytes=%llu reply_bytes=%llu "
                  "tokens=%llu entries=%llu distinct_experts=%llu "
                  "shapes=0:%llu,1:%llu,2:%llu,3:%llu,4:%llu,5:%llu,6:%llu,7:%llu,8:%llu",
                  side, (unsigned long long) calls_, (unsigned long long) successes_,
                  (unsigned long long) failures_, (unsigned long long) request_bytes_,
                  (unsigned long long) reply_bytes_, (unsigned long long) tokens_,
                  (unsigned long long) entries_, (unsigned long long) experts_,
                  (unsigned long long) token_shapes_[0],
                  (unsigned long long) token_shapes_[1], (unsigned long long) token_shapes_[2],
                  (unsigned long long) token_shapes_[3], (unsigned long long) token_shapes_[4],
                  (unsigned long long) token_shapes_[5], (unsigned long long) token_shapes_[6],
                  (unsigned long long) token_shapes_[7], (unsigned long long) token_shapes_[8]);
    std::string out = head;
    for (size_t i = 0; i < phases_.size(); ++i) {
        const auto& d = phases_[i];
        if (!d.count) continue;
        char phase[256];
        std::snprintf(phase, sizeof phase,
                      "; %s[n=%llu mean_us=%.1f p50_us=%llu p95_us=%llu p99_us=%llu max_us=%.1f]",
                      phase_name((RpcTimingPhase) i), (unsigned long long) d.count,
                      (double) d.total_ns / (double) d.count / 1000.0,
                      (unsigned long long) d.percentile_us(50),
                      (unsigned long long) d.percentile_us(95),
                      (unsigned long long) d.percentile_us(99), (double) d.max_ns / 1000.0);
        out += phase;
    }
    return out;
}

} // namespace strata::worker
