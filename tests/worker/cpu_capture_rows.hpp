#pragma once
// Shared offline diagnostic seam: one verified expert blob, supplied native activation scratch.
#include "strata/worker/cpu_capture.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/worker/local_source.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <array>
namespace strata::worker::diagnostic {
inline void row_check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
inline std::vector<size_t> capture_row_samples(const CpuCaptureFile& f, const char* tag) {
    std::set<size_t> selected;
    for (unsigned layer : {0u, 12u, 24u, 47u}) for (unsigned T : {1u, 2u}) {
        std::vector<size_t> found;
        for (size_t i = 0; i < f.records.size(); ++i) {
            const auto& r = f.records[i];
            if (r.sent && r.request.layer == layer && r.request.T == T && r.request.K == 10) found.push_back(i);
        }
        if (found.empty()) std::printf("%s layer=%u T=%u present=0\n", tag, layer, T);
        else { selected.insert(found.front()); selected.insert(found.back()); }
    }
    row_check(!selected.empty() && selected.size() <= 16, "real sample bound");
    return {selected.begin(), selected.end()};
}
// Optional issue #20 probe. All timings are serial native calls, excluding file reads,
// digest checks, allocation and cache displacement. No worker-pool speed is inferred.
struct CaptureSplitProfile {
    static constexpr size_t repeats = 7;
    using Times = std::array<double, repeats>;
    Times gu{}, quant{}, down{}, replay{};
    double read_ns = 0, copy_ns = 0, displaced_ns = 0;
    size_t entries = 0, groups = 0, paired = 0, weight_bytes = 0;
    std::vector<uint8_t> retained, displacement;
    volatile uint64_t sink = 0;
    CaptureSplitProfile() : displacement(64u << 20, 1) {}
    void displace() {
        uint64_t sum = 0;
        for (size_t i = 0; i < displacement.size(); i += 64) sum += displacement[i];
        sink = sum; // a cache-displaced probe, not a guaranteed DRAM/cold-cache measurement
    }
};
using CaptureClock = std::chrono::steady_clock;
inline double capture_ns(CaptureClock::time_point t) {
    return std::chrono::duration<double, std::nano>(CaptureClock::now()-t).count();
}
inline void split_expert_rows(const strata::kernels::cpu::NativeFmt& fmt,
    const std::vector<uint8_t>& blob, const InferRequest& q, uint16_t expert,
    const std::vector<uint8_t>& act, std::vector<float>& rows, CaptureSplitProfile& p) {
    using namespace strata::kernels::cpu;
    const size_t H = size_t(fmt.n_embd), N = q.mask.size();
    row_check(fmt.d_type == 8 && fmt.h_bytes == 680 && !fmt.tail_off && !fmt.had_d,
              "split probe Q8_0 down format");
    p.retained.resize(N * fmt.h_bytes);
    std::vector<size_t> indices;
    for (size_t i=0; i<N; ++i) if (q.mask[i] && q.ids[i]==expert) indices.push_back(i);
    row_check(!indices.empty() && indices.size()<=2, "split expert token bound");
    std::vector<float> hidden(fmt.n_ff), check(H), replay(indices.size()*H);
    std::vector<uint8_t> packed(fmt.h_bytes), down(blob.size()-fmt.down_off);
    auto begin=CaptureClock::now();
    std::memcpy(down.data(), blob.data()+fmt.down_off, down.size());
    p.copy_ns+=capture_ns(begin); p.weight_bytes+=down.size();
    auto local=fmt; local.down_off=0; // diagnostic down-only copy; native code is unchanged
    std::vector<const void*> hq; std::vector<float*> out;
    for (size_t j=0; j<indices.size(); ++j) {
        const size_t i=indices[j]; const void* a=act.data()+(i/q.K)*kNativeActBytes;
        float* h=hidden.data(); float* result=rows.data()+i*H;
        for (size_t rep=0; rep<CaptureSplitProfile::repeats; ++rep) {
            begin=CaptureClock::now(); native_gu_rows(fmt,blob.data(),&a,1,&h,0,(int)fmt.n_ff);
            p.gu[rep]+=capture_ns(begin);
            begin=CaptureClock::now(); native_quant_h(fmt,h,packed.data()); p.quant[rep]+=capture_ns(begin);
            if (!rep) std::memcpy(p.retained.data()+i*fmt.h_bytes,packed.data(),fmt.h_bytes);
            else row_check(!std::memcmp(p.retained.data()+i*fmt.h_bytes,packed.data(),fmt.h_bytes), "retained hidden bytes");
            const void* input=packed.data(); float* dest=rep ? check.data() : result;
            begin=CaptureClock::now(); native_down_rows(fmt,blob.data(),&input,1,&dest,0,(int)H);
            p.down[rep]+=capture_ns(begin);
            row_check(!rep || !std::memcmp(result,check.data(),H*sizeof(float)), "repeated full row bytes");
        }
        hq.push_back(p.retained.data()+i*fmt.h_bytes); out.push_back(replay.data()+j*H);
    }
    auto exact=[&] {
        for (size_t j=0; j<indices.size(); ++j)
            row_check(!std::memcmp(out[j], rows.data()+indices[j]*H,H*sizeof(float)), "split full F32 row bytes");
    };
    // First touch after 64 MiB sequential cache displacement, then seven warm calls.
    p.displace(); begin=CaptureClock::now();
    native_down_rows(local,down.data(),hq.data(),(int)hq.size(),out.data(),0,(int)H);
    p.displaced_ns+=capture_ns(begin); exact();
    for (size_t rep=0; rep<CaptureSplitProfile::repeats; ++rep) {
        begin=CaptureClock::now();
        native_down_rows(local,down.data(),hq.data(),(int)hq.size(),out.data(),0,(int)H);
        p.replay[rep]+=capture_ns(begin); exact();
    }
    for (size_t j=0; j<indices.size(); ++j)
        std::memcpy(rows.data()+indices[j]*H,out[j],H*sizeof(float));
    p.entries+=indices.size(); ++p.groups; p.paired+=indices.size()==2;
}
template<class Guard>
void compute_capture_rows(const CpuCaptureFile& f, const LocalManifest& local, LocalSource& source,
                          const InferRequest& q, const std::vector<uint8_t>& act,
                          std::vector<float>& rows, Guard guard, CaptureSplitProfile* split = nullptr) {
    using namespace strata::kernels::cpu;
    const size_t H = f.setup.n_embd, N = size_t(q.T) * q.K;
    row_check(f.setup.gu_type == 8 && f.setup.d_type == 8 && f.setup.n_ff == 640 && H == 2560 && q.T && q.T <= 2 && q.K && q.K <= 10 && q.layer < local.layers.size() &&
        q.ids.size() == N && q.mask.size() == N && q.weights.size() == N &&
        act.size() == q.T * kNativeActBytes, "sample shape bound");
    NativeFmt fmt; std::string err;
    row_check(native_fmt(8, 8, H, f.setup.n_ff, fmt, err) && fmt.bytes <= 5222400, "sample native blob bound");
    rows.assign(N * H, 0.f);
    std::vector<float> hidden(fmt.n_ff);
    std::vector<uint8_t> blob(fmt.bytes), hq(fmt.h_bytes);
    std::set<uint16_t> experts;
    for (size_t i = 0; i < N; ++i) {
        row_check(q.ids[i] < f.setup.n_expert && q.mask[i] <= 1 && std::isfinite(q.weights[i]) && (!q.mask[i] || f.setup.owns(q.layer, q.ids[i])), "sample ownership");
        if (q.mask[i]) experts.insert(q.ids[i]);
    }
    for (auto e : experts) {
        guard(); size_t dst = 0;
        const auto read_begin=split ? CaptureClock::now() : CaptureClock::time_point{};
        for (const auto& range : local.layers[q.layer]) {
            row_check(range.bytes <= blob.size() - dst, "native role bounds");
            for (size_t at = 0; at < range.bytes;) {
                guard(); const size_t n = std::min<size_t>(256u << 10, range.bytes - at);
                if (!source.read(range.file, range.offset + uint64_t(e) * range.bytes + at, blob.data() + dst + at, n, err))
                    throw std::runtime_error(err);
                at += n;
            }
            dst += range.bytes;
        }
        if (split) split->read_ns+=capture_ns(read_begin);
        row_check(dst == blob.size() && blob_digest(blob) == local.digests.at(size_t(q.layer) * f.setup.n_expert + e), "selected source digest mismatch");
        if (split) { split_expert_rows(fmt,blob,q,e,act,rows,*split); continue; }
        for (size_t i = 0; i < N; ++i) if (q.mask[i] && q.ids[i] == e) {
            const void* a = act.data() + (i / q.K) * kNativeActBytes; float* h = hidden.data();
            native_gu_rows(fmt, blob.data(), &a, 1, &h, 0, (int)fmt.n_ff);
            native_quant_h(fmt, hidden.data(), hq.data()); const void* packed = hq.data(); float* out = rows.data() + i * H;
            native_down_rows(fmt, blob.data(), &packed, 1, &out, 0, (int)H);
        }
    }
    row_check(std::all_of(rows.begin(), rows.end(), [](float v) { return std::isfinite(v); }), "nonfinite native rows");
}
}
