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
template<class Guard>
void compute_capture_rows(const CpuCaptureFile& f, const LocalManifest& local, LocalSource& source,
                          const InferRequest& q, const std::vector<uint8_t>& act,
                          std::vector<float>& rows, Guard guard) {
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
        row_check(dst == blob.size() && blob_digest(blob) == local.digests.at(size_t(q.layer) * f.setup.n_expert + e), "selected source digest mismatch");
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
