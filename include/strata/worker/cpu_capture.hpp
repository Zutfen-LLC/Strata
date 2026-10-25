#pragma once
// Diagnostic file only. SEW1, worker implementation and inference arithmetic are unchanged.
#include "strata/worker/expert_wire.hpp"
#include <functional>
#include <array>
#include <memory>

namespace strata::worker {
inline constexpr size_t kCpuCaptureLimit = 96u << 20, kCpuCaptureHeader = 144;
struct CpuCaptureWindow { uint32_t ordinal=0, phase=0, T=0; uint64_t pos=0; };
struct CpuCaptureRecord {
    uint32_t window=0, token_begin=0;
    bool sent=false;
    InferRequest request;
    std::vector<float> partials;
};
struct CpuCaptureFile {
    SetupManifest setup;
    Digest128 identity, index, line, tokens, logits;
    uint32_t prompt_tokens=0, max_new=0, vocab=0;
    std::vector<CpuCaptureWindow> windows;
    std::vector<CpuCaptureRecord> records;
};
// Fully validate before connecting to a worker. Integrity digest is accidental-corruption detection,
// not authentication. Files/activations stay private; no paths or expert weight payloads are in this format.
bool decode_cpu_capture(const uint8_t*, size_t, CpuCaptureFile&, std::string&);
bool read_cpu_capture(const std::string&, CpuCaptureFile&, std::string&);
std::vector<int8_t> cpu_capture_owners(const SetupManifest&);

class CpuCapture {
public:
    bool prepare(const SetupManifest&, Digest128 identity, Digest128 index,
                 const std::string& path, std::string&, size_t limit=kCpuCaptureLimit);
    bool request(const std::string& line, const std::vector<int64_t>& tokens, uint64_t max_new);
    bool window(uint64_t pos, uint32_t T, uint32_t phase);
    void slice(uint32_t begin) { token_begin_=begin; }
    bool begin(const InferRequest&); // after tier input copies, before waking the socket owner
    bool finish(const float* partials); // after finish/drain, before exposing the result
    bool seal(const float* logits, size_t vocab); // host only, after existing ver.run/copy_logits
    bool active() const { return prepared_ && !sealed_; }
    const char* error() const { return error_ ? error_ : "CPU capture failure"; }
    size_t storage_bytes() const { return capacity_; }
private:
    bool fail(const char* s) { if(!error_)error_=s; return false; }
    bool room(size_t n);
    bool covered() const;
    void u32(uint32_t); void u64(uint64_t); void bytes(const void*, size_t);
    std::unique_ptr<uint8_t[]> data_;
    std::vector<uint8_t> payload_; // bounded codec scratch, reserved before any callback
    SetupManifest setup_;
    std::string path_;
    size_t capacity_=0, used_=0;
    uint32_t windows_=0, records_=0, prompt_=0, token_begin_=0, reply_values_=0;
    uint64_t next_pos_=0;
    CpuCaptureWindow current_;
    std::array<uint16_t,kMaxLayers> coverage_{};
    uint16_t pending_layer_=0,pending_span_=0;
    bool prepared_=false, bound_=false, pending_=false, pending_sent_=false, decoded_=false, sealed_=false;
    const char* error_=nullptr;
};

struct CpuReplayReport {
    uint32_t full_checked=0, grouped_checked=0, first_difference=UINT32_MAX, detail_record=UINT32_MAX, contribution_calls=0;
    double max_abs=0, max_rel=0, relative_l2=0, detail_max_abs=0;
    bool detail_envelope=true, entry_order_bits=true, peer_bits=true;
};
// exchange gets a validated captured request with a selected mask. Implementations rewrite only
// fresh session/sequence identity; no setup/upload/unload or retries belong in this function.
using CpuReplayExchange = std::function<bool(const InferRequest&, std::vector<float>&, std::string&)>;
using CpuReplayObserver = std::function<void(size_t record, int peer, const std::vector<float>&)>;
bool replay_cpu_capture(const CpuCaptureFile&, const CpuReplayExchange&, CpuReplayReport&, std::string&,
                        const CpuReplayObserver& masked = {});
}
