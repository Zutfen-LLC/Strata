#include "strata/worker/gpu_wire.hpp"
#include "wire_sock.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

namespace strata::worker::gpu_codec {
void put(std::vector<uint8_t>& p, uint64_t v, size_t n) {
    for (size_t b = 0; b < n; ++b) p.push_back((uint8_t)(v >> (8 * b)));
}
uint64_t Cursor::get(size_t n) {
    if (!ok || n > 8 || i > p.size() || n > p.size() - i) { ok = false; return 0; }
    uint64_t v = 0;
    for (size_t b = 0; b < n; ++b) v |= (uint64_t)p[i++] << (8 * b);
    return v;
}
bool send(uintptr_t fd, uint16_t msg, uint16_t status, const std::vector<uint8_t>& p,
          int ms, const std::function<bool()>& cancel, std::string& err, TlsSession* tls) {
    if (p.size() > kGpuFrameMax || ms <= 0) { err = "CUDA frame exceeds bound or deadline"; return false; }
    // Match the CPU transport: queue the complete frame through one bounded
    // writer. Separate header/body sends can leave a short write awaiting ACK.
    std::vector<uint8_t> frame;
    frame.reserve(kHeaderBytes + p.size());
    put(frame, kGpuWireMagic, 4); put(frame, msg, 2); put(frame, status, 2); put(frame, p.size());
    const auto began=std::chrono::steady_clock::now();
    auto remaining=[&] { return ms-(int)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-began).count(); };
    frame.insert(frame.end(), p.begin(), p.end());
    const int left = remaining(); // Copy time cannot renew the original I/O budget.
    if (left <= 0 || !(tls ? tls->send_all(frame.data(), frame.size(), left, cancel) :
        sock::send_all_controlled((sock_fd)fd, frame.data(), frame.size(), left, cancel))) {
        err = "CUDA worker send disconnected or exceeded deadline"; return false;
    }
    return true;
}
bool receive(uintptr_t fd, WireFrame& f, size_t cap, int ms,
             const std::function<bool()>& cancel, std::string& err, int response_wait_ms, TlsSession* tls) {
    auto read = [&](void* p, size_t n, int budget) {
        return tls ? tls->recv_all(p,n,budget,cancel) : sock::recv_all_controlled((sock_fd)fd,p,n,budget,cancel);
    };
    if(ms<=0){err="CUDA frame requires a positive I/O deadline";return false;}
    std::vector<uint8_t> h(kHeaderBytes);
    size_t first = 0;
    // SETUP/READY may hash or load the entire assigned set. Waiting for their first
    // response byte uses the remaining whole-setup budget; once a frame begins,
    // its remaining header and payload retain the short total frame-I/O budget.
    if (response_wait_ms > 0) {
        if (!read(h.data(),1,response_wait_ms)) {
            err="CUDA setup response missing, disconnected or total setup deadline"; return false;
        }
        first=1;
    }
    const auto began=std::chrono::steady_clock::now();
    auto remaining=[&] { return ms-(int)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now()-began).count(); };
    if (ms <= 0 || !read(h.data()+first, h.size()-first, ms)) {
        err = "CUDA worker reply missing, disconnected or deadline"; return false;
    }
    Cursor c{h};
    const uint32_t magic = (uint32_t)c.get(4);
    f.msg = (uint16_t)c.get(2); f.status = (uint16_t)c.get(2);
    const uint64_t n = c.get();
    size_t operation_cap=0;
    switch(f.msg) {
    case GPU_SETUP: operation_cap=kGpuPlanMax+8; break;
    case GPU_SETUP_ACK: operation_cap=52+(size_t)kMaxLayers*kMaxExperts; break;
    case GPU_UPLOAD: operation_cap=60+kUploadChunkMax; break;
    case GPU_INFER: case GPU_INFER_ROWS: operation_cap=48+kInferPayloadMax; break;
    case GPU_INFER_ROWS_ACK: operation_cap=64+kEntryRowsPayloadMax; break;
    case GPU_INFER_ACK: operation_cap=64+kReplyPayloadMax; break;
    case GPU_STATUS_ACK: operation_cap=256; break;
    case GPU_HELLO: case GPU_HELLO_ACK: case GPU_UPLOAD_ACK: case GPU_READY: case GPU_READY_ACK:
    case GPU_STATUS: case GPU_STOP: case GPU_STOP_ACK: operation_cap=64; break;
    default: err="unknown CUDA frame type"; return false;
    }
    cap=std::min(cap,operation_cap);
    if (!c.done() || magic != kGpuWireMagic || n > cap || n > kGpuFrameMax) {
        err = "unsupported or malformed CUDA protocol (requires SGW1 CUDA/q8_1 capability)"; return false;
    }
    f.payload.resize((size_t)n);
    if (n && (remaining()<=0 || !read(f.payload.data(), f.payload.size(), remaining()))) {
        err = "incomplete CUDA worker frame"; return false;
    }
    return true;
}
std::vector<uint8_t> binding(uint64_t session, const GpuPlan& p) {
    std::vector<uint8_t> o;
    put(o, session); put(o, p.model_id); put(o, p.layout_id.a); put(o, p.layout_id.b);
    put(o, p.placement_id.a); put(o, p.placement_id.b);
    return o;
}
bool binding(Cursor& c, uint64_t s, const GpuPlan& p) {
    const auto session = c.get(), model = c.get(), la = c.get(), lb = c.get(), pa = c.get(), pb = c.get();
    return c.ok && s != 0 && session == s && model == p.model_id && la == p.layout_id.a &&
           lb == p.layout_id.b && pa == p.placement_id.a && pb == p.placement_id.b;
}
bool encode_request(const GpuPlan& p, const InferRequest& r, std::vector<uint8_t>& o) {
    if (r.model_id != p.model_id || r.partition_id != p.placement_id.a || !r.session || !r.sequence)
        return false;
    std::vector<uint8_t> body;
    if (!encode_infer(r, p.n_embd, body)) return false;
    o = binding(r.session, p); o.insert(o.end(), body.begin(), body.end());
    return true;
}
bool decode_request(const GpuPlan& p, const std::vector<uint8_t>& o, InferRequest& r) {
    Cursor c{o};
    const auto session = c.get(); c.i = 0;
    if (!binding(c, session, p) || c.i > o.size()) return false;
    return decode_infer(o.data() + c.i, o.size() - c.i, p.n_layers, p.n_expert, p.n_embd, r) &&
           r.session == session && r.model_id == p.model_id && r.partition_id == p.placement_id.a;
}
std::vector<uint8_t> encode_reply(const GpuPlan& p, Digest128 request, const InferReply& r) {
    auto o = binding(r.session, p); put(o, request.a); put(o, request.b);
    auto body = encode_infer_ack(r); o.insert(o.end(), body.begin(), body.end()); return o;
}
bool decode_reply(const GpuPlan& p, Digest128 request, const InferRequest& r,
                  const std::vector<uint8_t>& o, InferReply& out) {
    Cursor c{o};
    if (!binding(c, r.session, p)) return false;
    const auto a = c.get(), b = c.get();
    if (!c.ok || a != request.a || b != request.b || c.i > o.size()) return false;
    InferReply reply;
    if (!decode_infer_ack(o.data() + c.i, o.size() - c.i, p.n_embd, r.T, reply) ||
        reply.session != r.session || reply.sequence != r.sequence) return false;
    for (float v : reply.partials) if (!std::isfinite(v)) return false;
    out = std::move(reply); return true;
}
bool encode_rows_reply(const GpuPlan& p, Digest128 request, const InferRequest& r,
                       const EntryRowsReply& rows, std::vector<uint8_t>& out) {
    out.clear();
    if(!p.n_embd || p.n_embd>kMaxNEmbd || !r.T || r.T>kMaxT || !r.K || r.K>kMaxK ||
       r.mask.size()!=(size_t)r.T*r.K || rows.indices.size()>kEntryRowsMax ||
       rows.session!=r.session || rows.sequence!=r.sequence ||
       rows.rows.size()!=rows.indices.size()*(size_t)p.n_embd) return false;
    size_t j=0;
    std::vector<float> dense(r.mask.size()*(size_t)p.n_embd);
    for(size_t i=0;i<r.mask.size();++i) if(r.mask[i]) {
        if(j>=rows.indices.size() || rows.indices[j]!=i) return false;
        std::copy_n(rows.rows.data()+j*p.n_embd,p.n_embd,dense.data()+i*p.n_embd); ++j;
    }
    if(j!=rows.indices.size()) return false;
    std::vector<uint8_t> body;
    if(!encode_infer_rows_ack(r,p.n_embd,dense.data(),body)) return false;
    out=binding(r.session,p); put(out,request.a); put(out,request.b);
    out.insert(out.end(),body.begin(),body.end()); return true;
}
bool decode_rows_reply(const GpuPlan& p, Digest128 request, const InferRequest& r,
                       const std::vector<uint8_t>& wire, EntryRowsReply& out) {
    Cursor c{wire};
    if(!binding(c,r.session,p)) return false;
    const auto a=c.get(),b=c.get();
    if(!c.ok || a!=request.a || b!=request.b) return false;
    EntryRowsReply rows;
    if(!decode_infer_rows_ack(wire.data()+c.i,wire.size()-c.i,p.n_embd,r,rows) ||
       rows.session!=r.session || rows.sequence!=r.sequence) return false;
    out=std::move(rows); return true;
}
std::vector<uint8_t> counters(const GpuCounters& c) {
    std::vector<uint8_t> p;
    for (auto v : {c.assigned_bytes,c.cached_bytes,c.transferred_bytes,c.persisted_bytes,
        c.retained_disk_bytes,c.verified_bytes,c.reused_bytes,c.host_staging_peak,
        c.gpu_allocated_bytes,c.gpu_peak_bytes,c.gpu_weight_bytes,c.gpu_scratch_bytes,
        c.gpu_allocation_count,c.gpu_free_before,c.gpu_free_after,c.gpu_runtime_reserve,
        c.requests,c.entries,c.failures,c.service_ns}) put(p,v);
    return p;
}
bool counters(const std::vector<uint8_t>& p, GpuCounters& c) {
    Cursor r{p};
    c.assigned_bytes=r.get(); c.cached_bytes=r.get(); c.transferred_bytes=r.get();
    c.persisted_bytes=r.get(); c.retained_disk_bytes=r.get(); c.verified_bytes=r.get(); c.reused_bytes=r.get();
    c.host_staging_peak=r.get(); c.gpu_allocated_bytes=r.get(); c.gpu_peak_bytes=r.get();
    c.gpu_weight_bytes=r.get(); c.gpu_scratch_bytes=r.get(); c.gpu_allocation_count=r.get();
    c.gpu_free_before=r.get(); c.gpu_free_after=r.get(); c.gpu_runtime_reserve=r.get();
    c.requests=r.get(); c.entries=r.get(); c.failures=r.get(); c.service_ns=r.get(); return r.done();
}
} // namespace strata::worker::gpu_codec
