// src/worker/expert_client.cpp - inferswarm #348: the client's frame io and protocol steps.
#include "strata/worker/expert_client.hpp"

#include "wire_sock.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace strata::worker {
namespace {

bool fd_ok(uintptr_t fd) { return fd != ~(uintptr_t) 0; }

}  // namespace

ExpertWorkerClient::~ExpertWorkerClient() { close(); }

bool ExpertWorkerClient::connect_host(const std::string& host, uint16_t port, int deadline_ms,
                                      std::string& err) {
    close();
    sock_fd fd = kBadFd;
    if (!sock::connect_host(host, port, deadline_ms, fd, err)) return false;
    fd_ = (uintptr_t) fd;
    deadline_ms_ = deadline_ms;
    return true;
}

bool ExpertWorkerClient::connect_local(const std::string &host, uint16_t port, int deadline_ms,
                                       const std::function<bool()> &cancel, std::string &err) {
    close();
    sock_fd fd = kBadFd;
    if (!sock::connect_numeric_controlled(host, port, deadline_ms, cancel, fd, err))
        return false;
    fd_ = (uintptr_t)fd;
    deadline_ms_ = deadline_ms;
    return true;
}

bool ExpertWorkerClient::send_frame(uint16_t msg, uint16_t st, const std::vector<uint8_t>& payload,
                                    std::string& err) {
    if (setup_cancelled_ && setup_cancelled_()) {
        err = "setup cancelled or timed out";
        return false;
    }
    if (!fd_ok(fd_)) {
        err = "not connected";
        return false;
    }
    std::vector<uint8_t> buf;
    encode_frame(buf, msg, st, payload.data(), payload.size());
    if (!sock::send_all_controlled((sock_fd)fd_, buf.data(), buf.size(), deadline_ms_, setup_cancelled_)) {
        err = "send failed or timed out";
        return false;
    }
    return true;
}

bool ExpertWorkerClient::recv_frame(uint16_t want_msg, WireFrame& f, std::string& err) {
    if (setup_cancelled_ && setup_cancelled_()) {
        err = "setup cancelled or timed out";
        return false;
    }
    uint8_t head[kHeaderBytes];
    if (!sock::recv_all_controlled((sock_fd)fd_, head, sizeof head, deadline_ms_, setup_cancelled_)) {
        err = "connection closed, short frame or deadline while waiting for the reply";
        return false;
    }
    const uint32_t magic = (uint32_t) head[0] | (uint32_t) head[1] << 8 | (uint32_t) head[2] << 16 |
                           (uint32_t) head[3] << 24;
    f.msg = (uint16_t) (head[4] | head[5] << 8);
    f.status = (uint16_t) (head[6] | head[7] << 8);
    uint64_t len = 0;
    for (int b = 0; b < 8; ++b) len |= (uint64_t) head[8 + b] << (8 * b);
    size_t cap = 0;
    switch (f.msg) {
    case MSG_LOCAL_SETUP_ACK:
        cap = 32;
        break;
    case MSG_LOCAL_LOAD_ACK:
        cap = 256;
        break;
    case MSG_SETUP_ACK:
        cap = 64;
        break;
    case MSG_UPLOAD_ACK:
        cap = 32;
        break;
    case MSG_READY_ACK:
        cap = 32;
        break;
    case MSG_INFER_ACK:
        cap = kReplyPayloadMax;
        break;
    case MSG_STATUS_ACK:
        cap = 128;
        break;
    default:
        err = "unexpected reply type";
        return false;
    }
    if (magic != kWireMagic || f.msg != want_msg || len > cap) {
        err = "malformed reply";
        return false;
    }
    if (setup_cancelled_ && setup_cancelled_()) {
        err = "setup cancelled or timed out";
        return false;
    }
    f.payload.resize((size_t) len);
    if (len > 0 && !sock::recv_all_controlled((sock_fd)fd_, f.payload.data(), f.payload.size(), deadline_ms_,
                                              setup_cancelled_)) {
        err = "reply payload cut short";
        return false;
    }
    if (setup_cancelled_ && setup_cancelled_()) {
        err = "setup cancelled or timed out";
        return false;
    }
    return true;
}

namespace {

uint64_t cur_u64(const uint8_t* p, size_t n, size_t& i) {
    uint64_t v = 0;
    if (i + 8 <= n) {
        for (int b = 0; b < 8; ++b) v |= (uint64_t) p[i + b] << (8 * b);
        i += 8;
    }
    return v;
}
uint32_t cur_u32(const uint8_t* p, size_t n, size_t& i) {
    uint32_t v = 0;
    if (i + 4 <= n) {
        for (int b = 0; b < 4; ++b) v |= (uint32_t) p[i + b] << (8 * b);
        i += 4;
    }
    return v;
}

void push_u64(std::vector<uint8_t>& o, uint64_t v) {
    for (int b = 0; b < 8; ++b) o.push_back((uint8_t) (v >> (8 * b)));
}
void push_u32(std::vector<uint8_t>& o, uint32_t v) {
    for (int b = 0; b < 4; ++b) o.push_back((uint8_t) (v >> (8 * b)));
}
void push_u16(std::vector<uint8_t>& o, uint16_t v) {
    o.push_back((uint8_t) v);
    o.push_back((uint8_t) (v >> 8));
}

}  // namespace

bool ExpertWorkerClient::setup(const SetupManifest& m, std::string& err) {
    return setup_impl(m, nullptr, false, err);
}
bool ExpertWorkerClient::setup_local(const SetupManifest &m, Digest128 id, bool fallback, std::string &err) {
    return setup_impl(m, &id, fallback, err);
}
bool ExpertWorkerClient::setup_impl(const SetupManifest &m, const Digest128 *id, bool fallback,
                                    std::string &err) {
    local_ = false;
    local_stats_ = {};
    session_ = 0;
    blob_bytes_ = 0;
    chunk_cap_ = 0;
    n_embd_ = 0;
    std::vector<uint8_t> payload;
    if (!encode_setup_begin(m, payload)) {
        err = "the manifest is outside the protocol's bounds";
        return false;
    }
    if (id) {
        std::vector<uint8_t> head;
        push_u32(head, kLocalVersion);
        push_u32(head, fallback ? 1 : 0);
        push_u64(head, id->a);
        push_u64(head, id->b);
        head.insert(head.end(), payload.begin(), payload.end());
        payload.swap(head);
    }
    if (!send_frame(id ? MSG_LOCAL_SETUP : MSG_SETUP_BEGIN, ST_OK, payload, err))
        return false;
    WireFrame f;
    if (!recv_frame(id ? MSG_LOCAL_SETUP_ACK : MSG_SETUP_ACK, f, err)) {
        if (id) {
            err = "local setup negotiation failed (peer must support local-v1): " + err;
            close();
        }
        return false;
    }
    last_status_ = f.status;
    if (f.status != ST_OK) {
        err = std::string("setup refused: ") + status_name(f.status);
        return false;
    }
    size_t i = 0;
    const uint64_t session = cur_u64(f.payload.data(), f.payload.size(), i);
    const uint64_t bytes = cur_u64(f.payload.data(), f.payload.size(), i);
    const uint32_t cap = cur_u32(f.payload.data(), f.payload.size(), i);
    const uint32_t version = id ? cur_u32(f.payload.data(), f.payload.size(), i) : 0;
    const uint32_t batch = id ? cur_u32(f.payload.data(), f.payload.size(), i) : 0;
    if (f.payload.size() != (id ? 28u : 20u) ||
        (id && (version != kLocalVersion || batch != kLocalBatchMax)) || session == 0 || bytes == 0 ||
        bytes > std::numeric_limits<uint32_t>::max() || cap == 0 || cap > kUploadChunkMax) {
        err = "invalid setup reply: session, blob length or data chunk cap";
        last_status_ = ST_PROTOCOL;
        close();
        return false;
    }
    session_ = session;
    blob_bytes_ = (uint32_t) bytes;
    chunk_cap_ = cap;
    n_embd_ = m.n_embd;
    local_ = id != nullptr;
    return true;
}

bool ExpertWorkerClient::upload(int layer, int expert, const uint8_t *blob, uint32_t blob_bytes,
                                std::string &err, const Digest128 *expected) {
    if (session_ == 0) {
        err = "no session: call setup first";
        return false;
    }
    if (blob_bytes != blob_bytes_) {
        err = "blob length does not match the session's layout";
        return false;
    }
    const Digest128 want = blob_digest(blob, blob_bytes);
    if (expected && want != *expected) {
        err = "upload source differs from local manifest";
        last_status_ = ST_DIGEST_MISMATCH;
        return false;
    }
    uint32_t off = 0;
    while (off < blob_bytes) {
        const uint32_t chunk =
            std::min(chunk_cap_, blob_bytes - off);
        std::vector<uint8_t> p;
        p.reserve(kUploadHeadBytes + chunk);
        push_u64(p, session_);
        push_u16(p, (uint16_t) layer);
        push_u16(p, (uint16_t) expert);
        push_u32(p, blob_bytes);
        push_u32(p, off);
        push_u32(p, chunk);
        push_u64(p, want.a);
        push_u64(p, want.b);
        p.insert(p.end(), blob + off, blob + off + chunk);
        if (!send_frame(MSG_UPLOAD, ST_OK, p, err)) return false;
        WireFrame f;
        if (!recv_frame(MSG_UPLOAD_ACK, f, err)) return false;
        last_status_ = f.status;
        if (f.status != ST_OK) {
            err = std::string("upload refused at byte ") + std::to_string(off) + ": " + status_name(f.status);
            return false;
        }
        size_t i = 0;
        const uint64_t sid = cur_u64(f.payload.data(), f.payload.size(), i);
        const uint32_t filled = cur_u32(f.payload.data(), f.payload.size(), i);
        if (f.payload.size() != 12 || sid != session_ || filled != off + chunk) {
            err = "upload reply does not match the session or next offset";
            last_status_ = ST_PROTOCOL;
            close();
            return false;
        }
        off += chunk;
    }
    return true;
}

bool ExpertWorkerClient::load_local(const std::vector<std::pair<uint16_t, uint16_t>> &slots,
                                    std::vector<uint8_t> &hits, std::string &err) {
    hits.clear();
    if (!local_ || !session_ || slots.empty() || slots.size() > kLocalBatchMax) {
        err = "invalid local load state or count";
        return false;
    }
    std::vector<uint8_t> p;
    push_u64(p, session_);
    push_u32(p, slots.size());
    for (const auto &s : slots) {
        push_u16(p, s.first);
        push_u16(p, s.second);
    }
    if (!send_frame(MSG_LOCAL_LOAD, ST_OK, p, err))
        return false;
    WireFrame f;
    if (!recv_frame(MSG_LOCAL_LOAD_ACK, f, err)) {
        close();
        return false;
    }
    last_status_ = f.status;
    if (f.status != ST_OK) {
        err = std::string("local load refused: ") + status_name(f.status);
        return false;
    }
    size_t i = 0;
    const auto sid = cur_u64(f.payload.data(), f.payload.size(), i);
    const auto count = cur_u32(f.payload.data(), f.payload.size(), i);
    LocalStats stats;
    stats.bytes = cur_u64(f.payload.data(), f.payload.size(), i);
    stats.hits = cur_u64(f.payload.data(), f.payload.size(), i);
    stats.misses = cur_u64(f.payload.data(), f.payload.size(), i);
    stats.read_ns = cur_u64(f.payload.data(), f.payload.size(), i);
    stats.verify_ns = cur_u64(f.payload.data(), f.payload.size(), i);
    stats.setup_ns = cur_u64(f.payload.data(), f.payload.size(), i);
    if (sid != session_ || count != slots.size() || i != 60 || f.payload.size() != 60 + slots.size()) {
        err = "local load reply identity or length mismatch";
        last_status_ = ST_PROTOCOL;
        close();
        return false;
    }
    hits.assign(f.payload.begin() + i, f.payload.end());
    for (auto h : hits)
        if (h > 1) {
            err = "invalid local hit flag";
            last_status_ = ST_PROTOCOL;
            close();
            return false;
        }
    local_stats_ = stats;
    return true;
}

bool ExpertWorkerClient::mark_ready(std::string& err) {
    if (session_ == 0) {
        err = "no session";
        return false;
    }
    std::vector<uint8_t> p;
    push_u64(p, session_);
    if (!send_frame(MSG_READY, ST_OK, p, err)) return false;
    WireFrame f;
    if (!recv_frame(MSG_READY_ACK, f, err)) return false;
    last_status_ = f.status;
    if (f.status != ST_OK) {
        size_t i = 8;
        const uint64_t missing = cur_u64(f.payload.data(), f.payload.size(), i);
        err = std::string("not ready: ") + status_name(f.status) + " (" + std::to_string(missing) +
              " slots missing)";
        return false;
    }
    if (local_) {
        size_t i = 0;
        const auto sid = cur_u64(f.payload.data(), f.payload.size(), i);
        const auto missing = cur_u64(f.payload.data(), f.payload.size(), i);
        if (f.payload.size() != 16 || sid != session_ || missing != 0) {
            err = "local readiness reply does not match the verified session";
            last_status_ = ST_PROTOCOL;
            close();
            return false;
        }
    }
    return true;
}

bool ExpertWorkerClient::infer(const InferRequest& r, InferReply& reply, std::string& err) {
    std::vector<uint8_t> p;
    if (!encode_infer(r, n_embd_, p)) {
        err = "the request is outside the protocol's bounds";
        return false;
    }
    if (!send_frame(MSG_INFER, ST_OK, p, err)) return false;
    WireFrame f;
    if (!recv_frame(MSG_INFER_ACK, f, err)) return false;
    last_status_ = f.status;
    if (f.status != ST_OK) {
        err = std::string("inference refused: ") + status_name(f.status);
        return false;
    }
    if (!decode_infer_ack(f.payload.data(), f.payload.size(), n_embd_, r.T, reply)) {
        err = "the reply's shape does not match the request";
        return false;
    }
    if (reply.session != session_ || reply.sequence != r.sequence) {
        err = "the reply's session or sequence does not match the request";
        return false;
    }
    return true;
}

bool ExpertWorkerClient::status(WorkerStatusInfo& out, std::string& err) {
    if (!send_frame(MSG_STATUS, ST_OK, {}, err)) return false;
    WireFrame f;
    if (!recv_frame(MSG_STATUS_ACK, f, err)) return false;
    last_status_ = f.status;
    if (f.status != ST_OK) {
        err = "status refused";
        return false;
    }
    const uint8_t* p = f.payload.data();
    const size_t n = f.payload.size();
    size_t i = 0;
    out.session = cur_u64(p, n, i);
    out.setup_weight_bytes = cur_u64(p, n, i);
    out.infer_in_bytes = cur_u64(p, n, i);
    out.infer_out_bytes = cur_u64(p, n, i);
    out.requests = cur_u64(p, n, i);
    out.failures = cur_u64(p, n, i);
    out.cpu_service_ns = cur_u64(p, n, i);
    if (i + 3 > n) {
        err = "short status reply";
        return false;
    }
    out.last_error = (uint16_t) (p[i] | p[i + 1] << 8);
    i += 2;
    out.ready = p[i++];
    return true;
}

bool ExpertWorkerClient::shutdown(std::string& err) {
    if (!send_frame(MSG_SHUTDOWN, ST_OK, {}, err)) return false;
    WireFrame f;
    if (!recv_frame(MSG_STATUS_ACK, f, err)) return false;
    return f.status == ST_OK;
}

void ExpertWorkerClient::close() {
    if (fd_ok(fd_)) {
        sock_fd fd = (sock_fd) fd_;
        sock::close_fd(fd);
    }
    fd_ = ~(uintptr_t) 0;
    local_ = false;
    setup_cancelled_ = {};
    session_ = 0;
    blob_bytes_ = 0;
    chunk_cap_ = 0;
    n_embd_ = 0;
}

}  // namespace strata::worker
