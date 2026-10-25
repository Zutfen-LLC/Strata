// src/worker/expert_wire.cpp - inferswarm #348: the frame codecs and the blob digest.
#include "strata/worker/expert_wire.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <utility>

namespace strata::worker {
namespace {

// Little-endian put/get, explicit on every platform: the wire is bytes, not this host's word order.
void put_u16(std::vector<uint8_t>& o, uint16_t v) { o.push_back((uint8_t) v); o.push_back((uint8_t) (v >> 8)); }
void put_u32(std::vector<uint8_t>& o, uint32_t v) {
    for (int b = 0; b < 4; ++b) o.push_back((uint8_t) (v >> (8 * b)));
}
void put_u64(std::vector<uint8_t>& o, uint64_t v) {
    for (int b = 0; b < 8; ++b) o.push_back((uint8_t) (v >> (8 * b)));
}
void put_f32(std::vector<uint8_t>& o, float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    put_u32(o, u);
}
// Only float payload bytes use this path; headers/indices remain explicit little-endian fields.
// memcpy tolerates the row schema's unaligned byte offsets and preserves every float bit.
constexpr bool bulk_f32 = sizeof(float)==4 && std::numeric_limits<float>::is_iec559 &&
                          std::endian::native==std::endian::little;
void put_f32s(std::vector<uint8_t>& o,const float* values,size_t count) {
    if constexpr(bulk_f32) {
        if(!count)return;
        const size_t at=o.size();o.resize(at+count*4);
        std::memcpy(o.data()+at,values,count*4);
    } else for(size_t i=0;i<count;++i)put_f32(o,values[i]);
}

struct Cursor {
    const uint8_t* p;
    size_t n, i = 0;
    bool take(size_t bytes) { if (i + bytes > n) return false; return true; }
    uint16_t u16() { if (!take(2)) return 0; uint16_t v = (uint16_t) (p[i] | (p[i + 1] << 8)); i += 2; return v; }
    uint32_t u32() {
        if (!take(4)) return 0;
        uint32_t v = 0;
        for (int b = 0; b < 4; ++b) v |= (uint32_t) p[i + b] << (8 * b);
        i += 4;
        return v;
    }
    uint64_t u64() {
        if (!take(8)) return 0;
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b) v |= (uint64_t) p[i + b] << (8 * b);
        i += 8;
        return v;
    }
    float f32() { uint32_t u = u32(); float v; std::memcpy(&v, &u, 4); return v; }
    void bytes(void* dst, size_t k) { if (!take(k)) return; std::memcpy(dst, p + i, k); i += k; }
    void f32s(float* dst,size_t count) {
        if constexpr(bulk_f32) {if(count)bytes(dst,count*4);}
        else for(size_t j=0;j<count;++j)dst[j]=f32();
    }
};

}  // namespace

void encode_frame(std::vector<uint8_t>& out, uint16_t msg, uint16_t status, const void* payload,
                  size_t n) {
    out.resize(kHeaderBytes + n);
    out[0] = 'S';
    out[1] = 'E';
    out[2] = 'W';
    out[3] = '1';
    out[4] = (uint8_t) msg;
    out[5] = (uint8_t) (msg >> 8);
    out[6] = (uint8_t) status;
    out[7] = (uint8_t) (status >> 8);
    for (int b = 0; b < 8; ++b) out[8 + b] = (uint8_t) (n >> (8 * b));
    if (n > 0) std::memcpy(out.data() + kHeaderBytes, payload, n);
}

Digest128 blob_digest(const void* data, size_t n) {
    // Two FNV-1a passes with different offset bases and a mirrored byte order: cheap (one pass each), and
    // the pair catches the single-bit, truncation and swap failures an upload channel produces.
    const uint8_t* p = (const uint8_t*) data;
    Digest128 d{0xcbf29ce484222325ull, 0x9e3779b97f4a7c15ull};
    for (size_t i = 0; i < n; ++i) {
        d.a = (d.a ^ p[i]) * 0x100000001b3ull;
        d.b = (d.b ^ p[n - 1 - i]) * 0x100000001b3ull;
    }
    return d;
}

bool encode_setup_begin(const SetupManifest& m, std::vector<uint8_t>& o) {
    if (m.n_layers < 1 || m.n_layers > kMaxLayers || m.n_expert < 1 || m.n_expert > kMaxExperts) return false;
    if (m.n_embd < 1 || m.n_embd > kMaxNEmbd || m.n_ff < 1 || m.n_ff > kMaxNFf) return false;
    const size_t stride = ((size_t) m.n_expert + 7) / 8;
    if (m.partition.size() != (size_t) m.n_layers * stride) return false;
    o.clear();
    o.reserve(96 + m.partition.size());
    put_u64(o, m.nonce);
    put_u64(o, m.model_id);
    for (size_t i = 0; i < 48; ++i) o.push_back((uint8_t) m.model_name[i]);
    for (size_t i = 0; i < 16; ++i) o.push_back((uint8_t) m.quant_name[i]);
    put_u16(o, m.gu_type);
    put_u16(o, m.d_type);
    put_u16(o, m.n_layers);
    put_u16(o, m.n_expert);
    put_u16(o, m.n_embd);
    put_u16(o, m.n_ff);
    put_u64(o, m.partition_id);
    o.insert(o.end(), m.partition.begin(), m.partition.end());
    return o.size() <= kSetupFrameMax - kHeaderBytes;
}

bool decode_setup_begin(const uint8_t* p, size_t n, SetupManifest& m) {
    Cursor c{p, n};
    m.nonce = c.u64();
    m.model_id = c.u64();
    c.bytes(m.model_name, 48);
    m.model_name[47] = 0;
    c.bytes(m.quant_name, 16);
    m.quant_name[15] = 0;
    m.gu_type = c.u16();
    m.d_type = c.u16();
    m.n_layers = c.u16();
    m.n_expert = c.u16();
    m.n_embd = c.u16();
    m.n_ff = c.u16();
    m.partition_id = c.u64();
    if (m.n_layers < 1 || m.n_layers > kMaxLayers || m.n_expert < 1 || m.n_expert > kMaxExperts) return false;
    if (m.n_embd < 1 || m.n_embd > kMaxNEmbd || m.n_ff < 1 || m.n_ff > kMaxNFf) return false;
    const size_t stride = ((size_t) m.n_expert + 7) / 8;
    if (n - c.i != (size_t) m.n_layers * stride) return false;
    m.partition.assign(p + c.i, p + n);
    return true;
}

bool encode_infer(const InferRequest& r, uint16_t n_embd, std::vector<uint8_t>& out, bool packed) {
    if (r.T < 1 || r.T > kMaxT || r.K < 1 || r.K > kMaxK) return false;
    if (n_embd < 1 || n_embd > kMaxNEmbd) return false;
    const size_t n = (size_t) r.T * r.K;
    if (r.mask.size() != n || r.ids.size() != n || r.weights.size() != n) return false;
    if (!packed && r.activations.size() != (size_t) r.T * n_embd) return false;
    if (packed && (n_embd != 2560 || r.T > 2 || !r.packed || r.packed->T != r.T ||
        r.packed->active.size() != r.T || r.packed->fields.size() != (size_t)r.T*kPackedRowBytes)) return false;
    out.clear();
    out.reserve(kInferFixed + n * 7 + r.activations.size() * 4);
    put_u64(out, r.session);
    put_u64(out, r.model_id);
    put_u64(out, r.partition_id);
    put_u64(out, r.sequence);
    put_u16(out, r.layer);
    out.push_back(r.T);
    out.push_back(r.K);
    for (size_t i = 0; i < n; ++i) out.push_back(r.mask[i] != 0 ? 1 : 0);
    for (size_t i = 0; i < n; ++i) put_u16(out, r.ids[i]);
    for (size_t i = 0; i < n; ++i) put_f32(out, r.weights[i]);
    if (!packed) {
        for (size_t i = 0; i < r.activations.size(); ++i) put_f32(out, r.activations[i]);
    } else {
        put_u16(out,kPackedVersion);put_u16(out,8);put_u16(out,2560);put_u16(out,80);
        for (size_t t=0;t<r.T;++t) {
            bool active=false;
            for (size_t j=0;j<r.K;++j) {if(r.mask[t*r.K+j]>1)return false;active |= r.mask[t*r.K+j]!=0;}
            if (!active) continue;
            if (r.packed->active[t]!=1) return false;
            const auto* row=r.packed->fields.data()+t*kPackedRowBytes;
            for(size_t b=0;b<80;++b)if((row[b*34+1]&0x7c)==0x7c)return false;
            out.insert(out.end(),row,row+kPackedRowBytes);
        }
    }
    return out.size() <= kInferPayloadMax;
}

bool decode_infer(const uint8_t* p, size_t n, uint16_t n_layers, uint16_t n_expert, uint16_t n_embd,
                  InferRequest& r, bool packed) {
    r.packed.reset();
    if (n < kInferFixed) return false;   // the fixed fields must be whole before any field is read
    Cursor c{p, n};
    r.session = c.u64();
    r.model_id = c.u64();
    r.partition_id = c.u64();
    r.sequence = c.u64();
    r.layer = c.u16();
    r.T = p[c.i++];
    r.K = p[c.i++];
    if (r.T < 1 || r.T > kMaxT || r.K < 1 || r.K > kMaxK) return false;
    const size_t entries = (size_t) r.T * r.K;
    const size_t metadata=kInferFixed+entries*7;
    if (!n_embd || n_embd>kMaxNEmbd || n<metadata || n>kInferPayloadMax) return false;
    if (!packed && n != metadata + (size_t) r.T * n_embd * 4) return false;
    if (packed && (n_embd!=2560 || r.T>2 || n<metadata+8))return false;
    if (!c.take(entries)) return false;
    r.mask.assign(p + c.i, p + c.i + entries);
    c.i += entries;
    r.ids.resize(entries);
    for (size_t i = 0; i < entries; ++i) r.ids[i] = c.u16();
    r.weights.resize(entries);
    for (size_t i = 0; i < entries; ++i) r.weights[i] = c.f32();
    if (!packed) {
        r.activations.resize((size_t) r.T * n_embd);
        for (size_t i = 0; i < r.activations.size(); ++i) r.activations[i] = c.f32();
    } else {
        r.activations.clear();
        if(c.u16()!=kPackedVersion || c.u16()!=8 || c.u16()!=2560 || c.u16()!=80)return false;
        auto a=std::make_shared<PackedActivation>();a->T=r.T;a->active.assign(r.T,0);
        a->fields.assign((size_t)r.T*kPackedRowBytes,0);
        for(size_t t=0;t<r.T;++t) {
            for(size_t j=0;j<r.K;++j) {if(r.mask[t*r.K+j]>1)return false;a->active[t]|=r.mask[t*r.K+j];}
            if(!a->active[t])continue;
            if(!c.take(kPackedRowBytes))return false;
            auto* row=a->fields.data()+t*kPackedRowBytes;c.bytes(row,kPackedRowBytes);
            for(size_t b=0;b<80;++b)if((row[b*34+1]&0x7c)==0x7c)return false;
        }
        if(c.i!=n)return false;
        r.packed=std::move(a);
    }
    if (r.layer >= n_layers) return false;
    for (size_t i = 0; i < entries; ++i)
        if (r.ids[i] >= n_expert) return false;
    return true;
}

std::vector<uint8_t> encode_infer_ack(const InferReply& r) {
    std::vector<uint8_t> o;
    encode_infer_ack(r, o);
    return o;
}

void encode_infer_ack(const InferReply& r, std::vector<uint8_t>& o) {
    o.clear();
    o.reserve(16 + r.partials.size() * 4);
    put_u64(o, r.session);
    put_u64(o, r.sequence);
    put_f32s(o,r.partials.data(),r.partials.size());
}

bool decode_infer_ack(const uint8_t* p, size_t n, uint16_t n_embd, uint8_t T, InferReply& r) {
    Cursor c{p, n};
    r.session = c.u64();
    r.sequence = c.u64();
    if (n != 16 + (size_t) T * n_embd * 4) return false;
    r.partials.resize((size_t) T * n_embd);
    c.f32s(r.partials.data(),r.partials.size());
    return true;
}

bool encode_infer_rows_ack(const InferRequest& q,uint16_t H,const float* rows,std::vector<uint8_t>& out) {
    out.clear();
    const size_t entries=(size_t)q.T*q.K;
    if(!H || H>kMaxNEmbd || !q.T || q.T>kMaxT || !q.K || q.K>kMaxK || q.mask.size()!=entries)return false;
    const size_t count=std::count_if(q.mask.begin(),q.mask.end(),[](uint8_t v){return v!=0;});
    if(count && !rows)return false;
    out.reserve(20+count*(2+(size_t)H*4));
    put_u64(out,q.session);put_u64(out,q.sequence);put_u16(out,kEntryRowsVersion);put_u16(out,(uint16_t)count);
    for(size_t i=0;i<entries;++i) {
        if(!q.mask[i])continue;
        put_u16(out,(uint16_t)i);
        const float* row=rows+i*H;
        if(!std::all_of(row,row+H,[](float v){return std::isfinite(v);})) {out.clear();return false;}
        put_f32s(out,row,H);
    }
    return true;
}
bool decode_infer_rows_ack(const uint8_t* p,size_t n,uint16_t H,const InferRequest& q,EntryRowsReply& out) {
    EntryRowsReply r;r.indices.swap(out.indices);r.rows.swap(out.rows);r.indices.clear();r.rows.clear();
    const size_t entries=(size_t)q.T*q.K;
    if(!H || H>kMaxNEmbd || !q.T || q.T>kMaxT || !q.K || q.K>kMaxK || q.mask.size()!=entries || n<20)return false;
    Cursor c{p,n};r.session=c.u64();r.sequence=c.u64();
    if(c.u16()!=kEntryRowsVersion)return false;
    const size_t count=c.u16();
    if(count!=size_t(std::count_if(q.mask.begin(),q.mask.end(),[](uint8_t v){return v!=0;})) || n!=20+count*(2+(size_t)H*4))return false;
    r.indices.reserve(count);r.rows.reserve(count*H);
    for(size_t i=0;i<entries;++i) {
        if(!q.mask[i])continue;
        if(c.u16()!=i)return false; // exact ascending set, not merely unique in-range indices
        r.indices.push_back((uint16_t)i);
        const size_t at=r.rows.size();r.rows.resize(at+H);c.f32s(r.rows.data()+at,H);
        if(!std::all_of(r.rows.begin()+at,r.rows.end(),[](float v){return std::isfinite(v);}))return false;
    }
    out=std::move(r);return true;
}

const char* status_name(uint16_t s) {
    switch (s) {
        case ST_OK: return "ok";
        case ST_PROTOCOL: return "protocol";
        case ST_WRONG_STATE: return "wrong-state";
        case ST_BAD_REQUEST: return "bad-request";
        case ST_UNSUPPORTED_LAYOUT: return "unsupported-layout";
        case ST_OVER_BUDGET: return "over-budget";
        case ST_DUPLICATE: return "duplicate";
        case ST_NOT_OWNED: return "not-owned";
        case ST_DIGEST_MISMATCH: return "digest-mismatch";
        case ST_NOT_READY: return "not-ready";
        case ST_INCOMPLETE: return "incomplete";
        case ST_STALE_SESSION: return "stale-session";
        case ST_NONFINITE: return "nonfinite";
        case ST_DEADLINE: return "deadline";
        case ST_LOCAL_INVALID:
            return "local-invalid";
        case ST_CACHE_MISMATCH: return "cache-mismatch";
        case ST_LOCAL_UNAVAILABLE:
            return "local-unavailable";
        case ST_INTERNAL: return "internal";
    }
    return "unknown";
}

}  // namespace strata::worker
