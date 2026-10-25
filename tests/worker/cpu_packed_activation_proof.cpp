// Bounded offline Q8_0 proof. No sockets, worker, arena, or protocol changes.
#include "local_fixture.hpp"
#include "cpu_capture_rows.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <climits>
#include <cstddef>
#include <limits>
#include <sstream>

using namespace strata::kernels::cpu;
using Bytes = std::vector<uint8_t>;
namespace {
constexpr size_t H = 2560, Packed = 2720, MaxArtifact = 16u << 20;
const auto started = std::chrono::steady_clock::now();
void need(bool ok, const char* s) { if (!ok) throw std::runtime_error(s); }
uint64_t counter(const char* path, const char* key) {
    std::ifstream in(path); std::string line;
    while (std::getline(in, line)) {
        std::istringstream row(line); std::string name; uint64_t value;
        if (row >> name >> value && name == key) return value;
    }
    throw std::runtime_error(std::string("missing resource counter: ") + key);
}
void guard() {
    static const auto oom = counter("/proc/vmstat", "oom_kill");
    need(counter("/proc/vmstat", "oom_kill") == oom, "new host OOM kill");
    need(counter("/proc/self/status", "VmSwap:") == 0, "owned swap must stay zero");
    need(counter("/proc/self/status", "VmRSS:") <= 2u << 20, "owned RSS above 2 GiB");
    need(counter("/proc/meminfo", "MemAvailable:") >= 8u << 20, "available memory below 8 GiB");
    need(std::chrono::steady_clock::now() - started < std::chrono::seconds(180), "180 second deadline");
}
void geometry(const NativeFmt& f) {
    static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(block_q8_0) == 34 && offsetof(block_q8_0, d) == 0 && offsetof(block_q8_0, qs) == 2);
    need(native_experts_available() && f.gu_type == 8 && f.d_type == 8 && f.gu_act == 8 &&
         f.d_act == 8 && f.n_embd == H && f.n_ff == 640 && f.act_bytes == Packed &&
         f.h_bytes == 680 && f.bytes == 5222400 && !f.had_block && !f.had_gu && !f.had_d &&
         !f.gu_signs && !f.d_signs && !f.tail_off && ggml_blck_size(GGML_TYPE_Q8_0) == 32 &&
         ggml_type_size(GGML_TYPE_Q8_0) == 34, "UNSUPPORTED native format/rotation/CPU geometry");
}
void u32(Bytes& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8*i))); }
void floats(Bytes& b, const std::vector<float>& v) {
    for (float x : v) { need(std::isfinite(x), "nonfinite output"); uint32_t bits; std::memcpy(&bits, &x, 4); u32(b, bits); }
}
bool scale_ok(uint16_t d) { return (d & 0x7c00) != 0x7c00; }
// Diagnostic file fields only: FP16 scale bits in little endian followed by 32 signed-byte bits.
// Access the declared block fields through memcpy, never transmit structs or scratch padding.
Bytes fields(const uint8_t* scratch, bool validate = true) {
    Bytes out; out.reserve(Packed);
    for (size_t i = 0; i < H / 32; ++i) {
        block_q8_0 block; std::memcpy(&block, scratch + i * sizeof(block), sizeof(block));
        need(!validate || scale_ok(block.d), "UNSUPPORTED nonfinite native scale from finite input");
        out.push_back(uint8_t(block.d)); out.push_back(uint8_t(block.d >> 8));
        const auto* q = reinterpret_cast<const uint8_t*>(block.qs); out.insert(out.end(), q, q + 32);
    }
    return out;
}
Bytes restore(const Bytes& in) {
    need(in.size() == Packed, "packed length/block count"); Bytes out(kNativeActBytes, 0);
    for (size_t i = 0; i < H / 32; ++i) {
        block_q8_0 block{}; block.d = uint16_t(in[i*34]) | uint16_t(in[i*34+1]) << 8;
        need(scale_ok(block.d), "malformed nonfinite scale");
        std::memcpy(block.qs, in.data() + i*34+2, 32);
        std::memcpy(out.data() + i*sizeof(block), &block, sizeof(block));
    }
    return out;
}
Bytes native(const NativeFmt& f, const InferRequest& q) {
    geometry(f);
    need(q.T && q.T <= 2 && q.K && q.K <= 10 && q.mask.size() == size_t(q.T)*q.K &&
         q.activations.size() == size_t(q.T)*H, "activation shape");
    need(std::all_of(q.activations.begin(), q.activations.end(), [](float v){return std::isfinite(v);}), "nonfinite F32 input");
    Bytes act(q.T*kNativeActBytes, 0);
    for (size_t t = 0; t < q.T; ++t) {
        bool active = false;
        for (size_t j = 0; j < q.K; ++j) { need(q.mask[t*q.K+j] <= 1, "nonbinary mask"); active |= q.mask[t*q.K+j] != 0; }
        if (active) native_quant_act(f, q.activations.data()+t*H, act.data()+t*kNativeActBytes);
    }
    return act;
}
Bytes roundtrip(const NativeFmt& f, const InferRequest& q, const Bytes& act, Bytes* artifact = nullptr) {
    geometry(f); need(act.size() == q.T*kNativeActBytes, "scratch length");
    Bytes restored(act.size(), 0);
    for (size_t t = 0; t < q.T; ++t) {
        bool active = false; for (size_t j = 0; j < q.K; ++j) active |= q.mask[t*q.K+j] != 0;
        if (!active) continue;
        auto packed = fields(act.data()+t*kNativeActBytes); auto row = restore(packed);
        std::memcpy(restored.data()+t*kNativeActBytes, row.data(), row.size());
        if (artifact) artifact->insert(artifact->end(), packed.begin(), packed.end());
    }
    need(act == restored, "native useful bytes or inactive/padding bytes differ"); return restored;
}
bool bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size()*sizeof(float));
}
std::vector<float> partial(const InferRequest& q, const std::vector<float>& rows) {
    need(rows.size() == size_t(q.T)*q.K*H, "row shape"); std::vector<float> out(q.T*H, 0.f);
    for (size_t i = 0; i < q.mask.size(); ++i) if (q.mask[i]) for (size_t h = 0; h < H; ++h) {
        const size_t at = (i/q.K)*H+h;
        out[at] = out[at] + q.weights[i] * rows[i*H+h];
    }
    return out;
}
template<class F> void refuses(F fn) {
    bool rejected = false; try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    need(rejected, "negative regression unexpectedly accepted");
}
void edge_checks(const NativeFmt& f, Bytes& artifact) {
    InferRequest q; q.T = 2; q.K = 2; q.mask = {1,0,0,0}; q.activations.resize(2*H);
    const float cases[] = {0.f, -0.f, std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(), 1.f, 65504.f, std::numeric_limits<float>::max(), -.125f, .5f, 127.f};
    for (size_t c = 0; c < sizeof(cases)/sizeof(*cases); ++c) {
        std::fill(q.activations.begin(), q.activations.end(), cases[c]);
        if (c == 8) for (size_t j=0;j<H;++j) q.activations[j] = float(int(j%255)-127)/127.f;
        if (c == 9) for (size_t j=0;j<H;++j) q.activations[j] = j%32 ? (j%2 ? -1.5f : 2.5f) : 127.f;
        auto a = native(f,q); auto raw = fields(a.data(), false);
        bool supported = true;
        for (size_t i = 0; i < H/32; ++i) supported &= scale_ok(uint16_t(raw[i*34]) | uint16_t(raw[i*34+1]) << 8);
        u32(artifact, uint32_t(c)); u32(artifact, supported); artifact.insert(artifact.end(), raw.begin(), raw.end());
        if (supported) roundtrip(f,q,a); else refuses([&]{restore(raw);});
        std::printf("PACKED_EDGE case=%zu finite_input=1 native_scale_supported=%d\n",c,int(supported));
    }
    q.mask = {0,0,0,0}; auto zero = native(f,q);
    need(zero == Bytes(2*kNativeActBytes,0), "zero work scratch"); roundtrip(f,q,zero);
    q.mask = {0,0,0,1}; std::fill(q.activations.begin(),q.activations.end(),-.125f); roundtrip(f,q,native(f,q));
    q.activations[0] = std::numeric_limits<float>::infinity(); refuses([&]{native(f,q);});
    q.activations[0] = std::numeric_limits<float>::quiet_NaN(); refuses([&]{native(f,q);});
    Bytes valid(Packed,0); valid[0]=0; valid[1]=0x80; valid[2]=0xff; // negative zero FP16 and signed quantized byte
    auto restored = restore(valid); need(fields(restored.data()) == valid, "signed zero field preservation");
    valid[0]=0; valid[1]=0x3c; restored=restore(valid); block_q8_0 block;
    std::memcpy(&block,restored.data(),sizeof(block)); need(block.d == 0x3c00 && block.qs[0] == -1, "explicit little endian fields");
    auto bad=valid; bad.pop_back(); refuses([&]{restore(bad);}); bad=valid; bad.push_back(0); refuses([&]{restore(bad);});
    for (uint16_t scale : {uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e00)}) {
        bad=valid; bad[0]=uint8_t(scale); bad[1]=uint8_t(scale>>8); refuses([&]{restore(bad);});
    }
    auto wrong=f; wrong.gu_act=15; refuses([&]{geometry(wrong);}); wrong=f; wrong.had_gu=true; refuses([&]{geometry(wrong);});
    wrong=f; wrong.act_bytes--; refuses([&]{geometry(wrong);});
}
Bytes read_bytes(const char* path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate); need(bool(in), "open reference"); auto n=in.tellg();
    need(n > 0 && n <= std::streamoff(MaxArtifact), "reference size bound"); Bytes b(static_cast<size_t>(n));
    in.seekg(0); in.read(reinterpret_cast<char*>(b.data()),b.size()); need(bool(in),"read reference"); return b;
}
void write_bytes(const char* path, const Bytes& b) {
    need(!fs::exists(path) && b.size() <= MaxArtifact, "fresh bounded output required");
    std::ofstream out(path,std::ios::binary); out.write(reinterpret_cast<const char*>(b.data()),b.size()); out.close(); need(bool(out),"write artifact");
}
void check_only() {
    Fixture fixture; geometry(fixture.fmt); Bytes edges; edge_checks(fixture.fmt,edges);
    CpuCaptureFile f; f.setup=fixture.setup(); InferRequest q; q.layer=0; q.T=2; q.K=2;
    q.ids={0,1,0,1}; q.mask={1,1,0,1}; q.weights={-.0f,-.25f,.5f,.5f}; q.activations.assign(2*H,.125f);
    LocalSource source; std::string err; require(source.open(fixture.root.string(),fixture.m,err),err);
    auto a=native(fixture.fmt,q); auto b=roundtrip(fixture.fmt,q,a); std::vector<float> left,right;
    diagnostic::compute_capture_rows(f,fixture.m,source,q,a,left,guard);
    diagnostic::compute_capture_rows(f,fixture.m,source,q,b,right,guard);
    need(bits(left,right),"synthetic native row bits"); auto p=partial(q,left);
    std::vector<const uint8_t*> blobs={fixture.blobs[0].data(),fixture.blobs[1].data(),nullptr,fixture.blobs[1].data()}; std::vector<float> direct;
    require(ExpertWorkerService::direct_native_partial(fixture.fmt,2,2,q.mask.data(),q.weights.data(),q.activations.data(),blobs.data(),direct,err),err);
    need(bits(p,direct) && bits(p,partial(q,right)),"independent ordered partial bits");
    right[0]+=1.f; refuses([&]{need(bits(left,right),"reference mismatch");});
    const auto range=fixture.m.layers[0][0]; source.close_files(); fixture.flip(fixture.m.files[range.file].name.c_str(),range.offset+11);
    refuses([&]{diagnostic::compute_capture_rows(f,fixture.m,source,q,a,left,guard);});
    q.mask.assign(4,0); a=native(fixture.fmt,q);
    diagnostic::compute_capture_rows(f,fixture.m,source,q,a,left,guard);
    need(left == std::vector<float>(4*H,0.f) && bits(partial(q,left),std::vector<float>(2*H,0.f)),
         "zero-work rows/ordered partial without weight reads");
    std::puts("PACKED_CHECK ok=1 sockets=0 children=0");
}
}
int main(int argc,char** argv) {
    const bool check=argc==2 && std::string(argv[1])=="--check";
    const bool pack=argc==4 && std::string(argv[1])=="--pack";
    const bool rows=argc==7 && std::string(argv[1])=="--rows";
    if (!check && !pack && !rows) { std::fprintf(stderr,"usage: --check | --pack CAPTURE OUT | --rows CAPTURE MANIFEST ROOT PACKED_REFERENCE ROW_OUT\n"); return 2; }
    std::signal(SIGALRM,SIG_DFL); ::alarm(180);
    try {
        guard(); need(native_experts_available(),"UNSUPPORTED native CPU build");
#if !defined(__x86_64__)
        need(false,"UNSUPPORTED CPU: this proof is bounded to retained Linux x86-64 builds");
#endif
        if (check) { check_only(); return 0; }
        CpuCaptureFile f; std::string err; require(read_cpu_capture(argv[2],f,err),err);
        need(f.setup.gu_type==8 && f.setup.d_type==8 && f.setup.n_embd==H && f.setup.n_ff==640,"UNSUPPORTED capture format");
        NativeFmt fmt; require(native_fmt(8,8,H,640,fmt,err),err); geometry(fmt);
        Bytes packed={'S','P','A','1'}, output={'S','R','A','1'}; edge_checks(fmt,packed);
        const auto samples=diagnostic::capture_row_samples(f,"PACKED_STRATUM");
        // Include the original production request bytes' digest to bind IDs, weights, masks and F32 bits.
        for (size_t i:samples) {
            guard(); const auto& r=f.records[i]; Bytes request; need(encode_infer(r.request,H,request),"request binding encode");
            auto d=blob_digest(request); u32(packed,uint32_t(i));
            for (uint64_t v:{d.a,d.b}) { u32(packed,uint32_t(v)); u32(packed,uint32_t(v>>32)); }
            roundtrip(fmt,r.request,native(fmt,r.request),&packed);
        }
        if (pack) { guard(); write_bytes(argv[3],packed); std::printf("PACKED_DONE ok=1 records=%zu bytes=%zu weights_read=0\n",samples.size(),packed.size()); return 0; }
        // Cross-build byte gate BEFORE opening any model asset. Restoring locally generated bytes after
        // exact comparison consumes the same fields as the supplied reference, without trusting its offsets.
        need(read_bytes(argv[5])==packed,"packed-byte reference mismatch");
        LocalManifest local; require(read_local_manifest(argv[3],local,err),err);
        need(local.matches(f.setup) && local.identity==f.identity && local.index_digest==f.index,"capture/manifest binding");
        LocalSource source; require(source.open(argv[4],local,err),err); const auto owners=cpu_capture_owners(f.setup);
        for (size_t i:samples) {
            guard(); const auto& r=f.records[i]; const auto& w=f.windows.at(r.window-1); u32(output,uint32_t(i));
            for (int peer=-1;peer<2;++peer) {
                auto q=r.request;
                if (peer>=0) for (size_t j=0;j<q.mask.size();++j)
                    q.mask[j]=q.mask[j] && owners.at(size_t(q.layer)*f.setup.n_expert+q.ids[j])==peer;
                auto a=native(fmt,q); auto b=roundtrip(fmt,q,a); std::vector<float> baseline,restored;
                diagnostic::compute_capture_rows(f,local,source,q,a,baseline,guard);
                diagnostic::compute_capture_rows(f,local,source,q,b,restored,guard);
                need(bits(baseline,restored),"native row bits mismatch"); auto p=partial(q,baseline), received=partial(q,restored);
                need(bits(p,received),"ordered partial bits mismatch");
                if (peer==-1) need(bits(p,r.partials),"retained capture partial bits mismatch");
                u32(output,uint32_t(peer+1)); floats(output,restored); floats(output,received);
                std::printf("PACKED_ROWS record=%zu peer=%d phase=%u position=%llu layer=%u T=%u row_bits=1 partial_bits=1 capture_bits=%d\n",i,peer,w.phase,(unsigned long long)w.pos,q.layer,q.T,peer==-1);
            }
        }
        source.close_files(); guard(); write_bytes(argv[6],output);
        std::printf("PACKED_ROWS_DONE ok=1 records=%zu packed_reference_bits=1 same_build_rows_bits=1 retained_capture_bits=1 cross_build_rows=REQUIRES_CMP sockets=0 worker_loads=0 model_GEN=0\n",samples.size()); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"PACKED_FAIL %s\n",e.what()); return 1; }
}
