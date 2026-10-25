// Offline real-row compressibility diagnostic. No socket, child, worker setup, or protocol extension.
#include "local_fixture.hpp"
#include "strata/worker/cpu_capture.hpp"
#include "cpu_capture_rows.hpp"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
using Bytes = std::vector<uint8_t>;
struct ZSTD_CCtx_s;
struct ZSTD_DCtx_s;
enum snappy_status { SNAPPY_OK = 0, SNAPPY_INVALID_INPUT = 1, SNAPPY_BUFFER_TOO_SMALL = 2 };
namespace {
// Literal success checks must not construct heap-backed std::strings in measured loops.
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void need(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
constexpr size_t MaxFrame = 20 + 20 * (2 + 2560 * 4), HeaderAllowance = 8;
constexpr int Warmups = 16, Iterations = 64;
const auto Started = Clock::now();
uint64_t number(const char* path, const char* key) {
    std::ifstream input(path); std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line); std::string name; uint64_t value;
        if (row >> name >> value && name == key) return value;
    }
    throw std::runtime_error(std::string("missing resource counter: ") + key);
}
void deadline() {
    need(Clock::now() - Started < std::chrono::seconds(180), "180 second work deadline");
    static const uint64_t oom = number("/proc/vmstat", "oom_kill");
    need(number("/proc/vmstat", "oom_kill") == oom, "new host OOM kill");
    need(number("/proc/self/status", "VmSwap:") == 0, "owned VmSwap must stay zero");
    need(number("/proc/self/status", "VmRSS:") <= 2u << 20, "owned RSS exceeds 2 GiB");
    need(number("/proc/meminfo", "MemAvailable:") >= 8u << 20, "MemAvailable below 8 GiB");
}
double ns(Clock::time_point begin) { return std::chrono::duration<double, std::nano>(Clock::now() - begin).count(); }
void escape(const void* p) { asm volatile("" : : "g"(p) : "memory"); }
struct Library {
    void* handle = nullptr;
    explicit Library(const char* path) {
        need(fs::path(path).is_absolute() && fs::is_regular_file(path), "absolute installed library required");
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        need(handle != nullptr, "dlopen failed");
    }
    ~Library() { if (handle) dlclose(handle); }
    Library(const Library&) = delete;
    template<class T> T symbol(const char* name) {
        void* p = dlsym(handle, name); need(p != nullptr, std::string("missing stable symbol: ") + name);
        static_assert(sizeof(T) == sizeof(p)); T f; std::memcpy(&f, &p, sizeof(f)); return f;
    }
};
// Signatures checked against official LZ4 v1.10.0 lz4.h and Zstd v1.5.7 zstd.h.
// Only public stable functions, no internal struct definitions/experimental entrypoints.
struct Codecs {
    Library lz4, extra;
    std::unique_ptr<Library> snappy;
    size_t (*sBound)(size_t) = nullptr;
    snappy_status (*sEncode)(const char*, size_t, char*, size_t*) = nullptr;
    snappy_status (*sDecode)(const char*, size_t, char*, size_t*) = nullptr;
    int (*lzBound)(int); int (*lzStateSize)();
    int (*lzEncode)(void*, const char*, char*, int, int, int);
    int (*lzDecode)(const char*, char*, int, int);
    size_t (*zBound)(size_t); unsigned (*zError)(size_t);
    ZSTD_CCtx_s* (*zCreateC)(); ZSTD_DCtx_s* (*zCreateD)();
    size_t (*zFreeC)(ZSTD_CCtx_s*); size_t (*zFreeD)(ZSTD_DCtx_s*);
    size_t (*zEncode)(ZSTD_CCtx_s*, void*, size_t, const void*, size_t, int);
    size_t (*zDecode)(ZSTD_DCtx_s*, void*, size_t, const void*, size_t);
    ZSTD_CCtx_s* cc = nullptr; ZSTD_DCtx_s* dc = nullptr;
    std::vector<uint64_t> state;
    Codecs(const char* a, const char* b) : lz4(a), extra(b) {
        need(lz4.symbol<int(*)()>("LZ4_versionNumber")() == 11000, "expected installed LZ4 1.10.0");
        lzBound = lz4.symbol<decltype(lzBound)>("LZ4_compressBound");
        lzStateSize = lz4.symbol<decltype(lzStateSize)>("LZ4_sizeofState");
        lzEncode = lz4.symbol<decltype(lzEncode)>("LZ4_compress_fast_extState");
        lzDecode = lz4.symbol<decltype(lzDecode)>("LZ4_decompress_safe");
        const int n = lzStateSize(); need(n > 0 && n <= 1 << 20, "LZ4 state bound");
        state.resize((n + 7) / 8);
        need(extra.symbol<unsigned(*)()>("ZSTD_versionNumber")() == 10507, "expected installed Zstd 1.5.7");
        zBound = extra.symbol<decltype(zBound)>("ZSTD_compressBound");
        zError = extra.symbol<decltype(zError)>("ZSTD_isError");
        zCreateC = extra.symbol<decltype(zCreateC)>("ZSTD_createCCtx");
        zCreateD = extra.symbol<decltype(zCreateD)>("ZSTD_createDCtx");
        zFreeC = extra.symbol<decltype(zFreeC)>("ZSTD_freeCCtx");
        zFreeD = extra.symbol<decltype(zFreeD)>("ZSTD_freeDCtx");
        zEncode = extra.symbol<decltype(zEncode)>("ZSTD_compressCCtx");
        zDecode = extra.symbol<decltype(zDecode)>("ZSTD_decompressDCtx");
        if (const char* path = std::getenv("STRATA_BENCH_SNAPPY_SO")) {
            snappy = std::make_unique<Library>(path);
            // Stable snappy-c.h C ABI, checked against official 1.2.2 declarations.
            sBound = snappy->symbol<decltype(sBound)>("snappy_max_compressed_length");
            sEncode = snappy->symbol<decltype(sEncode)>("snappy_compress");
            sDecode = snappy->symbol<decltype(sDecode)>("snappy_uncompress");
        }
        cc = zCreateC(); dc = zCreateD();
        if (!cc || !dc) { zFreeC(cc); zFreeD(dc); cc = nullptr; dc = nullptr; throw std::runtime_error("Zstd context creation"); }
    }
    ~Codecs() { if (cc) zFreeC(cc); if (dc) zFreeD(dc); }
    const char* name() const { return "zstd1"; }
    int modes() const { return snappy ? 5 : 4; }
    size_t bound(size_t n) {
        need(n > 20 && n <= MaxFrame, "bounded nonempty row payload");
        const int l = lzBound((int)n); const auto z = zBound(n);
        need(l > 0 && z && !zError(z), "codec bound failed");
        const auto value = std::max({size_t(l), z, snappy ? sBound(n) : size_t(0)});
        need(value <= 2 * MaxFrame, "compressed output bound"); return value;
    }
};

// Keep the 20-byte reply header and every 2-byte index in place. Transpose only each row's
// four IEEE754 byte lanes; this byte permutation does not inspect, round, or alter float bits.
void shuffle(const Bytes& in, Bytes& out, bool inverse) {
    need(in.size() >= 20 && in.size() <= MaxFrame, "shuffle payload bound");
    const size_t count = in[18] | size_t(in[19]) << 8;
    need(count && count <= 20 && in.size() == 20 + count * (2 + 2560 * 4), "shuffle exact row shape");
    out.resize(in.size()); std::memcpy(out.data(), in.data(), 20);
    for (size_t r = 0; r < count; ++r) {
        const size_t at = 20 + r * (2 + 2560 * 4);
        out[at] = in[at]; out[at + 1] = in[at + 1];
        for (size_t h = 0; h < 2560; ++h) for (size_t lane = 0; lane < 4; ++lane) {
            const size_t normal = at + 2 + h * 4 + lane, transposed = at + 2 + lane * 2560 + h;
            out[inverse ? normal : transposed] = in[inverse ? transposed : normal];
        }
    }
}
struct Trial {
    Codecs& codecs; Bytes raw, shuffled, packed, unpacked, decoded;
    size_t packed_size = 0;
    explicit Trial(Codecs& c, Bytes input) : codecs(c), raw(std::move(input)),
        shuffled(raw.size()), packed(c.bound(raw.size())), unpacked(raw.size()), decoded(raw.size()) {}
    void encode(int mode) {
        if (!mode) { std::memcpy(packed.data(), raw.data(), raw.size()); packed_size = raw.size(); return; }
        if (mode == 4) {
            packed_size = packed.size();
            need(codecs.sEncode((const char*)raw.data(), raw.size(), (char*)packed.data(), &packed_size) == SNAPPY_OK, "Snappy encode failed");
            return;
        }
        const Bytes* input = &raw;
        if (mode == 2) { shuffle(raw, shuffled, false); input = &shuffled; }
        if (mode == 3) {
            packed_size = codecs.zEncode(codecs.cc, packed.data(), packed.size(), input->data(), input->size(), 1);
            need(!codecs.zError(packed_size), "Zstd encode failed");
        } else {
            const int n = codecs.lzEncode(codecs.state.data(), (const char*)input->data(), (char*)packed.data(),
                (int)input->size(), (int)packed.size(), 1);
            need(n > 0, "LZ4 encode failed"); packed_size = n;
        }
    }
    void decode(int mode) {
        if (!mode) { std::memcpy(decoded.data(), packed.data(), raw.size()); return; }
        if (mode == 4) {
            size_t n = decoded.size();
            need(codecs.sDecode((const char*)packed.data(), packed_size, (char*)decoded.data(), &n) == SNAPPY_OK && n == raw.size(), "Snappy exact decode failed");
            return;
        }
        auto& out = mode == 2 ? unpacked : decoded;
        if (mode == 3) {
            const auto n = codecs.zDecode(codecs.dc, out.data(), out.size(), packed.data(), packed_size);
            need(!codecs.zError(n) && n == raw.size(), "Zstd exact decode failed");
        } else need(codecs.lzDecode((const char*)packed.data(), (char*)out.data(), (int)packed_size,
            (int)out.size()) == (int)raw.size(), "LZ4 exact decode failed");
        if (mode == 2) shuffle(unpacked, decoded, true);
    }
    void exact(int mode) { encode(mode); decode(mode); need(raw == decoded, "lossless byte gate failed"); }
    size_t capacity() const { return raw.capacity() + shuffled.capacity() + packed.capacity() + unpacked.capacity() + decoded.capacity(); }
};

// Independent row computation reused from the existing record0 microfixture, generalized only to
// one selected record. One verified native blob is reused; no worker, resident arena or local load.
void rows_for(const CpuCaptureFile& f, const LocalManifest& local, LocalSource& source,
               const CpuCaptureRecord& r, std::vector<float>& rows) {
    using namespace strata::kernels::cpu;
    const auto& q = r.request; const size_t H = f.setup.n_embd, N = size_t(q.T) * q.K;
    need(H == 2560 && q.T && q.T <= 2 && q.K && q.K <= 10 && q.layer < local.layers.size() &&
         q.activations.size() == q.T * H && r.partials.size() == q.T * H &&
         q.ids.size() == N && q.mask.size() == N && q.weights.size() == N, "sample input shape");
    NativeFmt fmt; std::string err;
    need(native_fmt(8, 8, H, f.setup.n_ff, fmt, err), err);
    Bytes act(q.T * kNativeActBytes);
    for (size_t t = 0; t < q.T; ++t) native_quant_act(fmt, q.activations.data() + t * H, act.data() + t * kNativeActBytes);
    for (size_t i = 0; i < N; ++i)
        need(q.mask[i] == (f.setup.owns(q.layer, q.ids[i]) ? 1 : 0), "sample ownership");
    need(std::any_of(q.mask.begin(), q.mask.end(), [](uint8_t m) { return m != 0; }), "sample has no CPU entries");
    diagnostic::compute_capture_rows(f, local, source, q, act, rows, deadline);
    std::vector<float> partial(q.T * H, 0.f);
    for (size_t i = 0; i < N; ++i) if (q.mask[i]) for (size_t h = 0; h < H; ++h) {
        const size_t at = (i / q.K) * H + h;
        partial[at] = partial[at] + q.weights[i] * rows[i * H + h];
    }
    need(std::memcmp(partial.data(), r.partials.data(), partial.size() * 4) == 0, "independent full/capture bits failed");
}
std::vector<size_t> samples(const CpuCaptureFile& f) {
    return diagnostic::capture_row_samples(f, "COMPRESSION_STRATUM");
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end()); return values.at((size_t)std::ceil(fraction * values.size()) - 1);
}
void benchmark(Codecs& codecs, const Bytes& payload, size_t record, int peer) {
    const char* names[] = {"raw_copy", "lz4", "shuffle_lz4", codecs.name(), "snappy"};
    Trial trial(codecs, payload);
    for (int mode = 0; mode < codecs.modes(); ++mode) {
        const auto setup = Clock::now();
        trial.exact(mode); // first context growth/initialization outside steady timing
        for (int i = 0; i < Warmups; ++i) { trial.encode(mode); trial.decode(mode); }
        const double setup_ns = ns(setup);
        const auto retained = trial.capacity(); std::vector<double> encode, decode;
        encode.reserve(Iterations); decode.reserve(Iterations);
        for (int i = 0; i < Iterations; ++i) {
            deadline(); auto begin = Clock::now(); trial.encode(mode); escape(trial.packed.data()); encode.push_back(ns(begin));
            begin = Clock::now(); trial.decode(mode); escape(trial.decoded.data()); decode.push_back(ns(begin));
        }
        need(trial.capacity() == retained && retained <= 2u << 20, "retained codec scratch bound");
        trial.exact(mode);
        const double e95 = percentile(encode, .95), d95 = percentile(decode, .95);
        // 8-byte hypothetical compression metadata; original outer header stays 16 bytes.
        const size_t encoded_wire = trial.packed_size + (mode ? HeaderAllowance : 0) + kHeaderBytes;
        const size_t raw_wire = payload.size() + kHeaderBytes;
        const double saved = (double(raw_wire) - encoded_wire) * 8.; // ns at exactly 1 Gb/s
        const bool choose = mode && encoded_wire < raw_wire && e95 + d95 <= .8 * saved;
        std::printf("COMPRESSION_FRAME record=%zu peer=%d codec=%s raw_bytes=%zu compressed_bytes=%zu wire_bytes=%zu setup_warmup_ns=%.0f "
            "bound_bytes=%zu encode_p50_ns=%.3f encode_p95_ns=%.3f decode_p50_ns=%.3f decode_p95_ns=%.3f "
            "encode_min_ns=%.3f encode_max_ns=%.3f decode_min_ns=%.3f decode_max_ns=%.3f "
            "link_saved_ns=%.3f net_saved_p95_ns=%.3f select=%d exact=1 scratch_bytes=%zu\n",
            record, peer, names[mode], payload.size(), trial.packed_size, encoded_wire, setup_ns, mode ? codecs.bound(payload.size()) + HeaderAllowance + kHeaderBytes : raw_wire,
            percentile(encode, .5), e95, percentile(decode, .5), d95, *std::min_element(encode.begin(), encode.end()),
            *std::max_element(encode.begin(), encode.end()), *std::min_element(decode.begin(), decode.end()),
            *std::max_element(decode.begin(), decode.end()), saved, saved - e95 - d95, choose, retained);
    }
}

void self_check(Codecs& codecs) {
    InferRequest q; q.session = 1; q.sequence = 2; q.T = 1; q.K = 2; q.mask = {1, 1};
    std::vector<float> rows(2 * 2560);
    const uint32_t edge[] = {0, 0x80000000u, 1, 0x007fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu};
    for (size_t i = 0; i < rows.size(); ++i) std::memcpy(&rows[i], &edge[i % 7], 4);
    Bytes raw; need(encode_infer_rows_ack(q, 2560, rows.data(), raw), "edge row codec");
    Trial trial(codecs, raw); for (int mode = 0; mode < codecs.modes(); ++mode) trial.exact(mode);
    Bytes a, b; shuffle(raw, a, false); shuffle(a, b, true); need(raw == b, "signed zero/edge shuffle exact");
    for (int mode = 1; mode < codecs.modes(); ++mode) {
        trial.exact(mode); trial.packed_size = 1; bool rejected = false;
        try { trial.decode(mode); } catch (const std::exception&) { rejected = true; }
        need(rejected, "truncated compressed input refused");
    }
    bool refused = false; raw.pop_back(); try { shuffle(raw, a, false); } catch (const std::exception&) { refused = true; }
    need(refused, "truncated shuffle refusal");
    q.T = 2; q.K = 10; q.mask.assign(20, 1); rows.resize(20 * 2560);
    need(encode_infer_rows_ack(q, 2560, rows.data(), raw) && raw.size() == MaxFrame, "maximum sample frame");
    Trial maximum(codecs, raw); for (int mode = 0; mode < codecs.modes(); ++mode) maximum.exact(mode);
    raw[18] = raw[19] = 255; refused = false;
    try { shuffle(raw, a, false); } catch (const std::exception&) { refused = true; }
    need(refused, "excess row count refused");
    CpuCaptureFile selection;
    for (unsigned layer : {0u, 12u, 24u, 47u}) for (unsigned T : {1u, 2u}) for (int n = 0; n < 3; ++n) {
        CpuCaptureRecord sample; sample.sent = true; sample.request.layer = layer; sample.request.T = T;
        sample.request.K = 10; selection.records.push_back(sample);
    }
    need(samples(selection).size() == 16, "sample first/last selection bound");
    q.T = 1; q.K = 2; q.mask = {1, 1};
    // Synthetic Q8 source checks independent reconstruction/digest/reference refusal, not compressibility.
    Fixture fixture; CpuCaptureFile f; f.setup = fixture.setup(); f.identity = fixture.m.identity; f.index = fixture.m.index_digest;
    CpuCaptureRecord r; r.sent = true; r.request = q; auto& request = r.request;
    request.layer = 0; request.ids = {0, 1}; request.weights = {.5f, -.25f}; request.activations.assign(2560, .125f);
    std::vector<const uint8_t*> blobs = {fixture.blobs[0].data(), fixture.blobs[1].data()}; std::string err;
    need(ExpertWorkerService::direct_native_partial(fixture.fmt, 1, 2, request.mask.data(), request.weights.data(),
        request.activations.data(), blobs.data(), r.partials, err), err);
    LocalSource source; need(source.open(fixture.root.string(), fixture.m, err), err);
    rows_for(f, fixture.m, source, r, rows);
    r.partials[0] += 1.f; refused = false;
    try { rows_for(f, fixture.m, source, r, rows); } catch (const std::exception&) { refused = true; }
    need(refused, "reference mismatch refusal"); r.partials[0] -= 1.f;
    const auto range = fixture.m.layers[0][0]; source.close_files(); fixture.flip(fixture.m.files[range.file].name.c_str(), range.offset + 11);
    refused = false; try { rows_for(f, fixture.m, source, r, rows); } catch (const std::exception&) { refused = true; }
    need(refused, "corrupt selected source refusal");
    std::puts("CPU_REPLY_COMPRESSION_CHECK ok=1 synthetic_compressibility_claim=0 sockets=0 children=0");
}
}
int main(int argc, char** argv) {
    const bool check_env = argc == 2 && std::string(argv[1]) == "--check";
    const bool check_only = check_env || (argc == 4 && std::string(argv[1]) == "--check");
    if (!check_only && !(argc == 7 && std::string(argv[1]) == "--capture")) {
        std::fprintf(stderr, "usage: cpu_reply_compression_bench --check LZ4_SO EXTRA_SO | --capture CAPTURE MANIFEST MODEL_ROOT LZ4_SO EXTRA_SO\n"); return 2;
    }
    std::signal(SIGALRM, SIG_DFL); ::alarm(180);
    try {
        const char* lz4 = check_env ? std::getenv("STRATA_BENCH_LZ4_SO") : argv[check_only ? 2 : 5];
        const char* extra = check_env ? std::getenv("STRATA_BENCH_EXTRA_SO") : argv[check_only ? 3 : 6];
        if (check_env && (!lz4 || !extra)) {
            std::puts("CPU_REPLY_COMPRESSION_CHECK skipped: explicit STRATA_BENCH_LZ4_SO/STRATA_BENCH_EXTRA_SO required"); return 77;
        }
        deadline(); // Unconfigured general CTest runs skip before Linux resource admission.
        auto setup = Clock::now(); Codecs codecs(lz4, extra);
        std::printf("COMPRESSION_CANDIDATE codec=snappy status=%s performance_verdict=NONE\n", codecs.snappy ? "AVAILABLE" : "SKIPPED_NO_BINDING");
        std::printf("COMPRESSION_SETUP libraries_ns=%.0f lz4=11000 extra=%s warmups=16 iterations=64 header_allowance=8 link_bps=1000000000\n", ns(setup), codecs.name());
        if (check_only) { self_check(codecs); return 0; }
        for (int i : {2, 3, 4}) need(fs::path(argv[i]).is_absolute(), "absolute input bindings");
        CpuCaptureFile capture; LocalManifest local; std::string err;
        need(read_cpu_capture(argv[2], capture, err) && read_local_manifest(argv[3], local, err), err);
        need(local.matches(capture.setup) && local.identity == capture.identity && local.index_digest == capture.index, "capture/manifest identity");
        LocalSource source; need(source.open(argv[4], local, err), err);
        const auto selected = samples(capture); const auto owners = cpu_capture_owners(capture.setup);
        struct Sample { size_t index; std::vector<float> rows; };
        std::vector<Sample> prepared;
        const auto reconstruction = Clock::now();
        for (size_t i : selected) {
            Sample s{i, {}}; rows_for(capture, local, source, capture.records[i], s.rows);
            prepared.push_back(std::move(s));
            const auto& r = capture.records[i]; const auto& window = capture.windows.at(r.window - 1);
            std::printf("COMPRESSION_REFERENCE record=%zu phase=%u position=%llu layer=%u T=%u K=%u full_capture_bits=1\n",
                i, window.phase, (unsigned long long)window.pos, r.request.layer, r.request.T, r.request.K);
        }
        source.close_files();
        std::printf("COMPRESSION_SETUP reconstruction_ns=%.0f records=%zu all_references_passed=1\n", ns(reconstruction), prepared.size());
        // No codec timing until ALL independent full references have passed.
        size_t frames = 0;
        for (const auto& sample : prepared) for (int peer = 0; peer < 2; ++peer) {
            auto request = capture.records[sample.index].request;
            for (size_t j = 0; j < request.mask.size(); ++j)
                request.mask[j] = request.mask[j] && owners.at(size_t(request.layer) * capture.setup.n_expert + request.ids[j]) == peer;
            if (std::none_of(request.mask.begin(), request.mask.end(), [](uint8_t m) { return m != 0; })) continue;
            Bytes payload; need(encode_infer_rows_ack(request, capture.setup.n_embd, sample.rows.data(), payload), "production row encode");
            benchmark(codecs, payload, sample.index, peer); ++frames;
        }
        need(frames && frames <= 32, "frame count bound");
        deadline();
        std::printf("CPU_REPLY_COMPRESSION_DONE ok=1 records=%zu frames=%zu codecs=%d socket_contacts=0 worker_loads=0 model_GEN=0\n", prepared.size(), frames, codecs.modes());
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "CPU_REPLY_COMPRESSION_FAIL %s\n", e.what()); return 1; }
}
