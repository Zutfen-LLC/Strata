// One deterministic Q8_0 expert. Measurement, not a cross-backend tolerance gate.
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef STRATA_AMD_PROBE_VULKAN
#include "ggml-vulkan.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace cpu = strata::kernels::cpu;
constexpr int H = 2560, F = 640;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE binary32");
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
uint64_t digest(const void* data, size_t bytes) {
    auto p = static_cast<const uint8_t*>(data);
    uint64_t v = 14695981039346656037ull;
    for (size_t i = 0; i < bytes; ++i) { v ^= p[i]; v *= 1099511628211ull; }
    return v;
}
void save(const std::filesystem::path& path, const void* data, size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    require(bool(file), "cannot create " + path.string());
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    file.close(); require(bool(file), "cannot finish " + path.string());
}
void finite(const std::vector<float>& v) {
    for (float x : v) require(std::isfinite(x), "nonfinite expert value");
}
struct Rows { std::vector<float> gate, up, hidden, out; };
void validate(const Rows& r) {
    require(r.gate.size() == F && r.up.size() == F && r.hidden.size() == F && r.out.size() == H,
            "expert result shape mismatch");
    finite(r.gate); finite(r.up); finite(r.hidden); finite(r.out);
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size()*sizeof(float));
}
void exact_repeat(const Rows& a, const Rows& b) {
    require(same(a.gate,b.gate) && same(a.up,b.up) && same(a.hidden,b.hidden) && same(a.out,b.out),
            "same-backend repeat changed bits");
}
std::vector<uint8_t> fixture(const cpu::NativeFmt& fmt) {
    require(fmt.bytes == 5222400 && fmt.gu_row == 2720 && fmt.d_row == 680, "unexpected Q8 layout");
    std::vector<uint8_t> bytes(fmt.bytes);
    // Every 34-byte Q8_0 block has scale 2^-9, exactly representable in FP16.
    // Codes stay in [-15,15]. No RNG/library-dependent quantization.
    uint32_t state = 0x56333430u;
    for (size_t off = 0; off < bytes.size(); off += 34) {
        bytes[off] = 0; bytes[off+1] = 0x18;
        for (size_t j = 0; j < 32; ++j) {
            state = state*1664525u + 1013904223u;
            const int code = int((state >> 16) % 31u)-15;
            bytes[off+2+j] = static_cast<uint8_t>(code);
        }
    }
    return bytes;
}
std::vector<float> input(bool zero) {
    std::vector<float> x(H, 0.f);
    if (!zero) {
        uint32_t state = 0x51385f30u;
        for (auto& v : x) { state = state*1664525u+1013904223u; v = (int((state>>16)%127u)-63)/128.f; }
    }
    return x;
}
Rows native_reference(const cpu::NativeFmt& fmt, const std::vector<uint8_t>& blob,
                      const std::vector<float>& x, std::vector<uint8_t>& aq, std::vector<uint8_t>& hq) {
    aq.resize(fmt.act_bytes); hq.resize(fmt.h_bytes);
    cpu::native_quant_act(fmt, x.data(), aq.data());
    Rows r; r.gate.resize(F); r.up.resize(F); r.hidden.resize(F); r.out.resize(H);
    const void* acts[] = {aq.data()}; float* hidden[] = {r.hidden.data()};
    cpu::native_gu_rows(fmt, blob.data(), acts, 1, hidden, 0, F);
    cpu::native_quant_h(fmt, r.hidden.data(), hq.data());
    const void* hs[] = {hq.data()}; float* outputs[] = {r.out.data()};
    cpu::native_down_rows(fmt, blob.data(), hs, 1, outputs, 0, H);
    // Expose the actual reference projection dots for stage diagnostics. Its hidden/output
    // above come from production native_gu_rows/native_down_rows, not this transcription.
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->vec_dot;
    for (int row = 0; row < F; ++row) {
        dot(H, &r.gate[row], 0, blob.data()+row*fmt.gu_row, 0, aq.data(), 0, 1);
        dot(H, &r.up[row], 0, blob.data()+fmt.up_off+row*fmt.gu_row, 0, aq.data(), 0, 1);
        const float recomposed = (r.gate[row]/(1.f+std::exp(-r.gate[row])))*r.up[row];
        require(!std::memcmp(&recomposed,&r.hidden[row],sizeof(float)), "native hidden projection seam mismatch");
    }
    validate(r); return r;
}

class ExpertGraph {
    Context ctx_{nullptr,ggml_free};
    Buffer buf_{nullptr,ggml_backend_buffer_free};
    ggml_cgraph* graph_ = nullptr;
    ggml_tensor *x_ = nullptr, *g_ = nullptr, *u_ = nullptr, *hidden_ = nullptr, *out_ = nullptr;
    ggml_backend_t backend_;
    std::vector<float> read(ggml_tensor* t) {
        std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(float)); return v;
    }
public:
    ExpertGraph(ggml_backend_t backend, const cpu::NativeFmt& fmt, const std::vector<uint8_t>& blob):backend_(backend) {
        ggml_init_params p{2u<<20,nullptr,true};
        ctx_.reset(ggml_init(p)); require(bool(ctx_), "ggml metadata allocation failed");
        auto* gate = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_Q8_0,H,F);
        auto* up = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_Q8_0,H,F);
        auto* down = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_Q8_0,F,H);
        x_ = ggml_new_tensor_2d(ctx_.get(),GGML_TYPE_F32,H,1);
        g_ = ggml_mul_mat(ctx_.get(),gate,x_); u_ = ggml_mul_mat(ctx_.get(),up,x_);
        hidden_ = ggml_mul(ctx_.get(),ggml_silu(ctx_.get(),g_),u_);
        out_ = ggml_mul_mat(ctx_.get(),down,hidden_);
        graph_ = ggml_new_graph(ctx_.get()); ggml_build_forward_expand(graph_,out_);
        // Direct backend execution only. Unsupported operations stop; no scheduler/CPU fallback.
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i)
            require(ggml_backend_supports_op(backend_,ggml_graph_node(graph_,i)), "backend declines expert operation");
        buf_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(),backend_));
        require(bool(buf_), "backend expert allocation failed");
        require(ggml_backend_buffer_get_size(buf_.get()) < (16u<<20), "unexpected backend tensor allocation above 16 MiB");
        ggml_backend_tensor_set(gate,blob.data(),0,fmt.up_off);
        ggml_backend_tensor_set(up,blob.data()+fmt.up_off,0,fmt.up_off);
        ggml_backend_tensor_set(down,blob.data()+fmt.down_off,0,fmt.bytes-fmt.down_off);
    }
    size_t buffer_bytes() const { return ggml_backend_buffer_get_size(buf_.get()); }
    Rows run(const std::vector<float>& x) {
        ggml_backend_tensor_set(x_,x.data(),0,x.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend_,graph_) == GGML_STATUS_SUCCESS,"backend expert compute failed");
        ggml_backend_synchronize(backend_);
        Rows r{read(g_),read(u_),read(hidden_),read(out_)}; validate(r); return r;
    }
};
void metric(std::ofstream& file, const std::string& pair, const char* stage,
            const std::vector<float>& got, const std::vector<float>& ref) {
    require(got.size() == ref.size() && !got.empty(), "metric shape"); finite(got); finite(ref);
    double square = 0, base = 0, maxabs = 0; size_t bits = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = double(got[i])-ref[i]; square += d*d; base += double(ref[i])*ref[i];
        maxabs = std::max(maxabs,std::fabs(d)); bits += bool(std::memcmp(&got[i],&ref[i],4));
    }
    // Explicit zero-reference handling; finite, bounded fixtures cannot overflow these sums.
    require(base > 0 || square == 0, "nonzero result against zero reference");
    const double rel = base > 0 ? std::sqrt(square/base) : 0;
    file << pair << '\t' << stage << '\t' << got.size() << '\t' << bits << '\t'
         << std::setprecision(17) << maxabs << '\t' << rel << '\n';
    std::printf("OBSERVED %s %s values=%zu bit_different_values=%zu max_abs=%.9g relative_L2=%.9g\n",
                pair.c_str(),stage,got.size(),bits,maxabs,rel);
}
void compare(std::ofstream& f, const std::string& pair, const Rows& got, const Rows& ref) {
    metric(f,pair,"gate",got.gate,ref.gate); metric(f,pair,"up",got.up,ref.up);
    metric(f,pair,"hidden",got.hidden,ref.hidden); metric(f,pair,"out",got.out,ref.out);
}
void save_rows(const std::filesystem::path& dir, const std::string& name, const Rows& r) {
    const std::array<std::pair<const char*,const std::vector<float>*>,4> parts = {{{"gate",&r.gate},{"up",&r.up},{"hidden",&r.hidden},{"out",&r.out}}};
    for (const auto& p : parts) save(dir/(name+"."+p.first+".f32"),p.second->data(),p.second->size()*4);
}
int run(int argc, char** argv) {
    const uint16_t endian = 1;
    require(*reinterpret_cast<const uint8_t*>(&endian) == 1,"fixture requires little-endian Q8/F32 storage");
    std::string backend,output; int device = -1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i]; require(i+1 < argc,"missing option value"); const std::string value = argv[++i];
        if (arg == "--backend") backend = value;
        else if (arg == "--out-dir") output = value;
        else if (arg == "--device") {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,"invalid device");
            const auto n = std::stoull(value); require(n < 16,"device out of range"); device = int(n);
        } else throw std::runtime_error("unknown option: "+arg);
    }
    require((backend == "cpu" || backend == "vulkan") && !output.empty(),"require --backend cpu|vulkan --out-dir NEW_PATH");
    require((backend == "cpu" && device == -1) || (backend == "vulkan" && device >= 0),"Vulkan requires --device; CPU forbids it");
#ifndef STRATA_AMD_PROBE_VULKAN
    require(backend == "cpu","Vulkan not compiled; no CPU fallback");
#endif
    require(std::filesystem::create_directory(output),"output directory already exists");
    cpu::NativeFmt fmt; std::string why;
    require(cpu::native_fmt(8,8,H,F,fmt,why),why);
    require(fmt.gu_act == 8 && fmt.d_act == 8,"CPU reference activation type is not Q8_0");
    auto blob = fixture(fmt); save(std::filesystem::path(output)/"expert.q8_0",blob.data(),blob.size());
    Backend cpu_backend(ggml_backend_cpu_init(),ggml_backend_free);
    require(bool(cpu_backend),"CPU backend unavailable"); ggml_backend_cpu_set_n_threads(cpu_backend.get(),1);
    Backend vk_backend(nullptr,ggml_backend_free);
#ifdef STRATA_AMD_PROBE_VULKAN
    if (backend == "vulkan") {
        require(device < ggml_backend_vk_get_device_count(),"requested Vulkan device absent");
        vk_backend.reset(ggml_backend_vk_init(size_t(device))); require(bool(vk_backend),"Vulkan init failed");
    }
#endif
    ggml_backend_t selected = backend == "cpu" ? cpu_backend.get() : vk_backend.get();
    ExpertGraph cpu_graph(cpu_backend.get(),fmt,blob);
    std::unique_ptr<ExpertGraph> vk_graph;
    if (vk_backend) vk_graph = std::make_unique<ExpertGraph>(selected,fmt,blob);
    auto& selected_graph = vk_graph ? *vk_graph : cpu_graph;
    std::ofstream manifest(std::filesystem::path(output)/"manifest.txt"), metrics(std::filesystem::path(output)/"metrics.tsv");
    require(bool(manifest) && bool(metrics),"cannot create evidence");
    manifest << "schema=1\nH=2560\nFF=640\nT=1\nweight_type=Q8_0\ncpu_activation=Q8_0\n"
             << "backend=" << ggml_backend_name(selected) << "\ndevice=" << device
             << "\ndescription=" << ggml_backend_dev_description(ggml_backend_get_device(selected))
             << "\nweight_bytes=" << blob.size() << "\nweight_fnv1a64=" << std::hex << digest(blob.data(),blob.size()) << std::dec
             << "\nbackend_tensor_bytes=" << selected_graph.buffer_bytes() << "\n";
    metrics << "comparison\tstage\tvalues\tbit_different_values\tmax_abs\trelative_L2\n";
    for (bool zero : {false,true}) {
        const std::string name = zero ? "zero" : "signal"; auto x = input(zero);
        std::vector<uint8_t> aq,hq,aq2,hq2;
        const Rows native = native_reference(fmt,blob,x,aq,hq);
        const Rows native2 = native_reference(fmt,blob,x,aq2,hq2); exact_repeat(native,native2);
        require(aq == aq2 && hq == hq2,"native packed repeat changed bytes");
        const Rows cpu_rows = cpu_graph.run(x); exact_repeat(cpu_rows,cpu_graph.run(x));
        const Rows selected_rows = vk_graph ? vk_graph->run(x) : cpu_rows;
        if (vk_graph) exact_repeat(selected_rows,vk_graph->run(x));
        if (zero) for (float v : selected_rows.out) require(v == 0,"zero expert produced nonzero output");
        else require(std::any_of(native.out.begin(),native.out.end(),[](float v){return v != 0;}),"degenerate signal fixture");
        compare(metrics,name+".cpu_graph_vs_native",cpu_rows,native);
        compare(metrics,name+".selected_vs_native",selected_rows,native);
        compare(metrics,name+".selected_vs_cpu_graph",selected_rows,cpu_rows);
        const auto dir = std::filesystem::path(output);
        save(dir/(name+".input.f32"),x.data(),x.size()*4); save(dir/(name+".native_input.q8_0"),aq.data(),aq.size());
        save(dir/(name+".native_hidden.q8_0"),hq.data(),hq.size());
        save_rows(dir,name+".native",native); save_rows(dir,name+".cpu_graph",cpu_rows); save_rows(dir,name+".selected",selected_rows);
        manifest << name << "_input_fnv1a64=" << std::hex << digest(x.data(),x.size()*4) << std::dec << '\n';
    }
    metrics.close(); manifest.close(); require(bool(metrics) && bool(manifest),"evidence write failed");
    const std::string done = "completed finite direct-backend expert measurement; no numerical tolerance acceptance\n";
    save(std::filesystem::path(output)/"COMPLETED",done.data(),done.size());
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    try { return run(argc,argv); }
    catch (const std::exception& e) { std::fprintf(stderr,"expert probe: %s\n",e.what()); return 1; }
}
