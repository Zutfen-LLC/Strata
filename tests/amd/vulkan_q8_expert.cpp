// One deterministic Q8_0 expert. Measurement, not a cross-backend tolerance gate.
#include "../../src/worker/vulkan_expert_graph.hpp"
#include "strata/worker/vulkan_executor.hpp"
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
using namespace strata::worker::vulkan_detail;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE binary32");
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
        const float recomposed = host_swiglu(r.gate[row],r.up[row]);
        require(!std::memcmp(&recomposed,&r.hidden[row],sizeof(float)), "native hidden projection seam mismatch");
    }
    validate(r); return r;
}


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
#include "vulkan_executor_check.hpp"
int run(int argc, char** argv) {
    const uint16_t endian = 1;
    require(*reinterpret_cast<const uint8_t*>(&endian) == 1,"fixture requires little-endian Q8/F32 storage");
    std::string backend,output; int device = -1; bool native_boundaries = false, check_executor = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--executor-check") { require(!check_executor,"duplicate executor check"); check_executor=true; continue; }
        if (arg == "--native-q8-boundaries") { require(!native_boundaries,"duplicate native-boundary control"); native_boundaries = true; continue; }
        require(i+1 < argc,"missing option value"); const std::string value = argv[++i];
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
    require(!check_executor || native_boundaries,"executor check requires --native-q8-boundaries");
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
    std::unique_ptr<ExpertGraph> controlled_graph;
    if (vk_backend || native_boundaries) controlled_graph = std::make_unique<ExpertGraph>(selected,fmt,blob,native_boundaries);
    auto& selected_graph = controlled_graph ? *controlled_graph : cpu_graph;
    std::ofstream manifest(std::filesystem::path(output)/"manifest.txt"), metrics(std::filesystem::path(output)/"metrics.tsv");
    require(bool(manifest) && bool(metrics),"cannot create evidence");
    manifest << "schema=1\nH=2560\nFF=640\nT=1\nweight_type=Q8_0\ncpu_activation=Q8_0\n"
             << "mode=" << (native_boundaries ? "hybrid_native_q8_boundaries" : "raw_f32") << '\n'
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
        const Rows selected_rows = controlled_graph ? controlled_graph->run(x) : cpu_rows;
        const Boundaries boundaries = selected_graph.boundaries();
        if (controlled_graph) exact_repeat(selected_rows,controlled_graph->run(x));
        if (native_boundaries) {
            const auto& again = selected_graph.boundaries();
            require(boundaries.input_q == aq && boundaries.input_q == again.input_q &&
                    boundaries.hidden_q == again.hidden_q && same(boundaries.input_f,again.input_f) &&
                    same(boundaries.down_f,again.down_f),"hybrid boundary bytes changed on repeat");
        }
        if (check_executor && !zero) executor_check(backend,device,output,blob,x,selected_rows.out);
        if (zero) for (float v : selected_rows.out) require(v == 0,"zero expert produced nonzero output");
        else require(std::any_of(native.out.begin(),native.out.end(),[](float v){return v != 0;}),"degenerate signal fixture");
        compare(metrics,name+".cpu_graph_vs_native",cpu_rows,native);
        compare(metrics,name+".selected_vs_native",selected_rows,native);
        compare(metrics,name+".selected_vs_cpu_graph",selected_rows,cpu_rows);
        const auto dir = std::filesystem::path(output);
        save(dir/(name+".input.f32"),x.data(),x.size()*4); save(dir/(name+".native_input.q8_0"),aq.data(),aq.size());
        save(dir/(name+".native_hidden.q8_0"),hq.data(),hq.size());
        save_rows(dir,name+".native",native); save_rows(dir,name+".cpu_graph",cpu_rows); save_rows(dir,name+".selected",selected_rows);
        if (native_boundaries) {
            save(dir/(name+".selected_input.q8_0"),boundaries.input_q.data(),boundaries.input_q.size());
            save(dir/(name+".selected_input.f32"),boundaries.input_f.data(),boundaries.input_f.size()*4);
            save(dir/(name+".selected_hidden.q8_0"),boundaries.hidden_q.data(),boundaries.hidden_q.size());
            save(dir/(name+".selected_down_input.f32"),boundaries.down_f.data(),boundaries.down_f.size()*4);
        }
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
