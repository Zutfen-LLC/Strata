// Explicit experimental loopback worker; never a production/default backend selection.
#include "strata/worker/vulkan_executor.hpp"
#include "wire_sock.hpp"
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>
namespace {
std::atomic<bool> interrupted{false};
void signal_stop(int) { interrupted.store(true); }
void usage() {
    std::fprintf(stderr,"usage: strata-vulkan-expert-worker --plan FILE --cache-root DIR "
      "[--port 0] [--device 0] [--tensor-budget-mib 16] [--cache-budget-mib 64] "
      "[--deadline-ms 10000] [--setup-ms 600000]\n"
      "  127.0.0.1 only; Q8_0 H2560/FF640 T1 indexed rows; native host Q8 boundaries\n");
}
}
int main(int argc,char** argv) {
    using namespace strata::worker;
    GpuWorkerOptions o; o.vulkan=true; o.runtime_reserve=0; o.cache_budget=64ull<<20;
    std::string path; uint16_t port=0; int device=0; uint64_t budget=16ull<<20;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(i+1 >= argc) { usage(); return 2; }
        const char* value=argv[++i];
        if(arg == "--plan") path=value;
        else if(arg == "--cache-root") o.cache_root=value;
        else {
            errno=0; char* end=nullptr; const auto n=std::strtoull(value,&end,10);
            if(!*value || *value == '-' || *end || errno) { usage(); return 2; }
            if(arg == "--port" && n <= 65535) port=uint16_t(n);
            else if(arg == "--device" && n <= 15) device=int(n);
            else if(arg == "--tensor-budget-mib" && n && n <= (1ull<<20)) budget=n<<20;
            else if(arg == "--cache-budget-mib" && n && n <= (1ull<<30)) o.cache_budget=n<<20;
            else if(arg == "--deadline-ms" && n && n <= 600000) o.deadline_ms=int(n);
            else if(arg == "--setup-ms" && n && n <= 86400000) o.setup_ms=int(n);
            else { usage(); return 2; }
        }
    }
    if(path.empty() || o.cache_root.empty()) { usage(); return 2; }
    std::string err;
    if(!read_gpu_plan(path,o.plan,err)) { std::fprintf(stderr,"vulkan-worker: %s\n",err.c_str()); return 2; }
    GpuWorkerService service(o,create_vulkan_gpu_executor(device,budget));
    if(!service.listen(port,err)) { std::fprintf(stderr,"vulkan-worker: %s\n",err.c_str()); return 1; }
    std::signal(SIGINT,signal_stop); std::signal(SIGTERM,signal_stop);
    std::fprintf(stderr,"vulkan-worker listening 127.0.0.1:%u device=%d tensor_budget=%llu "
      "Vulkan/native-Q8-host-boundaries approximate; total VRAM unbounded by tensor budget\n",
      port,device,(unsigned long long)budget);
    std::atomic<bool> ended{false};
    std::thread signals([&] {
        while(!ended.load() && !interrupted.load()) sock::sleep_ms(50);
        if(interrupted.load()) service.stop();
    });
    service.run(); ended.store(true); signals.join(); return 0;
}
