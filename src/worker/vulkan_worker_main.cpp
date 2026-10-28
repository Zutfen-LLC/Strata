// Explicit experimental worker; never a production/default backend selection.
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
      "[--deadline-ms 10000] [--setup-ms 600000] [--tcp-nodelay]\n"
      "[--bind-ip IPv4 --allow-peer-ip IPv4] [--tls-ca FILE --tls-cert FILE --tls-key FILE --tls-peer SAN_DNS]\n"
      "  Q8_0 H2560/FF640 T1..8 indexed rows; private LAN requires mTLS; native host Q8 boundaries\n");
}
}
int main(int argc,char** argv) {
    using namespace strata::worker;
    GpuWorkerOptions o; o.vulkan=true; o.runtime_reserve=0; o.cache_budget=64ull<<20;
    std::string path; uint16_t port=0; int device=0; uint64_t budget=16ull<<20;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg == "--tcp-nodelay") {o.accepted_tcp_nodelay=true;continue;}
        if(i+1 >= argc) { usage(); return 2; }
        const char* value=argv[++i];
        if(arg.rfind("--tls-",0)==0 && !*value) {usage();return 2;}
        if(arg == "--plan") path=value;
        else if(arg == "--cache-root") o.cache_root=value;
        else if(arg == "--bind-ip") o.bind_ip=value;
        else if(arg == "--allow-peer-ip") o.allow_peer_ips.emplace_back(value);
        else if(arg == "--tls-ca") o.tls.ca_file=value;
        else if(arg == "--tls-cert") o.tls.cert_file=value;
        else if(arg == "--tls-key") o.tls.key_file=value;
        else if(arg == "--tls-peer") o.tls.peer_dns=value;
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
    std::fprintf(stderr,"vulkan-worker listening %s:%u device=%d tensor_budget=%llu "
      "Vulkan/native-Q8-host-boundaries approximate transport=%s accepted_nodelay=%s; total VRAM unbounded by tensor budget\n",
      o.bind_ip.c_str(),port,device,(unsigned long long)budget,o.tls.enabled()?"mTLS":"plaintext-test",o.accepted_tcp_nodelay?"on":"off");
    std::atomic<bool> ended{false};
    std::thread signals([&] {
        while(!ended.load() && !interrupted.load()) sock::sleep_ms(50);
        if(interrupted.load()) service.stop();
    });
    service.run(); ended.store(true); signals.join(); return 0;
}
