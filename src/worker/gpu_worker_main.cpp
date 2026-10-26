#include "strata/worker/gpu_worker.hpp"
#include "wire_sock.hpp"
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
std::atomic<bool> interrupted{false};
void signal_stop(int){interrupted.store(true);}
void usage(){std::fprintf(stderr,"usage: strata-cuda-expert-worker --plan FILE --cache-root DIR "
  "[--port 7432] [--device 0] [--cache-budget-mib 24576] [--reserve-mib 1024] "
  "[--deadline-ms 10000] [--setup-ms 600000] [--tcp-nodelay] [--plaintext-test]\n"
  "  [--bind-ip IPv4 --allow-peer-ip IPv4 [--allow-peer-ip IPv4 ...]]\n"
  "  [--tls-ca FILE --tls-cert FILE --tls-key FILE --tls-peer COORDINATOR_SAN_DNS]\n"
  "  default loopback only; private LAN requires exact coordinator IPs; fixed operator plan; no automatic cleanup\n");}
}
int main(int argc,char** argv){
    strata::worker::GpuWorkerOptions options;std::string plan_path;uint16_t port=7432;int device=0;
    for(int i=1;i<argc;++i){
        const std::string a=argv[i];
        if(a=="--plaintext-test"){options.plaintext_test=true;continue;}
        if(a=="--tcp-nodelay"){options.accepted_tcp_nodelay=true;continue;}
        if(i+1>=argc){usage();return 2;}
        const char* value=argv[++i];
        if(a.rfind("--tls-",0)==0 && !*value){usage();return 2;}
        if(a=="--plan")plan_path=value;
        else if(a=="--cache-root")options.cache_root=value;
        else if(a=="--tls-ca")options.tls.ca_file=value;
        else if(a=="--tls-cert")options.tls.cert_file=value;
        else if(a=="--tls-key")options.tls.key_file=value;
        else if(a=="--tls-peer")options.tls.peer_dns=value;
        else if(a=="--bind-ip")options.bind_ip=value;
        else if(a=="--allow-peer-ip")options.allow_peer_ips.emplace_back(value);
        else{
            errno=0;char* end=nullptr;const auto n=std::strtoull(value,&end,10);
            if(!*value || *value=='-' || *end || errno){usage();return 2;}
            if(a=="--port" && n<=65535)port=(uint16_t)n;
            else if(a=="--device" && n<=15)device=(int)n;
            else if(a=="--cache-budget-mib" && n && n<=(1ull<<30))options.cache_budget=n<<20;
            else if(a=="--reserve-mib" && n>=64 && n<=(1ull<<20))options.runtime_reserve=n<<20;
            else if(a=="--deadline-ms" && n && n<=600000)options.deadline_ms=(int)n;
            else if(a=="--setup-ms" && n && n<=86400000)options.setup_ms=(int)n;
            else{usage();return 2;}
        }
    }
    if(plan_path.empty() || options.cache_root.empty()){usage();return 2;}
    std::string err;
    if(!strata::worker::read_gpu_plan(plan_path,options.plan,err)){std::fprintf(stderr,"cuda-worker: %s\n",err.c_str());return 2;}
    strata::worker::GpuWorkerService service(options,strata::worker::create_cuda_gpu_executor(device));
    if(!service.listen(port,err)){std::fprintf(stderr,"cuda-worker: %s\n",err.c_str());return 1;}
    std::signal(SIGINT,signal_stop);std::signal(SIGTERM,signal_stop);
    std::fprintf(stderr,"cuda-worker listening %s:%u device=%d model=%llx layout=%llx:%llx "
      "placement=%llx:%llx assigned=%llu cache_budget=%llu reserve=%llu CUDA/q8_1 approximate transport=%s accepted_nodelay=%s\n",options.bind_ip.c_str(),port,device,
      (unsigned long long)options.plan.model_id,(unsigned long long)options.plan.layout_id.a,
      (unsigned long long)options.plan.layout_id.b,(unsigned long long)options.plan.placement_id.a,
      (unsigned long long)options.plan.placement_id.b,(unsigned long long)options.plan.assigned_bytes(),
      (unsigned long long)options.cache_budget,(unsigned long long)options.runtime_reserve,options.tls.enabled()?"mTLS":"plaintext-test",options.accepted_tcp_nodelay?"on":"off");
    std::atomic<bool> ended{false};
    std::thread signals([&]{while(!ended.load()&&!interrupted.load())strata::worker::sock::sleep_ms(50);if(interrupted.load())service.stop();});
    service.run();ended.store(true);signals.join();return 0;
}
