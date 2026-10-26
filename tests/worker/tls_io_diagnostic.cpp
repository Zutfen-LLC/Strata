// Linux-only, model-free SGW1 diagnostic. Linker wrappers affect this executable only.
// Numbers describe loopback, never LAN throughput or model performance.
#include "tls_fixture.hpp"
#include "strata/worker/gpu_wire.hpp"
#include "wire_sock.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <limits>
#include <thread>
#include <vector>
using namespace strata::worker;
using Clock = std::chrono::steady_clock;
namespace {
struct Sizes {
    uint64_t calls=0,bytes=0,again=0,min=UINT64_MAX,max=0;
    void record(ssize_t result,int error) {
        ++calls;
        if(result<0&&(error==EAGAIN||error==EWOULDBLOCK))++again;
        if(result>0){bytes+=(uint64_t)result;min=std::min(min,(uint64_t)result);max=std::max(max,(uint64_t)result);}
    }
};
struct Counts { Sizes send,recv;uint64_t polls=0,poll_ns=0; };
thread_local Counts* recording=nullptr;
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
struct Fd {sock_fd value=kBadFd;~Fd(){sock::close_fd(value);}};
struct Join {std::thread& thread;~Join(){if(thread.joinable())thread.join();}};
struct Release {std::atomic<bool>& flag;~Release(){flag.store(true);}};
struct Result {Counts client,worker;double ms=0;};
struct Frames {std::vector<uint8_t> request,reply;int T,K;};
Frames frames(int T,int K) {
    GpuPlan p;p.model_id=1;p.layout_id={2,3};p.placement_id={4,5};p.n_layers=1;p.n_expert=10;p.n_embd=2560;
    InferRequest r;r.model_id=p.model_id;r.partition_id=p.placement_id.a;r.session=7;r.sequence=1;r.T=T;r.K=K;r.layer=0;
    r.activations.assign((size_t)T*p.n_embd,0.125f);r.mask.assign(T*K,1);r.weights.assign(T*K,1.f/K);
    for(int i=0;i<T*K;++i)r.ids.push_back(i%10);
    Frames f;f.T=T;f.K=K;require(gpu_codec::encode_request(p,r,f.request),"request encode");
    InferRequest decoded;require(gpu_codec::decode_request(p,f.request,decoded),"request codec");
    EntryRowsReply rows;rows.session=r.session;rows.sequence=r.sequence;
    for(int i=0;i<T*K;++i)rows.indices.push_back(i);
    rows.rows.assign((size_t)T*K*p.n_embd,0.25f);
    require(gpu_codec::encode_rows_reply(p,blob_digest(f.request),r,rows,f.reply),"reply encode");
    EntryRowsReply decoded_rows;
    require(gpu_codec::decode_rows_reply(p,blob_digest(f.request),r,f.reply,decoded_rows)&&decoded_rows.rows==rows.rows,"reply codec");
    return f;
}
Result run(const TlsFixture& certificates,const Frames& f,bool tls,bool nodelay,int rounds) {
    Fd listener,client;uint16_t port=0;std::string err;
    require(sock::listen_loopback(port,listener.value,err),err);
    require(sock::connect_numeric_controlled("127.0.0.1",port,2000,{},client.value,err),err);
    Result result;std::exception_ptr worker_error;std::atomic<bool> measured{false};
    std::thread worker([&]{try{
        Fd peer;require(sock::accept_one(listener.value,peer.value,2000,nodelay),"accept");
        int actual=-1;socklen_t length=sizeof actual;
        require(getsockopt(peer.value,IPPROTO_TCP,TCP_NODELAY,&actual,&length)==0 && actual==int(nodelay),"accepted NODELAY policy");
        TlsSession session;std::string worker_why;
        if(tls)require(session.handshake(peer.value,certificates.config("worker","coordinator.test"),true,2000,{},worker_why),worker_why);
        for(int i=-2;i<rounds;++i){
            if(i==0)while(!measured.load())std::this_thread::yield();
            recording=i>=0?&result.worker:nullptr;
            WireFrame in;std::string why;
            require(gpu_codec::receive(peer.value,in,kGpuFrameMax,2000,{},why,0,tls?&session:nullptr),why);
            require(in.msg==GPU_INFER_ROWS&&in.status==ST_OK&&in.payload==f.request,"request bytes");
            require(gpu_codec::send(peer.value,GPU_INFER_ROWS_ACK,ST_OK,f.reply,2000,{},why,tls?&session:nullptr),why);
        }
        recording=nullptr;
    }catch(...){recording=nullptr;worker_error=std::current_exception();}});
    Join join{worker};Release release{measured};TlsSession session;
    if(tls)require(session.handshake(client.value,certificates.config("coordinator","worker.test"),false,2000,{},err),err);
    Clock::time_point began;
    for(int i=-2;i<rounds;++i){
        if(i==0){recording=&result.client;began=Clock::now();measured.store(true);}
        require(gpu_codec::send(client.value,GPU_INFER_ROWS,ST_OK,f.request,2000,{},err,tls?&session:nullptr),err);
        WireFrame reply;require(gpu_codec::receive(client.value,reply,kGpuFrameMax,2000,{},err,0,tls?&session:nullptr),err);
        require(reply.msg==GPU_INFER_ROWS_ACK&&reply.status==ST_OK&&reply.payload==f.reply,"reply bytes");
    }
    result.ms=std::chrono::duration<double,std::milli>(Clock::now()-began).count();recording=nullptr;
    worker.join();if(worker_error)std::rethrow_exception(worker_error);return result;
}
void print(const char* mode,bool nodelay,const Frames& f,int rounds,int trial,const char* side,const Result& r,const Counts& c){
    auto minimum=[](const Sizes& s){return s.min==UINT64_MAX?0:s.min;};
    std::printf("%s,%d,%d,%d,%zu,%zu,%d,%d,%s,%.6f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.6f\n",
      mode,nodelay,f.T,f.K,f.request.size()+kHeaderBytes,f.reply.size()+kHeaderBytes,rounds,trial,side,r.ms,
      (unsigned long long)c.send.calls,(unsigned long long)c.send.bytes,(unsigned long long)minimum(c.send),(unsigned long long)c.send.max,(unsigned long long)c.send.again,
      (unsigned long long)c.recv.calls,(unsigned long long)c.recv.bytes,(unsigned long long)minimum(c.recv),(unsigned long long)c.recv.max,(unsigned long long)c.recv.again,
      (unsigned long long)c.polls,c.poll_ns/1e6);
}
}
extern "C" ssize_t __real_send(int,const void*,size_t,int);
extern "C" ssize_t __real_recv(int,void*,size_t,int);
extern "C" int __real_poll(pollfd*,nfds_t,int);
extern "C" ssize_t __wrap_send(int fd,const void* p,size_t n,int flags){
    const auto result=__real_send(fd,p,n,flags);const int error=errno;
    if(recording)recording->send.record(result,error);
    errno=error;return result;
}
extern "C" ssize_t __wrap_recv(int fd,void* p,size_t n,int flags){
    const auto result=__real_recv(fd,p,n,flags);const int error=errno;
    if(recording)recording->recv.record(result,error);
    errno=error;return result;
}
extern "C" int __wrap_poll(pollfd* f,nfds_t n,int timeout){
    const auto began=recording?Clock::now():Clock::time_point{};const int result=__real_poll(f,n,timeout);const int error=errno;
    if(recording){++recording->polls;recording->poll_ns+=(uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-began).count();}
    errno=error;return result;
}
int main(int argc,char** argv){try{
    int rounds=16;
    if(argc==3&&std::string(argv[1])=="--rounds"){
        char* end=nullptr;long n=std::strtol(argv[2],&end,10);require(*argv[2]&&!*end&&n>=1&&n<=128,"rounds 1..128");rounds=(int)n;
    }else require(argc==1,"usage: tls_io_diagnostic [--rounds 1..128]");
    TlsFixture certificates;
    std::puts("# LOOPBACK ONLY: pre-encoded valid SGW1 frames; no model/executor; handshake + two warmups excluded; instrumented timings; no pass/fail timing threshold");
    std::puts("mode,accepted_nodelay,T,K,request_frame_bytes,reply_frame_bytes,rounds,trial,side,elapsed_ms,send_calls,sent_bytes,send_min,send_max,send_eagain,recv_calls,received_bytes,recv_min,recv_max,recv_eagain,poll_calls,poll_wait_ms");
    for(auto shape:std::vector<std::pair<int,int>>{{1,1},{1,2},{1,8},{8,10}}){
        const auto f=frames(shape.first,shape.second);
        for(int trial=0;trial<2;++trial)for(bool nodelay:{false,true})for(bool tls:{false,true}){
            const auto r=run(certificates,f,tls,nodelay,rounds);
            print(tls?"tls":"plain",nodelay,f,rounds,trial,"client",r,r.client);
            print(tls?"tls":"plain",nodelay,f,rounds,trial,"worker",r,r.worker);
        }
    }
    return 0;
}catch(const std::exception& e){recording=nullptr;std::fprintf(stderr,"FAIL diagnostic: %s\n",e.what());return 1;}}
