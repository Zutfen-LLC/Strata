// CPU-only SGW1 transport candidate. No executor library, weights or CUDA.
#include "strata/worker/gpu_wire.hpp"
#include "wire_sock.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <netinet/tcp.h>
#include <sys/resource.h>
#include <sys/socket.h>

using namespace strata::worker;
using Clock = std::chrono::steady_clock;
namespace {
constexpr int io_ms = 2000, samples = 8;
void require(bool value, const std::string& why) { if (!value) throw std::runtime_error(why); }
struct Fd {
    sock_fd value = kBadFd;
    Fd() = default;
    explicit Fd(sock_fd v): value(v) {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    ~Fd() { sock::close_fd(value); }
};
struct Join {
    std::thread& thread;
    ~Join() { if (thread.joinable()) thread.join(); }
};
struct Pair {
    Fd left, right;
    Pair() {
        int fds[2]; require(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair");
        left.value = fds[0]; right.value = fds[1];
    }
};
void small_buffer(sock_fd fd) {
    int bytes = 4096;
    require(::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bytes, sizeof bytes) == 0, "small send buffer");
}
// Frozen bf51f09e sender, fixture-only: isolate framing from all other changes.
bool split_send(sock_fd fd, uint16_t msg, const std::vector<uint8_t>& p, std::string& err) {
    if (p.size() > kGpuFrameMax) return false;
    std::vector<uint8_t> h;
    gpu_codec::put(h, kGpuWireMagic, 4); gpu_codec::put(h, msg, 2);
    gpu_codec::put(h, ST_OK, 2); gpu_codec::put(h, p.size());
    const auto began = Clock::now();
    const auto cancel = [] { return false; }; // Same cancellation polling as the real GPU exchange.
    auto remaining = [&] { return io_ms - (int)std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-began).count(); };
    if (!sock::send_all_controlled(fd, h.data(), h.size(), io_ms, cancel) ||
        (!p.empty() && (remaining() <= 0 || !sock::send_all_controlled(fd, p.data(), p.size(), remaining(), cancel)))) {
        err = "original split send failed"; return false;
    }
    return true;
}
bool emit(int arm, sock_fd fd, uint16_t msg, const std::vector<uint8_t>& p, std::string& err) {
    return arm == 1 ? gpu_codec::send(fd, msg, ST_OK, p, io_ms, [] {return false;}, err) : split_send(fd, msg, p, err);
}
std::vector<uint8_t> bytes(size_t n) {
    std::vector<uint8_t> p(n);
    for (size_t i=0; i<n; ++i) p[i] = (uint8_t)(i*37+11);
    return p;
}
void wire_bytes(size_t n, uint16_t status=ST_OK) {
    Pair pair; small_buffer(pair.left.value);
    const auto payload = bytes(n);
    std::vector<uint8_t> expected;
    // Independent wire oracle: CPU frame layout, replacing only SEW1 magic.
    encode_frame(expected, GPU_SETUP, status, payload.data(), payload.size());
    expected[0]='S'; expected[1]='G'; expected[2]='W'; expected[3]='1';
    bool sent=false; std::string error;
    std::thread sender([&] { sent=gpu_codec::send(pair.left.value,GPU_SETUP,status,payload,io_ms,{},error); });
    Join join{sender};
    std::vector<uint8_t> actual(expected.size());
    // A much smaller receiver chunk plus SO_SNDBUF forces partial transport I/O.
    for (size_t off=0; off<actual.size();) {
        const size_t count=std::min<size_t>(997,actual.size()-off);
        require(sock::recv_all(pair.right.value,actual.data()+off,count,io_ms),"fragmented raw receive"); off+=count;
    }
    sender.join(); require(sent,error); require(actual==expected,"exact original header/payload bytes");
}
void fragmented_receive() {
    Pair pair; const auto payload=bytes(8192); std::vector<uint8_t> frame;
    encode_frame(frame,GPU_INFER,ST_OK,payload.data(),payload.size()); frame[1]='G';
    bool sent=true;
    std::thread sender([&] {
        for (size_t off=0; off<frame.size();) {
            const size_t n=std::min<size_t>(7,frame.size()-off);
            if (!sock::send_all(pair.left.value,frame.data()+off,n,io_ms)) {sent=false;break;} off+=n;
        }
    });
    Join join{sender}; WireFrame result; std::string error;
    require(gpu_codec::receive(pair.right.value,result,kGpuFrameMax,io_ms,{},error),error);
    sender.join(); require(sent && result.msg==GPU_INFER && result.payload==payload,"fragmented codec receive");
}
void refusal_checks() {
    Pair pair; std::string error; const std::vector<uint8_t> small{1,2,3};
    require(!gpu_codec::send(pair.left.value,GPU_HELLO,ST_OK,small,0,{},error),"zero deadline refused");
    require(!gpu_codec::send(pair.left.value,GPU_HELLO,ST_OK,small,io_ms,[]{return true;},error),"pre-write cancellation");
    auto large=bytes(kGpuFrameMax+1);
    require(!gpu_codec::send(pair.left.value,GPU_SETUP,ST_OK,large,io_ms,{},error),"oversized payload refused");
    uint8_t b; require(::recv(pair.right.value,&b,1,MSG_DONTWAIT)<0 && (errno==EAGAIN || errno==EWOULDBLOCK),"refusals wrote no bytes");
    large.resize(kGpuFrameMax); small_buffer(pair.left.value);
    int checks=0;
    require(!gpu_codec::send(pair.left.value,GPU_SETUP,ST_OK,large,io_ms,[&]{return ++checks>=3;},error),"mid-write cancellation");
    require(checks>=3,"cancellation rechecked during partial write");
    Pair blocked; small_buffer(blocked.left.value);
    require(!gpu_codec::send(blocked.left.value,GPU_SETUP,ST_OK,large,25,{},error),"blocked peer exhausts total deadline");
}
void close_cases() {
    for (int kind=0; kind<4; ++kind) {
        Pair pair; std::vector<uint8_t> raw; const std::vector<uint8_t> payload{1,2,3,4};
        encode_frame(raw,GPU_HELLO,ST_OK,payload.data(),payload.size()); raw[1]='G';
        if (kind==0) raw.clear();
        if (kind==1) raw.resize(8);
        if (kind==2) raw.resize(kHeaderBytes+2);
        if (kind==3) raw[0]^=1;
        require(sock::send_all(pair.left.value,raw.data(),raw.size(),io_ms),"close-case send");
        sock::close_fd(pair.left.value);
        WireFrame result; std::string error;
        require(!gpu_codec::receive(pair.right.value,result,kGpuFrameMax,io_ms,{},error),"bad/closed frame never accepted");
        const char* names[]={"ordinary_eof","truncated_header","truncated_payload","malformed_magic"};
        std::printf("CLOSE source=%s sent_bytes=%zu receive=false error=%s\n",names[kind],raw.size(),error.c_str());
    }
}
void correctness() {
    for (size_t n : {size_t(0),size_t(12),size_t(10394),size_t(kGpuPlanMax+8),size_t(kGpuFrameMax)}) wire_bytes(n);
    wire_bytes(12,ST_BAD_REQUEST);
    fragmented_receive(); refusal_checks(); close_cases();
    std::puts("GPU_FRAME_CORRECTNESS_PASS_V1 live_models=0 cuda=0");
}
GpuPlan plan() {
    // Codec identity/geometry only; no asset plan validation, cache or weights.
    GpuPlan p; p.model_id=1; p.layout_id={2,3}; p.placement_id={4,5};
    p.n_layers=1; p.n_expert=10; p.n_embd=kMaxNEmbd; p.n_ff=kMaxNFf; return p;
}
InferRequest request(const GpuPlan& p,int T,uint64_t sequence) {
    InferRequest r; r.session=123; r.model_id=p.model_id; r.partition_id=p.placement_id.a;
    r.sequence=sequence; r.layer=0; r.T=(uint8_t)T; r.K=10;
    r.mask.assign((size_t)T*10,1); r.ids.resize(r.mask.size()); r.weights.assign(r.mask.size(),0.1f);
    for (size_t i=0;i<r.ids.size();++i) r.ids[i]=(uint16_t)(i%10);
    r.activations.resize((size_t)T*p.n_embd);
    for (size_t i=0;i<r.activations.size();++i) r.activations[i]=(float)(int(i%11)-5)*0.01f;
    return r;
}
void rows_transport() {
    const auto p=plan();auto r=request(p,kMaxT,1);r.K=kMaxK;
    r.mask.assign((size_t)r.T*r.K,1);
    EntryRowsReply rows;rows.session=r.session;rows.sequence=r.sequence;
    for(size_t i=0;i<r.mask.size();++i)rows.indices.push_back((uint16_t)i);
    rows.rows.resize(rows.indices.size()*p.n_embd);
    for(size_t i=0;i<rows.rows.size();++i)rows.rows[i]=i%7 ? (float)(int(i%31)-15)/128.f : -0.f;
    const Digest128 digest{17,19};std::vector<uint8_t> wire;
    require(gpu_codec::encode_rows_reply(p,digest,r,rows,wire),"maximal row reply encode");
    require(wire.size()==64+kEntryRowsPayloadMax,"row operation cap exact maximum");
    Pair pair;std::string error;bool sent=false;
    std::thread sender([&]{sent=gpu_codec::send(pair.left.value,GPU_INFER_ROWS_ACK,ST_OK,wire,io_ms,{},error);});
    Join join{sender};WireFrame frame;std::string why;
    require(gpu_codec::receive(pair.right.value,frame,kGpuFrameMax,io_ms,{},why),why);
    sender.join();require(sent,error);EntryRowsReply result;
    require(frame.payload==wire && gpu_codec::decode_rows_reply(p,digest,r,frame.payload,result),"maximum rows cross actual socket");
    require(result.indices==rows.indices && result.rows.size()==rows.rows.size() &&
            !std::memcmp(result.rows.data(),rows.rows.data(),rows.rows.size()*sizeof(float)),"raw F32 row bytes and all original indices survive transport");
    auto bad=wire;bad[48]^=1;
    require(!gpu_codec::decode_rows_reply(p,digest,r,bad,result),"row request digest mismatch refused");
    bad=wire;bad[32]^=1;
    require(!gpu_codec::decode_rows_reply(p,digest,r,bad,result),"row placement mismatch refused");
}
std::vector<uint8_t> reply(const GpuPlan& p,const WireFrame& in) {
    if (in.msg==GPU_STATUS) {
        require(in.payload==gpu_codec::binding(123,p),"dummy status identity");
        auto out=in.payload; gpu_codec::put(out,1,4); auto counters=gpu_codec::counters(GpuCounters{});
        out.insert(out.end(),counters.begin(),counters.end()); return out;
    }
    require(in.msg==GPU_INFER,"dummy inference message"); InferRequest r;
    require(gpu_codec::decode_request(p,in.payload,r),"real inference request codec");
    InferReply out; out.session=r.session; out.sequence=r.sequence;
    out.partials.assign((size_t)r.T*p.n_embd,0.125f+(float)r.sequence/1024.f);
    return gpu_codec::encode_reply(p,blob_digest(in.payload),out);
}
constexpr int shapes[]={0,1,2,kMaxT}; // control, real T=1/2, maximum compiled T; K=10, H=2560
const char* arms[]={"original_split","contiguous","split_accepted_nodelay"};
void serve(sock_fd listener) {
    const auto p=plan();
    for (int arm=0; arm<3; ++arm) {
        Fd peer; require(sock::accept_one(listener,peer.value,10000),"dummy accept");
        if (arm==2) { int one=1; require(::setsockopt(peer.value,IPPROTO_TCP,TCP_NODELAY,&one,sizeof one)==0,"fixture-only accepted NODELAY"); }
        int actual=-1; socklen_t n=sizeof actual;
        require(::getsockopt(peer.value,IPPROTO_TCP,TCP_NODELAY,&actual,&n)==0 && actual==(arm==2),"isolated accepted socket option");
        WireFrame hello; std::string error;
        require(gpu_codec::receive(peer.value,hello,kGpuFrameMax,io_ms,{},error),error);
        require(hello.msg==GPU_HELLO && hello.payload==std::vector<uint8_t>{(uint8_t)arm},"arm handshake");
        require(emit(arm,peer.value,GPU_HELLO_ACK,hello.payload,error),error);
        for (int T : shapes) for (int i=0;i<=samples;++i) {
            WireFrame in; require(gpu_codec::receive(peer.value,in,kGpuFrameMax,io_ms,{},error),error);
            require(in.msg==(T ? GPU_INFER : GPU_STATUS),"fixed shape order");
            InferRequest decoded;
            if (T) require(gpu_codec::decode_request(p,in.payload,decoded) && decoded.T==T && decoded.K==10,"fixed real inference shape");
            const auto out=reply(p,in); require(emit(arm,peer.value,in.msg+1,out,error),error);
        }
    }
}
void client(uint16_t port) {
    const auto p=plan();
    for (int arm=0; arm<3; ++arm) {
        Fd fd; std::string error;
        require(sock::connect_numeric_controlled("127.0.0.1",port,io_ms,{},fd.value,error),error);
        int actual=0; socklen_t option_bytes=sizeof actual;
        require(::getsockopt(fd.value,IPPROTO_TCP,TCP_NODELAY,&actual,&option_bytes)==0 && actual==1,"existing client NODELAY");
        require(emit(arm,fd.value,GPU_HELLO,{(uint8_t)arm},error),error);
        WireFrame hello; require(gpu_codec::receive(fd.value,hello,kGpuFrameMax,io_ms,{},error),error);
        require(hello.msg==GPU_HELLO_ACK && hello.payload==std::vector<uint8_t>{(uint8_t)arm},"handshake reply");
        const auto began=Clock::now(); double sum=0; uint64_t sequence=0;
        for (int T : shapes) {
            std::vector<double> times; size_t request_bytes=0,response_bytes=0;
            for (int i=0;i<=samples;++i) {
                WireFrame in; in.msg=T ? GPU_INFER : GPU_STATUS;
                const auto r=request(p,T ? T : 1,++sequence);
                if (T) require(gpu_codec::encode_request(p,r,in.payload),"inference encode");
                else in.payload=gpu_codec::binding(123,p);
                const auto expected=reply(p,in); const auto start=Clock::now();
                require(emit(arm,fd.value,in.msg,in.payload,error),error);
                WireFrame out; require(gpu_codec::receive(fd.value,out,kGpuFrameMax,io_ms,{},error),error);
                require(out.msg==in.msg+1 && out.status==ST_OK && out.payload==expected,"identical dummy reply bytes");
                if (T) { InferReply decoded; require(gpu_codec::decode_reply(p,blob_digest(in.payload),r,out.payload,decoded),"real reply validation"); }
                const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
                if (i) { times.push_back(ms); sum+=ms; }
                request_bytes=kHeaderBytes+in.payload.size(); response_bytes=kHeaderBytes+out.payload.size();
            }
            std::sort(times.begin(),times.end());
            const double median=(times[samples/2-1]+times[samples/2])/2;
            std::printf("LATENCY arm=%s T=%d K=%d H=%d samples=%d warmup=1 request_bytes=%zu reply_bytes=%zu median_ms=%.3f p95_ms=%.3f max_ms=%.3f\n",
                arms[arm],T,T ? 10 : 0,kMaxNEmbd,samples,request_bytes,response_bytes,median,
                times[(size_t)std::ceil(0.95*samples)-1],times.back());
        }
        std::printf("ARM arm=%s measured_rtt_sum_ms=%.3f elapsed_with_warmups_ms=%.3f\n",arms[arm],sum,
            std::chrono::duration<double,std::milli>(Clock::now()-began).count());
    }
    std::puts("GPU_FRAME_LATENCY_COMPLETE_V1 timing_assertions=0 models=0 cuda=0");
}
uint16_t port(const char* text) {
    char* end=nullptr; errno=0; const auto n=std::strtoul(text,&end,10);
    require(*text && !errno && !*end && n>0 && n<=65535 && *text!='-',"nonzero existing loopback port"); return (uint16_t)n;
}
} // namespace
int main(int argc,char** argv) {
    const rlimit memory{256ull<<20,256ull<<20}, cpu{10,10}, output{1ull<<20,1ull<<20};
    if (::setrlimit(RLIMIT_AS,&memory) || ::setrlimit(RLIMIT_CPU,&cpu) || ::setrlimit(RLIMIT_FSIZE,&output)) return 2;
    ::alarm(60);
    try {
        if (argc==1) { correctness(); rows_transport(); return 0; }
        if (argc==2 && std::string(argv[1])=="--latency") {
            Fd listener; uint16_t bound=0; std::string error;
            require(sock::listen_loopback(bound,listener.value,error),error); std::exception_ptr failure;
            std::thread server([&] {try {serve(listener.value);} catch (...) {failure=std::current_exception();}});
            Join join{server}; client(bound); server.join(); if (failure) std::rethrow_exception(failure); return 0;
        }
        if (argc==3 && std::string(argv[1])=="--serve") {
            Fd listener; auto bound=port(argv[2]); std::string error;
            require(sock::listen_loopback(bound,listener.value,error),error);
            std::printf("GPU_FRAME_DUMMY_LISTENING_V1 address=127.0.0.1 port=%u\n",bound); std::fflush(stdout);
            serve(listener.value); std::puts("GPU_FRAME_DUMMY_SERVER_COMPLETE_V1"); return 0;
        }
        if (argc==3 && std::string(argv[1])=="--connect") { client(port(argv[2])); return 0; }
        throw std::runtime_error("usage: gpu_frame_test [--latency | --serve PORT | --connect PORT]");
    } catch (const std::exception& error) { std::fprintf(stderr,"FAIL %s\n",error.what()); return 1; }
}
